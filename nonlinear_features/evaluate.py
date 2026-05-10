"""Evaluation: restricted R², support size, receptive field spread, Ising coupling."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .sae import TopKSAE
from .manifolds import ManifoldInstance


@dataclass
class ManifoldEvalResult:
    """Evaluation results for a single manifold instance."""
    type_name: str
    variant_idx: int
    embedding_dim: int  # k_i
    restricted_r2: dict[int, float]  # n_atoms -> R² score
    support_size: int
    receptive_field_spread: float
    greedy_atom_indices: list[int]


def greedy_atom_selection(
    decoder_weights: torch.Tensor,  # (d, c) - columns are decoder directions
    true_contributions: torch.Tensor,  # (n_i, d) - manifold contributions
    n_select: int,
) -> list[int]:
    """Greedily select decoder atoms that explain the most variance of true_contributions.

    At each step, pick the atom whose decoder direction explains the most
    residual variance, then project out that direction.
    """
    residual = true_contributions.clone()  # (n_i, d)
    selected = []

    for _ in range(n_select):
        # Project residual onto each decoder column
        # decoder_weights: (d, c), residual: (n_i, d)
        projections = residual @ decoder_weights  # (n_i, c)

        # Variance explained by each atom
        var_explained = projections.var(dim=0) * decoder_weights.pow(2).sum(dim=0)
        # Actually: project residual onto each unit-norm column, measure residual reduction
        # For unit-norm columns: variance explained = var(residual @ d_j)
        var_explained = projections.pow(2).sum(dim=0)  # sum of squared projections

        # Zero out already-selected atoms
        for idx in selected:
            var_explained[idx] = -1

        best_atom = var_explained.argmax().item()
        selected.append(best_atom)

        # Remove this atom's contribution from residual
        d_j = decoder_weights[:, best_atom]  # (d,)
        proj_coeffs = residual @ d_j  # (n_i,)
        residual = residual - proj_coeffs.unsqueeze(-1) * d_j.unsqueeze(0)

    return selected


def compute_restricted_r2(
    model: TopKSAE,
    eval_data: torch.Tensor,  # (n_eval, d)
    active_masks: torch.Tensor,  # (n_eval, m) bool
    contributions: torch.Tensor,  # (n_eval, m, d)
    instances: list[ManifoldInstance],
    n_atoms_range: tuple[int, int] = (-2, 3),  # relative to k_i
    device: str = "cpu",
) -> list[ManifoldEvalResult]:
    """Compute restricted R² for all manifold instances.

    Following the paper (eq 14): for each manifold, greedily select decoder
    directions by residual variance, then measure how well the restricted
    codes (projected through those decoder directions) reconstruct the
    manifold's true contributions.

    We use an affine fit (codes @ decoder_cols + bias) to account for the
    fact that SAE codes are non-negative and may have offsets.
    """
    model = model.to(device).eval()
    eval_data = eval_data.to(device)
    active_masks = active_masks.to(device)
    contributions = contributions.to(device)

    decoder_weights = model.W_dec.weight.data  # (d, c)

    # Encode all eval data
    with torch.no_grad():
        codes = model.encode(eval_data)  # (n_eval, c)

    results = []

    for i, inst in enumerate(instances):
        # Select rows where manifold i is active
        mask = active_masks[:, i]
        if mask.sum() < 10:
            continue

        codes_i = codes[mask]  # (n_i, c)
        true_i = contributions[mask, i]  # (n_i, d)

        k_i = inst.embedding_dim
        max_atoms = k_i + n_atoms_range[1]
        min_atoms = max(1, k_i + n_atoms_range[0])

        # Greedy atom selection (based on decoder directions explaining true variance)
        selected = greedy_atom_selection(decoder_weights, true_i, max_atoms)

        # Compute R² for each number of atoms
        mean_i = true_i.mean(dim=0)
        total_var = (true_i - mean_i).pow(2).sum().item()

        r2_scores = {}
        for n in range(min_atoms, max_atoms + 1):
            atoms_n = selected[:n]

            # Restricted decode: project codes through selected decoder directions
            # Use affine reconstruction: recon = codes_restricted @ D_selected^T + bias
            # where bias absorbs the mean offset from non-negative codes
            D_selected = decoder_weights[:, atoms_n]  # (d, n)
            codes_restricted = codes_i[:, atoms_n]  # (n_i, n)

            # Least-squares affine fit: minimize ||true_i - codes @ D^T - b||²
            # Equivalent to centering and fitting
            codes_centered = codes_restricted - codes_restricted.mean(dim=0, keepdim=True)
            true_centered = true_i - mean_i

            # Reconstruct via decoder directions with centered codes
            recon_centered = codes_centered @ D_selected.T  # (n_i, d)
            residual_var = (true_centered - recon_centered).pow(2).sum().item()
            r2 = 1 - residual_var / max(total_var, 1e-10)
            r2_scores[n] = r2

        # Support size: unique atoms firing on >=10% of manifold's points
        firing_counts = (codes_i.abs() > 0).float().sum(dim=0)  # (c,)
        threshold = 0.1 * mask.sum().item()
        min_count = 30
        support_atoms = ((firing_counts >= threshold) & (firing_counts >= min_count))
        support_size = support_atoms.sum().item()

        # Receptive field spread
        rf_spread = compute_receptive_field_spread(
            codes_i, true_i, support_atoms, device
        )

        results.append(ManifoldEvalResult(
            type_name=inst.type_name,
            variant_idx=inst.variant_idx,
            embedding_dim=k_i,
            restricted_r2=r2_scores,
            support_size=int(support_size),
            receptive_field_spread=rf_spread,
            greedy_atom_indices=selected,
        ))

    return results


def compute_receptive_field_spread(
    codes_i: torch.Tensor,  # (n_i, c)
    true_i: torch.Tensor,  # (n_i, d)
    support_atoms: torch.Tensor,  # (c,) bool
    device: str = "cpu",
    max_points_for_distance: int = 2000,
) -> float:
    """Compute median receptive field spread across support atoms."""
    support_indices = support_atoms.nonzero(as_tuple=True)[0]
    if len(support_indices) == 0:
        return 0.0

    # Manifold's own mean pairwise distance (subsample for efficiency)
    n = min(len(true_i), max_points_for_distance)
    subsample = true_i[:n]
    dists = torch.cdist(subsample, subsample)
    manifold_mean_dist = dists.sum() / (n * (n - 1))

    if manifold_mean_dist < 1e-10:
        return 0.0

    spreads = []
    for j in support_indices:
        # Points where atom j fires
        firing_mask = codes_i[:, j].abs() > 0
        if firing_mask.sum() < 2:
            continue
        points = true_i[firing_mask]
        n_pts = min(len(points), max_points_for_distance)
        pts = points[:n_pts]
        d = torch.cdist(pts, pts)
        mean_pairwise = d.sum() / (n_pts * (n_pts - 1)) if n_pts > 1 else 0.0
        spreads.append(mean_pairwise.item())

    if not spreads:
        return 0.0

    median_spread = sorted(spreads)[len(spreads) // 2]
    return median_spread / manifold_mean_dist.item()


def compute_ising_coupling(
    codes: torch.Tensor,  # (n, c)
    regularization: float = 0.01,
) -> torch.Tensor:
    """Fit pairwise Ising model via pseudo-likelihood maximization.

    Returns coupling matrix J (c, c).

    Simplified version using correlation-based approximation for speed.
    For the full PLM approach, use scipy or dedicated Ising fitting libraries.
    """
    # Binarize codes
    s = (codes.abs() > 0).float()  # (n, c)

    # Center
    s_centered = s - s.mean(dim=0, keepdim=True)

    # Empirical covariance
    n = s.shape[0]
    cov = (s_centered.T @ s_centered) / n  # (c, c)

    # Precision matrix (inverse covariance) as proxy for conditional independence
    # Add regularization for stability
    eye = torch.eye(cov.shape[0], device=cov.device)
    precision = torch.linalg.inv(cov + regularization * eye)

    # Coupling matrix: symmetrize and zero diagonal
    J = -(precision - torch.diag(precision.diag()))
    J = (J + J.T) / 2
    J.fill_diagonal_(0)

    return J


def aggregate_results(
    results: list[ManifoldEvalResult],
) -> dict:
    """Aggregate evaluation results across manifold instances."""
    # R² at embedding dimension for each manifold
    r2_at_ki = []
    for r in results:
        k_i = r.embedding_dim
        if k_i in r.restricted_r2:
            r2_at_ki.append(r.restricted_r2[k_i])

    avg_support_size = sum(r.support_size for r in results) / max(len(results), 1)
    avg_rf_spread = sum(r.receptive_field_spread for r in results) / max(len(results), 1)

    return {
        "mean_r2_at_ki": sum(r2_at_ki) / max(len(r2_at_ki), 1) if r2_at_ki else 0.0,
        "r2_at_ki_values": r2_at_ki,
        "avg_support_size": avg_support_size,
        "avg_receptive_field_spread": avg_rf_spread,
        "n_manifolds_evaluated": len(results),
    }
