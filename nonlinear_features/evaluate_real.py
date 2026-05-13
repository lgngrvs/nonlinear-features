"""Evaluation for real model activations (non-synthetic).

Restricted R² is computed in the **concept-specific subspace** defined by the
manifold labels.  For a 1-D concept like temperature we find the activation
direction most correlated with the label and measure R² there.  For multi-D
concepts (colors: HSV) we orthogonalize the per-label-dimension regression
directions.  Atoms are selected by their code–label correlation rather than
purely geometric decoder alignment.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class RealManifoldResult:
    name: str
    n_samples: int
    embedding_dim_estimate: int
    restricted_r2: dict[int, float]
    support_size: int
    receptive_field_spread: float
    greedy_atom_indices: list[int]
    pca_var_explained: list[float]


# ---------------------------------------------------------------------------
# Concept-direction helpers
# ---------------------------------------------------------------------------

def _expand_labels(labels: torch.Tensor) -> torch.Tensor:
    """Expand labels to handle circular dimensions.

    Multi-dim labels whose first dimension spans [0, 1] (e.g. HSV hue) are
    expanded by replacing that dim with sin(2π*h) and cos(2π*h), so that
    linear correlation captures the circular structure.
    """
    if labels.ndim == 1:
        return labels
    first = labels[:, 0]
    if first.min() >= -0.05 and first.max() <= 1.05 and first.std() > 0.15:
        angle = 2 * torch.pi * first
        expanded = [torch.sin(angle), torch.cos(angle)]
        for i in range(1, labels.shape[1]):
            expanded.append(labels[:, i])
        return torch.stack(expanded, dim=1)
    return labels


def compute_concept_direction(
    acts_centered: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Return (d_in, k) orthonormal directions most correlated with labels.

    For scalar labels (k=1): OLS regression direction.
    For multi-dim labels (k>1): one OLS direction per (expanded) label dimension,
    QR-orthogonalized.  Circular first dimensions (e.g. hue in [0,1]) are
    automatically expanded to sin/cos before regression.
    """
    if labels.ndim == 1:
        lnorm = (labels - labels.mean()) / (labels.std() + 1e-8)
        d = acts_centered.T @ lnorm / len(lnorm)
        d = d / (d.norm() + 1e-10)
        return d.unsqueeze(1)  # (d_in, 1)

    labels_exp = _expand_labels(labels)
    k = labels_exp.shape[1]
    dirs = []
    for i in range(k):
        li = labels_exp[:, i]
        li = (li - li.mean()) / (li.std() + 1e-8)
        d = acts_centered.T @ li / len(li)
        d = d / (d.norm() + 1e-10)
        dirs.append(d)
    D = torch.stack(dirs, dim=1)  # (d_in, k)
    Q, _ = torch.linalg.qr(D)
    return Q[:, :k]  # (d_in, k)


def select_atoms_by_label_correlation(
    codes: torch.Tensor,
    labels: torch.Tensor,
    n_select: int,
) -> list[int]:
    """Return indices of SAE atoms with highest correlation to concept labels."""
    n = codes.shape[0]
    codes_std = codes.std(dim=0)
    active = codes_std > 1e-6
    codes_c = codes - codes.mean(dim=0)

    labels_exp = _expand_labels(labels) if labels.ndim > 1 else labels

    if labels_exp.ndim == 1:
        lnorm = (labels_exp - labels_exp.mean()) / (labels_exp.std() + 1e-8)
        corrs = (codes_c.T @ lnorm) / (n * codes_std.clamp(min=1e-8))
        corrs[~active] = 0
        score = corrs.abs()
    else:
        score = torch.zeros(codes.shape[1], device=codes.device)
        for i in range(labels_exp.shape[1]):
            li = labels_exp[:, i]
            li = (li - li.mean()) / (li.std() + 1e-8)
            c = (codes_c.T @ li) / (n * codes_std.clamp(min=1e-8))
            c[~active] = 0
            score = torch.max(score, c.abs())

    k = min(n_select, int((score > 0.05).sum().item()), codes.shape[1])
    if k == 0:
        k = min(n_select, int(active.sum().item()))
    if k == 0:
        return []
    _, idx = torch.topk(score, k)
    return idx.tolist()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def estimate_intrinsic_dim(activations: torch.Tensor, threshold: float = 0.90) -> int:
    centered = activations - activations.mean(dim=0)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    var = S ** 2
    cumvar = var.cumsum(0) / var.sum()
    return int((cumvar < threshold).sum().item()) + 1


def _get_pca_variance(acts: torch.Tensor, n_components: int = 10) -> list[float]:
    centered = acts - acts.mean(dim=0)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    var = S[:n_components] ** 2
    total = (S ** 2).sum()
    return (var / total).tolist()


def compute_restricted_r2_real(
    sae,
    manifold_activations: dict[str, torch.Tensor],
    manifold_labels: dict[str, np.ndarray] | None = None,
    max_atoms: int = 16,
    device: str = "cpu",
) -> list[RealManifoldResult]:
    """Compute restricted R² for manifold activations.

    R² is measured in the concept-specific subspace (label-regression
    direction) so that background activation variance doesn't dilute the score.
    Atom selection uses code–label correlation when labels are available.
    """
    sae = sae.to(device).eval()

    # Decoder weights as (d_in, d_sae) — columns are unit-normed directions
    if hasattr(sae, 'W_dec') and isinstance(sae.W_dec, torch.nn.Parameter):
        decoder_weights = sae.W_dec.data.T.to(device)   # (d_in, d_sae)
    elif hasattr(sae, 'W_dec') and hasattr(sae.W_dec, 'weight'):
        decoder_weights = sae.W_dec.weight.data.to(device)
    else:
        raise ValueError("Cannot find decoder weights on SAE")

    results = []

    for name, acts in manifold_activations.items():
        acts = acts.to(device)
        n = acts.shape[0]

        k_est = estimate_intrinsic_dim(acts)
        pca_var = _get_pca_variance(acts, n_components=min(10, n - 1))

        acts_centered = acts - acts.mean(dim=0)

        # Encode
        with torch.no_grad():
            codes = sae.encode(acts)  # (n, d_sae)

        # --- Atom selection ---
        labels_t = None
        if manifold_labels is not None and name in manifold_labels:
            raw = manifold_labels[name]
            labels_t = torch.tensor(raw, dtype=torch.float32).to(device)
            selected = select_atoms_by_label_correlation(codes, labels_t, max_atoms)
        else:
            # Geometric fallback: greedy on decoder alignment
            selected = _greedy_geometric(decoder_weights, acts_centered, max_atoms)

        if not selected:
            selected = list(range(min(max_atoms, codes.shape[1])))

        # --- Concept subspace ---
        if labels_t is not None:
            concept_dir = compute_concept_direction(acts_centered, labels_t)  # (d_in, k)
            acts_concept = acts_centered @ concept_dir                         # (n, k)
            total_var = acts_concept.pow(2).sum().item()
        else:
            acts_concept = acts_centered
            total_var = acts_centered.pow(2).sum().item()

        # --- R² at each number of atoms ---
        # When labels are available: affine OLS from codes → concept subspace
        # (mirrors the synthetic evaluation and guarantees R² >= 0 on training data).
        # When no labels: fall back to decoder-based reconstruction in activation space.
        r2_scores = {}
        acts_concept_cpu = acts_concept.cpu()
        for n_atoms_use in range(1, len(selected) + 1):
            atoms = selected[:n_atoms_use]
            codes_sel = codes[:, atoms].cpu()       # (n, n_atoms_use)

            if labels_t is not None:
                # Affine OLS: acts_concept ≈ codes_sel @ A + b
                # Use gelsd (SVD-based) driver for stability when selected atoms are
                # collinear (common with wide SAEs like 65k whose features are specialized).
                X = torch.cat([codes_sel, torch.ones(n, 1)], dim=-1)
                W = torch.linalg.lstsq(X, acts_concept_cpu, driver="gelsd").solution
                recon_concept = X @ W
                residual = (acts_concept_cpu - recon_concept).pow(2).sum().item()
            else:
                D_sel = decoder_weights[:, atoms]
                codes_c = codes_sel.to(device) - codes_sel.to(device).mean(dim=0)
                recon = codes_c @ D_sel.T
                residual = (acts_concept.cpu() - recon.cpu()).pow(2).sum().item()

            r2_scores[n_atoms_use] = 1 - residual / max(total_var, 1e-10)

        # --- Support size ---
        firing_counts = (codes.abs() > 0).float().sum(dim=0)
        threshold_count = max(0.05 * n, 5)
        support_size = int((firing_counts >= threshold_count).sum().item())

        # --- Receptive field spread ---
        rf_spread = _compute_rf_spread_real(codes, acts_centered, firing_counts, threshold_count)

        results.append(RealManifoldResult(
            name=name,
            n_samples=n,
            embedding_dim_estimate=k_est,
            restricted_r2=r2_scores,
            support_size=support_size,
            receptive_field_spread=rf_spread,
            greedy_atom_indices=selected,
            pca_var_explained=pca_var,
        ))

    return results


def _greedy_geometric(
    decoder_weights: torch.Tensor,
    acts_centered: torch.Tensor,
    n_select: int,
) -> list[int]:
    residual = acts_centered.clone()
    selected = []
    for _ in range(n_select):
        projections = residual @ decoder_weights
        var_exp = projections.pow(2).sum(dim=0)
        for idx in selected:
            var_exp[idx] = -1
        best = var_exp.argmax().item()
        selected.append(best)
        d_j = decoder_weights[:, best]
        proj_coeffs = residual @ d_j
        residual = residual - proj_coeffs.unsqueeze(-1) * d_j.unsqueeze(0)
    return selected


def _compute_rf_spread_real(
    codes: torch.Tensor,
    acts_centered: torch.Tensor,
    firing_counts: torch.Tensor,
    threshold_count: float,
    max_points: int = 2000,
) -> float:
    support_mask = firing_counts >= threshold_count
    support_indices = support_mask.nonzero(as_tuple=True)[0]
    if len(support_indices) == 0:
        return 0.0
    n = min(len(acts_centered), max_points)
    sub = acts_centered[:n]
    dists = torch.cdist(sub, sub)
    manifold_mean_dist = dists.sum() / (n * (n - 1))
    if manifold_mean_dist < 1e-10:
        return 0.0
    spreads = []
    for j in support_indices[:100]:
        firing_mask = codes[:, j].abs() > 0
        if firing_mask.sum() < 2:
            continue
        pts = acts_centered[firing_mask][:max_points]
        np_ = len(pts)
        d = torch.cdist(pts, pts)
        mean_pw = d.sum() / (np_ * (np_ - 1)) if np_ > 1 else torch.tensor(0.0)
        spreads.append(mean_pw.item())
    if not spreads:
        return 0.0
    median_spread = sorted(spreads)[len(spreads) // 2]
    return median_spread / manifold_mean_dist.item()


# ---------------------------------------------------------------------------
# Ising coupling
# ---------------------------------------------------------------------------

def compute_ising_coupling_real(
    sae,
    manifold_activations: dict[str, torch.Tensor],
    device: str = "cpu",
    regularization: float = 0.01,
    n_steps: int = 2000,
) -> tuple[torch.Tensor, list[int]]:
    """Compute Ising coupling matrix from SAE codes on real manifold activations.

    Shuffles the concatenated activations so all manifolds contribute equally,
    then uses all samples (not just the first 10k).
    Returns (J_full, active_indices) so callers can visualize the active submatrix.
    """
    sae = sae.to(device).eval()
    all_acts = torch.cat(list(manifold_activations.values()), dim=0)
    perm = torch.randperm(len(all_acts))
    all_acts = all_acts[perm].to(device)

    with torch.no_grad():
        codes = sae.encode(all_acts)

    from .evaluate import compute_ising_coupling
    J_active, active_idx_tensor = compute_ising_coupling(
        codes, lam=regularization, device=device,
        n_steps=n_steps, n_samples=len(codes),
    )
    active_idx = active_idx_tensor.tolist()
    return J_active, active_idx
