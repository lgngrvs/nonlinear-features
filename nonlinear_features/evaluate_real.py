"""Evaluation for real model activations (non-synthetic).

Adapts the restricted R² and Ising coupling pipeline for the setting where
each activation belongs to exactly one manifold (no superposition mixing).

The paper computes restricted R² as:
1. For each manifold, center the activations.
2. Greedily select SAE decoder directions that explain the most variance.
3. Project SAE codes (centered) through selected decoder columns.
4. R² = 1 - ||residual||² / ||centered_truth||²

This module works with any SAE that has .encode() and decoder weights accessible.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class RealManifoldResult:
    """Evaluation result for a single manifold from real model activations."""
    name: str
    n_samples: int
    embedding_dim_estimate: int
    restricted_r2: dict[int, float]
    support_size: int
    receptive_field_spread: float
    greedy_atom_indices: list[int]
    pca_var_explained: list[float]


def estimate_intrinsic_dim(activations: torch.Tensor, threshold: float = 0.90) -> int:
    """Estimate intrinsic dimensionality via PCA variance threshold."""
    centered = activations - activations.mean(dim=0)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    var = S ** 2
    cumvar = var.cumsum(0) / var.sum()
    return int((cumvar < threshold).sum().item()) + 1


def greedy_atom_selection_real(
    decoder_weights: torch.Tensor,  # (d_in, d_sae) or (d_sae, d_in) depending on convention
    activations_centered: torch.Tensor,  # (n, d_in) — centered manifold activations
    n_select: int,
) -> list[int]:
    """Greedily select decoder atoms that explain the most manifold variance.

    decoder_weights should be (d_in, d_sae) — columns are decoder directions.
    """
    residual = activations_centered.clone()
    selected = []

    for _ in range(n_select):
        projections = residual @ decoder_weights  # (n, d_sae)
        var_explained = projections.pow(2).sum(dim=0)  # (d_sae,)

        for idx in selected:
            var_explained[idx] = -1

        best = var_explained.argmax().item()
        selected.append(best)

        d_j = decoder_weights[:, best]  # (d_in,)
        proj_coeffs = residual @ d_j  # (n,)
        residual = residual - proj_coeffs.unsqueeze(-1) * d_j.unsqueeze(0)

    return selected


def compute_restricted_r2_real(
    sae,
    manifold_activations: dict[str, torch.Tensor],
    max_atoms: int = 16,
    device: str = "cpu",
) -> list[RealManifoldResult]:
    """Compute restricted R² for manifold activations through a pre-trained SAE.

    Args:
        sae: An SAE module with .encode() method and decoder weights.
             Supports both TopKSAE (.W_dec.weight -> (d_in, d_sae))
             and JumpReLUSAE (.W_dec -> (d_sae, d_in)).
        manifold_activations: {name: (n_samples, d_in)} tensors of activations
            belonging to each manifold.
        max_atoms: Maximum number of atoms for greedy selection.
        device: Computation device.

    Returns:
        List of RealManifoldResult for each manifold.
    """
    sae = sae.to(device).eval()

    # Get decoder weights as (d_in, d_sae) — columns are decoder directions
    if hasattr(sae, 'W_dec') and isinstance(sae.W_dec, torch.nn.Parameter):
        # JumpReLUSAE: W_dec is (d_sae, d_in), so transpose
        decoder_weights = sae.W_dec.data.T.to(device)
    elif hasattr(sae, 'W_dec') and hasattr(sae.W_dec, 'weight'):
        # TopKSAE: W_dec.weight is (d_in, d_sae) — Linear(c, d) stores (d, c)
        decoder_weights = sae.W_dec.weight.data.to(device)
    else:
        raise ValueError("Cannot find decoder weights on SAE")

    results = []

    for name, acts in manifold_activations.items():
        acts = acts.to(device)
        n = acts.shape[0]

        # Estimate intrinsic dimensionality
        k_est = estimate_intrinsic_dim(acts)
        pca_var = _get_pca_variance(acts, n_components=min(10, n - 1))

        # Center activations (the "truth" for R²)
        mean_acts = acts.mean(dim=0)
        acts_centered = acts - mean_acts
        total_var = acts_centered.pow(2).sum().item()

        # Encode through SAE
        with torch.no_grad():
            codes = sae.encode(acts)  # (n, d_sae)

        # Greedy atom selection on centered activations
        n_atoms = min(max_atoms, decoder_weights.shape[1])
        selected = greedy_atom_selection_real(decoder_weights, acts_centered, n_atoms)

        # Compute R² at each number of atoms
        r2_scores = {}
        for n_atoms_use in range(1, n_atoms + 1):
            atoms = selected[:n_atoms_use]
            D_sel = decoder_weights[:, atoms]  # (d_in, n_atoms_use)
            codes_sel = codes[:, atoms]  # (n, n_atoms_use)

            # Center codes (affine correction for non-negative activations)
            codes_centered = codes_sel - codes_sel.mean(dim=0, keepdim=True)

            # Reconstruct via selected decoder directions
            recon = codes_centered @ D_sel.T  # (n, d_in)
            residual_var = (acts_centered - recon).pow(2).sum().item()
            r2 = 1 - residual_var / max(total_var, 1e-10)
            r2_scores[n_atoms_use] = r2

        # Support size: latents firing on >= 5% of this manifold's samples
        firing_counts = (codes.abs() > 0).float().sum(dim=0)
        threshold_count = max(0.05 * n, 5)
        support_size = int((firing_counts >= threshold_count).sum().item())

        # Receptive field spread
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


def _get_pca_variance(acts: torch.Tensor, n_components: int = 10) -> list[float]:
    """Get PCA variance explained ratios."""
    centered = acts - acts.mean(dim=0)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    var = S[:n_components] ** 2
    total = (S ** 2).sum()
    return (var / total).tolist()


def _compute_rf_spread_real(
    codes: torch.Tensor,
    acts_centered: torch.Tensor,
    firing_counts: torch.Tensor,
    threshold_count: float,
    max_points: int = 2000,
) -> float:
    """Compute receptive field spread for real activations."""
    support_mask = firing_counts >= threshold_count
    support_indices = support_mask.nonzero(as_tuple=True)[0]

    if len(support_indices) == 0:
        return 0.0

    n = min(len(acts_centered), max_points)
    subsample = acts_centered[:n]
    dists = torch.cdist(subsample, subsample)
    manifold_mean_dist = dists.sum() / (n * (n - 1))

    if manifold_mean_dist < 1e-10:
        return 0.0

    spreads = []
    for j in support_indices[:100]:  # cap at 100 atoms for speed
        firing_mask = codes[:, j].abs() > 0
        if firing_mask.sum() < 2:
            continue
        pts = acts_centered[firing_mask][:max_points]
        n_pts = len(pts)
        d = torch.cdist(pts, pts)
        mean_pw = d.sum() / (n_pts * (n_pts - 1)) if n_pts > 1 else torch.tensor(0.0)
        spreads.append(mean_pw.item())

    if not spreads:
        return 0.0

    median_spread = sorted(spreads)[len(spreads) // 2]
    return median_spread / manifold_mean_dist.item()


def compute_ising_coupling_real(
    sae,
    manifold_activations: dict[str, torch.Tensor],
    device: str = "cpu",
    regularization: float = 0.01,
) -> torch.Tensor:
    """Compute Ising coupling matrix from SAE codes on real manifold activations.

    Pools all manifold activations, encodes them, and computes pairwise
    conditional independence structure in the code space.
    """
    sae = sae.to(device).eval()

    all_acts = torch.cat(list(manifold_activations.values()), dim=0).to(device)

    with torch.no_grad():
        codes = sae.encode(all_acts)

    # Reuse the synthetic Ising computation
    from .evaluate import compute_ising_coupling
    return compute_ising_coupling(codes, regularization=regularization)
