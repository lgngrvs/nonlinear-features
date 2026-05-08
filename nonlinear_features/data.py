"""Data generation: sparse mixture sampling from manifold superposition."""

import torch
from tqdm import trange

from .manifolds import ManifoldInstance


def generate_dataset_fast(
    instances: list[ManifoldInstance],
    n_samples: int,
    L0: int = 4,
    sigma_eps: float = 1e-5,
    seed: int = 0,
    store_contributions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized dataset generation. Returns (data, active_masks, contributions).

    Generates all manifold samples in bulk, then assembles superpositions.
    """
    m = len(instances)
    d = instances[0].V.shape[1]

    torch.manual_seed(seed)

    # Pre-sample embedded points for each manifold (over-allocate for safety)
    expected_per = int(1.5 * L0 * n_samples / m) + 1000
    print(f"  Pre-sampling {expected_per} points per manifold...")
    embedded = []
    for inst in instances:
        embedded.append(inst.sample_embedded(expected_per))
    embedded = torch.stack(embedded)  # (m, expected_per, d)

    # Generate active sets in batch: for each sample, pick L0 of m manifolds
    # We generate a (n_samples, m) random matrix and pick top-L0 per row
    rng = torch.Generator().manual_seed(seed + 1)
    rand_scores = torch.rand(n_samples, m, generator=rng)
    _, active_indices = rand_scores.topk(L0, dim=1)  # (n_samples, L0)

    # Build active masks
    active_masks = torch.zeros(n_samples, m, dtype=torch.bool)
    active_masks.scatter_(1, active_indices, True)

    # For each manifold, figure out which samples use it and assign point indices
    usage_counts = torch.zeros(m, dtype=torch.long)
    # Build a mapping: for each (sample, manifold) pair that's active, which pre-sampled point to use
    point_indices = torch.zeros(n_samples, m, dtype=torch.long)
    for i in range(m):
        mask = active_masks[:, i]
        count = mask.sum().item()
        idx = torch.arange(count) % expected_per
        point_indices[mask, i] = idx

    # Assemble contributions and data
    print(f"  Assembling {n_samples:,} superposition samples...")
    data = torch.zeros(n_samples, d)

    if store_contributions:
        contributions = torch.zeros(n_samples, m, d)

    # Process per-manifold (vectorized over samples)
    for i in range(m):
        mask = active_masks[:, i]  # (n_samples,) bool
        if mask.sum() == 0:
            continue
        indices = point_indices[mask, i]  # which pre-sampled point
        contribs = embedded[i][indices]  # (n_active, d)
        data[mask] += contribs
        if store_contributions:
            contributions[mask, i] = contribs

    # Add noise
    noise = torch.randn(n_samples, d) * sigma_eps
    data += noise

    if store_contributions:
        return data, active_masks, contributions
    else:
        return data, active_masks, torch.empty(0)
