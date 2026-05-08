"""Manifold zoo: 8 manifold types with parametric embeddings.

Each manifold is defined by:
- intrinsic dimension d_i (number of free parameters)
- embedding dimension k_i (dimension of ambient subspace)
- parametric embedding gamma_i(theta) -> R^k_i
- variant parameters that control shape/scale
"""

import torch
import math
from dataclasses import dataclass
from typing import Callable


@dataclass
class ManifoldType:
    name: str
    intrinsic_dim: int  # d_i
    embedding_dim: int  # k_i
    sample_fn: Callable  # (n, **params) -> (n, k_i) tensor
    variant_params: list[dict]


def sample_circle(n: int, r: float = 1.0) -> torch.Tensor:
    theta = torch.rand(n) * 2 * math.pi
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


def sample_sphere(n: int, r: float = 1.0) -> torch.Tensor:
    # Uniform sampling on sphere via normalized Gaussians
    z = torch.randn(n, 3)
    z = z / z.norm(dim=-1, keepdim=True)
    return r * z


def sample_torus(n: int, R: float = 2.0, r: float = 0.5) -> torch.Tensor:
    # Clifford torus embedding in R^4
    theta = torch.rand(n) * 2 * math.pi
    phi = torch.rand(n) * 2 * math.pi
    return torch.stack([
        (R + r * torch.cos(phi)) * torch.cos(theta),
        (R + r * torch.cos(phi)) * torch.sin(theta),
        r * torch.sin(phi) * torch.cos(theta),
        r * torch.sin(phi),
    ], dim=-1)


def sample_mobius(n: int, w: float = 0.5) -> torch.Tensor:
    # Möbius strip: phi in [0, 2pi), t in [-w, w]
    phi = torch.rand(n) * 2 * math.pi
    t = (torch.rand(n) * 2 - 1) * w
    return torch.stack([
        (1 + t * torch.cos(phi / 2)) * torch.cos(phi),
        (1 + t * torch.cos(phi / 2)) * torch.sin(phi),
        t * torch.sin(phi / 2),
    ], dim=-1)


def sample_swiss_roll(n: int, theta_max: float = 3 * math.pi, h_max: float = 3.0) -> torch.Tensor:
    # theta in [1.5*pi, theta_max], h in [0, h_max]
    theta = torch.rand(n) * (theta_max - 1.5 * math.pi) + 1.5 * math.pi
    h = torch.rand(n) * h_max
    return torch.stack([
        theta * torch.cos(theta),
        h,
        theta * torch.sin(theta),
    ], dim=-1)


def sample_helix(n: int, alpha: float = 0.3, r: float = 1.0, turns: int = 3) -> torch.Tensor:
    theta = torch.rand(n) * 2 * math.pi * turns
    return torch.stack([
        r * torch.cos(theta),
        r * torch.sin(theta),
        alpha * theta,
    ], dim=-1)


def sample_flat_disk(n: int, R: float = 1.0) -> torch.Tensor:
    # Uniform sampling on disk via sqrt trick
    theta = torch.rand(n) * 2 * math.pi
    r = torch.sqrt(torch.rand(n)) * R
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


def sample_segment(n: int, length: float = 1.0) -> torch.Tensor:
    t = torch.rand(n) * length
    return t.unsqueeze(-1)


MANIFOLD_TYPES = [
    ManifoldType(
        name="circle", intrinsic_dim=1, embedding_dim=2,
        sample_fn=sample_circle,
        variant_params=[{"r": r} for r in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]],
    ),
    ManifoldType(
        name="sphere", intrinsic_dim=2, embedding_dim=3,
        sample_fn=sample_sphere,
        variant_params=[{"r": r} for r in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]],
    ),
    ManifoldType(
        name="torus", intrinsic_dim=2, embedding_dim=4,
        sample_fn=sample_torus,
        variant_params=[
            {"R": 2, "r": 0.5}, {"R": 2, "r": 1}, {"R": 3, "r": 1},
            {"R": 3, "r": 1.5}, {"R": 4, "r": 1}, {"R": 4, "r": 2},
        ],
    ),
    ManifoldType(
        name="mobius", intrinsic_dim=2, embedding_dim=3,
        sample_fn=sample_mobius,
        variant_params=[{"w": w} for w in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]],
    ),
    ManifoldType(
        name="swiss_roll", intrinsic_dim=2, embedding_dim=3,
        sample_fn=sample_swiss_roll,
        variant_params=[
            {"theta_max": 2 * math.pi, "h_max": 1.5},
            {"theta_max": 2.5 * math.pi, "h_max": 2.0},
            {"theta_max": 3 * math.pi, "h_max": 3.0},
            {"theta_max": 3.5 * math.pi, "h_max": 4.0},
            {"theta_max": 4 * math.pi, "h_max": 5.0},
            {"theta_max": 4.5 * math.pi, "h_max": 6.0},
        ],
    ),
    ManifoldType(
        name="helix", intrinsic_dim=1, embedding_dim=3,
        sample_fn=sample_helix,
        variant_params=[{"alpha": a} for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]],
    ),
    ManifoldType(
        name="flat_disk", intrinsic_dim=2, embedding_dim=2,
        sample_fn=sample_flat_disk,
        variant_params=[{"R": r} for r in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]],
    ),
    ManifoldType(
        name="segment", intrinsic_dim=1, embedding_dim=1,
        sample_fn=sample_segment,
        variant_params=[{"length": l} for l in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]],
    ),
]


@dataclass
class ManifoldInstance:
    """A specific manifold instance with its embedding matrix and normalization."""
    type_name: str
    intrinsic_dim: int
    embedding_dim: int
    variant_idx: int
    params: dict
    sample_fn: Callable
    # Set after construction:
    V: torch.Tensor | None = None  # (k_i, d) orthonormal embedding
    mean: torch.Tensor | None = None  # (k_i,) calibration mean
    scale: float = 1.0  # RMS norm for normalization

    def sample_raw(self, n: int) -> torch.Tensor:
        return self.sample_fn(n, **self.params)

    def sample_normalized(self, n: int) -> torch.Tensor:
        """Sample and apply normalization: (gamma(theta) - mu) / sigma."""
        raw = self.sample_raw(n)
        return (raw - self.mean) / self.scale

    def sample_embedded(self, n: int) -> torch.Tensor:
        """Sample, normalize, and embed into R^d."""
        normed = self.sample_normalized(n)
        return normed @ self.V  # (n, k_i) @ (k_i, d) -> (n, d)


def build_manifold_instances(
    d: int = 128,
    calibration_samples: int = 50_000,
    seed: int = 42,
) -> list[ManifoldInstance]:
    """Build all 48 manifold instances (8 types x 6 variants) with embeddings."""
    rng = torch.Generator().manual_seed(seed)
    instances = []

    for mtype in MANIFOLD_TYPES:
        for vi, params in enumerate(mtype.variant_params):
            inst = ManifoldInstance(
                type_name=mtype.name,
                intrinsic_dim=mtype.intrinsic_dim,
                embedding_dim=mtype.embedding_dim,
                variant_idx=vi,
                params=params,
                sample_fn=mtype.sample_fn,
            )

            # Calibration: compute mean and RMS norm
            with torch.no_grad():
                cal = inst.sample_raw(calibration_samples)
                mu = cal.mean(dim=0)
                centered = cal - mu
                sigma = (centered.norm(dim=-1) ** 2).mean().sqrt()
                inst.mean = mu
                inst.scale = sigma.item()

            # Random orthonormal embedding V_i in R^(k_i x d)
            G = torch.randn(d, mtype.embedding_dim, generator=rng)
            Q, _ = torch.linalg.qr(G)
            inst.V = Q[:, :mtype.embedding_dim].T  # (k_i, d)

            instances.append(inst)

    return instances
