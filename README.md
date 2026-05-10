# Nonlinear Features: Manifold SAE Replication

Replication of the synthetic experiment from **"Do Sparse Autoencoders Capture Concept Manifolds?"** ([arXiv:2604.28119](https://arxiv.org/abs/2604.28119)).

## What this does

Tests whether TopK Sparse Autoencoders can recover manifold structure from data generated in **manifold superposition** — where observations are sparse mixtures of points sampled from multiple embedded manifolds.

## Key findings (replicated)

The SAE exhibits three distinct regimes as the sparsity budget `k` increases:

1. **Tiling/Shattering** (low k): atoms are shared across manifolds, no clean block structure
2. **Capture** (intermediate k ≈ 8–10): atoms cleanly span individual manifold subspaces, block-diagonal Ising coupling emerges
3. **Dilution** (high k): atoms over-tile manifolds redundantly, fragmenting structure

Peak subspace capture (R² ≈ 0.60) occurs around k=8–10 with our setup.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib tqdm einops scipy scikit-learn networkx python-louvain
```

## Running

```bash
# Full experiment (9 SAEs, ~30 min on MPS/GPU)
python run_synthetic.py --device mps

# Quick test (fewer samples, one k value)
python run_synthetic.py --n-train 200000 --n-eval 20000 --k-values 8 --epochs 3 --device mps
```

## Architecture

```
nonlinear_features/
├── manifolds.py   # 8 manifold types × 6 variants = 48 instances
├── data.py        # Vectorized superposition sampling: x = Σ γ̃_i(θ_i) V_i + ε
├── sae.py         # TopK SAE with unit-norm decoder, dead neuron reanimation
├── train.py       # Adam (lr=3e-3), batch 1024, 10 epochs per k
├── evaluate.py    # Restricted R², support size, RF spread, Ising coupling
└── plot.py        # Visualization utilities
```

## Experiment details

- **Ambient dimension**: d=128, dictionary size c=512 (4× expansion)
- **Manifold zoo**: circles, spheres, tori, Möbius strips, Swiss rolls, helices, flat disks, line segments
- **Data**: 2M training / 100k eval samples, L0=4 manifolds active per sample, σ_ε=10⁻⁵
- **Sparsity sweep**: k ∈ {3, 4, 6, 8, 10, 14, 16, 20, 25}
- **Evaluation**: greedy atom selection → restricted R² (centered codes), Ising PLM with EBIC + Louvain

## Future directions

- Replicate on Gemma 2 12B + GemmaScope 2 SAEs (real model activations)
- Unsupervised manifold discovery on temperature, color, day-of-week features
- Cross-layer transcoders / CLTs
