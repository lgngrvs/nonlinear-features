"""End-to-end Gemma 3 12B manifold evaluation pipeline.

Step 1: Load pre-harvested activations (from run_gemma_pca.py --save-activations)
Step 2: Load GemmaScope 2 JumpReLU SAE from HuggingFace
Step 3: Compute restricted R² and Ising coupling
Step 4: Visualize results

Usage:
    # After running: python run_gemma_pca.py --save-activations --device cuda
    python run_gemma_eval.py --activations-dir figures/gemma_pca --device cuda
    python run_gemma_eval.py --activations-dir figures/gemma_pca --device mps
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nonlinear_features.jumprelu_sae import JumpReLUSAE
from nonlinear_features.evaluate_real import (
    compute_restricted_r2_real,
    compute_ising_coupling_real,
)


def load_activations(activations_dir: str) -> dict[str, torch.Tensor]:
    """Load saved activation tensors from run_gemma_pca.py."""
    manifold_acts = {}
    act_dir = Path(activations_dir)
    for path in sorted(act_dir.glob("activations_*.pt")):
        name = path.stem.replace("activations_", "")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            manifold_acts[name] = data["activations"]
        else:
            manifold_acts[name] = data
        print(f"  Loaded {name}: {manifold_acts[name].shape}")
    return manifold_acts


def plot_restricted_r2(results, save_dir: str):
    """Plot restricted R² curves for each manifold."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4), squeeze=False)

    for ax, r in zip(axes[0], results):
        n_atoms = sorted(r.restricted_r2.keys())
        r2_vals = [r.restricted_r2[n] for n in n_atoms]

        ax.plot(n_atoms, r2_vals, "o-", linewidth=2, markersize=5)
        ax.axvline(x=r.embedding_dim_estimate, color="red", linestyle="--",
                   alpha=0.7, label=f"k_est={r.embedding_dim_estimate}")
        ax.set_xlabel("# decoder atoms")
        ax.set_ylabel("Restricted R²")
        ax.set_title(f"{r.name}\n(n={r.n_samples}, support={r.support_size})")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(save_dir) / "restricted_r2_gemma.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_ising_matrix(J: torch.Tensor, save_dir: str):
    """Plot the Ising coupling matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    J_np = J.cpu().numpy()
    vmax = np.percentile(np.abs(J_np), 99)
    im = ax.imshow(J_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Ising coupling matrix J (SAE latent pairs)")
    ax.set_xlabel("Latent index")
    ax.set_ylabel("Latent index")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = Path(save_dir) / "ising_coupling_gemma.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Gemma 3 12B manifold evaluation")
    parser.add_argument("--activations-dir", type=str, default="figures/gemma_pca",
                        help="Directory with saved activations from run_gemma_pca.py")
    parser.add_argument("--sae-repo", type=str, default="google/gemma-scope-2-12b-it")
    parser.add_argument("--sae-layer", type=int, default=24)
    parser.add_argument("--sae-site", type=str, default="resid_post",
                        choices=["resid_post", "attn_out", "mlp_out"])
    parser.add_argument("--sae-width", type=str, default="16k")
    parser.add_argument("--sae-l0", type=str, default="medium",
                        choices=["small", "medium", "big"])
    parser.add_argument("--sae-local-path", type=str, default=None,
                        help="Path to local SAE checkpoint (skips HF download)")
    parser.add_argument("--max-atoms", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="figures/gemma_eval")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load activations
    print("\n=== Loading activations ===")
    manifold_acts = load_activations(args.activations_dir)
    if not manifold_acts:
        print("ERROR: No activation files found. Run run_gemma_pca.py --save-activations first.")
        return

    d_in = next(iter(manifold_acts.values())).shape[1]
    print(f"  d_in={d_in}, {len(manifold_acts)} manifolds")

    # 2. Load GemmaScope SAE
    print("\n=== Loading GemmaScope SAE ===")
    if args.sae_local_path:
        print(f"  Loading from local path: {args.sae_local_path}")
        sae = JumpReLUSAE.from_pretrained(args.sae_local_path, device=device)
    else:
        print(f"  Downloading from {args.sae_repo}")
        print(f"  Layer {args.sae_layer}, site={args.sae_site}, width={args.sae_width}, l0={args.sae_l0}")
        sae = JumpReLUSAE.from_huggingface(
            repo_id=args.sae_repo,
            layer=args.sae_layer,
            site=args.sae_site,
            width=args.sae_width,
            l0=args.sae_l0,
            device=device,
        )
    print(f"  SAE loaded: d_in={sae.d_in}, d_sae={sae.d_sae}")

    # Quick sparsity check
    sample_acts = next(iter(manifold_acts.values()))[:64].to(device)
    stats = sae.sparsity_stats(sample_acts)
    print(f"  Sparsity stats (sample): L0={stats['l0_mean']:.1f} ± {stats['l0_std']:.1f}, "
          f"alive={stats['frac_alive_latents']:.1%}")

    # 3. Compute restricted R²
    print("\n=== Computing restricted R² ===")
    results = compute_restricted_r2_real(
        sae, manifold_acts, max_atoms=args.max_atoms, device=device,
    )

    print("\nResults:")
    for r in results:
        k_est = r.embedding_dim_estimate
        r2_at_k = r.restricted_r2.get(k_est, r.restricted_r2.get(min(r.restricted_r2.keys()), 0))
        print(f"  {r.name:15s}  k_est={k_est:2d}  R²@k={r2_at_k:.4f}  "
              f"support={r.support_size:4d}  RF={r.receptive_field_spread:.3f}")

    plot_restricted_r2(results, args.save_dir)

    # 4. Ising coupling
    print("\n=== Computing Ising coupling ===")
    J = compute_ising_coupling_real(sae, manifold_acts, device=device)
    print(f"  J shape: {J.shape}, |J|_max={J.abs().max():.4f}")
    plot_ising_matrix(J, args.save_dir)

    # 5. Save results
    summary = {
        "model": args.sae_repo,
        "layer": args.sae_layer,
        "site": args.sae_site,
        "width": args.sae_width,
        "l0": args.sae_l0,
        "manifolds": [
            {
                "name": r.name,
                "n_samples": r.n_samples,
                "k_estimate": r.embedding_dim_estimate,
                "restricted_r2": {str(k): round(v, 4) for k, v in r.restricted_r2.items()},
                "support_size": r.support_size,
                "rf_spread": round(r.receptive_field_spread, 4),
                "pca_var_top3": sum(r.pca_var_explained[:3]),
            }
            for r in results
        ],
    }
    summary_path = Path(args.save_dir) / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
