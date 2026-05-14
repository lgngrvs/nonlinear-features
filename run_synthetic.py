"""Run the synthetic manifold superposition experiment.

Replicates the synthetic experiment from
"Do Sparse Autoencoders Capture Concept Manifolds?" (arXiv:2604.28119)
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from nonlinear_features.manifolds import build_manifold_instances
from nonlinear_features.data import generate_dataset_fast
from nonlinear_features.train import train_sae
from nonlinear_features.evaluate import (
    compute_restricted_r2,
    compute_ising_coupling,
    aggregate_results,
)


def main():
    parser = argparse.ArgumentParser(description="Synthetic manifold SAE experiment")
    parser.add_argument("--d", type=int, default=128, help="Ambient dimension")
    parser.add_argument("--c", type=int, default=512, help="Dictionary size")
    parser.add_argument("--n-train", type=int, default=2_000_000, help="Training samples")
    parser.add_argument("--n-eval", type=int, default=100_000, help="Eval samples (paper uses 1M, reduced for speed)")
    parser.add_argument("--L0", type=int, default=4, help="Number of active manifolds per sample")
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 4, 6, 8, 10, 14, 16, 20, 25])
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--loss-fn", type=str, default="mse", choices=["l1", "mse"])
    parser.add_argument("--abs-max-atoms", type=int, default=50,
                        help="Evaluate restricted R² for 1..abs_max_atoms atoms")
    parser.add_argument("--ising-k", type=int, default=25,
                        help="Compute Ising coupling for this K value (0 to skip)")
    parser.add_argument("--ising-steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
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
    print(f"Using device: {device}")

    # 1. Build manifold instances
    print("\n=== Building manifold instances ===")
    t0 = time.time()
    instances = build_manifold_instances(d=args.d, seed=args.seed)
    print(f"Built {len(instances)} manifold instances in {time.time()-t0:.1f}s")
    for inst in instances[:8]:
        print(f"  {inst.type_name} (d_i={inst.intrinsic_dim}, k_i={inst.embedding_dim}, "
              f"params={inst.params}, scale={inst.scale:.3f})")

    # 2. Generate training data (no contributions needed - saves ~49GB)
    print(f"\n=== Generating {args.n_train:,} training samples (L0={args.L0}) ===")
    t0 = time.time()
    train_data, _, _ = generate_dataset_fast(
        instances, n_samples=args.n_train, L0=args.L0, seed=0,
        store_contributions=False,
    )
    print(f"Generated in {time.time()-t0:.1f}s, shape={train_data.shape}")
    print(f"  Data stats: mean={train_data.mean():.4f}, std={train_data.std():.4f}, "
          f"norm={train_data.norm(dim=-1).mean():.4f}")

    # 3. Generate eval data (with metadata)
    print(f"\n=== Generating {args.n_eval:,} eval samples ===")
    t0 = time.time()
    eval_data, eval_masks, eval_contribs = generate_dataset_fast(
        instances, n_samples=args.n_eval, L0=args.L0, seed=999,
    )
    print(f"Generated in {time.time()-t0:.1f}s")

    # 4. Train SAEs across sparsity budgets
    all_results = {}
    os.makedirs(args.save_dir, exist_ok=True)

    for k in args.k_values:
        print(f"\n{'='*60}")
        print(f"Training SAE with k={k}")
        print(f"{'='*60}")

        model, metrics = train_sae(
            train_data, d=args.d, c=args.c, k=k,
            lr=args.lr, batch_size=args.batch_size,
            epochs=args.epochs, device=device,
            loss_fn=args.loss_fn,
        )

        # Save checkpoint
        torch.save(model.state_dict(), f"{args.save_dir}/sae_k{k}.pt")

        # 5. Evaluate
        print(f"\n--- Evaluating k={k} ---")
        eval_results = compute_restricted_r2(
            model, eval_data, eval_masks, eval_contribs,
            instances, abs_max_atoms=args.abs_max_atoms, device=device,
        )
        agg = aggregate_results(eval_results)

        print(f"  Mean R² at k_i: {agg['mean_r2_at_ki']:.4f}")
        print(f"  Avg support size: {agg['avg_support_size']:.1f}")
        print(f"  Avg RF spread: {agg['avg_receptive_field_spread']:.4f}")

        # Per-type breakdown
        type_r2 = {}
        for r in eval_results:
            k_i = r.embedding_dim
            r2 = r.restricted_r2.get(k_i, 0.0)
            type_r2.setdefault(r.type_name, []).append(r2)
        for name, vals in sorted(type_r2.items()):
            print(f"    {name}: R²={sum(vals)/len(vals):.4f} (n={len(vals)})")

        all_results[k] = {
            "metrics": metrics,
            "eval_agg": agg,
            "per_manifold": [
                {
                    "type": r.type_name,
                    "variant": r.variant_idx,
                    "k_i": r.embedding_dim,
                    "r2": r.restricted_r2,
                    "support_size": r.support_size,
                    "rf_spread": r.receptive_field_spread,
                }
                for r in eval_results
            ],
        }

    # 6. Summary
    print(f"\n{'='*60}")
    print("SUMMARY: R² at k_i across sparsity budgets")
    print(f"{'='*60}")
    for k in args.k_values:
        if k in all_results:
            r2 = all_results[k]["eval_agg"]["mean_r2_at_ki"]
            supp = all_results[k]["eval_agg"]["avg_support_size"]
            rf = all_results[k]["eval_agg"]["avg_receptive_field_spread"]
            print(f"  k={k:3d}  R²={r2:.4f}  support={supp:6.1f}  RF_spread={rf:.4f}")

    # 7. Ising coupling for the requested K value
    if args.ising_k > 0 and args.ising_k in all_results:
        print(f"\n{'='*60}")
        print(f"Computing Ising coupling for k={args.ising_k}")
        print(f"{'='*60}")
        # Reload the checkpoint for this K
        ckpt_path = f"{args.save_dir}/sae_k{args.ising_k}.pt"
        if os.path.exists(ckpt_path):
            from nonlinear_features.sae import TopKSAE
            ising_model = TopKSAE(args.d, args.c, args.ising_k).to(device)
            ising_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            ising_model.eval()
        else:
            print(f"  No checkpoint at {ckpt_path}, reusing last-trained model")
            ising_model = model

        with torch.no_grad():
            eval_codes = ising_model.encode(eval_data.to(device))

        J_active, active_idx = compute_ising_coupling(
            eval_codes, lam=0.01, device=device,
            n_steps=args.ising_steps, n_samples=len(eval_codes),
        )
        n_active = len(active_idx)
        print(f"  Active atoms: {n_active}, |J|_max={J_active.abs().max():.4f}")

        # Plot Ising matrix
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#111")
        ax.set_facecolor("#1a1a1a")
        J_np = J_active.cpu().numpy()
        nonzero = J_np[J_np != 0]
        vmax = float(np.percentile(np.abs(nonzero), 99)) if len(nonzero) > 0 else 1.0
        im = ax.imshow(J_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Ising coupling — {n_active} active atoms  (k={args.ising_k})",
                     color="#ddd")
        ax.set_xlabel("Atom index", color="#aaa")
        ax.set_ylabel("Atom index", color="#aaa")
        ax.tick_params(colors="#aaa")
        plt.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        ising_path = f"{args.save_dir}/ising_coupling_k{args.ising_k}.png"
        plt.savefig(ising_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  Saved {ising_path}")

        all_results[f"ising_k{args.ising_k}"] = {
            "n_active": n_active,
            "j_max": float(J_active.abs().max()),
        }
    else:
        print(f"\n  Skipping Ising (ising_k={args.ising_k} not in k_values or disabled)")

    # Save results
    # Convert for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    with open(f"{args.save_dir}/results.json", "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {args.save_dir}/results.json")


if __name__ == "__main__":
    main()
