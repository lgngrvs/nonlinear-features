"""Plotting utilities for the synthetic experiment."""

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_r2_vs_sparsity(results_path: str, save_path: str | None = None):
    """Plot mean R² at k_i vs sparsity budget k (main result figure)."""
    with open(results_path) as f:
        results = json.load(f)

    k_values = sorted(int(k) for k in results.keys())
    r2_means = [results[str(k)]["eval_agg"]["mean_r2_at_ki"] for k in k_values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, r2_means, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Sparsity budget k", fontsize=13)
    ax.set_ylabel("Mean restricted R² at k_i", fontsize=13)
    ax.set_title("Subspace Capture vs Sparsity", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.5, label="VE threshold")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()


def plot_support_and_spread(results_path: str, save_path: str | None = None):
    """Plot support size and receptive field spread vs k."""
    with open(results_path) as f:
        results = json.load(f)

    k_values = sorted(int(k) for k in results.keys())
    support = [results[str(k)]["eval_agg"]["avg_support_size"] for k in k_values]
    spread = [results[str(k)]["eval_agg"]["avg_receptive_field_spread"] for k in k_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(k_values, support, "s-", linewidth=2, markersize=8, color="tab:blue")
    ax1.set_xlabel("Sparsity budget k", fontsize=13)
    ax1.set_ylabel("Avg support size", fontsize=13)
    ax1.set_title("Support Size vs Sparsity", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, spread, "^-", linewidth=2, markersize=8, color="tab:orange")
    ax2.set_xlabel("Sparsity budget k", fontsize=13)
    ax2.set_ylabel("Avg receptive field spread", fontsize=13)
    ax2.set_title("Receptive Field Spread vs Sparsity", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_r2_by_manifold_type(results_path: str, save_path: str | None = None):
    """Plot R² broken down by manifold type for each k."""
    with open(results_path) as f:
        results = json.load(f)

    k_values = sorted(int(k) for k in results.keys())
    manifold_types = set()
    for k in k_values:
        for m in results[str(k)]["per_manifold"]:
            manifold_types.add(m["type"])
    manifold_types = sorted(manifold_types)

    fig, ax = plt.subplots(figsize=(10, 6))
    for mtype in manifold_types:
        r2_vals = []
        for k in k_values:
            type_r2 = []
            for m in results[str(k)]["per_manifold"]:
                if m["type"] == mtype:
                    k_i = m["k_i"]
                    if str(k_i) in m["r2"]:
                        type_r2.append(m["r2"][str(k_i)])
            r2_vals.append(np.mean(type_r2) if type_r2 else 0.0)
        ax.plot(k_values, r2_vals, "o-", label=mtype, linewidth=1.5, markersize=6)

    ax.set_xlabel("Sparsity budget k", fontsize=13)
    ax.set_ylabel("Mean R² at k_i", fontsize=13)
    ax.set_title("Subspace Capture by Manifold Type", fontsize=14)
    ax.legend(ncol=2, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/results.json"
    plot_r2_vs_sparsity(path, "r2_vs_sparsity.png")
    plot_support_and_spread(path, "support_spread.png")
    plot_r2_by_manifold_type(path, "r2_by_type.png")
