"""Plotting utilities for the synthetic experiment."""

import json
import matplotlib.pyplot as plt
import numpy as np


def _load_results(results_path: str) -> tuple[list[int], list[float], list[float], list[float]]:
    """Load results and return (k_values, r2s, supports, rf_spreads)."""
    with open(results_path) as f:
        results = json.load(f)

    k_values = sorted(int(k) for k in results.keys())
    r2_means, supports, spreads = [], [], []
    for k in k_values:
        r = results[str(k)]
        if "eval_agg" in r:
            r2_means.append(r["eval_agg"]["mean_r2_at_ki"])
            supports.append(r["eval_agg"]["avg_support_size"])
            spreads.append(r["eval_agg"]["avg_receptive_field_spread"])
        else:
            r2_means.append(r["mean_r2_at_ki"])
            supports.append(r["avg_support_size"])
            spreads.append(r["avg_rf_spread"])
    return k_values, r2_means, supports, spreads


def plot_synthetic_main(results_path: str, save_path: str | None = None):
    """3-panel figure: R², support size, RF spread vs k."""
    k_values, r2_means, supports, spreads = _load_results(results_path)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    ax1.plot(k_values, r2_means, "o-", linewidth=2, markersize=7, color="tab:blue")
    ax1.set_xlabel("Sparsity budget k")
    ax1.set_ylabel("Mean restricted R² at k_i")
    ax1.set_title("Subspace Capture")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_values, supports, "s-", linewidth=2, markersize=7, color="tab:green")
    ax2.set_xlabel("Sparsity budget k")
    ax2.set_ylabel("Avg support size")
    ax2.set_title("Support Size")
    ax2.grid(True, alpha=0.3)

    ax3.plot(k_values, spreads, "^-", linewidth=2, markersize=7, color="tab:orange")
    ax3.set_xlabel("Sparsity budget k")
    ax3.set_ylabel("Avg RF spread (normalized)")
    ax3.set_title("Receptive Field Spread")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.close()


def plot_r2_vs_sparsity(results_path: str, save_path: str | None = None):
    """Plot mean R² at k_i vs sparsity budget k (main result figure)."""
    k_values, r2_means, _, _ = _load_results(results_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, r2_means, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Sparsity budget k", fontsize=13)
    ax.set_ylabel("Mean restricted R² at k_i", fontsize=13)
    ax.set_title("Subspace Capture vs Sparsity", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.close()


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
    plt.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_final/results.json"
    plot_synthetic_main(path, "figures/synthetic_main.png")
    plot_r2_by_manifold_type(path, "figures/r2_by_type.png")
    plot_support_and_spread(path, "figures/support_spread.png")
