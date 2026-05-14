"""Plot Figure 4 panels (A, B, C) from synthetic experiment results.

Figure 4A left:  Aggregate mean R²@k_i vs SAE sparsity K.
Figure 4A right: Per-manifold-type subplots, each showing R²(# atoms)
                 for multiple K values (narrow window around k_i).
Figure 4B:       Single averaged path through (support_size, RF_spread) as K varies,
                 with shattering / capture / dilution regime labels.

Usage:
    python plot_synthetic_fig4.py checkpoints_mse/results.json
    python plot_synthetic_fig4.py checkpoints_mse/results.json --max-atoms 15
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


MANIFOLD_COLORS = {
    "circle":     "#e6194b",
    "sphere":     "#3cb44b",
    "torus":      "#4363d8",
    "mobius":     "#f58231",
    "swiss_roll": "#911eb4",
    "helix":      "#42d4f4",
    "flat_disk":  "#f032e6",
    "segment":    "#bfef45",
}

# K values to show in the per-type R²(atoms) subplots
HIGHLIGHT_KS = [3, 4, 6, 10, 25]


def load_results(path: str):
    with open(path) as f:
        return json.load(f)


def plot_fig4a(results: dict, save_path: str, max_atoms: int = 15):
    """Figure 4A: two-panel figure matching the paper.

    Left: aggregate mean R²@k_i vs SAE sparsity K.
    Right: 8 per-type subplots each showing R²(atoms) for HIGHLIGHT_KS values.
    """
    k_values = sorted(int(k) for k in results if k.isdigit())

    # --- Left panel data ---
    mean_r2_per_k = []
    for k in k_values:
        agg = results[str(k)]["eval_agg"]
        mean_r2_per_k.append(agg["mean_r2_at_ki"])

    # --- Right panel data: per type, per K, R²(atoms) ---
    type_names = sorted(set(m["type"] for m in results[str(k_values[0])]["per_manifold"]))

    # type -> k -> list of r2_dicts (one per variant)
    type_k_r2: dict[str, dict[int, list[dict]]] = {}
    for mtype in type_names:
        type_k_r2[mtype] = {}
        for k in k_values:
            per_manifold = results[str(k)]["per_manifold"]
            type_k_r2[mtype][k] = [m["r2"] for m in per_manifold if m["type"] == mtype]

    # Color map for K values
    k_cmap = plt.cm.viridis
    k_norm = plt.Normalize(vmin=min(HIGHLIGHT_KS), vmax=max(HIGHLIGHT_KS))

    # Build figure: left panel + 2 rows × 4 cols of type subplots
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(2, 5, figure=fig, wspace=0.35, hspace=0.45)
    ax_left = fig.add_subplot(gs[:, 0])  # left panel spans both rows

    ax_left.plot(k_values, mean_r2_per_k, "o-", color="#333", linewidth=2, markersize=6)
    ax_left.axvline(x=4, color="steelblue", linestyle="--", linewidth=1, alpha=0.7, label="capture (k=4)")
    ax_left.set_xlabel("SAE sparsity budget k", fontsize=11)
    ax_left.set_ylabel("Mean R² @ k_i atoms", fontsize=11)
    ax_left.set_title("Aggregate capture\nvs. sparsity", fontsize=11)
    ax_left.set_ylim(0, 1.05)
    ax_left.legend(fontsize=8, loc="lower right")
    ax_left.grid(True, alpha=0.3)

    # Right subplots: 2 rows × 4 cols
    n_types = len(type_names)  # 8
    right_axes = [fig.add_subplot(gs[row, col + 1])
                  for row in range(2) for col in range(4)]

    atoms_range = list(range(1, max_atoms + 1))

    for idx, mtype in enumerate(type_names):
        ax = right_axes[idx]
        color = MANIFOLD_COLORS.get(mtype, "#888")

        for k in HIGHLIGHT_KS:
            if k not in type_k_r2[mtype]:
                continue
            r2_list = type_k_r2[mtype][k]
            # Average across variants
            atom_vals: dict[int, list[float]] = {}
            for r2_dict in r2_list:
                for a_str, v in r2_dict.items():
                    a = int(a_str)
                    if a in atoms_range:
                        atom_vals.setdefault(a, []).append(v)
            xs = sorted(atom_vals.keys())
            ys = [np.mean(atom_vals[a]) for a in xs]
            k_color = k_cmap(k_norm(k))
            ax.plot(xs, ys, "-", color=k_color, linewidth=1.5, alpha=0.85, label="k=%d" % k)

        # Mark k_i
        if type_k_r2[mtype][k_values[0]]:
            ki_val = results[str(k_values[0])]["per_manifold"]
            ki = next((m["k_i"] for m in ki_val if m["type"] == mtype), None)
            if ki is not None and ki <= max_atoms:
                ax.axvline(x=ki, color=color, linestyle=":", linewidth=1, alpha=0.6)

        ax.set_title(mtype.replace("_", " "), fontsize=9, color=color)
        ax.set_ylim(-0.1, 1.05)
        ax.set_xlim(0, max_atoms + 0.5)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("R²", fontsize=8)
        ax.set_xlabel("# atoms", fontsize=8)
        ax.grid(True, alpha=0.25)

    # Colorbar / legend for K values
    sm = plt.cm.ScalarMappable(cmap=k_cmap, norm=k_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=right_axes, shrink=0.6, pad=0.01)
    cbar.set_label("SAE sparsity k", fontsize=9)
    cbar.set_ticks(HIGHLIGHT_KS)

    fig.suptitle("Figure 4A: Subspace Capture vs. Sparsity (synthetic)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 4A -> %s" % save_path)


def plot_fig4b(results: dict, save_path: str):
    """Figure 4B: single averaged path through (support_size, RF_spread) as K varies.

    One path total (averaged over all manifold types), with regime annotations.
    """
    k_values = sorted(int(k) for k in results if k.isdigit())

    support_per_k = []
    rf_per_k = []
    for k in k_values:
        agg = results[str(k)]["eval_agg"]
        support_per_k.append(agg["avg_support_size"])
        rf_per_k.append(agg["avg_receptive_field_spread"])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Color path by K value
    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=min(k_values), vmax=max(k_values))

    for i in range(len(k_values) - 1):
        ax.plot(
            support_per_k[i:i+2], rf_per_k[i:i+2],
            "-", color=cmap(norm(k_values[i])), linewidth=2.5, alpha=0.8,
        )

    # Scatter points
    sc = ax.scatter(support_per_k, rf_per_k,
                    c=k_values, cmap=cmap, norm=norm, s=70, zorder=5)

    # Annotate K values
    for k, x, y in zip(k_values, support_per_k, rf_per_k):
        ax.annotate("k=%d" % k, (x, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=7.5, color=cmap(norm(k)))

    # Regime labels
    # Shattering: small support, small RF (low k)
    ax.text(support_per_k[0] + 0.3, rf_per_k[0] - 0.04,
            "shattering", fontsize=8, color="#e44", ha="left", style="italic")
    # Capture: intermediate (k=4 is index 1)
    cap_idx = k_values.index(4) if 4 in k_values else 1
    ax.text(support_per_k[cap_idx] - 0.5, rf_per_k[cap_idx] + 0.025,
            "capture", fontsize=8, color="#4a4", ha="right", style="italic")
    # Dilution: large support, large RF (high k)
    ax.text(support_per_k[-1] - 2, rf_per_k[-1] + 0.01,
            "dilution", fontsize=8, color="#44e", ha="right", style="italic")

    plt.colorbar(sc, ax=ax, label="SAE sparsity k", shrink=0.85)

    ax.set_xlabel("Support size (# atoms firing on ≥10% of manifold)", fontsize=11)
    ax.set_ylabel("RF spread (median RF diam / manifold diam)", fontsize=11)
    ax.set_title("Figure 4B: Support Size vs. RF Spread as Sparsity k Varies", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved Figure 4B -> %s" % save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", help="Path to results.json")
    parser.add_argument("--max-atoms", type=int, default=15,
                        help="Max atoms to show in Fig 4A right panels")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Output directory (defaults to same dir as results)")
    args = parser.parse_args()

    results = load_results(args.results_path)
    save_dir = Path(args.save_dir or Path(args.results_path).parent)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_fig4a(results, str(save_dir / "fig4a.png"), max_atoms=args.max_atoms)
    plot_fig4b(results, str(save_dir / "fig4b.png"))


if __name__ == "__main__":
    main()
