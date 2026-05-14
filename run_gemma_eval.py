"""End-to-end Gemma 3 12B manifold evaluation pipeline.

Step 1: Load pre-harvested activations (from run_gemma_pca.py --save-activations)
Step 2: Load GemmaScope 2 JumpReLU SAE from HuggingFace
Step 3: Compute restricted R² and Ising coupling
Step 4: Visualize results

Usage:
    # After running: python run_gemma_pca.py --save-activations --device cuda
    python run_gemma_eval.py --activations-dir figures/gemma_pca --device cuda
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from nonlinear_features.jumprelu_sae import JumpReLUSAE
from nonlinear_features.evaluate_real import (
    compute_restricted_r2_real,
    compute_ising_coupling_real,
    compute_restricted_r2_multi_n,
    compute_figure4b_path,
    _expand_labels,
    select_atoms_by_label_correlation,
)

CONCEPT_ORDER = ["years", "temperature", "days", "colors"]
CANONICAL_MANIFOLDS = {"colors", "days", "temperature", "years"}
CONCEPT_COLORS = {
    "years":       "#4CAF50",
    "temperature": "#F44336",
    "days":        "#2196F3",
    "colors":      "#FF9800",
    "unknown":     "#888888",
}


def load_activations(activations_dir: str, manifolds: set[str] | None = None):
    """Load saved activation tensors and labels from run_gemma_pca.py.

    Only loads the canonical manifold files by default (colors, days, temperature,
    years) to avoid accidentally pulling in encoding-variant files.
    """
    manifold_filter = manifolds if manifolds is not None else CANONICAL_MANIFOLDS
    manifold_acts = {}
    manifold_labels = {}
    act_dir = Path(activations_dir)
    for path in sorted(act_dir.glob("activations_*.pt")):
        name = path.stem.replace("activations_", "")
        if name not in manifold_filter:
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            manifold_acts[name] = data["activations"]
            if "labels" in data:
                labels = data["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.numpy()
                manifold_labels[name] = labels
        else:
            manifold_acts[name] = data
        print(f"  Loaded {name}: {manifold_acts[name].shape}")
    return manifold_acts, manifold_labels


def plot_restricted_r2(results, save_dir: str, tag: str = "gemma"):
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
    path = Path(save_dir) / f"restricted_r2_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def compute_atom_concept_assignments(
    sae,
    manifold_acts: dict[str, torch.Tensor],
    manifold_labels: dict[str, np.ndarray],
    active_indices: list[int],
    device: str = "cpu",
) -> dict[int, dict]:
    """For each active atom, find which concept's labels it correlates most with.

    Returns a dict: atom_idx → {"concept": str, "score": float, "all_scores": dict}
    """
    sae = sae.to(device).eval()
    scores: dict[int, dict[str, float]] = {idx: {} for idx in active_indices}

    for name, acts in manifold_acts.items():
        if name not in manifold_labels:
            continue
        acts = acts.to(device)
        with torch.no_grad():
            codes = sae.encode(acts)

        labels = torch.tensor(manifold_labels[name], dtype=torch.float32).to(device)
        labels_exp = _expand_labels(labels) if labels.ndim > 1 else labels

        n = codes.shape[0]
        codes_std = codes.std(dim=0)
        codes_c = codes - codes.mean(dim=0)

        if labels_exp.ndim == 1:
            li = (labels_exp - labels_exp.mean()) / (labels_exp.std() + 1e-8)
            corrs = (codes_c.T @ li) / (n * codes_std.clamp(min=1e-8))
            for idx in active_indices:
                scores[idx][name] = corrs[idx].abs().item()
        else:
            concept_scores = torch.zeros(codes.shape[1], device=device)
            for i in range(labels_exp.shape[1]):
                li = labels_exp[:, i]
                li = (li - li.mean()) / (li.std() + 1e-8)
                c = (codes_c.T @ li) / (n * codes_std.clamp(min=1e-8))
                concept_scores = torch.max(concept_scores, c.abs())
            for idx in active_indices:
                scores[idx][name] = concept_scores[idx].item()

    assignments = {}
    for idx in active_indices:
        s = scores[idx]
        if s:
            best = max(s, key=s.get)
            best_score = s[best]
        else:
            best = "unknown"
            best_score = 0.0
        assignments[idx] = {"concept": best, "score": best_score, "all_scores": s}

    return assignments


def plot_ising_matrix(
    J_active: torch.Tensor,
    active_indices: list[int],
    save_dir: str,
    tag: str = "gemma",
    atom_assignments: dict | None = None,
):
    """Plot the active-atom Ising coupling matrix, sorted by concept.

    J_active is the (p×p) matrix for the p active atoms (in the order they
    appear in active_indices).  If atom_assignments is provided the rows/cols
    are reordered so that atoms belonging to the same concept are adjacent,
    revealing any block-diagonal structure.
    """
    if not active_indices:
        print("  No active atoms for Ising plot.")
        return

    p = len(active_indices)
    J_np = J_active.cpu().numpy()   # (p, p)

    if atom_assignments is not None:
        concept_order_map = {c: i for i, c in enumerate(CONCEPT_ORDER + ["unknown"])}
        # Sort positions (0..p-1) inside J_active by concept assignment of their atom
        positions = list(range(p))
        positions.sort(
            key=lambda pos: (
                concept_order_map.get(atom_assignments[active_indices[pos]]["concept"], 99),
                -atom_assignments[active_indices[pos]]["score"],
            )
        )
        J_sorted = J_np[np.ix_(positions, positions)]
        concept_labels = [atom_assignments[active_indices[pos]]["concept"] for pos in positions]
    else:
        J_sorted = J_np
        concept_labels = None

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    nonzero = J_sorted[J_sorted != 0]
    vmax = float(np.percentile(np.abs(nonzero), 99)) if len(nonzero) > 0 else 1.0
    im = ax.imshow(J_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    if concept_labels is not None:
        prev = concept_labels[0]
        for k, c in enumerate(concept_labels):
            if c != prev:
                ax.axhline(y=k - 0.5, color="white", linewidth=1.2, alpha=0.8)
                ax.axvline(x=k - 0.5, color="white", linewidth=1.2, alpha=0.8)
                prev = c


    jmax = float(np.abs(J_sorted).max())
    ax.set_title(f"Ising coupling — {p} active atoms (sorted by concept)\n"
                 f"|J|_max={jmax:.4f}  SAE width={tag}", color="#ddd")
    ax.set_xlabel("Atom (sorted by concept)", color="#aaa")
    ax.set_ylabel("Atom (sorted by concept)", color="#aaa")
    ax.tick_params(colors="#aaa")
    plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    path = Path(save_dir) / f"ising_coupling_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {path}")


def plot_restricted_r2_multi_n(
    multi_n_results: dict[str, dict[int, dict[int, float]]],
    n_select_values: list[int],
    save_dir: str,
    tag: str = "gemma",
):
    """Plot k vs restricted R² curves for multiple top-n atom pool sizes (Fig 4A style)."""
    manifolds = [m for m in CONCEPT_ORDER if m in multi_n_results]
    fig, axes = plt.subplots(1, len(manifolds), figsize=(4 * len(manifolds), 4), squeeze=False)

    cmap = plt.cm.plasma
    colors_n = [cmap(i / max(len(n_select_values) - 1, 1)) for i in range(len(n_select_values))]

    for ax, name in zip(axes[0], manifolds):
        for color, n_sel in zip(colors_n, n_select_values):
            if n_sel not in multi_n_results[name]:
                continue
            curve = multi_n_results[name][n_sel]
            ks = sorted(curve.keys())
            r2s = [curve[k] for k in ks]
            ax.plot(ks, r2s, "o-", linewidth=1.5, markersize=3, color=color,
                    label=f"n={n_sel}", alpha=0.85)
        ax.axhline(y=0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xlabel("# decoder atoms (k)")
        ax.set_ylabel("Restricted R²")
        ax.set_title(name)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(save_dir) / f"restricted_r2_multi_n_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_figure4b(
    paths: dict[str, list[tuple[int, float]]],
    k_values: list[int],
    save_dir: str,
    tag: str = "gemma",
):
    """Figure 4B: support size vs RF spread path as K (post-hoc sparsity) varies."""
    fig, ax = plt.subplots(figsize=(6, 5))

    markers = ["o", "s", "^", "D"]
    for (name, path_pts), marker in zip(paths.items(), markers):
        color = CONCEPT_COLORS.get(name, "#888")
        xs = [p[0] for p in path_pts]
        ys = [p[1] for p in path_pts]
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(xs, ys, c=[color] * len(xs), marker=marker, s=50, zorder=4,
                   label=name)
        # Annotate a few K values
        for i, (x, y, k) in enumerate(zip(xs, ys, k_values)):
            if i in {0, len(k_values) // 2, len(k_values) - 1}:
                ax.annotate(f"K={k}", (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=6, color=color)

    ax.set_xlabel("Support size (# active atoms on manifold)")
    ax.set_ylabel("RF spread (median RF diam / manifold diam)")
    ax.set_title(f"Support size vs RF spread  (varying K)\nSAE width={tag}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(save_dir) / f"figure4b_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Gemma manifold evaluation")
    parser.add_argument("--activations-dir", type=str, default="figures/gemma_pca")
    parser.add_argument("--sae-repo", type=str, default="google/gemma-scope-2-12b-it")
    parser.add_argument("--sae-layer", type=int, default=24)
    parser.add_argument("--sae-site", type=str, default="resid_post",
                        choices=["resid_post", "attn_out", "mlp_out"])
    parser.add_argument("--sae-width", type=str, default="16k")
    parser.add_argument("--sae-l0", type=str, default="medium",
                        choices=["small", "medium", "big"])
    parser.add_argument("--sae-local-path", type=str, default=None)
    parser.add_argument("--max-atoms", type=int, default=32)
    parser.add_argument("--multi-n", type=str, default="16,32,64,128",
                        help="Comma-separated n_select values for multi-n R² plot")
    parser.add_argument("--k-values", type=str, default="5,10,15,20,30,45,60,80",
                        help="Comma-separated K values for Figure 4B sweep")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-dir", type=str, default="figures/gemma_eval")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for output filenames (default: sae_width)")
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

    tag = args.tag or args.sae_width
    os.makedirs(args.save_dir, exist_ok=True)
    multi_n_values = [int(x) for x in args.multi_n.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]

    # 1. Load activations
    print("\n=== Loading activations ===")
    manifold_acts, manifold_labels = load_activations(args.activations_dir)
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

    sample_acts = next(iter(manifold_acts.values()))[:64].to(device)
    stats = sae.sparsity_stats(sample_acts)
    print(f"  Sparsity: L0={stats['l0_mean']:.1f} ± {stats['l0_std']:.1f}, "
          f"alive={stats['frac_alive_latents']:.1%}")

    # 3. Restricted R²
    print("\n=== Computing restricted R² ===")
    results = compute_restricted_r2_real(
        sae, manifold_acts, manifold_labels=manifold_labels,
        max_atoms=args.max_atoms, device=device,
    )
    print("\nResults:")
    for r in results:
        k_est = r.embedding_dim_estimate
        r2_at_k = r.restricted_r2.get(k_est, r.restricted_r2.get(min(r.restricted_r2.keys()), 0))
        print(f"  {r.name:15s}  k_est={k_est:2d}  R²@k={r2_at_k:.4f}  "
              f"support={r.support_size:4d}  RF={r.receptive_field_spread:.3f}")
    plot_restricted_r2(results, args.save_dir, tag=tag)

    # 4. Multi-n restricted R² (Fig 4A style)
    print(f"\n=== Multi-n restricted R² (n = {multi_n_values}) ===")
    multi_n_results = compute_restricted_r2_multi_n(
        sae, manifold_acts, manifold_labels, multi_n_values, device=device,
    )
    plot_restricted_r2_multi_n(multi_n_results, multi_n_values, args.save_dir, tag=tag)

    # 5. Figure 4B: support size vs RF spread path
    print(f"\n=== Figure 4B: K sweep {k_values} ===")
    fig4b_paths = compute_figure4b_path(sae, manifold_acts, k_values, device=device)
    plot_figure4b(fig4b_paths, k_values, args.save_dir, tag=tag)

    # 6. Ising coupling — restricted to concept-atom union for cleaner ~75-atom matrix
    print("\n=== Computing Ising coupling ===")
    # Collect union of greedy-selected atoms across all manifolds
    concept_atoms = sorted(set().union(*[r.greedy_atom_indices for r in results]))
    print(f"  Restricting Ising to {len(concept_atoms)} concept atoms "
          f"(union of greedy selections, max_atoms={args.max_atoms})")
    J_active, active_indices = compute_ising_coupling_real(
        sae, manifold_acts, device=device, concept_atom_indices=concept_atoms,
    )
    print(f"  J_active shape: {J_active.shape}, active atoms: {len(active_indices)}, "
          f"|J|_max={J_active.abs().max():.4f}")

    # 7. Assign atoms to concepts for sorted Ising plot
    print("  Computing atom → concept assignments...")
    atom_assignments = compute_atom_concept_assignments(
        sae, manifold_acts, manifold_labels, active_indices, device=device
    )
    concept_counts = {}
    for v in atom_assignments.values():
        concept_counts[v["concept"]] = concept_counts.get(v["concept"], 0) + 1
    print(f"  Concept assignments: {concept_counts}")

    plot_ising_matrix(J_active, active_indices, args.save_dir, tag=tag,
                      atom_assignments=atom_assignments)

    # 8. Save summary
    summary = {
        "model": args.sae_repo,
        "layer": args.sae_layer,
        "site": args.sae_site,
        "width": args.sae_width,
        "l0": args.sae_l0,
        "d_sae": sae.d_sae,
        "l0_mean": round(stats["l0_mean"], 1),
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
        "ising": {
            "active_atoms": len(active_indices),
            "j_max": round(J_active.abs().max().item(), 4),
            "concept_assignments": concept_counts,
        },
    }
    summary_path = Path(args.save_dir) / f"eval_summary_{tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
