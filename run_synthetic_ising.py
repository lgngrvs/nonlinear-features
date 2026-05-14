"""Compute and plot the Ising coupling matrix for a specific SAE checkpoint.

Replicates Figure 4C from Bhalla et al.: Ising coupling at the capture sweet-spot
(k=4), with atoms sorted by their ground-truth manifold assignment.

Usage:
    python run_synthetic_ising.py --k 4 --save-dir checkpoints_mse
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nonlinear_features.manifolds import build_manifold_instances
from nonlinear_features.data import generate_dataset_fast
from nonlinear_features.sae import TopKSAE


def compute_ising_coupling(
    codes: torch.Tensor,
    lam: float = 0.001,
    n_steps: int = 2000,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit pairwise Ising model via joint pseudo-likelihood (PLL) maximization.

    Returns (J_active, active_idx): coupling matrix and active atom indices.
    """
    s = torch.sign(codes)
    s[codes == 0] = -1.0

    firing_rate = (s > 0).float().mean(dim=0)
    active = (firing_rate > 0.01) & (firing_rate < 0.99)
    active_idx = active.nonzero(as_tuple=True)[0]
    s_active = s[:, active_idx].to(device)
    p = len(active_idx)
    print("  Active atoms: %d" % p)

    J = torch.zeros(p, p, device=device, requires_grad=True)
    h = torch.zeros(p, device=device, requires_grad=True)
    opt = torch.optim.Adam([J, h], lr=0.01)

    def soft_threshold(x, t):
        return torch.sign(x) * torch.clamp(x.abs() - t, min=0)

    for step in range(n_steps):
        opt.zero_grad()
        J_sym = (J + J.T) / 2
        J_sym = J_sym - torch.diag(J_sym.diag())
        field = s_active @ J_sym + h.unsqueeze(0)
        pll = F.logsigmoid(2 * s_active * field).mean()
        (-pll).backward()
        opt.step()
        with torch.no_grad():
            J.data = soft_threshold(J.data, lam)
            J.data = (J.data + J.data.T) / 2
            J.data.fill_diagonal_(0)
        if (step + 1) % 500 == 0:
            print("  step %d/%d  |J|_max=%.4f" % (step + 1, n_steps, J.abs().max().item()))

    return J.detach().cpu(), active_idx.cpu()


def assign_atoms_to_manifolds(
    codes: torch.Tensor,        # (n, c)
    active_masks: torch.Tensor, # (n, m)
    active_idx: torch.Tensor,   # (p,) indices into c
    instances: list,
) -> np.ndarray:
    """Assign each active atom to the manifold it correlates most with.

    Returns (p,) array of manifold indices (0..m-1), ordered by manifold.
    """
    codes_active = codes[:, active_idx].cpu().float()  # (n, p)
    masks_float = active_masks.float().cpu()            # (n, m)
    n, p = codes_active.shape
    m = masks_float.shape[1]

    # Pearson correlation between each atom's activation and each manifold's mask
    codes_c = codes_active - codes_active.mean(dim=0, keepdim=True)
    masks_c = masks_float - masks_float.mean(dim=0, keepdim=True)
    codes_std = codes_c.std(dim=0).clamp(min=1e-8)
    masks_std = masks_c.std(dim=0).clamp(min=1e-8)

    corr = (codes_c.T @ masks_c) / (n * codes_std.unsqueeze(1) * masks_std.unsqueeze(0))
    # corr: (p, m)

    atom_manifold = corr.argmax(dim=1).numpy()  # (p,) — index into m manifolds
    return atom_manifold


def plot_ising(J_np: np.ndarray, atom_manifold: np.ndarray, instances: list,
               active_idx: torch.Tensor, k: int, save_path: str):
    """Plot the Ising coupling matrix sorted by ground-truth manifold assignment."""
    # Sort atoms by manifold assignment (and by manifold index within each manifold)
    sort_order = np.argsort(atom_manifold, kind="stable")
    J_sorted = J_np[np.ix_(sort_order, sort_order)]
    manifold_sorted = atom_manifold[sort_order]

    # Build manifold labels for tick marks
    type_names = [inst.type_name for inst in instances]
    manifold_labels = [type_names[i] if i < len(type_names) else "?" for i in manifold_sorted]

    # Find manifold boundaries for block separators
    boundaries = [0]
    for i in range(1, len(manifold_sorted)):
        if manifold_sorted[i] != manifold_sorted[i - 1]:
            boundaries.append(i)
    boundaries.append(len(manifold_sorted))

    p = J_sorted.shape[0]
    nonzero = J_sorted[J_sorted != 0]
    vmax = float(np.percentile(np.abs(nonzero), 99)) if len(nonzero) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    im = ax.imshow(J_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Draw block separator lines
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="#aaa", linewidth=0.6, alpha=0.7)
        ax.axvline(b - 0.5, color="#aaa", linewidth=0.6, alpha=0.7)

    # Label blocks at midpoints
    for i in range(len(boundaries) - 1):
        mid = (boundaries[i] + boundaries[i + 1]) / 2
        mtype = type_names[manifold_sorted[boundaries[i]]] if manifold_sorted[boundaries[i]] < len(type_names) else "?"
        ax.text(mid, -1.5, mtype[:6], ha="center", va="bottom", fontsize=6,
                color="#ccc", rotation=45)
        ax.text(-1.5, mid, mtype[:6], ha="right", va="center", fontsize=6,
                color="#ccc")

    ax.set_title("Ising coupling — %d active atoms  (k=%d, sorted by manifold)" % (p, k),
                 color="#ddd", fontsize=11)
    ax.set_xlabel("Atom index (sorted by manifold)", color="#aaa", fontsize=10)
    ax.set_ylabel("Atom index (sorted by manifold)", color="#aaa", fontsize=10)
    ax.tick_params(colors="#aaa", labelbottom=False, labelleft=False)
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved %s" % save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4, help="SAE K to compute Ising for")
    parser.add_argument("--save-dir", type=str, default="checkpoints_mse")
    parser.add_argument("--n-eval", type=int, default=50_000)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--c", type=int, default=512)
    parser.add_argument("--L0", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("Device: %s" % device)

    # Build manifold instances (same seed as training)
    print("Building manifold instances...")
    instances = build_manifold_instances(d=args.d, seed=args.seed)
    print("  %d instances" % len(instances))

    # Generate eval data (same seed as in run_synthetic.py)
    print("Generating %d eval samples..." % args.n_eval)
    eval_data, eval_masks, _ = generate_dataset_fast(
        instances, n_samples=args.n_eval, L0=args.L0, seed=999,
        store_contributions=False,
    )
    print("  eval_data shape:", eval_data.shape, "eval_masks shape:", eval_masks.shape)

    # Load SAE checkpoint
    ckpt = os.path.join(args.save_dir, "sae_k%d.pt" % args.k)
    print("Loading checkpoint: %s" % ckpt)
    model = TopKSAE(args.d, args.c, args.k).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Encode eval data
    print("Encoding eval data...")
    with torch.no_grad():
        codes = model.encode(eval_data.to(device)).cpu()
    print("  codes shape:", codes.shape, "nnz fraction:", (codes != 0).float().mean().item())

    # Fit Ising model
    print("Fitting Ising model (%d steps)..." % args.n_steps)
    J_active, active_idx = compute_ising_coupling(
        codes, lam=args.lam, n_steps=args.n_steps, device=device,
    )
    print("  |J|_max=%.4f" % J_active.abs().max().item())

    # Assign atoms to manifolds by correlation
    print("Assigning atoms to manifolds...")
    atom_manifold = assign_atoms_to_manifolds(codes, eval_masks, active_idx, instances)
    unique, counts = np.unique(atom_manifold, return_counts=True)
    for u, c_val in zip(unique, counts):
        mtype = instances[u].type_name if u < len(instances) else "?"
        print("  manifold %d (%s): %d atoms" % (u, mtype, c_val))

    # Plot
    J_np = J_active.numpy()
    save_path = os.path.join(args.save_dir, "ising_coupling_k%d_sorted.png" % args.k)
    plot_ising(J_np, atom_manifold, instances, active_idx, args.k, save_path)
    print("Done.")


if __name__ == "__main__":
    main()
