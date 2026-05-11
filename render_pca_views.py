"""Render multiple rotation views of saved Gemma PCA activations."""

import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

SAVE_DIR = Path("figures/gemma_pca")

VIEWS = [
    (30, -60),
    (30, -120),
    (30, 30),
    (30, 150),
    (70, -60),
    (10, -90),
]

MANIFOLDS = ["colors", "temperature", "days", "years"]


def load_and_project(name):
    data = torch.load(SAVE_DIR / f"activations_{name}.pt", weights_only=False)
    acts = data["activations"].numpy()
    labels = data["labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    pca = PCA(n_components=3)
    projected = pca.fit_transform(acts)
    return projected, labels, pca.explained_variance_ratio_


def make_colors(labels, name):
    if labels.ndim == 2:
        return np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in labels])
    norm = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
    cmap = plt.cm.hsv if name == "days" else plt.cm.viridis
    return cmap(norm)[:, :3]


for name in MANIFOLDS:
    print(f"Rendering {name}...")
    projected, labels, var = load_and_project(name)
    colors = make_colors(labels, name)

    n_views = len(VIEWS)
    fig = plt.figure(figsize=(5 * n_views, 4))
    fig.suptitle(f"{name} — Gemma 3 12B  (top-3 var: {sum(var):.1%})", fontsize=13)

    for i, (elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, n_views, i + 1, projection="3d")
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                   c=colors, s=1, alpha=0.5)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(f"PC1 {var[0]:.1%}", fontsize=7)
        ax.set_ylabel(f"PC2 {var[1]:.1%}", fontsize=7)
        ax.set_zlabel(f"PC3 {var[2]:.1%}", fontsize=7)
        ax.set_title(f"el={elev}° az={azim}°", fontsize=8)
        ax.tick_params(labelsize=6)

    plt.tight_layout()
    out = SAVE_DIR / f"pca_{name}_multiview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

print("Done.")
