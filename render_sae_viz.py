"""Two complementary SAE visualizations for each concept manifold.

Plot A — SAE-space PCA:
    Encode activations through the SAE, keep only the top-N SAE atom codes
    (selected by label correlation), then PCA-reduce that N-dim space to 3D.
    This shows "the manifold as the SAE sees it."

Plot B — SAE decoder directions in activation PCA space:
    Run PCA on the raw activations (3840D → 3D), then overlay arrows for each
    of the top-N SAE decoder columns projected into that PCA coordinate system.
    Shows which directions in the PCA space each SAE feature is "pointing at."

Usage:
    uv run render_sae_viz.py --sae-local-path /path/to/params.safetensors
    uv run render_sae_viz.py --sae-local-path ... --n-atoms 8 --output-dir figures/gemma_eval
"""

import argparse
import colorsys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.decomposition import PCA

from nonlinear_features.jumprelu_sae import JumpReLUSAE
from nonlinear_features.evaluate_real import (
    _expand_labels,
    select_atoms_by_label_correlation,
)

MANIFOLDS = ["colors", "temperature", "days", "years"]
TITLES = {
    "colors":      "Colors (HSV)",
    "temperature": "Temperature (°F)",
    "days":        "Days of week",
    "years":       "Years (1825–2024)",
}

ARROW_COLORS = [
    "#FF4444", "#44AAFF", "#44FF88", "#FFD700",
    "#FF88FF", "#FF8800", "#00DDDD", "#AAAAAA",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(acts_dir: str) -> dict:
    result = {}
    for name in MANIFOLDS:
        path = Path(acts_dir) / f"activations_{name}.pt"
        d = torch.load(path, weights_only=False)
        acts = d["activations"]
        labels = d["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        result[name] = {"acts": acts, "labels": labels}
    return result


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def make_point_colors(labels: np.ndarray, name: str):
    """Return (color_data, colorscale_name_or_None) for scatter points."""
    if name == "colors":
        colors = []
        for h, s, v in labels:
            r, g, b = colorsys.hsv_to_rgb(float(h), float(s), float(v))
            colors.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
        return colors, None
    elif name == "days":
        return labels.tolist(), "hsv"
    else:
        return labels.tolist(), "Viridis"


def scatter3d(proj: np.ndarray, labels: np.ndarray, name: str) -> go.Scatter3d:
    color_data, colorscale = make_point_colors(labels, name)
    marker = dict(size=2, opacity=0.6)
    if colorscale is None:
        marker["color"] = color_data
    else:
        marker.update(color=color_data, colorscale=colorscale,
                      colorbar=dict(thickness=10, len=0.6))
    return go.Scatter3d(
        x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
        mode="markers", marker=marker, name="activations", showlegend=False,
    )


# ---------------------------------------------------------------------------
# Plot A: SAE-space PCA
# ---------------------------------------------------------------------------

def build_sae_space_fig(name: str, acts: torch.Tensor, labels: np.ndarray,
                         codes: torch.Tensor, selected: list[int]) -> go.Figure:
    codes_sel = codes[:, selected].cpu().numpy()   # (n, N)
    pca = PCA(n_components=min(3, codes_sel.shape[1]))
    proj = pca.fit_transform(codes_sel)
    var = pca.explained_variance_ratio_

    trace = scatter3d(proj, labels, name)
    fig = go.Figure(trace)
    subtitle = (f"SAE codes of top-{len(selected)} atoms | "
                f"PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}  "
                f"(top-3: {sum(var[:3]):.1%})")
    fig.update_layout(
        title=dict(text=f"<b>{TITLES[name]} — SAE Space</b><br><sup>{subtitle}</sup>", x=0.5),
        scene=dict(xaxis_title="SAE-PC1", yaxis_title="SAE-PC2", zaxis_title="SAE-PC3",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))),
        margin=dict(l=0, r=0, t=80, b=0), height=680,
        paper_bgcolor="#111", plot_bgcolor="#1a1a1a",
        font=dict(color="#ddd"),
    )
    return fig


# ---------------------------------------------------------------------------
# Plot B: SAE decoder directions in activation PCA space
# ---------------------------------------------------------------------------

def _atom_corr_strength(codes: torch.Tensor, selected: list[int],
                         labels: np.ndarray, name: str) -> list[float]:
    """Return max-abs label correlation for each selected atom."""
    labels_t = torch.tensor(labels, dtype=torch.float32)
    labels_exp = _expand_labels(labels_t) if labels_t.ndim > 1 else labels_t

    n = codes.shape[0]
    codes_std = codes.std(dim=0).clamp(min=1e-8)
    codes_c = codes - codes.mean(dim=0)

    strengths = []
    for atom_idx in selected:
        if labels_exp.ndim == 1:
            li = (labels_exp - labels_exp.mean()) / (labels_exp.std() + 1e-8)
            r = ((codes_c[:, atom_idx] @ li) / (n * codes_std[atom_idx])).abs().item()
        else:
            r = 0.0
            for i in range(labels_exp.shape[1]):
                li = labels_exp[:, i]
                li = (li - li.mean()) / (li.std() + 1e-8)
                c = ((codes_c[:, atom_idx] @ li) / (n * codes_std[atom_idx])).abs().item()
                r = max(r, c)
        strengths.append(r)
    return strengths


def build_sae_vectors_fig(name: str, acts: torch.Tensor, labels: np.ndarray,
                           selected: list[int], decoder_weights: torch.Tensor,
                           codes: torch.Tensor) -> go.Figure:
    acts_np = acts.cpu().numpy()
    pca = PCA(n_components=3)
    proj = pca.fit_transform(acts_np)        # (n, 3)
    var = pca.explained_variance_ratio_

    # Project decoder columns into PCA space
    # pca.components_: (3, d_in), decoder_weights: (d_in, d_sae)
    D_sel = decoder_weights[:, selected].cpu().numpy()  # (d_in, N)
    pca_vecs = pca.components_ @ D_sel                  # (3, N)

    # Scale arrows to ~25% of mean data range, further scaled by correlation
    data_range = np.array([proj[:, i].max() - proj[:, i].min() for i in range(3)]).mean()
    arrow_scale = 0.25 * data_range

    corr_strengths = _atom_corr_strength(codes, selected, labels, name)
    max_r = max(corr_strengths) if corr_strengths else 1.0

    centroid = proj.mean(axis=0)

    traces = [scatter3d(proj, labels, name)]

    for k, atom_idx in enumerate(selected):
        v = pca_vecs[:, k]           # (3,) in PCA space
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            continue

        rel_r = corr_strengths[k] / max_r if max_r > 0 else 0.5
        tip = centroid + (v / v_norm) * arrow_scale * (0.4 + 0.6 * rel_r)

        # Proportion of decoder direction captured by top-3 PCs
        pca_capture = min(v_norm, 1.0)

        color = ARROW_COLORS[k % len(ARROW_COLORS)]
        traces.append(go.Scatter3d(
            x=[centroid[0], tip[0]],
            y=[centroid[1], tip[1]],
            z=[centroid[2], tip[2]],
            mode="lines",
            line=dict(color=color, width=8),
            name=f"atom {atom_idx}  r={corr_strengths[k]:.2f}",
            hovertemplate=(
                f"<b>Atom {atom_idx}</b><br>"
                f"label corr: {corr_strengths[k]:.3f}<br>"
                f"PCA capture: {pca_capture:.2f}<br>"
                f"PCA vec: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"
                "<extra></extra>"
            ),
        ))
        # Arrowhead as cone
        traces.append(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[float(v[0])], v=[float(v[1])], w=[float(v[2])],
            sizemode="scaled", sizeref=arrow_scale * 0.25,
            colorscale=[[0, color], [1, color]],
            showscale=False, showlegend=False,
            hoverinfo="skip",
        ))

    subtitle = (f"Activation PCA + top-{len(selected)} SAE decoder directions | "
                f"PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}")
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=f"<b>{TITLES[name]} — SAE Directions in Activation PCA Space</b><br><sup>{subtitle}</sup>", x=0.5),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))),
        margin=dict(l=0, r=0, t=80, b=0), height=680,
        paper_bgcolor="#111", plot_bgcolor="#1a1a1a",
        font=dict(color="#ddd"),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", font=dict(size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _tabbed_html(sections: list[tuple[str, list[tuple[str, str]]]], page_title: str) -> str:
    """Build a dark-theme HTML page with two groups of tabs."""
    all_buttons = []
    all_divs = []

    for section_label, figs in sections:
        all_buttons.append(
            f'<div class="section-label">{section_label}</div>'
        )
        for tab_name, html in figs:
            btn_id = f"{section_label}_{tab_name}"
            all_buttons.append(
                f'<button class="tab-btn" onclick="showTab(\'{btn_id}\')" id="btn-{btn_id}">'
                f'{TITLES.get(tab_name, tab_name)}</button>'
            )
            all_divs.append(
                f'<div class="tab-pane" id="tab-{btn_id}" style="display:none">{html}</div>'
            )

    first_id = f"{sections[0][0]}_{sections[0][1][0][0]}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{page_title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #0f0f0f; color: #eee; }}
  h1 {{ text-align: center; padding: 16px 0 4px; font-size: 1.2em; color: #ccc; }}
  .tab-bar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
               justify-content: center; padding: 4px 8px 12px; border-bottom: 1px solid #333; }}
  .section-label {{ color: #888; font-size: 0.75em; text-transform: uppercase;
                    letter-spacing: .08em; margin: 4px 4px 0; }}
  .tab-btn {{ padding: 6px 16px; border: 1px solid #444; border-radius: 5px;
              background: #1e1e1e; color: #bbb; cursor: pointer; font-size: 0.88em;
              transition: background .12s; }}
  .tab-btn:hover {{ background: #2a2a2a; }}
  .tab-btn.active {{ background: #3a6ea8; color: #fff; border-color: #3a6ea8; }}
  .tab-pane {{ padding: 0 12px 12px; }}
</style>
</head>
<body>
<h1>{page_title}</h1>
<div class="tab-bar">
{''.join(all_buttons)}
</div>
{''.join(all_divs)}
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-pane').forEach(d => d.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).style.display = 'block';
  document.getElementById('btn-' + id).classList.add('active');
}}
showTab('{first_id}');
document.getElementById('btn-{first_id}').classList.add('active');
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAE visualization (SAE-space + decoder directions)")
    parser.add_argument("--activations-dir", default="figures/gemma_pca")
    parser.add_argument("--sae-local-path", required=True,
                        help="Path to GemmaScope params.safetensors")
    parser.add_argument("--n-atoms", type=int, default=8)
    parser.add_argument("--output-dir", default="figures/gemma_eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default="16k", help="Tag for output filename")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading SAE from {args.sae_local_path}")
    sae = JumpReLUSAE.from_pretrained(args.sae_local_path, device=device)
    decoder_weights = sae.W_dec.data.T  # (d_in, d_sae), keep on CPU for numpy ops
    if decoder_weights.is_cuda:
        decoder_weights_cpu = decoder_weights.cpu()
    else:
        decoder_weights_cpu = decoder_weights
    print(f"SAE: d_in={sae.d_in}, d_sae={sae.d_sae}")

    print(f"Loading activations from {args.activations_dir}")
    data = load_data(args.activations_dir)

    sae_space_figs = []
    sae_vector_figs = []

    for name in MANIFOLDS:
        print(f"\n--- {name} ---")
        acts = data[name]["acts"].to(device)
        labels = data[name]["labels"]
        labels_t = torch.tensor(labels, dtype=torch.float32).to(device)

        with torch.no_grad():
            codes = sae.encode(acts)   # (n, d_sae)

        selected = select_atoms_by_label_correlation(codes, labels_t, args.n_atoms)
        print(f"  Top-{len(selected)} atoms: {selected}")

        print("  Building SAE-space PCA plot...")
        fig_a = build_sae_space_fig(name, acts, labels, codes, selected)
        sae_space_figs.append((name, fig_a.to_html(full_html=False, include_plotlyjs=False)))

        print("  Building SAE decoder directions plot...")
        fig_b = build_sae_vectors_fig(name, acts, labels, selected,
                                       decoder_weights_cpu, codes.cpu())
        sae_vector_figs.append((name, fig_b.to_html(full_html=False, include_plotlyjs=False)))

    out_path = Path(args.output_dir) / f"sae_viz_{args.tag}.html"
    html = _tabbed_html(
        sections=[
            ("SAE Space (codes → PCA)", sae_space_figs),
            ("SAE Directions in Activation PCA", sae_vector_figs),
        ],
        page_title=f"SAE Visualization — GemmaScope {args.tag}",
    )
    out_path.write_text(html)
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
