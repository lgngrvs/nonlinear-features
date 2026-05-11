"""Interactive 3D PCA visualization for Gemma activation manifolds.

Produces a single self-contained HTML file with one tab per manifold,
fully rotatable/zoomable in the browser.
"""

import colorsys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

SAVE_DIR = Path("figures/gemma_pca")
OUT_HTML = SAVE_DIR / "pca_interactive.html"

MANIFOLDS = ["colors", "temperature", "days", "years"]

TITLES = {
    "colors": "Colors (hex → HSV paraboloid)",
    "temperature": "Temperature (°F line)",
    "days": "Days of week (circular)",
    "years": "Years (helix)",
}


def load_and_project(name):
    data = torch.load(SAVE_DIR / f"activations_{name}.pt", weights_only=False)
    acts = data["activations"].numpy()
    labels = data["labels"]
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    pca = PCA(n_components=3)
    projected = pca.fit_transform(acts)
    return projected, labels, pca.explained_variance_ratio_


def make_rgba_strings(labels, name):
    """Return list of 'rgb(r,g,b)' strings and a colorscale name for the colorbar."""
    if labels.ndim == 2:
        # Colors manifold: use HSV directly
        rgbs = []
        for h, s, v in labels:
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            rgbs.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)})")
        return rgbs, None
    else:
        return labels, "hsv" if name == "days" else "Viridis"


def build_trace(name):
    projected, labels, var = load_and_project(name)
    color_data, colorscale = make_rgba_strings(labels, name)

    if colorscale is None:
        # Literal RGB strings — no colorbar
        trace = go.Scatter3d(
            x=projected[:, 0], y=projected[:, 1], z=projected[:, 2],
            mode="markers",
            marker=dict(size=2, color=color_data, opacity=0.7),
            name=name,
            text=[f"HSV ({l[0]:.2f}, {l[1]:.2f}, {l[2]:.2f})" for l in labels],
            hovertemplate="%{text}<extra></extra>",
        )
    else:
        trace = go.Scatter3d(
            x=projected[:, 0], y=projected[:, 1], z=projected[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=color_data,
                colorscale=colorscale,
                opacity=0.7,
                colorbar=dict(thickness=12, len=0.6),
            ),
            name=name,
            text=[f"{l:.2f}" for l in labels],
            hovertemplate="value: %{text}<extra></extra>",
        )

    subtitle = (
        f"n={len(projected):,}  |  "
        f"PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}  "
        f"(top-3: {sum(var):.1%})"
    )
    return trace, subtitle


# One figure per manifold, assembled into tabs via HTML buttons
figures_html = []

for name in MANIFOLDS:
    print(f"Building {name}...")
    trace, subtitle = build_trace(name)

    fig = go.Figure(trace)
    fig.update_layout(
        title=dict(text=f"<b>{TITLES[name]}</b><br><sup>{subtitle}</sup>", x=0.5),
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=700,
    )
    figures_html.append((name, fig.to_html(full_html=False, include_plotlyjs=False)))

# Stitch into a tabbed single-page HTML
tab_buttons = "\n".join(
    f'<button class="tab-btn" onclick="showTab(\'{name}\')" id="btn-{name}">'
    f'{TITLES[name]}</button>'
    for name, _ in figures_html
)

tab_divs = "\n".join(
    f'<div class="tab-pane" id="tab-{name}" style="display:none">{html}</div>'
    for name, html in figures_html
)

first_name = figures_html[0][0]

html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gemma 3 12B — Activation Manifolds (PCA)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #0f0f0f; color: #eee; }}
  h1 {{ text-align: center; padding: 18px 0 4px; font-size: 1.3em; color: #ddd; }}
  .tab-bar {{ display: flex; gap: 8px; justify-content: center; padding: 0 0 16px; flex-wrap: wrap; }}
  .tab-btn {{
    padding: 7px 18px; border: 1px solid #555; border-radius: 6px;
    background: #1e1e1e; color: #bbb; cursor: pointer; font-size: 0.9em;
    transition: background 0.15s;
  }}
  .tab-btn:hover {{ background: #2a2a2a; }}
  .tab-btn.active {{ background: #3a6ea8; color: #fff; border-color: #3a6ea8; }}
  .tab-pane {{ padding: 0 16px 16px; }}
</style>
</head>
<body>
<h1>Gemma 3 12B &mdash; Activation Manifolds (PCA, layer 24)</h1>
<div class="tab-bar">
{tab_buttons}
</div>
{tab_divs}
<script>
function showTab(name) {{
  document.querySelectorAll('.tab-pane').forEach(d => d.style.display = 'none');
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).style.display = 'block';
  document.getElementById('btn-' + name).classList.add('active');
}}
showTab('{first_name}');
document.getElementById('btn-{first_name}').classList.add('active');
</script>
</body>
</html>
"""

OUT_HTML.write_text(html_page)
print(f"\nSaved → {OUT_HTML}  ({OUT_HTML.stat().st_size / 1e6:.1f} MB)")
