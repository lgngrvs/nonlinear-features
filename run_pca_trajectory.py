"""
PCA trajectory plot: how does attn_out move through activation space as char_pos varies?

For each layer (0, 1) and each width (40,60,80,100,120):
  - collect attn_out activations at every char_pos step (3 chars / "ab " token)
  - run PCA on the combined points across all widths for that layer
  - plot the trajectory in PC1-PC2-PC3 space, one coloured curve per width
  - also plot explained variance and per-PC running variance to judge dimensionality

The key question: does the trajectory live in 1D (straight line), 2D (arc/circle),
or 3D (helix/spiral)?  RoPE uses sin/cos pairs so 2D arcs per frequency are expected.
"""

import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm import tqdm

CONTROL_UNIT   = "ab "
CONTROL_WIDTHS = (40, 60, 80, 100, 120)
WIDTH_COLORS   = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5"]


def load_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    device_map = "auto" if n_gpus > 1 else (device if device not in ("cpu","mps") else None)
    dtype = torch.bfloat16 if device not in ("cpu","mps") else torch.float32
    print(f"Loading {model_name}...")
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map)
    if device in ("cpu","mps"):
        model = model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def collect(model, tokenizer, device, layers=(0, 1),
            unit=CONTROL_UNIT, widths=CONTROL_WIDTHS, n_ctx=3):
    """
    Returns: {layer: {"X": (N,d), "char_pos": (N,), "width": (N,)}}
    """
    from sklearn.decomposition import PCA

    unit_len  = len(unit)
    unit_toks = tokenizer.encode(unit, add_special_tokens=False)
    decoder   = model.model.language_model.layers

    bufs = {l: {"X": [], "char_pos": [], "width": []} for l in layers}

    for width in widths:
        upl     = width // unit_len
        ctx     = ("\n".join([(unit * upl).rstrip()] * n_ctx)) + "\n"
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)

        line_ids: list[int] = []
        for n_units in tqdm(range(1, upl + 4), desc=f"w={width}"):
            line_ids = line_ids + unit_toks
            full_ids  = ctx_ids + line_ids
            tok_idx   = len(full_ids) - 1
            char_pos  = (n_units - 1) * unit_len

            captured = {}
            hooks = []
            for l in layers:
                def make_hook(li):
                    def h(module, inp, out):
                        t = out[0] if isinstance(out, tuple) else out
                        captured[li] = t[0, tok_idx].float().cpu().numpy()
                    return h
                hooks.append(
                    decoder[l].post_attention_layernorm.register_forward_hook(make_hook(l))
                )

            ids_t = torch.tensor([full_ids], dtype=torch.long).to(device)
            model(ids_t)
            for h in hooks:
                h.remove()

            for l in layers:
                bufs[l]["X"].append(captured[l])
                bufs[l]["char_pos"].append(char_pos)
                bufs[l]["width"].append(width)

    return {l: {k: np.array(v) for k, v in bufs[l].items()} for l in layers}


def trajectory_plot(data, layer, widths, save_dir):
    from sklearn.decomposition import PCA

    X        = data["X"]            # (N, d)
    char_pos = data["char_pos"]     # (N,)
    width_a  = data["width"]        # (N,)

    # Fit PCA on all points combined
    n_comp = min(10, X.shape[0] - 1, X.shape[1])
    pca    = PCA(n_components=n_comp)
    Z      = pca.fit_transform(X)   # (N, n_comp)
    ev     = pca.explained_variance_ratio_

    print(f"\n  L{layer} explained variance: {[f'{v:.3f}' for v in ev[:6]]}")
    print(f"  Cumulative: {np.cumsum(ev[:6]).round(3).tolist()}")

    # ── 3-D trajectory plot ──────────────────────────────────────────────────
    fig3d = go.Figure()
    for wi, w in enumerate(sorted(set(widths))):
        mask = width_a == w
        cp   = char_pos[mask]
        z    = Z[mask]
        sort = np.argsort(cp)
        cp_s, z_s = cp[sort], z[sort]

        fig3d.add_trace(go.Scatter3d(
            x=z_s[:, 0], y=z_s[:, 1], z=z_s[:, 2],
            mode="lines+markers",
            line=dict(color=WIDTH_COLORS[wi], width=4),
            marker=dict(
                size=5,
                color=cp_s,
                colorscale="Plasma",
                showscale=(wi == 0),
                colorbar=dict(title="char_pos", x=1.02) if wi == 0 else None,
            ),
            name=f"width={w}",
        ))

    fig3d.update_layout(
        title=f"attn_out L{layer}: trajectory in PCA space as char_pos varies<br>"
              f"<sup>PC1={ev[0]:.1%}  PC2={ev[1]:.1%}  PC3={ev[2]:.1%} variance — "
              f"cumulative: {ev[:3].sum():.1%}</sup>",
        scene=dict(
            xaxis_title=f"PC1 ({ev[0]:.1%})",
            yaxis_title=f"PC2 ({ev[1]:.1%})",
            zaxis_title=f"PC3 ({ev[2]:.1%})",
        ),
        height=700, width=900,
        legend=dict(x=0, y=1),
    )
    p3 = Path(save_dir) / f"attn_trajectory_L{layer}_3d.html"
    fig3d.write_html(str(p3))
    print(f"  Saved {p3}")

    # ── 2-D panels: PC1v2, PC1v3, PC2v3 ─────────────────────────────────────
    fig2d = make_subplots(rows=1, cols=3,
        subplot_titles=[
            f"PC1 vs PC2  (cum {ev[:2].sum():.1%})",
            f"PC1 vs PC3  (cum {ev[:3].sum():.1%})",
            f"PC2 vs PC3",
        ])
    pairs = [(0,1), (0,2), (1,2)]

    for col, (i, j) in enumerate(pairs, 1):
        for wi, w in enumerate(sorted(set(widths))):
            mask  = width_a == w
            cp    = char_pos[mask]
            z     = Z[mask]
            sort  = np.argsort(cp)
            cp_s  = cp[sort]
            z_s   = z[sort]
            fig2d.add_trace(go.Scatter(
                x=z_s[:, i], y=z_s[:, j],
                mode="lines+markers",
                marker=dict(size=5, color=WIDTH_COLORS[wi]),
                line=dict(color=WIDTH_COLORS[wi]),
                name=f"w={w}", showlegend=(col == 1),
            ), row=1, col=col)
        fig2d.update_xaxes(title_text=f"PC{i+1}", row=1, col=col)
        fig2d.update_yaxes(title_text=f"PC{j+1}", row=1, col=col)

    fig2d.update_layout(
        title=f"attn_out L{layer}: 2-D PCA projections of char_pos trajectory",
        height=450, width=1200,
    )
    p2 = Path(save_dir) / f"attn_trajectory_L{layer}_2d.html"
    fig2d.write_html(str(p2))
    print(f"  Saved {p2}")

    # ── Scree / cumulative variance ──────────────────────────────────────────
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=list(range(1, n_comp+1)), y=ev.tolist(),
        name="Individual", marker_color="#457b9d",
    ))
    fig_scree.add_trace(go.Scatter(
        x=list(range(1, n_comp+1)), y=np.cumsum(ev).tolist(),
        mode="lines+markers", name="Cumulative",
        line=dict(color="#e63946", width=2),
        yaxis="y2",
    ))
    fig_scree.update_layout(
        title=f"attn_out L{layer}: PCA scree plot (intrinsic dimensionality of position trajectory)",
        xaxis_title="PC", yaxis_title="Explained variance",
        yaxis2=dict(title="Cumulative", overlaying="y", side="right", range=[0,1]),
        height=400,
    )
    p_s = Path(save_dir) / f"attn_trajectory_L{layer}_scree.html"
    fig_scree.write_html(str(p_s))
    print(f"  Saved {p_s}")

    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="google/gemma-3-12b-it")
    ap.add_argument("--device",   default="auto")
    ap.add_argument("--save-dir", default="figures/linebreak_mech")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device: {device}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model, device)

    print("\nCollecting attn_out activations (L0 and L1)...")
    layer_data = collect(model, tok, device, layers=(0, 1))

    for l in (0, 1):
        print(f"\n{'='*55}\nLayer {l}\n{'='*55}")
        trajectory_plot(layer_data[l], l, CONTROL_WIDTHS, args.save_dir)


if __name__ == "__main__":
    main()
