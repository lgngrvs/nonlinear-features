"""
Per-PC R² and top-3 PCA visualisation for attn_out at layers 0 & 1.
Also probes raw token embeddings (before any attention) as a baseline.
"""

import argparse, json
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

CONTROL_UNIT  = "ab "
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
def collect_activations(model, tokenizer, device,
                        unit=CONTROL_UNIT, widths=CONTROL_WIDTHS, n_ctx=3):
    """
    Returns dict with keys:
      'embed'       : (N, d) – raw token embeddings before any layer
      'attn_out_0'  : (N, d) – post_attention_layernorm output, layer 0
      'attn_out_1'  : (N, d) – post_attention_layernorm output, layer 1
      'char_pos'    : (N,)   – int
      'width'       : (N,)   – int
    """
    unit_len  = len(unit)
    unit_toks = tokenizer.encode(unit, add_special_tokens=False)
    decoder   = model.model.language_model.layers

    buffers = {k: [] for k in ("embed", "attn_out_0", "attn_out_1", "char_pos", "width")}

    for width in widths:
        upl = width // unit_len          # units per line
        ctx = ("\n".join([(unit * upl).rstrip()] * n_ctx)) + "\n"
        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)

        line_ids: list[int] = []
        for n_units in tqdm(range(1, upl + 4), desc=f"w={width}"):
            line_ids = line_ids + unit_toks
            full_ids  = ctx_ids + line_ids
            tok_idx   = len(full_ids) - 1
            char_pos  = (n_units - 1) * unit_len

            captured = {}

            def make_hook(key):
                def h(module, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    captured[key] = t[0, tok_idx].float().cpu()
                return h

            h0 = decoder[0].post_attention_layernorm.register_forward_hook(make_hook("attn_out_0"))
            h1 = decoder[1].post_attention_layernorm.register_forward_hook(make_hook("attn_out_1"))

            ids_t = torch.tensor([full_ids], dtype=torch.long).to(device)
            out   = model(ids_t, output_hidden_states=True)

            h0.remove(); h1.remove()

            # raw embedding = hidden_states[0] (before any transformer layer)
            buffers["embed"].append(out.hidden_states[0][0, tok_idx].float().cpu().numpy())
            buffers["attn_out_0"].append(captured["attn_out_0"].numpy())
            buffers["attn_out_1"].append(captured["attn_out_1"].numpy())
            buffers["char_pos"].append(char_pos)
            buffers["width"].append(width)

    return {k: np.array(v) for k, v in buffers.items()}


def per_pc_r2(X, y, n_components=20):
    """R² of linear regression char_pos ~ PCᵢ for each component individually."""
    pca   = PCA(n_components=min(n_components, X.shape[0]-1, X.shape[1]))
    Z     = pca.fit_transform(X)          # (N, k)
    r2s   = []
    for i in range(Z.shape[1]):
        z  = Z[:, i:i+1]
        lr = LinearRegression().fit(z, y)
        ss_res = ((y - lr.predict(z))**2).sum()
        ss_tot = ((y - y.mean())**2).sum()
        r2s.append(1 - ss_res / ss_tot)
    return np.array(r2s), pca, Z


def plot_top3(Z, y_char, widths_arr, pca, r2s, label, save_dir):
    """3-column scatter: PC0, PC1, PC2 vs char_pos, coloured by width."""
    uniq_w = sorted(set(widths_arr))
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=[f"PC{i+1}  R²={r2s[i]:.4f}" for i in range(3)])

    for col, pc_idx in enumerate([0, 1, 2], 1):
        for wi, w in enumerate(uniq_w):
            mask = widths_arr == w
            fig.add_trace(go.Scatter(
                x=y_char[mask], y=Z[mask, pc_idx],
                mode="markers",
                marker=dict(color=WIDTH_COLORS[wi], size=6, opacity=0.8),
                name=f"w={w}", showlegend=(col == 1),
            ), row=1, col=col)
        fig.update_xaxes(title_text="char_pos", row=1, col=col)
        fig.update_yaxes(title_text=f"PC{col}", row=1, col=col)

    fig.update_layout(
        title=f"{label}: top-3 PCA axes vs char_pos  (coloured by width)",
        height=420, width=1100,
    )
    p = Path(save_dir) / f"pca_top3_{label.replace(' ','_')}.html"
    fig.write_html(str(p))
    print(f"  Saved {p}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="google/gemma-3-12b-it")
    ap.add_argument("--device",   default="auto")
    ap.add_argument("--save-dir", default="figures/linebreak_mech")
    ap.add_argument("--n-pcs",    type=int, default=20)
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model, device)

    print("\nCollecting activations...")
    data = collect_activations(model, tok, device)

    y        = data["char_pos"].astype(float)
    widths_a = data["width"]

    summary = {}

    for key, label in [
        ("embed",     "embed (before any attn)"),
        ("attn_out_0","attn_out L0"),
        ("attn_out_1","attn_out L1"),
    ]:
        X = data[key]
        r2s, pca, Z = per_pc_r2(X, y, n_components=args.n_pcs)
        cumvar = pca.explained_variance_ratio_.cumsum()

        print(f"\n{label}")
        print(f"  Explained var (first {args.n_pcs} PCs): {pca.explained_variance_ratio_[:5]}")
        for i in range(min(args.n_pcs, len(r2s))):
            bar = "#" * max(0, int(r2s[i] * 20))
            print(f"  PC{i+1:2d}: R²={r2s[i]:+.4f}  {bar}")

        summary[key] = {
            "per_pc_r2": r2s.tolist(),
            "explained_var": pca.explained_variance_ratio_.tolist(),
        }

        if key != "embed":
            plot_top3(Z, y, widths_a, pca, r2s, label, args.save_dir)

    out = Path(args.save_dir) / "pca_viz_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out}")


if __name__ == "__main__":
    main()
