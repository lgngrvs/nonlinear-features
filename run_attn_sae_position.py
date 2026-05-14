"""
Find SAE features in attn_out at layers 0 and 1 that encode char_pos.

For each layer:
  1. Collect attn_out activations via hook on post_attention_layernorm
  2. Pass through SAE encoder → sparse feature activations
  3. Rank features by Pearson r² with char_pos
  4. Plot top features: activation vs char_pos, coloured by width
  5. Show whether top features are monotone (position counter) or threshold (near-break)
"""

import argparse, json
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from tqdm import tqdm

from nonlinear_features.jumprelu_sae import JumpReLUSAE

CONTROL_UNIT   = "ab "
CONTROL_WIDTHS = (40, 60, 80, 100, 120)
WIDTH_COLORS   = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5"]
TOP_N          = 20


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
def collect_attn_sae_records(model, tokenizer, saes, device,
                              unit=CONTROL_UNIT, widths=CONTROL_WIDTHS, n_ctx=3):
    """
    One forward pass per position. Hooks capture attn_out at each requested layer.
    Returns list of dicts: {char_pos, width, remaining, feats: {layer_idx: np.array}}
    """
    unit_len  = len(unit)
    unit_toks = tokenizer.encode(unit, add_special_tokens=False)
    decoder   = model.model.language_model.layers
    layers    = sorted(saes.keys())

    records = []

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
                    # SAE is trained on input to o_proj (pre output-projection)
                    def h(module, inp):
                        t = inp[0] if isinstance(inp, tuple) else inp
                        captured[li] = t[0, tok_idx].float().cpu()
                    return h
                hooks.append(
                    decoder[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))
                )

            ids_t = torch.tensor([full_ids], dtype=torch.long).to(device)
            model(ids_t)
            for h in hooks:
                h.remove()

            # Pass each layer's attn_out through its SAE
            feats = {}
            for l in layers:
                with torch.no_grad():
                    f = saes[l].encode(captured[l].unsqueeze(0)).squeeze(0)
                feats[l] = f.numpy()

            records.append({
                "char_pos":  int(char_pos),
                "width":     int(width),
                "remaining": int(width - char_pos),
                "feats":     feats,
            })

    return records


def pearson_r2_per_feature(feats_matrix, y):
    """feats_matrix: (N, d_sae). y: (N,). Returns r² for each feature."""
    # Only consider features that are non-zero in at least 5% of samples
    active = (feats_matrix > 0).mean(0) >= 0.05
    r2 = np.zeros(feats_matrix.shape[1])
    if active.sum() == 0:
        return r2
    X = feats_matrix[:, active]
    ym = y - y.mean()
    Xm = X - X.mean(0)
    denom_y = (ym**2).sum()
    denom_X = (Xm**2).sum(0)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(denom_X > 0, (ym @ Xm) / np.sqrt(denom_y * denom_X), 0.0)
    r2[active] = r**2
    return r2


def plot_feature_profiles(records, layer, top_features, widths, save_dir, label):
    """For each top feature: activation vs char_pos, one line per width."""
    all_feats = np.stack([r["feats"][layer] for r in records])
    char_pos  = np.array([r["char_pos"]     for r in records])
    width_arr = np.array([r["width"]        for r in records])

    n = min(8, len(top_features))
    fig = make_subplots(rows=2, cols=4,
        subplot_titles=[f"F{fi}" for fi in top_features[:n]])

    uniq_w = sorted(set(widths))
    for idx, fi in enumerate(top_features[:n]):
        row, col = divmod(idx, 4)
        for wi, w in enumerate(uniq_w):
            mask = width_arr == w
            cp   = char_pos[mask]
            fv   = all_feats[mask, fi]
            sort = np.argsort(cp)
            fig.add_trace(go.Scatter(
                x=cp[sort], y=fv[sort],
                mode="lines+markers",
                marker=dict(size=5, color=WIDTH_COLORS[wi]),
                line=dict(color=WIDTH_COLORS[wi]),
                name=f"w={w}", showlegend=(idx == 0),
            ), row=row+1, col=col+1)
            # vertical line at break point
            fig.add_vline(x=float(w), line_dash="dot",
                          line_color=WIDTH_COLORS[wi], row=row+1, col=col+1)

    fig.update_layout(
        title=f"{label}: top position-correlated SAE features vs char_pos",
        height=700, width=1200,
    )
    p = Path(save_dir) / f"attn_sae_{label.replace(' ','_')}_profiles.html"
    fig.write_html(str(p))
    print(f"  Saved {p}")


def plot_r2_bar(r2, top_pos, top_neg, layer, save_dir):
    top_all = top_pos[:10] + top_neg[:5]
    labels  = [f"F{fi}" for fi in top_all]
    vals    = [float(r2[fi]) for fi in top_all]
    colors  = ["#e63946"] * 10 + ["#457b9d"] * 5

    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title=f"attn_out L{layer}: per-feature Pearson r² with char_pos<br>"
              "<sup>Red = positive correlation; blue = negative</sup>",
        xaxis_title="Feature", yaxis_title="r²",
        height=450,
    )
    p = Path(save_dir) / f"attn_sae_L{layer}_r2_bar.html"
    fig.write_html(str(p))
    print(f"  Saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",     default="google/gemma-3-12b-it")
    ap.add_argument("--device",    default="auto")
    ap.add_argument("--sae-base",  required=True,
                    help="Base path to gemma-scope snapshot, e.g. "
                         "/path/to/models--google--gemma-scope-2-12b-it/snapshots/<hash>")
    ap.add_argument("--save-dir",  default="figures/linebreak_mech")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device: {device}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model, device)

    saes = {}
    for l in (0, 1):
        sae_path = (Path(args.sae_base) /
                    f"attn_out_all/layer_{l}_width_16k_l0_small")
        print(f"Loading SAE layer {l} from {sae_path}...")
        saes[l] = JumpReLUSAE.from_pretrained(
            str(sae_path), device="cpu", dtype=torch.float32)
        saes[l].eval()
        print(f"  d_in={saes[l].d_in}, d_sae={saes[l].d_sae}")

    print("\nCollecting attn_out SAE records...")
    records = collect_attn_sae_records(model, tok, saes, device)
    print(f"  Total records: {len(records)}")

    y        = np.array([r["char_pos"] for r in records], dtype=float)
    width_a  = np.array([r["width"]    for r in records])
    summary  = {}

    for l in (0, 1):
        label = f"attn_out L{l}"
        print(f"\n{'='*50}\n{label}\n{'='*50}")

        feats_matrix = np.stack([r["feats"][l] for r in records])  # (N, d_sae)
        r2 = pearson_r2_per_feature(feats_matrix, y)

        top_pos = np.argsort(-r2)[:TOP_N].tolist()
        top_neg_idx = np.where(r2 > 0)[0]   # we only have r², look at signed r
        # Recompute signed correlation to find negatively correlated features
        active = (feats_matrix > 0).mean(0) >= 0.05
        ym = y - y.mean()
        signed_r = np.zeros(feats_matrix.shape[1])
        for fi in np.where(active)[0]:
            xm = feats_matrix[:, fi] - feats_matrix[:, fi].mean()
            dx = (xm**2).sum(); dy = (ym**2).sum()
            if dx > 0 and dy > 0:
                signed_r[fi] = (ym @ xm) / np.sqrt(dy * dx)

        top_pos_signed = np.argsort(-signed_r)[:TOP_N].tolist()
        top_neg_signed = np.argsort(signed_r)[:10].tolist()

        print(f"  Top features positively correlated with char_pos (r²):")
        for fi in top_pos_signed[:10]:
            print(f"    F{fi:5d}: r={signed_r[fi]:+.4f}  r²={r2[fi]:.4f}  "
                  f"mean_act={feats_matrix[:,fi].mean():.3f}  "
                  f"frac_active={( feats_matrix[:,fi]>0).mean():.2f}")
        print(f"  Top features negatively correlated:")
        for fi in top_neg_signed[:5]:
            if r2[fi] > 0.01:
                print(f"    F{fi:5d}: r={signed_r[fi]:+.4f}  r²={r2[fi]:.4f}")

        plot_feature_profiles(records, l, top_pos_signed, CONTROL_WIDTHS,
                              args.save_dir, label)
        plot_r2_bar(r2, top_pos_signed, top_neg_signed, l, args.save_dir)

        summary[f"L{l}"] = {
            "top_pos_features": top_pos_signed,
            "top_neg_features": top_neg_signed,
            "top_r2_values":    [float(r2[fi]) for fi in top_pos_signed],
            "top_signed_r":     [float(signed_r[fi]) for fi in top_pos_signed],
        }

    out = Path(args.save_dir) / "attn_sae_position_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out}")


if __name__ == "__main__":
    main()
