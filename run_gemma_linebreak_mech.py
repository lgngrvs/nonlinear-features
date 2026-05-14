"""Mechanistic analysis of Gemma 3 12B line-breaking via SAE features.

Experiments
-----------
1. Contrastive  — SAE features at char_pos≈34, width=40 vs width=80.
   Same character position, different expected break distance.
   Top features by mean_activation(near-break) − mean_activation(mid-line).

2. Profile      — Mean SAE feature activation vs char_pos, binned, for top
   contrastive features across multiple widths [40,60,80,100,120].
   Shows whether features ramp up toward break or are step-functions.

3. PCA          — All (text, width) positions projected through SAE encoder,
   then PCA-reduced to 3D. Colored by 'remaining = width − char_pos'.
   Tests whether PC1 ≈ remaining-chars-to-break.

4. Control      — "ab " repeated at fixed widths (40, 80). No semantic content;
   char_pos = n_units × 3. Shows pure position signal in top features.

5. Layer probe  — Ridge regression: char_pos ~ PCA(activation) at each layer.
   Cross-val R². Identifies where position enters the residual stream.

Usage
-----
    python run_gemma_linebreak_mech.py --sae-path /path/to/layer_24_width_16k_l0_medium
    python run_gemma_linebreak_mech.py --sae-path ... --device cuda --n-texts 6
"""

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from tqdm import tqdm

from nonlinear_features.jumprelu_sae import JumpReLUSAE


# ---------------------------------------------------------------------------
# Prose corpus (identical to run_gemma_linebreak_eval.py)
# ---------------------------------------------------------------------------

PROSE_TEXTS = [
    ("The quick brown fox jumps over the lazy dog. A stitch in time saves nine. "
     "Early to bed and early to rise makes a man healthy wealthy and wise. "
     "All that glitters is not gold. Actions speak louder than words."),
    ("It was the best of times it was the worst of times it was the age of wisdom "
     "it was the age of foolishness it was the epoch of belief it was the epoch of "
     "incredulity it was the season of light it was the season of darkness."),
    ("To be or not to be that is the question whether tis nobler in the mind to "
     "suffer the slings and arrows of outrageous fortune or to take arms against a "
     "sea of troubles and by opposing end them to die to sleep no more."),
    ("Ask not what your country can do for you ask what you can do for your country. "
     "One small step for man one giant leap for mankind. "
     "We choose to go to the moon not because it is easy but because it is hard."),
    ("The only thing we have to fear is fear itself. "
     "Four score and seven years ago our fathers brought forth on this continent "
     "a new nation conceived in liberty and dedicated to the proposition "
     "that all men are created equal and endowed with inalienable rights."),
    ("Photons are the quanta of electromagnetic radiation carrying energy inversely "
     "proportional to wavelength. The gradient descent algorithm iteratively moves "
     "parameters in the direction of steepest loss reduction scaled by a learning rate."),
    ("Neural networks learn hierarchical representations through successive "
     "transformations of the input data using learnable weight matrices and "
     "nonlinear activation functions such as the rectified linear unit or sigmoid."),
    ("Scientists discovered a new species of deep-sea fish exhibiting "
     "bioluminescence adapted for communication in the aphotic zone where "
     "no sunlight penetrates. The organism produces light through a "
     "luciferin-luciferase reaction catalyzed by specific enzymes."),
    ("The transformer architecture uses self-attention mechanisms to weigh "
     "the importance of different positions in the input sequence when computing "
     "representations allowing parallel computation and capturing long-range "
     "dependencies that recurrent models often struggle with."),
    ("Mechanistic interpretability aims to reverse-engineer the computations "
     "performed by neural networks by identifying circuits and representations "
     "that implement specific behaviors or capabilities within the model weights "
     "using tools such as activation patching and sparse autoencoders."),
    ("The mitochondria convert chemical energy from food into adenosine triphosphate "
     "through oxidative phosphorylation occurring in the inner mitochondrial membrane "
     "via electron transport chain complexes and the ATP synthase rotary mechanism."),
    ("Language models trained on large corpora develop emergent capabilities not "
     "explicitly optimized for during training such as arithmetic reasoning "
     "chain-of-thought problem solving and in-context learning from a small "
     "number of demonstrations provided in the prompt."),
]

CONTRASTIVE_WIDTHS = (40, 80)
PROFILE_WIDTHS = [40, 60, 80, 100, 120]
CONTROL_UNIT = "ab "   # 3 chars; char_pos = n_units × 3
CONTROL_WIDTHS = (40, 80)
TOP_N = 20
CHAR_POS_WINDOW = (28, 42)   # "near char_pos=34" bucket for contrastive


# ---------------------------------------------------------------------------
# Model + SAE loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    device_map = "auto" if n_gpus > 1 else (device if device not in ("cpu", "mps") else None)
    dtype = torch.bfloat16 if device not in ("cpu", "mps") else torch.float32
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map)
    if device in ("cpu", "mps"):
        model = model.to(device)
    model.eval()
    cfg = getattr(model.config, "text_config", model.config)
    n_layers = cfg.num_hidden_layers
    print(f"  d_model={cfg.hidden_size}, layers={n_layers}")
    return model, tokenizer, n_layers


def find_newline_id(tokenizer):
    for probe in ["word\n", "\nword", "\n"]:
        for tid in tokenizer.encode(probe, add_special_tokens=False):
            if "\n" in tokenizer.decode([tid]):
                return tid
    return None


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_hidden_states(model, ids_list, device, layers):
    """One forward pass → {layer: (seq_len, d_model) float32 CPU tensor}."""
    ids = torch.tensor([ids_list], dtype=torch.long).to(device)
    out = model(ids, output_hidden_states=True)
    # hidden_states[0]=embed, hidden_states[l+1]=after layer l
    return {l: out.hidden_states[l + 1][0].float().cpu() for l in layers}


def tokenize_with_offsets(text, tokenizer):
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return list(enc["input_ids"]), list(enc["offset_mapping"])


def collect_positions(wrapped, tokenizer, newline_id, min_lines=2):
    """Yield (tok_idx, char_pos_in_line, line_num) for non-newline tokens."""
    ids, offsets = tokenize_with_offsets(wrapped, tokenizer)
    nl_count = 0
    results = []
    for i, (tid, (tok_start, _)) in enumerate(zip(ids, offsets)):
        if tid == newline_id:
            nl_count += 1
            continue
        if nl_count < min_lines:
            continue
        last_nl = wrapped.rfind("\n", 0, tok_start)
        char_pos = tok_start - last_nl - 1 if last_nl >= 0 else tok_start
        results.append((i, char_pos, nl_count))
    return ids, results


def encode_sae(sae, residual_cpu):
    """residual_cpu: (d_model,) float32. Returns (d_sae,) float32 numpy."""
    with torch.no_grad():
        feat = sae.encode(residual_cpu.unsqueeze(0)).squeeze(0)
    return feat.numpy()


# ---------------------------------------------------------------------------
# Experiment 1 & 2: Contrastive + full profile
# ---------------------------------------------------------------------------

def collect_sae_records(model, tokenizer, sae, texts, widths, device, layer, min_lines=2):
    """
    For every (text, width), collect all valid token positions:
      {char_pos, width, remaining, feat (d_sae,)}
    One forward pass per (text, width).
    """
    newline_id = find_newline_id(tokenizer)
    records = []
    for text in tqdm(texts, desc="SAE record collection"):
        for width in widths:
            wrapped = textwrap.fill(text, width=width)
            ids, positions = collect_positions(wrapped, tokenizer, newline_id, min_lines)
            if not positions:
                continue
            hs = forward_hidden_states(model, ids, device, [layer])
            residuals = hs[layer]  # (seq_len, d_model)
            for (tok_idx, char_pos, _) in positions:
                if char_pos > width + 5:
                    continue
                feat = encode_sae(sae, residuals[tok_idx])
                records.append({
                    "char_pos": int(char_pos),
                    "width": int(width),
                    "remaining": int(width - char_pos),
                    "feat": feat,
                })
    return records


def run_contrastive(records, w_narrow=40, w_wide=80, window=CHAR_POS_WINDOW):
    """Compare mean SAE features at char_pos∈window for narrow vs wide width."""
    feats_narrow, feats_wide = [], []
    for r in records:
        if window[0] <= r["char_pos"] < window[1]:
            if r["width"] == w_narrow:
                feats_narrow.append(r["feat"])
            elif r["width"] == w_wide:
                feats_wide.append(r["feat"])

    if not feats_narrow or not feats_wide:
        raise ValueError(f"No records for contrastive: n={len(feats_narrow)}/{len(feats_wide)}")

    mean_n = np.stack(feats_narrow).mean(0)
    mean_w = np.stack(feats_wide).mean(0)
    delta = mean_n - mean_w  # positive = more active near break

    top_pos = np.argsort(-delta)[:TOP_N].tolist()
    top_neg = np.argsort(delta)[:TOP_N].tolist()

    print(f"\n  Contrastive: n_narrow={len(feats_narrow)}, n_wide={len(feats_wide)}")
    print(f"  Top features MORE active near break (width={w_narrow}, pos≈34):")
    for fi in top_pos[:8]:
        print(f"    F{fi:5d}: delta={delta[fi]:+.4f}  mean_{w_narrow}={mean_n[fi]:.4f}  mean_{w_wide}={mean_w[fi]:.4f}")
    print(f"  Top features LESS active near break:")
    for fi in top_neg[:8]:
        print(f"    F{fi:5d}: delta={delta[fi]:+.4f}  mean_{w_narrow}={mean_n[fi]:.4f}  mean_{w_wide}={mean_w[fi]:.4f}")

    return delta, mean_n, mean_w, top_pos, top_neg


# ---------------------------------------------------------------------------
# Experiment 3: SAE PCA
# ---------------------------------------------------------------------------

def run_sae_pca(records):
    """PCA on all SAE feature vectors across (text, width, char_pos)."""
    feats = np.stack([r["feat"] for r in records])
    pca = PCA(n_components=3)
    proj = pca.fit_transform(feats)
    print(f"\n  PCA: {len(records)} points, variance explained: "
          f"{pca.explained_variance_ratio_}")
    return proj, pca


# ---------------------------------------------------------------------------
# Experiment 4: Repeated-token control
# ---------------------------------------------------------------------------

def run_control(model, tokenizer, sae, device, layer,
                widths=CONTROL_WIDTHS, unit=CONTROL_UNIT, n_context_lines=3):
    """
    Generate [n_context_lines of 'unit' tokens at given width] then append
    unit tokens one-by-one and record SAE features at each position.
    """
    newline_id = find_newline_id(tokenizer)
    unit_len = len(unit)
    records = []

    for width in widths:
        units_per_line = width // unit_len
        context = ("\n".join([(unit * units_per_line).rstrip()] * n_context_lines)) + "\n"
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        unit_toks = tokenizer.encode(unit, add_special_tokens=False)

        line_ids: list[int] = []
        for n_units in range(1, units_per_line + 4):  # +3 past break
            line_ids = line_ids + unit_toks
            full_ids = context_ids + line_ids
            char_pos = (n_units - 1) * unit_len  # start char of current unit

            hs = forward_hidden_states(model, full_ids, device, [layer])
            tok_idx = len(full_ids) - 1
            feat = encode_sae(sae, hs[layer][tok_idx])
            records.append({
                "width": int(width),
                "char_pos": int(char_pos),
                "remaining": int(width - char_pos),
                "n_units": n_units,
                "feat": feat,
            })

        print(f"  Control width={width}: {units_per_line + 3} positions")
    return records


# ---------------------------------------------------------------------------
# Experiment 5: Layer probe
# ---------------------------------------------------------------------------

def run_layer_probe_control(model, tokenizer, device, n_layers,
                             unit=CONTROL_UNIT, widths=(40, 60, 80, 100, 120),
                             n_context_lines=3, n_pca_components=50):
    """
    Layer probe using repeated-token control sequences only.
    Content is constant ('ab ' repeated) so PCA captures position, not content.
    Char_pos is the only varying signal — gives clean R² estimates.
    Returns {layer: cross-val R²}.
    """
    step = max(1, n_layers // 12)
    probe_layers = list(range(0, n_layers, step))
    unit_len = len(unit)
    unit_toks = tokenizer.encode(unit, add_special_tokens=False)

    layer_data: dict[int, tuple[list, list]] = {l: ([], []) for l in probe_layers}

    for width in widths:
        units_per_line = width // unit_len
        context = ("\n".join([(unit * units_per_line).rstrip()] * n_context_lines)) + "\n"
        context_ids = tokenizer.encode(context, add_special_tokens=False)

        line_ids: list[int] = []
        for n_units in tqdm(range(1, units_per_line + 4),
                            desc=f"Layer probe control w={width}"):
            line_ids = line_ids + unit_toks
            full_ids = context_ids + line_ids
            char_pos = (n_units - 1) * unit_len

            hs = forward_hidden_states(model, full_ids, device, probe_layers)
            tok_idx = len(full_ids) - 1

            for l in probe_layers:
                layer_data[l][0].append(hs[l][tok_idx].numpy())
                layer_data[l][1].append(float(char_pos))

    r2_by_layer = {}
    for l in probe_layers:
        X_list, y_list = layer_data[l]
        if len(y_list) < 10:
            r2_by_layer[l] = float("nan")
            continue
        X = np.stack(X_list)
        y = np.array(y_list)
        n_comp = min(n_pca_components, X.shape[0] - 1, X.shape[1])
        X_pca = PCA(n_components=n_comp).fit_transform(X)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(Ridge(alpha=1.0), X_pca, y, cv=cv, scoring="r2")
        r2_by_layer[l] = float(scores.mean())
        print(f"  Layer {l:2d}: R²={r2_by_layer[l]:.3f}  (n={len(y)})")

    return r2_by_layer


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_contrastive(delta, top_pos, top_neg, save_dir):
    top_all = top_pos[:10] + top_neg[:10]
    labels = [f"F{fi}" for fi in top_all]
    vals = [float(delta[fi]) for fi in top_all]
    colors = ["#e63946" if v > 0 else "#457b9d" for v in vals]

    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors,
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        title="SAE features: activation(width=40, char_pos≈34) − activation(width=80, char_pos≈34)<br>"
              "<sup>Red = more active near break; blue = less active near break</sup>",
        xaxis_title="Feature", yaxis_title="Activation delta",
        height=500, margin=dict(t=80),
    )
    p = Path(save_dir) / "sae_contrastive.html"
    fig.write_html(str(p))
    print(f"  Saved {p}")


def plot_feature_profiles(records, top_features, widths, save_dir):
    """For each top feature: mean activation vs char_pos/width, one line per width."""
    all_feats = np.stack([r["feat"] for r in records])
    colors = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5"]

    for fi in top_features[:8]:
        fig = go.Figure()
        for ci, width in enumerate(widths):
            mask = np.array([r["width"] == width for r in records])
            if mask.sum() < 3:
                continue
            cp = np.array([r["char_pos"] for r in records])[mask]
            fv = all_feats[mask, fi]
            # Bin into 10 equal-width bins
            bins = np.linspace(0, width, 11)
            bcenters, bmeans = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = (cp >= lo) & (cp < hi)
                if m.any():
                    bcenters.append((lo + hi) / 2)
                    bmeans.append(fv[m].mean())
            # Normalize x to fraction of width for comparability
            fig.add_trace(go.Scatter(
                x=[c / width for c in bcenters], y=bmeans,
                mode="lines+markers", name=f"w={width}",
                line=dict(color=colors[ci % len(colors)]),
            ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="black", annotation_text="break")
        fig.update_layout(
            title=f"SAE Feature {fi}: activation vs position (fraction of line width)",
            xaxis_title="char_pos / width", yaxis_title="Mean activation",
            height=400,
        )
        p = Path(save_dir) / f"sae_feature_{fi}_profile.html"
        fig.write_html(str(p))
    print(f"  Saved feature profiles for features: {top_features[:8]}")


def plot_sae_pca(records, proj, pca, save_dir):
    remaining = np.array([r["remaining"] for r in records])
    widths_arr = np.array([r["width"] for r in records])
    ev = pca.explained_variance_ratio_

    for color_arr, color_label, fname, cscale in [
        (remaining, "Remaining chars to break", "sae_pca_remaining.html", "RdYlGn"),
        (widths_arr, "Line width", "sae_pca_width.html", "Viridis"),
    ]:
        fig = go.Figure(go.Scatter3d(
            x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
            mode="markers",
            marker=dict(size=3, color=color_arr, colorscale=cscale,
                        colorbar=dict(title=color_label), opacity=0.8),
            text=[f"w={r['width']}, cp={r['char_pos']}, rem={r['remaining']}"
                  for r in records],
        ))
        fig.update_layout(
            title=f"SAE-space PCA — colored by: {color_label}",
            scene=dict(
                xaxis_title=f"PC1 ({ev[0]:.1%})",
                yaxis_title=f"PC2 ({ev[1]:.1%})",
                zaxis_title=f"PC3 ({ev[2]:.1%})",
            ),
            height=700,
        )
        p = Path(save_dir) / fname
        fig.write_html(str(p))
        print(f"  Saved {p}")


def plot_control(control_records, top_features, save_dir):
    all_feats = np.stack([r["feat"] for r in control_records])
    widths = sorted(set(r["width"] for r in control_records))
    colors = ["#e63946", "#457b9d"]
    n_feat = min(6, len(top_features))
    fig = make_subplots(rows=1, cols=n_feat,
                        subplot_titles=[f"F{fi}" for fi in top_features[:n_feat]])

    for col, fi in enumerate(top_features[:n_feat], 1):
        for ci, width in enumerate(widths):
            mask = np.array([r["width"] == width for r in control_records])
            cp = np.array([r["char_pos"] for r in control_records])[mask]
            fv = all_feats[mask, fi]
            sort_idx = np.argsort(cp)
            fig.add_trace(go.Scatter(
                x=cp[sort_idx], y=fv[sort_idx],
                mode="lines+markers", name=f"w={width}",
                line=dict(color=colors[ci]),
                showlegend=(col == 1),
            ), row=1, col=col)
            fig.add_vline(x=float(width), line_dash="dash",
                          line_color=colors[ci], row=1, col=col)

    fig.update_layout(
        title="Control ('ab ' repeated): SAE feature vs char_pos — no semantic content",
        height=420,
    )
    p = Path(save_dir) / "sae_control_profiles.html"
    fig.write_html(str(p))
    print(f"  Saved {p}")


def plot_layer_probe(r2_by_layer, sae_layer, save_dir):
    layers = sorted(r2_by_layer)
    r2s = [r2_by_layer[l] for l in layers]
    colors = ["#e63946" if l == sae_layer else "#457b9d" for l in layers]

    fig = go.Figure(go.Bar(
        x=layers, y=r2s, marker_color=colors,
        text=[f"{v:.3f}" if not np.isnan(v) else "nan" for v in r2s],
        textposition="outside",
    ))
    fig.update_layout(
        title="Linear probe R²: char_pos ~ PCA(activation) by layer<br>"
              "<sup>Red bar = layer with available SAE (layer 24)</sup>",
        xaxis_title="Layer", yaxis_title="Cross-val R² (5-fold)",
        height=450, margin=dict(t=80),
    )
    p = Path(save_dir) / "layer_probe_r2.html"
    fig.write_html(str(p))
    print(f"  Saved {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sae-path", required=True,
                    help="Directory or .safetensors for GemmaScope SAE")
    ap.add_argument("--sae-layer", type=int, default=24)
    ap.add_argument("--save-dir", default="figures/linebreak_mech")
    ap.add_argument("--n-texts", type=int, default=len(PROSE_TEXTS))
    ap.add_argument("--skip-layer-probe", action="store_true")
    ap.add_argument("--only-layer-probe", action="store_true",
                    help="Skip SAE experiments; only re-run layer probe (no SAE needed)")
    ap.add_argument("--skip-control", action="store_true")
    args = ap.parse_args()

    if args.device == "auto":
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
    else:
        device = args.device
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    model, tokenizer, n_layers = load_model_and_tokenizer(args.model, device)

    texts = PROSE_TEXTS[:args.n_texts]

    if args.only_layer_probe:
        # Skip all SAE experiments; only re-run the layer probe
        print(f"\n{'='*60}\nLAYER PROBE (control, no SAE needed)\n{'='*60}")
        r2_by_layer = run_layer_probe_control(model, tokenizer, device, n_layers)
        plot_layer_probe(r2_by_layer, args.sae_layer, args.save_dir)
        # Merge into existing summary if present
        summary_path = Path(args.save_dir) / "mech_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {}
        summary["layer_probe_r2"] = {str(k): v for k, v in r2_by_layer.items()}
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nUpdated summary: {summary_path}")
        return

    print(f"Loading SAE from {args.sae_path}...")
    sae = JumpReLUSAE.from_pretrained(args.sae_path, device="cpu", dtype=torch.float32)
    sae.eval()
    print(f"  SAE: d_in={sae.d_in}, d_sae={sae.d_sae}")

    # ── Experiments 1 & 2: collect records for contrastive + profile + PCA ──
    print(f"\n{'='*60}\nCOLLECTING SAE RECORDS\n{'='*60}")
    all_widths = sorted(set(list(PROFILE_WIDTHS) + list(CONTRASTIVE_WIDTHS)))
    records = collect_sae_records(
        model, tokenizer, sae, texts, all_widths, device, args.sae_layer,
    )
    print(f"  Total records: {len(records)}")

    # Contrastive analysis
    print(f"\n{'='*60}\nCONTRASTIVE ANALYSIS\n{'='*60}")
    delta, mean_n, mean_w, top_pos, top_neg = run_contrastive(
        records, w_narrow=40, w_wide=80, window=CHAR_POS_WINDOW,
    )
    top_features = top_pos[:4] + top_neg[:4]
    plot_contrastive(delta, top_pos, top_neg, args.save_dir)
    plot_feature_profiles(records, top_features, PROFILE_WIDTHS, args.save_dir)

    # SAE PCA
    print(f"\n{'='*60}\nSAE PCA\n{'='*60}")
    proj, pca = run_sae_pca(records)
    plot_sae_pca(records, proj, pca, args.save_dir)

    # ── Experiment 4: Control ────────────────────────────────────────────────
    if not args.skip_control:
        print(f"\n{'='*60}\nREPEATED-TOKEN CONTROL\n{'='*60}")
        control_records = run_control(
            model, tokenizer, sae, device, args.sae_layer,
        )
        plot_control(control_records, top_features, args.save_dir)

    # ── Experiment 5: Layer probe (control sequences, no content confound) ───
    if not args.skip_layer_probe:
        print(f"\n{'='*60}\nLAYER PROBE (control)\n{'='*60}")
        r2_by_layer = run_layer_probe_control(model, tokenizer, device, n_layers)
        plot_layer_probe(r2_by_layer, args.sae_layer, args.save_dir)
    else:
        r2_by_layer = {}

    # ── Save summary JSON ────────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "sae_layer": args.sae_layer,
        "n_records": len(records),
        "top_features_near_break": top_pos[:TOP_N],
        "top_features_far_from_break": top_neg[:TOP_N],
        "delta_max": float(delta.max()),
        "delta_min": float(delta.min()),
        "delta_mean_abs": float(np.abs(delta).mean()),
        "layer_probe_r2": {str(k): v for k, v in r2_by_layer.items()},
    }
    out = Path(args.save_dir) / "mech_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to {args.save_dir}/")
    print(f"Summary: {out}")


if __name__ == "__main__":
    main()
