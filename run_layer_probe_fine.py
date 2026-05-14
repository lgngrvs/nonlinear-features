"""Fine-grained layer probe for layers 0–11, with shuffled CV and NaN checks."""

import argparse
import json
import numpy as np
import torch
import textwrap
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

CONTROL_UNIT = "ab "
CONTROL_WIDTHS = (40, 60, 80, 100, 120)


def load_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    device_map = "auto" if n_gpus > 1 else (device if device not in ("cpu", "mps") else None)
    dtype = torch.bfloat16 if device not in ("cpu", "mps") else torch.float32
    print(f"Loading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map)
    if device in ("cpu", "mps"):
        model = model.to(device)
    model.eval()
    cfg = getattr(model.config, "text_config", model.config)
    print(f"  n_layers={cfg.num_hidden_layers}, d_model={cfg.hidden_size}")
    return model, tok, cfg.num_hidden_layers


@torch.no_grad()
def forward_hidden_states(model, ids_list, device, layers, hook_type="resid_post"):
    ids = torch.tensor([ids_list], dtype=torch.long).to(device)

    if hook_type == "resid_post":
        out = model(ids, output_hidden_states=True)
        return {l: out.hidden_states[l + 1][0].float().cpu() for l in layers}

    # Hook-based extraction for attn_out / mlp_out
    captured = {}

    def make_hook(layer_idx):
        def hook(module, inp, output):
            # output is a tensor (post-norm contribution to residual)
            captured[layer_idx] = output[0].float().cpu() if isinstance(output, tuple) else output.float().cpu()
        return hook

    # Gemma 3 is multimodal: model.model.language_model.layers[l]
    decoder_layers = model.model.language_model.layers
    hooks = []
    for l in layers:
        if hook_type == "attn_out":
            mod = decoder_layers[l].post_attention_layernorm
        else:  # mlp_out
            mod = decoder_layers[l].post_feedforward_layernorm
        hooks.append(mod.register_forward_hook(make_hook(l)))

    model(ids)
    for h in hooks:
        h.remove()

    return {l: captured[l][0] for l in layers}  # (seq_len, d_model)


def collect_control_data(model, tokenizer, device, probe_layers,
                          unit=CONTROL_UNIT, widths=CONTROL_WIDTHS, n_context_lines=3,
                          hook_type="resid_post"):
    unit_len = len(unit)
    unit_toks = tokenizer.encode(unit, add_special_tokens=False)
    layer_data = {l: ([], []) for l in probe_layers}

    for width in widths:
        units_per_line = width // unit_len
        context = ("\n".join([(unit * units_per_line).rstrip()] * n_context_lines)) + "\n"
        context_ids = tokenizer.encode(context, add_special_tokens=False)

        line_ids: list[int] = []
        for n_units in tqdm(range(1, units_per_line + 4),
                            desc=f"  w={width}"):
            line_ids = line_ids + unit_toks
            full_ids = context_ids + line_ids
            char_pos = (n_units - 1) * unit_len

            hs = forward_hidden_states(model, full_ids, device, probe_layers, hook_type)
            tok_idx = len(full_ids) - 1

            for l in probe_layers:
                vec = hs[l][tok_idx].numpy()
                layer_data[l][0].append(vec)
                layer_data[l][1].append(float(char_pos))

    return layer_data


def probe_layer(X_list, y_list, n_pca_components=50, n_splits=5):
    X = np.stack(X_list)
    y = np.array(y_list)

    # Sanity checks
    nan_frac = np.isnan(X).mean()
    inf_frac = np.isinf(X).mean()
    print(f"    NaN={nan_frac:.4f}  Inf={inf_frac:.4f}  "
          f"mean={X.mean():.3f}  std={X.std():.3f}  "
          f"n={len(y)}  y_range=[{y.min():.0f},{y.max():.0f}]")

    if nan_frac > 0 or inf_frac > 0:
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n_comp = min(n_pca_components, X.shape[0] - 1, X.shape[1])
    X_pca = PCA(n_components=n_comp).fit_transform(X)

    # Shuffled KFold to avoid width-stratified train/test splits
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(Ridge(alpha=1.0), X_pca, y, cv=cv, scoring="r2")
    return float(scores.mean()), scores.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-12b-it")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--layers", default="0-11",
                    help="Layer range, e.g. '0-11' or '0-47'")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--hook-type", default="resid_post",
                    choices=["resid_post", "attn_out", "mlp_out"])
    ap.add_argument("--save", default="figures/linebreak_mech/layer_probe_fine.json")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    lo, hi = map(int, args.layers.split("-"))
    probe_layers = list(range(lo, hi + 1, args.step))
    print(f"Probing layers: {probe_layers}")

    model, tokenizer, n_layers = load_model(args.model, device)

    print("\nCollecting control activations (hook=%s)..." % args.hook_type)
    layer_data = collect_control_data(model, tokenizer, device, probe_layers,
                                      hook_type=args.hook_type)

    print("\nFitting probes:")
    results = {}
    for l in probe_layers:
        X_list, y_list = layer_data[l]
        print(f"  Layer {l:2d}:")
        r2, fold_scores = probe_layer(X_list, y_list)
        print(f"    R²={r2:.4f}  folds={[f'{s:.3f}' for s in fold_scores]}")
        results[l] = {"r2": r2, "fold_scores": fold_scores}

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.save}")

    print("\nSummary:")
    for l, v in results.items():
        bar = "#" * max(0, int(v["r2"] * 30))
        print(f"  L{l:2d}: {v['r2']:+.4f}  {bar}")


if __name__ == "__main__":
    main()
