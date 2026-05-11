"""Harvest activations from Gemma 3 12B and visualize manifold geometry via PCA.

Replicates the "ubiquity of manifolds" analysis (paper Appendix B, Table 1)
on Gemma 3 12B instead of Llama 3.1 8B. Extracts last-token activations from
a target layer, runs PCA, and produces 3D scatter plots colored by the
continuous label for each manifold concept.

Usage:
    python run_gemma_pca.py --device cuda   # GPU server
    python run_gemma_pca.py --device mps    # Apple Silicon
    python run_gemma_pca.py --device cpu    # CPU fallback
"""

import argparse
import colorsys
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Manifold prompt datasets (from paper Appendix B, Table 1)
# ---------------------------------------------------------------------------

def build_color_prompts():
    """~5800 hex color prompts; geometry = paraboloid (hue, sat, val)."""
    prompts, labels = [], []
    for h in range(0, 360, 5):
        for s_pct in range(10, 100, 10):
            for v_pct in range(10, 100, 10):
                s, v = s_pct / 100, v_pct / 100
                r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
                hex_code = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                prompts.append(f"The hex code {hex_code} is for the color")
                labels.append([h / 360, s, v])
    return prompts, np.array(labels), "colors"


def build_temperature_prompts():
    """~1500 temperature prompts; geometry = line."""
    prompts, labels = [], []
    for f in np.arange(-20, 130.1, 0.1):
        prompts.append(f"Today it's {f:.1f} degrees Fahrenheit outside")
        labels.append(float(f))
    return prompts, np.array(labels, dtype=float), "temperature"


def build_age_prompts():
    """~980 age prompts; geometry = line."""
    prompts, labels = [], []
    for age in np.arange(1, 99.1, 0.1):
        prompts.append(f"They are {age:.1f} years old.")
        labels.append(float(age))
    return prompts, np.array(labels, dtype=float), "age"


def build_days_prompts():
    """~10080 day-of-week prompts (every minute); geometry = circle."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    prompts, labels = [], []
    for di, day in enumerate(days):
        for hour in range(0, 24):
            for minute in range(0, 60):
                time_str = f"{hour:02d}:{minute:02d}"
                prompts.append(f"It's {time_str} on {day}")
                angle = (di * 24 * 60 + hour * 60 + minute) / (7 * 24 * 60) * 2 * np.pi
                labels.append(angle)
    return prompts, np.array(labels), "days"


def build_years_prompts():
    """~2400 year+month prompts; geometry = helix."""
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    prompts, labels = [], []
    for year in range(1825, 2025):
        for mi, month in enumerate(months):
            prompts.append(f"The date is {month} {year}")
            labels.append(year + mi / 12)
    return prompts, np.array(labels, dtype=float), "years"


MANIFOLD_BUILDERS = {
    "colors": build_color_prompts,
    "temperature": build_temperature_prompts,
    "age": build_age_prompts,
    "days": build_days_prompts,
    "years": build_years_prompts,
}


# ---------------------------------------------------------------------------
# Activation harvesting
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: str):
    """Load Gemma model with appropriate dtype for device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if device == "mps":
        dtype = torch.float32  # MPS bfloat16 support is spotty

    print(f"Loading {model_name} (dtype={dtype}, device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device if device != "mps" else None,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def harvest_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    device: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """Extract last-token hidden states from the specified layer.

    Returns: (n_prompts, d_model) tensor of activations.
    """
    all_acts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Harvesting"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, d)
        # Get last non-padding token activation for each sequence
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # last real token index
        batch_indices = torch.arange(hidden_states.size(0), device=device)
        last_token_acts = hidden_states[batch_indices, seq_lengths]

        all_acts.append(last_token_acts.float().cpu())

    return torch.cat(all_acts, dim=0)


# ---------------------------------------------------------------------------
# PCA + Visualization
# ---------------------------------------------------------------------------

def run_pca_and_plot(
    activations: torch.Tensor,
    labels: np.ndarray,
    manifold_name: str,
    save_dir: str,
):
    """Run PCA on activations and produce 3D scatter plot."""
    acts_np = activations.numpy()
    pca = PCA(n_components=3)
    projected = pca.fit_transform(acts_np)
    var_explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color by label
    if labels.ndim == 2:
        # Multi-dim label (colors: use HSV directly)
        colors = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in labels])
    else:
        # Scalar label: use colormap
        norm = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
        cmap = plt.cm.viridis if manifold_name != "days" else plt.cm.hsv
        colors = cmap(norm)[:, :3]

    ax.scatter(
        projected[:, 0], projected[:, 1], projected[:, 2],
        c=colors, s=3, alpha=0.7,
    )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1%})")
    ax.set_zlabel(f"PC3 ({var_explained[2]:.1%})")
    ax.set_title(f"{manifold_name} — Gemma 3 12B activation PCA")

    total_var = sum(var_explained)
    ax.text2D(0.02, 0.02, f"Top-3 PCs explain {total_var:.1%} variance",
              transform=ax.transAxes, fontsize=9)

    path = Path(save_dir) / f"pca_{manifold_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path} (var explained: {var_explained[0]:.3f}, {var_explained[1]:.3f}, {var_explained[2]:.3f})")

    return {
        "manifold": manifold_name,
        "n_samples": len(activations),
        "var_explained": var_explained.tolist(),
        "total_var_top3": float(total_var),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Gemma 3 12B activation PCA")
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=24,
                        help="Layer to extract activations from (Gemma 3 12B has 48 layers)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--manifolds", type=str, nargs="+",
                        default=["colors", "temperature", "days", "years"],
                        choices=list(MANIFOLD_BUILDERS.keys()))
    parser.add_argument("--save-dir", type=str, default="figures/gemma_pca")
    parser.add_argument("--save-activations", action="store_true",
                        help="Save raw activations for later analysis")
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

    os.makedirs(args.save_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    cfg = getattr(model.config, "text_config", model.config)
    d_model = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    print(f"Model loaded: d={d_model}, layers={n_layers}")
    print(f"Extracting from layer {args.layer}/{n_layers}")

    results = []
    for manifold_name in args.manifolds:
        print(f"\n{'='*50}")
        print(f"Manifold: {manifold_name}")
        print(f"{'='*50}")

        prompts, labels, name = MANIFOLD_BUILDERS[manifold_name]()
        print(f"  {len(prompts)} prompts, label shape: {labels.shape}")
        print(f"  Example: \"{prompts[0]}\"")

        activations = harvest_activations(
            model, tokenizer, prompts,
            layer=args.layer, device=device, batch_size=args.batch_size,
        )
        print(f"  Activations shape: {activations.shape}")

        if args.save_activations:
            act_path = Path(args.save_dir) / f"activations_{manifold_name}.pt"
            torch.save({"activations": activations, "labels": labels}, act_path)
            print(f"  Saved activations to {act_path}")

        result = run_pca_and_plot(activations, labels, manifold_name, args.save_dir)
        results.append(result)

    # Save summary
    summary_path = Path(args.save_dir) / "pca_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for r in results:
        print(f"  {r['manifold']:15s} n={r['n_samples']:5d}  "
              f"top-3 var={r['total_var_top3']:.1%}  "
              f"PC1={r['var_explained'][0]:.3f}")


if __name__ == "__main__":
    main()
