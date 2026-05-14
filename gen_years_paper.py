"""Generate year-only activations ("The date is {year}") matching paper Appendix B Table 1,
then plot PCA side-by-side with the existing month-included variant."""

import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

SAVE_DIR = Path("figures/gemma_pca")
MODEL_NAME = "google/gemma-3-12b-it"
LAYER = 24
BATCH_SIZE = 16


def harvest(model, tokenizer, prompts, layer, device, batch_size=16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    all_acts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Harvesting"):
        batch = prompts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[layer]
        seq_len = inputs["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(hs.size(0), device=device)
        all_acts.append(hs[idx, seq_len].float().cpu())
    return torch.cat(all_acts, dim=0)


def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device)
    mdl.eval()
    return mdl, tok


def plot_both(acts_paper, labels_paper, acts_month, labels_month, out_path):
    cmap = plt.cm.viridis

    def project(acts, n=3):
        pca = PCA(n_components=n)
        proj = pca.fit_transform(acts.numpy())
        return proj, pca.explained_variance_ratio_

    proj_p, var_p = project(acts_paper)
    proj_m, var_m = project(acts_month)

    def norm_colors(labels):
        n = (labels - labels.min()) / (labels.max() - labels.min() + 1e-8)
        return cmap(n)[:, :3]

    col_p = norm_colors(labels_paper)
    col_m = norm_colors(labels_month)

    views = [(30, -60), (30, -120), (70, -60)]
    n_views = len(views)

    fig = plt.figure(figsize=(4 * n_views * 2, 4.5))
    fig.suptitle('Years manifold: paper original vs month-injected', fontsize=13, y=1.01)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    gs_left = gs[0].subgridspec(1, n_views, wspace=0.05)
    gs_right = gs[1].subgridspec(1, n_views, wspace=0.05)

    for vi, (elev, azim) in enumerate(views):
        for (gs_sub, proj, var, col, title) in [
            (gs_left[vi],  proj_p, var_p, col_p, f'Paper: "The date is {{year}}" (n={len(labels_paper)})'),
            (gs_right[vi], proj_m, var_m, col_m, f'Ours: "The date is {{month}} {{year}}" (n={len(labels_month)})'),
        ]:
            ax = fig.add_subplot(gs_sub, projection='3d')
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=col, s=1, alpha=0.5)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel(f'PC1 {var[0]:.1%}', fontsize=6)
            ax.set_ylabel(f'PC2 {var[1]:.1%}', fontsize=6)
            ax.set_zlabel(f'PC3 {var[2]:.1%}', fontsize=6)
            ax.tick_params(labelsize=5)
            if vi == 1:
                ax.set_title(title, fontsize=8, pad=8)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    paper_acts_path = SAVE_DIR / "activations_years_paper.pt"

    if paper_acts_path.exists():
        print("Loading cached paper activations...")
        d = torch.load(paper_acts_path, weights_only=False)
        acts_paper = d["activations"]
        labels_paper = d["labels"]
    else:
        years = list(range(1826, 2025))  # 199 years, matching paper n=199
        prompts = [f"The date is {y}" for y in years]
        labels_paper = np.array(years, dtype=float)

        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda"
        model, tokenizer = load_model(device)
        acts_paper = harvest(model, tokenizer, prompts, LAYER, device, BATCH_SIZE)
        torch.save({"activations": acts_paper, "labels": labels_paper}, paper_acts_path)
        print(f"Saved {paper_acts_path}")
        del model

    # Load existing month-included activations
    d2 = torch.load(SAVE_DIR / "activations_years.pt", weights_only=False)
    acts_month = d2["activations"]
    labels_month = d2["labels"]

    out = SAVE_DIR / "pca_years_comparison.png"
    plot_both(acts_paper, labels_paper, acts_month, labels_month, out)
