"""
Ising coactivation matrix for linebreak SAE features.

Runs on:
  - attn_out L0  (o_proj input, 4096-dim)
  - attn_out L1  (o_proj input, 4096-dim)
  - resid_post L24 (3840-dim)

Collects SAE codes from prose texts at multiple widths, computes pairwise
Ising coupling via pseudo-likelihood, and plots sorted by signed Pearson r
with char_pos:
  pos_ramp  — features that increase with char_pos (r > 0.35)
  neg_ramp  — features that decrease with char_pos (r < -0.35)
  remaining — features that increase with (width − char_pos)  (r > 0.35)
  other     — everything else
"""

import argparse, json, sys, textwrap
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from nonlinear_features.jumprelu_sae import JumpReLUSAE

# ── prose corpus ─────────────────────────────────────────────────────────────
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

PROFILE_WIDTHS = [40, 60, 80, 100, 120]

CONCEPT_COLORS = {
    "pos_ramp":  "#e63946",
    "neg_ramp":  "#457b9d",
    "remaining": "#2a9d8f",
    "other":     "#888888",
}
CONCEPT_ORDER = ["pos_ramp", "neg_ramp", "remaining", "other"]
R_THRESH = 0.35


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n_gpus = torch.cuda.device_count() if device == "cuda" else 0
    device_map = "auto" if n_gpus > 1 else (device if device not in ("cpu","mps") else None)
    dtype = torch.bfloat16 if device not in ("cpu","mps") else torch.float32
    print(f"Loading {model_name}...", flush=True)
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map=device_map)
    if device in ("cpu","mps"):
        model = model.to(device)
    model.eval()
    return model, tok


def find_newline_id(tokenizer):
    for probe in ["word\n", "\nword", "\n"]:
        for tid in tokenizer.encode(probe, add_special_tokens=False):
            if "\n" in tokenizer.decode([tid]):
                return tid
    return None


# ── activation collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_codes(model, tokenizer, sae, hook_spec, device,
                  texts, widths, min_lines=2):
    """
    hook_spec: ("resid_post", layer_idx)  or  ("attn_o_proj", layer_idx)

    Returns codes (n, d_sae), char_pos (n,), width_arr (n,).
    """
    newline_id  = find_newline_id(tokenizer)
    decoder     = model.model.language_model.layers
    hook_type, layer_idx = hook_spec

    all_codes, all_cp, all_w = [], [], []

    for text in tqdm(texts, desc=f"  {hook_type} L{layer_idx}"):
        for width in widths:
            wrapped = textwrap.fill(text, width=width)
            enc     = tokenizer(wrapped, return_offsets_mapping=True,
                                add_special_tokens=False)
            ids     = list(enc["input_ids"])
            offsets = list(enc["offset_mapping"])
            prefix_nl = np.cumsum([1 if t == newline_id else 0 for t in ids])

            # Collect valid positions
            positions = []
            for i, (tid, (tok_start, _)) in enumerate(zip(ids, offsets)):
                if prefix_nl[i-1] if i > 0 else 0 < min_lines:
                    continue
                last_nl  = wrapped.rfind("\n", 0, tok_start)
                char_pos = tok_start - last_nl - 1 if last_nl >= 0 else tok_start
                if char_pos > width + 5:
                    continue
                positions.append((i, int(char_pos)))
            if not positions:
                continue

            # Forward pass with hook
            captured = {}
            if hook_type == "resid_post":
                ids_t = torch.tensor([ids], dtype=torch.long).to(device)
                out   = model(ids_t, output_hidden_states=True)
                hs    = out.hidden_states[layer_idx + 1][0].float().cpu()
                for (tok_idx, cp) in positions:
                    captured[tok_idx] = hs[tok_idx]
            else:  # attn_o_proj
                def make_hook():
                    def h(module, inp):
                        t = inp[0] if isinstance(inp, tuple) else inp
                        captured["act"] = t[0].float().cpu()   # (seq, d)
                    return h
                hook = decoder[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                    make_hook())
                ids_t = torch.tensor([ids], dtype=torch.long).to(device)
                model(ids_t)
                hook.remove()
                act = captured["act"]  # (seq, d)
                for (tok_idx, cp) in positions:
                    captured[tok_idx] = act[tok_idx]

            for (tok_idx, cp) in positions:
                if tok_idx not in captured:
                    continue
                vec  = captured[tok_idx]
                with torch.no_grad():
                    feat = sae.encode(vec.unsqueeze(0)).squeeze(0)
                all_codes.append(feat.numpy())
                all_cp.append(cp)
                all_w.append(width)

    codes    = np.stack(all_codes)   # (n, d_sae)
    char_pos = np.array(all_cp, dtype=float)
    width_a  = np.array(all_w,  dtype=float)
    return codes, char_pos, width_a


# ── Ising fitting ─────────────────────────────────────────────────────────────

def fit_ising(codes_np, device, n_steps=1500, lam=0.005, n_samples=5000):
    """codes_np: (n, d_sae) float32 numpy.  Returns J_active (p,p), active_idx."""
    codes = torch.from_numpy(codes_np)
    n_use = min(n_samples, len(codes))
    s     = torch.sign(codes[:n_use])
    s[codes[:n_use] == 0] = -1.0

    fr     = (s > 0).float().mean(0)
    active = (fr > 0.01) & (fr < 0.99)
    active_idx = active.nonzero(as_tuple=True)[0]
    s_active   = s[:, active_idx].to(device)
    p          = len(active_idx)
    print(f"    Active atoms: {p}", flush=True)

    J = torch.zeros(p, p, device=device, requires_grad=True)
    h = torch.zeros(p, device=device, requires_grad=True)
    opt = torch.optim.Adam([J, h], lr=0.01)

    def soft_thresh(x, t):
        return torch.sign(x) * torch.clamp(x.abs() - t, min=0)

    for step in tqdm(range(n_steps), desc="    Ising opt", leave=False):
        opt.zero_grad()
        Js = (J + J.T) / 2
        Js = Js - torch.diag(Js.diag())
        field = s_active @ Js + h.unsqueeze(0)
        pll   = F.logsigmoid(2 * s_active * field).mean()
        (-pll).backward()
        opt.step()
        with torch.no_grad():
            J.data = soft_thresh(J.data, lam)
            J.data = (J.data + J.data.T) / 2
            J.data.fill_diagonal_(0)

    return J.detach().cpu(), active_idx.cpu()


# ── concept assignment ────────────────────────────────────────────────────────

def assign_concepts(codes_np, char_pos, width_arr, active_idx):
    """
    Compute signed Pearson r of each active feature with:
      char_pos  → pos_ramp if r > R_THRESH, neg_ramp if r < -R_THRESH
      remaining → remaining if r > R_THRESH
    Returns {global_feat_idx: {"concept": str, "score": float}}
    """
    remaining = width_arr - char_pos
    ym_cp  = char_pos  - char_pos.mean()
    ym_rem = remaining - remaining.mean()
    dy_cp  = (ym_cp**2).sum()
    dy_rem = (ym_rem**2).sum()

    assignments = {}
    for local_i, global_i in enumerate(active_idx.tolist()):
        x  = codes_np[:, global_i]
        xm = x - x.mean()
        dx = (xm**2).sum()
        if dx < 1e-12:
            assignments[global_i] = {"concept": "other", "score": 0.0,
                                     "r_cp": 0.0, "r_rem": 0.0}
            continue
        r_cp  = float((ym_cp  @ xm) / np.sqrt(dy_cp  * dx))
        r_rem = float((ym_rem @ xm) / np.sqrt(dy_rem * dx))
        if r_cp > R_THRESH:
            concept, score = "pos_ramp", r_cp
        elif r_cp < -R_THRESH:
            concept, score = "neg_ramp", -r_cp
        elif r_rem > R_THRESH:
            concept, score = "remaining", r_rem
        else:
            concept, score = "other", max(abs(r_cp), abs(r_rem))
        assignments[global_i] = {"concept": concept, "score": score,
                                  "r_cp": r_cp, "r_rem": r_rem}
    return assignments


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_ising(J_active, active_idx, assignments, tag, save_dir):
    p = len(active_idx)
    J_np = J_active.numpy()

    concept_rank = {c: i for i, c in enumerate(CONCEPT_ORDER)}
    positions    = list(range(p))
    positions.sort(key=lambda pos: (
        concept_rank.get(assignments[active_idx[pos].item()]["concept"], 99),
        -assignments[active_idx[pos].item()]["score"],
    ))
    J_sorted       = J_np[np.ix_(positions, positions)]
    concept_labels = [assignments[active_idx[pos].item()]["concept"]
                      for pos in positions]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#1a1a1a")

    nonzero = J_sorted[J_sorted != 0]
    vmax = float(np.percentile(np.abs(nonzero), 99)) if len(nonzero) else 1.0
    im = ax.imshow(J_sorted, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Block separators
    prev = concept_labels[0]
    counts = {}
    for k, c in enumerate(concept_labels):
        counts[c] = counts.get(c, 0) + 1
        if c != prev:
            ax.axhline(y=k - 0.5, color="white", lw=1.2, alpha=0.8)
            ax.axvline(x=k - 0.5, color="white", lw=1.2, alpha=0.8)
            prev = c

    patches = [mpatches.Patch(color=CONCEPT_COLORS[c],
                               label=f"{c}  ({counts.get(c,0)})")
               for c in CONCEPT_ORDER if c in set(concept_labels)]
    ax.legend(handles=patches, loc="lower right", fontsize=9,
              framealpha=0.8, facecolor="#222", labelcolor="white")

    jmax = float(np.abs(J_sorted).max())
    ax.set_title(f"Ising coupling — {tag}  ({p} active features)\n"
                 f"|J|_max={jmax:.4f}  "
                 f"pos_ramp={counts.get('pos_ramp',0)}  "
                 f"neg_ramp={counts.get('neg_ramp',0)}  "
                 f"remaining={counts.get('remaining',0)}",
                 color="#ddd", fontsize=10)
    ax.set_xlabel("Feature (sorted by char_pos correlation)", color="#aaa")
    ax.set_ylabel("Feature (sorted by char_pos correlation)", color="#aaa")
    ax.tick_params(colors="#aaa")
    plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()

    path = Path(save_dir) / f"ising_linebreak_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="google/gemma-3-12b-it")
    ap.add_argument("--device",      default="auto")
    ap.add_argument("--sae-base",    required=True,
                    help="Snapshot root for attn_out SAEs "
                         "(e.g. /path/.hf-cache/models--google--gemma-scope-2-12b-it/snapshots/<hash>)")
    ap.add_argument("--sae-resid",   required=True,
                    help="Path to resid_post layer_24 SAE directory")
    ap.add_argument("--save-dir",    default="figures/linebreak_mech")
    ap.add_argument("--n-steps",     type=int, default=1500)
    ap.add_argument("--n-samples",   type=int, default=5000)
    ap.add_argument("--lam",         type=float, default=0.005)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else args.device
    print(f"Device: {device}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model, device)

    sae_specs = [
        ("attn_o_proj", 0,
         Path(args.sae_base) / "attn_out_all/layer_0_width_16k_l0_small",
         "attn_L0"),
        ("attn_o_proj", 1,
         Path(args.sae_base) / "attn_out_all/layer_1_width_16k_l0_small",
         "attn_L1"),
        ("resid_post",  24,
         Path(args.sae_resid),
         "resid_L24"),
    ]

    summary = {}

    for hook_type, layer_idx, sae_path, tag in sae_specs:
        print(f"\n{'='*55}\n{tag}  ({hook_type} layer {layer_idx})\n{'='*55}")

        print(f"  Loading SAE from {sae_path}...")
        sae = JumpReLUSAE.from_pretrained(str(sae_path), device="cpu",
                                          dtype=torch.float32)
        sae.eval()
        print(f"  d_in={sae.d_in}, d_sae={sae.d_sae}")

        print("  Collecting codes from prose texts...")
        codes, char_pos, width_arr = collect_codes(
            model, tok, sae, (hook_type, layer_idx), device,
            PROSE_TEXTS, PROFILE_WIDTHS,
        )
        print(f"  Collected {len(codes)} samples")

        print("  Fitting Ising model...")
        J_active, active_idx = fit_ising(
            codes.astype(np.float32), device,
            n_steps=args.n_steps, lam=args.lam, n_samples=args.n_samples,
        )

        print("  Assigning concepts...")
        assignments = assign_concepts(codes, char_pos, width_arr, active_idx)
        counts = {}
        for v in assignments.values():
            counts[v["concept"]] = counts.get(v["concept"], 0) + 1
        print(f"  Concept counts: {counts}")

        plot_ising(J_active, active_idx, assignments, tag, args.save_dir)

        summary[tag] = {
            "n_samples":     int(len(codes)),
            "n_active":      int(len(active_idx)),
            "concept_counts": counts,
            "J_max":         float(J_active.abs().max()),
            "J_mean_abs":    float(J_active.abs().mean()),
        }

        del sae  # free memory before loading next

    out = Path(args.save_dir) / "ising_linebreak_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out}")


if __name__ == "__main__":
    main()
