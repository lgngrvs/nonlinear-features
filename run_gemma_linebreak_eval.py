"""Behavioral eval: can Gemma perform fixed-width line breaking?

Tests whether Gemma can correctly insert newlines to format text at a
specified line width, replicating the line-breaking task studied in
"When Models Manipulate Manifolds: The Geometry of a Counting Task"
(arXiv 2601.04480), which found this behavior in Claude 3.5 Haiku.

The core task: given text and a line-width constraint, the model must
count characters on the current line and predict a newline token when
the next word would cause an overflow — a non-trivial geometric
computation over token sequences.

Two evaluation modes
--------------------
generation  Prompt the model to reformat unstructured text at a given
            width. Measures exact formatting match, line overflow rate,
            and word preservation against the textwrap oracle.

logit       Feed pre-formatted text without instruction wrapping and,
            at each line-break decision point, compare
            p(newline token) vs p(next-word token). Tests the model's
            *natural* next-token line-breaking behavior as in the paper.

Usage
-----
    python run_gemma_linebreak_eval.py --device cuda
    python run_gemma_linebreak_eval.py --model google/gemma-3-4b-it --device cuda
    python run_gemma_linebreak_eval.py --mode logit --device cpu --n-texts 5
    python run_gemma_linebreak_eval.py --mode generation --widths 40 80
"""

import argparse
import json
import os
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Prose corpus — 12 diverse passages, varied word lengths and sentence rhythm
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

LINE_WIDTHS = [40, 60, 80]


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

def wrap_oracle(text: str, width: int) -> str:
    """Ground-truth fixed-width word-wrap (no hyphenation)."""
    return textwrap.fill(text, width=width)


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

@dataclass
class GenResult:
    exact_match: bool         # pred.strip() == gold.strip()
    overflow_rate: float      # fraction of output lines exceeding width
    word_preservation: float  # fraction of gold words present in prediction


def compute_gen_result(pred: str, gold: str, width: int) -> GenResult:
    exact = pred.strip() == gold.strip()

    lines = pred.strip().splitlines()
    overflow_rate = (
        sum(1 for ln in lines if len(ln) > width) / len(lines)
        if lines else 1.0
    )
    gold_words = set(gold.lower().split())
    pred_words = set(pred.lower().split())
    word_preservation = (
        len(gold_words & pred_words) / len(gold_words) if gold_words else 1.0
    )
    return GenResult(
        exact_match=exact,
        overflow_rate=overflow_rate,
        word_preservation=word_preservation,
    )


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def find_newline_token_id(tokenizer) -> Optional[int]:
    """Return the token ID whose decoding contains '\\n', or None."""
    for probe in ["word\n", "\nword", "\n"]:
        for tid in tokenizer.encode(probe, add_special_tokens=False):
            if "\n" in tokenizer.decode([tid]):
                return tid
    return None


def collect_decision_points(
    wrapped: str,
    tokenizer,
    newline_id: int,
    min_context: int = 10,
    no_break_stride: int = 4,
) -> list[dict]:
    """
    Tokenize `wrapped` and return decision-point dicts.

    Break points    — positions where the true next token is a newline.
    No-break points — mid-line positions sampled every `no_break_stride`
                      tokens, skipping the ±2 neighbourhood of any newline.

    Each dict:
      input_ids — token ids preceding the decision token
      target_id — true next token (newline or continuation word)
      alt_id    — competing token (continuation word or newline)
      is_break  — True iff target is the newline token
    """
    ids = tokenizer.encode(wrapped, add_special_tokens=False)
    newline_pos = {i for i, t in enumerate(ids) if t == newline_id}
    points = []
    no_break_counter = 0

    for i in range(min_context, len(ids) - 1):
        tid = ids[i]
        if tid == newline_id:
            next_tid = ids[i + 1]
            if next_tid == newline_id:
                continue  # skip consecutive newlines
            points.append(dict(
                input_ids=ids[:i], target_id=tid, alt_id=next_tid, is_break=True,
            ))
        else:
            near = any((i + k) in newline_pos for k in range(-2, 3))
            if near:
                continue
            no_break_counter += 1
            if no_break_counter % no_break_stride == 0:
                points.append(dict(
                    input_ids=ids[:i], target_id=tid, alt_id=newline_id, is_break=False,
                ))

    return points


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device not in ("cpu", "mps") else torch.float32
    print(f"Loading {model_name} ({dtype}, device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device not in ("cpu", "mps") else None,
    )
    if device in ("cpu", "mps"):
        model = model.to(device)
    model.eval()
    cfg = getattr(model.config, "text_config", model.config)
    print(f"  d_model={cfg.hidden_size}, layers={cfg.num_hidden_layers}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation eval
# ---------------------------------------------------------------------------

_GEN_INSTRUCTION = (
    "Reformat the following text so that no line exceeds {width} characters. "
    "Break only at word boundaries. Preserve all words in their original order. "
    "Output only the reformatted text, nothing else.\n\n{text}"
)


def run_generation_eval(
    model,
    tokenizer,
    texts: list[str],
    widths: list[int],
    device: str,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Instruction-following eval: ask the model to reformat text at each width."""
    results = []
    for width in widths:
        samples = []
        for text in tqdm(texts, desc=f"gen width={width}"):
            gold = wrap_oracle(text, width)
            msg = _GEN_INSTRUCTION.format(width=width, text=text)

            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": msg}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt = msg + "\n\nFormatted text:\n"

            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            pred = tokenizer.decode(
                out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()

            r = compute_gen_result(pred, gold, width)
            samples.append({"gold": gold, "pred": pred, **asdict(r)})

        exact = float(np.mean([s["exact_match"] for s in samples]))
        overflow = float(np.mean([s["overflow_rate"] for s in samples]))
        word_p = float(np.mean([s["word_preservation"] for s in samples]))
        print(
            f"  width={width:2d}: exact={exact:.1%}  "
            f"overflow={overflow:.3f}  word_pres={word_p:.3f}"
        )
        results.append(dict(
            width=width,
            exact_match_rate=exact,
            mean_overflow_rate=overflow,
            mean_word_preservation=word_p,
            samples=samples,
        ))
    return results


# ---------------------------------------------------------------------------
# Logit probe eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def _logit_pair(
    model, input_ids: list, target_id: int, alt_id: int, device: str
) -> tuple[float, float]:
    """Return (log_p_target, log_p_alt) at the last position."""
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    lp = torch.log_softmax(model(x).logits[0, -1, :], dim=-1)
    return lp[target_id].item(), lp[alt_id].item()


def run_logit_eval(
    model,
    tokenizer,
    texts: list[str],
    widths: list[int],
    device: str,
    max_points_per_text: int = 20,
) -> list[dict]:
    """
    Natural next-token eval: at each line-break decision boundary in
    pre-formatted text, check whether p(newline) > p(next-word token).

    No chat template is used — we test raw next-token predictions to
    mirror the paper's setup on Claude Haiku.
    """
    newline_id = find_newline_token_id(tokenizer)
    if newline_id is None:
        print("WARNING: could not locate newline token; skipping logit eval.")
        return []
    print(f"Newline token: id={newline_id}  repr={repr(tokenizer.decode([newline_id]))}")

    results = []
    for width in widths:
        b_ok, b_mg = [], []
        nb_ok, nb_mg = [], []

        for text in tqdm(texts, desc=f"logit width={width}"):
            wrapped = wrap_oracle(text, width)
            pts = collect_decision_points(wrapped, tokenizer, newline_id)

            bpts = [p for p in pts if p["is_break"]][:max_points_per_text]
            nbpts = [p for p in pts if not p["is_break"]][:max_points_per_text]

            for p in bpts:
                lt, la = _logit_pair(
                    model, p["input_ids"], p["target_id"], p["alt_id"], device
                )
                b_ok.append(lt > la)
                b_mg.append(lt - la)

            for p in nbpts:
                lt, la = _logit_pair(
                    model, p["input_ids"], p["target_id"], p["alt_id"], device
                )
                nb_ok.append(lt > la)
                nb_mg.append(lt - la)

        ba = float(np.mean(b_ok)) if b_ok else float("nan")
        nba = float(np.mean(nb_ok)) if nb_ok else float("nan")
        bm = float(np.mean(b_mg)) if b_mg else float("nan")
        nbm = float(np.mean(nb_mg)) if nb_mg else float("nan")
        print(
            f"  width={width:2d}: "
            f"break_acc={ba:.3f} (n={len(b_ok)})  "
            f"no_break_acc={nba:.3f} (n={len(nb_ok)})  "
            f"break_margin={bm:.3f}"
        )
        results.append(dict(
            width=width,
            newline_token_id=newline_id,
            break_accuracy=ba,
            no_break_accuracy=nba,
            mean_break_margin=bm,
            mean_no_break_margin=nbm,
            n_break_points=len(b_ok),
            n_no_break_points=len(nb_ok),
        ))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Gemma fixed-width line-breaking eval")
    ap.add_argument("--model", default="google/gemma-3-12b-it",
                    help="HuggingFace model ID")
    ap.add_argument("--device", default="auto",
                    help="cuda | mps | cpu | auto")
    ap.add_argument("--mode", choices=["generation", "logit", "both"], default="both",
                    help="Which eval(s) to run")
    ap.add_argument("--widths", type=int, nargs="+", default=LINE_WIDTHS,
                    help="Line widths to test (characters)")
    ap.add_argument("--n-texts", type=int, default=len(PROSE_TEXTS),
                    help="Number of texts from the corpus to use")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="Max tokens for generation eval")
    ap.add_argument("--max-points-per-text", type=int, default=20,
                    help="Max decision points per text in logit eval")
    ap.add_argument("--save-dir", default="figures/linebreak_eval",
                    help="Directory for results JSON")
    args = ap.parse_args()

    if args.device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = args.device
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    texts = PROSE_TEXTS[:args.n_texts]

    summary: dict = {
        "model": args.model,
        "device": device,
        "n_texts": len(texts),
        "widths": args.widths,
    }

    if args.mode in ("generation", "both"):
        print("\n" + "=" * 60)
        print("GENERATION EVAL")
        print("=" * 60)
        summary["generation"] = run_generation_eval(
            model, tokenizer, texts, args.widths, device, args.max_new_tokens,
        )

    if args.mode in ("logit", "both"):
        print("\n" + "=" * 60)
        print("LOGIT PROBE EVAL")
        print("=" * 60)
        summary["logit"] = run_logit_eval(
            model, tokenizer, texts, args.widths, device, args.max_points_per_text,
        )

    out = Path(args.save_dir) / "linebreak_eval_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out}")

    print("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
    if "generation" in summary:
        print("Generation eval (instruction-following):")
        for r in summary["generation"]:
            print(
                f"  width={r['width']:2d}:  exact={r['exact_match_rate']:.1%}  "
                f"overflow={r['mean_overflow_rate']:.3f}  "
                f"word_pres={r['mean_word_preservation']:.3f}"
            )
        print(
            "  Metrics: exact=formatting matches oracle; overflow=lines exceeding width;\n"
            "           word_pres=fraction of words preserved (random baseline ~0.5)"
        )
    if "logit" in summary:
        print("Logit probe eval (natural next-token):")
        for r in summary["logit"]:
            print(
                f"  width={r['width']:2d}:  break_acc={r['break_accuracy']:.3f}  "
                f"no_break_acc={r['no_break_accuracy']:.3f}  "
                f"margin={r['mean_break_margin']:.3f}"
            )
        print(
            "  Metrics: break_acc=p(newline)>p(word) at break pts (random baseline 0.5);\n"
            "           no_break_acc=p(word)>p(newline) at mid-line pts;\n"
            "           margin=mean log-prob difference at break points"
        )


if __name__ == "__main__":
    main()
