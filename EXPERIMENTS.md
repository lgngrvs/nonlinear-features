# Gemma Line-Breaking Experiments

## Behavioral Evals
`run_gemma_linebreak_eval.py` · results: `figures/linebreak_eval_v2/linebreak_eval_results.json`

- Ran generation and logit-probe evals for Gemma 3 12B and 27B on the fixed-width line-breaking task; both models show "trigger-happy" newline behavior with p(newline) 3–424× higher than p(next word) mid-line, and nl_margin ramps sharply at 75–85% of target width.

## SAE Contrastive Analysis
`run_gemma_linebreak_mech.py` · figures: `figures/linebreak_mech/sae_contrastive.html`, `sae_feature_*.html` · data: `figures/linebreak_mech/mech_summary.json`

- Compared resid_post L24 SAE features at the same char_pos under width=40 (near-break) vs width=80 (mid-line); found F14066 as a binary near-break detector (mean_40=337, mean_80=0) and F885 as the largest-delta feature overall.

## SAE PCA + Control
`run_gemma_linebreak_mech.py` · figures: `figures/linebreak_mech/sae_pca_remaining.html`, `sae_control_profiles.html`

- Projected all (text, width, char_pos) SAE activations into 3D PCA space (PC1=44.6%) and ran the same analysis on repeated "ab " control tokens to isolate position signal from content.

## Layer Probe (corrected)
`run_layer_probe_fine.py`, `run_gemma_linebreak_mech.py` · data: `figures/linebreak_mech/layer_probe_fine.json`, `layer_probe_full_shuffled.json`

- Probed char_pos linear decodability from each layer's residual stream using control sequences and shuffled CV (fixing a fold-ordering bug that produced spurious negative R²); found R²>0.97 from layer 1 onward, declining gradually from layer 21 to ~0.73 at the final layer.

## Attn vs MLP Decomposition
`run_layer_probe_fine.py` · data: `figures/linebreak_mech/layer_probe_attn_out.json`, `layer_probe_mlp_out.json`

- Hooked `post_attention_layernorm` and `post_feedforward_layernorm` at layers 0–6; attention is the primary position writer (R²=0.856 at L0, jumping to 0.995 at L1), with MLP lagging slightly and catching up by layer 5.

## PCA Per-Axis R² and Embedding Baseline
`run_pca_viz.py` · figures: `figures/linebreak_mech/pca_top3_attn_out_L{0,1}.html` · data: `figures/linebreak_mech/pca_viz_summary.json`

- Confirmed raw token embeddings have R²=0.000 with char_pos (RoPE adds no position to embeddings); at L0 attn PC1 is the position axis (R²=0.54), at L1 attn position migrates to PC3 (R²=0.67) while PC1 captures width-context (R²=0.10).

## PCA Trajectory Geometry
`run_pca_trajectory.py` · figures: `figures/linebreak_mech/attn_trajectory_L{0,1}_{2d,3d,scree}.html`

- Visualised how attn_out moves through activation space as char_pos varies; both L0 and L1 are ~2D (PC1+PC2 ≈ 95–96%), with L0's PC1 being the position direction and L1's PC1 being a width-discriminating direction.

## Attn SAE Position Features
`run_attn_sae_position.py` · figures: `figures/linebreak_mech/attn_sae_*_profiles.html`, `attn_sae_L{0,1}_r2_bar.html` · data: `figures/linebreak_mech/attn_sae_position_summary.json`

- Loaded GemmaScope `attn_out_all` 16k-small SAEs for L0 and L1 (hook at `o_proj.input`); L0 best feature F299 r²=0.56, L1 best feature F632 r=+0.905 r²=0.82 with char_pos — a very clean monotone position-ramp feature.

## Ising Coactivation Matrices
`run_linebreak_ising.py` · figures: `figures/linebreak_mech/ising_linebreak_{attn_L0,attn_L1,resid_L24}.png` · data: `figures/linebreak_mech/ising_linebreak_summary.json`

- Fit pairwise Ising models on prose-text SAE codes for attn_out L0/L1 and resid_post L24 (825 samples each); concept labeling was sparse because content noise swamps position signal at R_THRESH=0.35 — needs re-run with control sequences or a lower threshold.
