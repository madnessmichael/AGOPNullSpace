# Beyond Behavioral Refusal Directions: Null-Space Constrained Safety Steering with Recursive Feature Machines

<p align="center">
  <b>AGOPN: Null-Space Constrained Safety Steering with Recursive Feature Machines</b>
</p>

<p align="center">
  <a href="#results">Results</a> •
  <a href="#method">Method</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#known-limitations">Known Limitations</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="figures/FigureAGOPNs.png" width="95%" alt="AGOPNullSpace pipeline overview"/>
</p>

> **EMNLP 2025 submission:** *Beyond Behavioral Refusal Directions: Null-Space Constrained Safety Steering with Recursive Feature Machines*

> **This README reflects the current top-K ridge-combo pipeline** (the method actually
> used to produce the results below). See `CLAUDE.md` for the full math, architecture,
> naming conventions, and operational notes if you're developing here rather than just
> reading about the method.

---

## Overview

**AGOPNullSpace** replaces the difference-in-means refusal direction used in standard
activation steering with a direction learned by a **Recursive Feature Machine (RFM)**
from the **Average Gradient Outer Product (AGOP)** of a kernel ridge regression. The
null-space constraint from [AlphaSteer (ICLR 2026)](https://github.com/AlphaLab-USTC/AlphaSteer)
is preserved as-is, keeping the utility-preservation guarantee intact while improving
the quality of the refusal direction — particularly against encoding-based attacks such
as Cipher.

Two changes since the original method description:

1. **The refusal direction is now a ridge-combination of the top-K AGOP eigenvectors**,
   not just the top-1 eigenvector. On held-out AUC, the top-1 eigenvector alone loses to
   plain DiffMean at nearly every layer on all 3 models; combining the top-K
   eigenvectors via a ridge fit recovers and then beats DiffMean (see
   [Results](#results)). This does not change the inference formula, gate, or
   null-space math — only how the direction `r` is computed.
2. **The refusal/compliance training labels now come from the target model's own
   behavior**, not an externally-labeled dataset. Prompts are drawn from SORRY-Bench,
   the model generates a real response, and a strict "is this an explicit, unhedged
   refusal" judge labels it — rather than reusing AlphaSteer's original
   `harmful_harmless_instructions` labels (which reflect what a human/dataset author
   considered harmful, not what the target model itself actually refuses).

---

## Key Idea

Standard activation steering computes a refusal direction as the mean difference
between activations of refused and compliant prompts (DiffMean). This linear estimate
works well when the two classes are linearly separable in Euclidean space, but degrades
on attacks that obfuscate the surface form of a prompt (Cipher, Base64, role-play
encoding).

<p align="center">
  <img src="figures/fig1_teaser_dim_llama3.1.png" width="49%" alt="DiffMean 3D: hard-refusal points scattered across its own axis, AUC 0.27, no separation"/>
  <img src="figures/fig1_teaser_agop_topk_llama3.1.png" width="49%" alt="AGOP top-K ridge-combo 3D: hard-refusal points form a distinct cluster along its own axis, AUC 0.79"/>
  <br>
  <em>Fig. 1 — Same real activations (llama3.1, layer 12, hardest real case for DiffMean:
  encoded jailbreaks — Caesar/morse/atbash/ascii), one 3D figure per method so each
  method's own separating axis (x) is unambiguous. The gray plane in each 3D plot and
  the dashed line in the 1D histogram beneath it mark the same thing — the ROC-optimal
  decision cut (Youden's J) on that method's own axis — so the separation doesn't have
  to be eyeballed off a bare point cloud. <b>Left (DiffMean):</b> hard-refusal points
  (orange) sit on both sides of the cut, mixed in with compliance — AUC 0.27, worse than
  random; no single cut on this axis works, because the malicious points form two
  separate clusters on opposite ends of it. <b>Right (AGOP top-K ridge-combo):</b> the
  same points fall almost entirely on one side of the cut — AUC 0.79. The shared y-axis
  in both (projection onto <code>u</code>, the real null-space gate) and z-axis (residual
  variance after removing each point's own encoding-style mean, so it isn't just showing
  "which cipher was used") are identical across the two figures — only the x-axis (each
  method's own direction) differs, by design, so the comparison is apples-to-apples. No
  synthetic/toy data anywhere. Reproduce with
  <code>python experimental/fig1_teaser_diffmean_vs_agop_topk.py</code>.</em>
</p>

<p align="center">
  <img src="figures/fig1_teaser_dim_qwen2.5.png" width="49%" alt="qwen2.5 DiffMean 3D, AUC 0.88"/>
  <img src="figures/fig1_teaser_agop_topk_qwen2.5.png" width="49%" alt="qwen2.5 AGOP top-K 3D, AUC 0.94"/>
  <br>
  <img src="figures/fig1_teaser_dim_gemma2.png" width="49%" alt="gemma2 DiffMean 3D, AUC 0.87"/>
  <img src="figures/fig1_teaser_agop_topk_gemma2.png" width="49%" alt="gemma2 AGOP top-K 3D, AUC 0.95"/>
  <br>
  <em>Fig. 2 — Same real-data method, qwen2.5 (top row, layer 12) and gemma2 (bottom
  row, layer 22). <b>These two models could not use the encoding-only subset Fig. 1
  uses</b>: checked directly against the source dataset, qwen2.5 has only 5 hard-refusal
  rows and gemma2 only 1 across all 1,760 caesar/morse/atbash/ascii prompts combined —
  both models essentially never produce a strict, unhedged refusal to an
  encoding-obfuscated prompt in the first place, a real finding in its own right, not an
  extraction bug. AUC on 1–5 positives isn't a meaningful comparison, so these two
  figures fall back to all 21 SORRY-Bench prompt styles instead (thousands of positives
  each) — still real data and a real per-model DIM-vs-AGOP gap, just not specifically an
  "encoding" story for these two. Gap is smaller than llama3.1's (qwen2.5: AUC
  0.88→0.94; gemma2: 0.87→0.95) because the broader 21-style task is easier for DIM to
  begin with — most styles are plain natural language, not obfuscated.</em>
</p>

AGOPNullSpace computes the direction as the **top-K eigenvectors of the AGOP matrix**,
ridge-combined into a single vector, via an iterative kernel regression loop:

```
For t = 1 … T:
    K_M(x,z) = exp(−(x−z)ᵀ M (x−z) / L)       # Laplace kernel with metric M
    α        = (K_M + λI)⁻¹ y                    # Kernel Ridge Regression
    G_t      = (1/n) Σᵢ ∇f(xᵢ) ∇f(xᵢ)ᵀ         # Average Gradient Outer Product
    M_{t+1}  = G_t / ‖G_t‖_F                      # metric update

{v_1, ..., v_K} = top_K_eigenvectors(M_T)
β               = RidgeClassifier(alpha=1.0).fit([X·v_1, ..., X·v_K], y).coef_
r_rfm           = normalize(Σ_c β_c · v_c)
```

This direction is then used in place of `r_dim` inside the AlphaSteer closed-form:

```
Δ̃* = R H_mᵀ P̂ᵀ (P̂ H_m H_mᵀ P̂ᵀ + α P̂ P̂ᵀ)⁺    # Eq. 9 from AlphaSteer
```

where `P̂ = Û Ûᵀ` is the null-space projector built from the lowest eigenvectors of the
benign covariance (60% of the spectrum, ρ=0.6). The null-space guarantee `Δ* H_b ≈ 0`
holds independently of which `r` is plugged in — this repo reformulates the closed-form
solve into a k×k (null-space-rank) Cholesky problem instead of a d×d pseudo-inverse, see
`CLAUDE.md` for why.

---

## Results

Evaluated on Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Gemma-2-9B-IT, judged with
GPT-4o-mini (`evaluation/jailbreak.py` for attack datasets, `evaluation/xstest.py` for
utility). All numbers below are from the **K=10 ridge-combo** steering matrices
(`steering_matrix_<model>_rfm_rc_full_hard_refusal_topk10.pt`), trained on the full
9,236-row SORRY-Bench refuse/comply set with the strict hard-refusal judge.

### Why top-K: held-out AUC vs DiffMean

The refusal/compliance concept direction's own discriminative quality, per layer,
before it ever gets used for steering (`experimental/notebooks/AGOPNs_RFM_steering_RD.ipynb`,
`experimental/checkpoints/k_sweep_summary_all_models.json`):

| K | llama3.1 AUC | layers beating DiffMean | qwen2.5 AUC | layers beating DiffMean | gemma2 AUC | layers beating DiffMean |
|---|---|---|---|---|---|---|
| 1 (old method) | 0.7913 | 0/26 | 0.7524 | 0/22 | 0.7819 | 2/36 |
| 2 | 0.8769 | 17/26 | 0.8391 | 5/22 | 0.8599 | 11/36 |
| 5 | 0.9187 | 20/26 | 0.9281 | 21/22 | 0.9172 | 25/36 |
| 7 | 0.9279 | 23/26 | 0.9359 | 22/22 | 0.9262 | 31/36 |
| **10** | **0.9394** | **25/26** | **0.9428** | **22/22** | **0.9327** | **35/36** |

The top-1 eigenvector alone (the method this repo originally used) loses to plain
DiffMean at nearly every layer, on all 3 models — this is why the ridge-combo extension
exists. Diminishing returns above K=7 (qwen2.5 is already 22/22 there); K=10 was picked
as the production default. qwen2.5 needs a larger K than llama3.1/gemma2 to close the
gap.

### Defense Success Rate + Utility, K=10 — full dose-response, per model

Judged with `llm_judge_evaluation.ipynb` (the official paper-number notebook). Each
table below rows the DSR-per-attack-dataset (`reject`-judged fraction) *and* the
utility metrics (XSTest full-compliance %, GSM8K/MATH500 exact-match accuracy %) at
the **same** steering strength, so a row is directly readable as one operating point's
full safety/utility tradeoff — read across a row, not just down a column.

#### llama3.1

| ε | aim | autodan | cipher | gcg | jailbroken | pair | renellm | **avg DSR** | XSTest compliance | GSM8K acc. | MATH500 acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 92.0 | 46.0 | 57.0 | 99.0 | 79.8 | 51.0 | 30.0 | 65.0 | 92.4 | 84.0 | 47.0 |
| 1.0 | 100.0 | 100.0 | 94.0 | 100.0 | 94.8 | 96.0 | 99.0 | 97.7 | 92.0 | 89.0 | 42.0 |
| **1.3** | 100.0 | 100.0 | 100.0 | 100.0 | 95.8 | 99.0 | 100.0 | **99.3** | 90.4 | 86.0 | 49.0 |

<p align="center">
  <img src="figures/dose_response_llama3.1_tradeoff.png" width="49%" alt="llama3.1 safety vs utility tradeoff"/>
  <img src="figures/dose_response_llama3.1_per_dataset.png" width="49%" alt="llama3.1 DSR per attack dataset"/>
</p>

**Saturates fastest and cheapest of the 3 models**: 97.7% avg DSR already at ε=1.0,
while every utility metric stays within 2–5pp of its unsteered baseline across the
entire tested range — the healthiest safety/utility tradeoff measured.

#### qwen2.5

| ε | aim | autodan | cipher | gcg | jailbroken | pair | renellm | **avg DSR** | XSTest compliance | GSM8K acc. | MATH500 acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 25.0 | 22.0 | 69.0 | 81.0 | 74.4 | 19.0 | 3.0 | 41.9 | 96.4 | 95.0 | 62.0 |
| 1.0 | 74.0 | 94.0 | 66.0 | 89.0 | 83.4 | 53.0 | 11.0 | 67.2 | 94.0 | 94.0 | 57.0 |
| 2.0 | 98.0 | 97.0 | 71.0 | 98.0 | 84.0 | 84.0 | 40.0 | 81.7 | 94.0 | 94.0 | 59.0 |
| 3.0 | 100.0 | 96.0 | 95.0 | 100.0 | 86.6 | 98.0 | 60.0 | 90.8 | 90.8 | 93.0 | 58.0 |
| 3.2 | 100.0 | 94.0 | 98.0 | 100.0 | 86.0 | 97.0 | 67.0 | 91.7 | 88.8 | 92.0 | 57.0 |
| 3.5 | 100.0 | 96.0 | 97.0 | 100.0 | 88.6 | 98.0 | 97.0 | 96.7 | 89.2 | 92.0 | 55.0 |
| 3.7 | 100.0 | 98.0 | 97.0 | 100.0 | 91.2 | 98.0 | 100.0 | 97.7 | 89.2 | 94.0 | 56.0 |
| **4.0** | 100.0 | 100.0 | 99.0 | 100.0 | 93.0 | 98.0 | 100.0 | **98.6** | 86.8 | 91.0 | **48.0** |

<p align="center">
  <img src="figures/dose_response_qwen2.5_tradeoff.png" width="49%" alt="qwen2.5 safety vs utility tradeoff"/>
  <img src="figures/dose_response_qwen2.5_per_dataset.png" width="49%" alt="qwen2.5 DSR per attack dataset"/>
</p>

**renellm needs a comparatively high dose to move** (3→11→40→60% through ε=3.0, only
breaking past 90% at ε≥3.5) and drags the average up late in the sweep. **MATH500
accuracy is the standout cost**: it tracks roughly flat 55–62% through ε=3.7, then
drops to 48% at the production strength ε=4.0 — the single largest utility hit
measured anywhere in this pipeline, bigger than the XSTest compliance drop at the same
strength (−9.6pp).

#### gemma2

| ε | aim | autodan | cipher | gcg | jailbroken | pair | renellm | **avg DSR** | XSTest compliance | GSM8K acc. | MATH500 acc. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0 | 6.0 | 73.0 | 94.0 | 68.8 | 18.0 | 7.0 | 38.1 | 82.0 | 89.0 | 41.0 |
| 2.0 | 3.0 | 48.0 | 75.0 | 99.0 | 77.6 | 56.0 | 31.0 | 55.7 | 80.0 | 88.0 | 42.0 |
| 4.0 | 77.0 | 92.0 | 72.0 | 99.0 | 86.8 | 80.0 | 74.0 | 83.0 | 77.2 | 87.0 | 43.0 |
| 6.0 | 100.0 | 100.0 | 71.0 | 100.0 | 99.4 | 92.0 | 93.0 | 93.6 | 75.2 | 87.0 | 40.0 |
| 8.0 | 100.0 | 100.0 | 72.0 | 100.0 | 99.4 | 98.0 | 98.0 | 95.3 | 72.0 | 87.0 | 39.0 |
| **12.0** | 100.0 | 100.0 | 72.0 | 100.0 | 100.0 | 99.0 | 100.0 | **95.9** | 63.2 | 83.0 | 37.0 |

<p align="center">
  <img src="figures/dose_response_gemma2_tradeoff.png" width="49%" alt="gemma2 safety vs utility tradeoff"/>
  <img src="figures/dose_response_gemma2_per_dataset.png" width="49%" alt="gemma2 DSR per attack dataset"/>
</p>

**Cipher is stuck across the entire dose range** (73→75→72→71→72→72 from ε=0 to
ε=12) — not sampling noise at the endpoints, a genuine plateau; more strength buys
nothing on this dataset for this model, unlike every other (model, dataset) pair.
**XSTest compliance degrades the most steeply of the 3 models** — a steady, almost
linear decline from 82% to 63.2% (**−18.8pp**) as strength increases, with no
plateau in sight by ε=12. Baseline (ε=0, unsteered) XSTest compliance is already only
82% — the base model over-refuses some XSTest prompts before any steering at all.

### Cross-model takeaways

- **llama3.1** > **qwen2.5** > **gemma2** in how cheaply (lowest strength, least
  utility cost) each model reaches near-ceiling DSR.
- **MATH500**, not XSTest, is the largest single utility cost measured — but only for
  qwen2.5, and only right at its production strength (48% vs a flat ~55–62% at every
  lower strength tested). GSM8K is comparatively robust everywhere (≤6pp drop at max
  strength, any model).
- Every model has at least one dataset that resists steering: gemma2/cipher (flat
  ~72% at every strength ≥2.0), qwen2.5/renellm (needs ε≥3.5 to break 90%). No
  single strength is uniformly "the" right choice per model — it's a genuine
  per-dataset tradeoff curve, not a step function.

> **Historical figures** (`figures/table1_dsr.png`, `fig2_dsr_sweep.png`,
> `fig3_cipher_highlight.png`, `fig4_radar.png`, `table2_utility.png`) visualize an
> **older DIM-vs-top-1-RFM comparison on a different training set**
> (`harmful_harmless_instructions`, not the SORRY-Bench refuse-compliance set above) —
> they predate the top-K method and the numbers in this section are not directly
> comparable to them. Regenerate via `scripts/gen_figures.py`/`scripts/gen_tables.py`
> against the tables above before reusing them in the paper.

---

## Method

The full pipeline:

**Step 1 — Build refuse/compliance labels from the target model's own behavior.**
For each of the 3 models, sample SORRY-Bench prompts (all 21 `prompt_style`
linguistic-mutation variants — base, slang, role-play, ascii, morse, caesar cipher,
translations, etc. — 9,236 rows after dropping 4 malformed source rows), generate a
real response, and judge it with a strict "is this an explicit, unhedged refusal"
criterion (not "is this harmful" — a looser jailbreak-content judge would also label a
hedge-then-comply response as refusal, which is too loose a criterion for training a
*refusal* concept vector specifically). `src/build_refusal_compliance_sorrybench.py`.

**Step 2 — Collect activations.** Extract hidden-state tensors for the refusal/
compliance set (`r`'s training data) and reuse AlphaSteer's original 14k-benign /
2k-malicious set for the gate `u` and null-space `Q` — only the *source* of the
direction `r` changed from AlphaSteer's original pipeline, not the gate-fitting data.

**Step 3 — Run the RFM/AGOP loop, keep the top-K eigenvectors.** Per steering layer:
fit the kernel ridge regression, extract the top-K eigenvectors of the resulting AGOP
matrix (not just the top-1), and ridge-combine them into a single direction `r`
(see [Key Idea](#key-idea)).

**Step 4 — Build the null-space projector.** Benign activation covariance → keep the
60% lowest eigenvectors (ρ=0.6) → `P̂ = Û Ûᵀ`.

**Step 5 — Solve the steering matrix in closed form.** AlphaSteer Eq. 9 with the top-K
ridge-combo `r` in place of `r_dim`, reformulated as a k×k (null-space-rank) solve
rather than the d×d pseudo-inverse — guarantees `Δ* H_b ≈ 0` (utility preservation) by
construction, not by floating-point cancellation.

**Step 6 — Inference.** At the last non-padding prompt token during prefill, propagated
to later decode steps via the KV cache:
```
gate = σ(a·(uᵀh_last − 0.5))     # sigmoid gate, slope a=10.0
h'   = h + strength · gate · r
```

---

## Repository Structure

```
AGOPNullSpace/
│
├── src/
│   ├── build_refusal_compliance_sorrybench.py   # Step 1: SORRY-Bench refuse/comply labels
│   ├── extract_refusal_compliance_embeddings.py # Step 2: activations for the r-training set
│   ├── extract_embeddings.py                    # activations for the gate/null-space set
│   ├── rfm_refusal_vector.py                    # Step 3: AGOP/RFM + top-K ridge-combo direction
│   ├── calc_steering_matrix_rfm_rc.py           # Steps 3-5, active pipeline entry point
│   ├── calc_steering_matrix_rfm_rc_no_nullspace.py  # no-null-space ablation
│   ├── calc_steering_matrix.py                  # AlphaSteer baseline (DiffMean), legacy
│   ├── generate_response.py                     # Step 6: single production generation entry point
│   ├── AlphaSteerModel/{AlphaLlama,AlphaQwen,AlphaGemma}.py  # steering-patched HF model classes
│   └── utils/{const,steering_utils,mask_utils,embedding_utils}.py
│
├── config/<model>_<variant>_rfm/<dataset>.yaml   # one YAML per (model, variant, dataset)
│                                                  # variant suffix encodes the ablation, see CLAUDE.md
│
├── data/
│   ├── embeddings/<model>/                       # activation tensors, incl. refusal_compliance* subdirs
│   ├── steering_matrix/                          # computed steering matrices (*.pt, *_meta.json)
│   ├── responses/<model>/                        # main production generation output
│   └── responses_ridge-combo_top-K/<model>/       # top-K ridge-combo generation output (this README's numbers)
│
├── evaluation/
│   ├── jailbreak.py            # attack-dataset judge (reject/jailbreak)
│   ├── hard_refusal_judge.py   # strict refusal judge, for Step 1's training labels
│   ├── xstest.py               # utility judge (3-way compliance classification)
│   └── summarize_results.py, TheLastJudgment*.sh   # paper-table aggregation
│
├── llm_judge_evaluation.ipynb   # official paper-number judging notebook (repo root)
├── inference_sample.ipynb       # ground-truth reference implementation (monkey-patched, not re-implemented)
│
├── scripts/*.sh                 # original AlphaSteer-era orchestration
├── scripts_claude/*.sh          # refuse-compliance sweep orchestration (put new scripts here)
│
├── experimental/                # in-progress R&D notebooks/checkpoints/figures — not production artifacts
│   └── notebooks/AGOPNs_RFM_steering_RD.ipynb   # the top-K ridge-combo K-sweep source
│
└── requirements.txt
```

See `CLAUDE.md` for naming conventions (`config/`/steering-matrix filenames), the full
math derivation, and operational notes (GPU allocation, non-determinism, known bugs and
their fixes).

---

## Installation

```bash
git clone https://github.com/<your-org>/AGOPNullSpace.git
cd AGOPNullSpace
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.6.0`, `transformers==4.52.4`, and installs the
`xRFM` library directly from git (`dmbeaglehole/xRFM@773fae8`) — this is required, not
optional; `compute_concept_vector_layer()` uses it for the RFM/AGOP fit.

---

## Usage

### 1. Build the refuse/compliance training set (once per model)

```bash
python src/build_refusal_compliance_sorrybench.py --model_name llama3.1 --device cuda:0 \
    --full --hard_refusal
# --full: all 21 SORRY-Bench prompt_style variants (9,236 rows) instead of the 440-row base set
# --hard_refusal: strict "explicit unhedged refusal" judge (used for all numbers in this README)
python src/extract_refusal_compliance_embeddings.py --model_name llama3.1 --device cuda:0
```

### 2. Compute the top-K ridge-combo steering matrix

```bash
python src/calc_steering_matrix_rfm_rc.py \
    --model_name llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --rc_subdir refusal_compliance_full_hard_refusal \
    --n_components 10 --combine_topk \
    --save_path data/steering_matrix/steering_matrix_llama3.1_rfm_rc_full_hard_refusal_topk10.pt \
    --device cuda
```
Omit `--n_components`/`--combine_topk` for the old top-1-eigenvector method (not
recommended — see [Results](#results)).

### 3. Generate responses

```bash
python src/generate_response.py --config_path config/llama3.1_rc_hr_topk10_rfm/aim.yaml
```
One YAML per (model, dataset); `strength` inside the config is a comma-joined list
swept in a single process. See `scripts_claude/*.sh` for orchestrating many
(model, dataset) jobs across multiple GPUs.

### 4. Evaluate

```bash
# attack datasets → reject/jailbreak
python evaluation/jailbreak.py ...
# utility (XSTest) → 3-way compliance classification
python evaluation/xstest.py ...
```
Or use `llm_judge_evaluation.ipynb` (repo root) directly — it mirrors the two scripts
above exactly and is what produced the numbers in [Results](#results).

---

## Steering Strength Convention

**Strength scale is not comparable across models** — each model's production sweep
found its own effective range empirically; there is no shared unit:

| Model | Production strengths swept | Best DSR at |
|---|---|---|
| llama3.1 | 0.0, 1.0, 1.3, 1.5, 1.7 | 1.3 |
| qwen2.5 | 0.0, 4.0, 5.0, 6.0, 7.0 | 4.0 |
| gemma2 | 0.0, 8.0, 9.0, 10.0, 11.0, 12.0 | 12.0 (but see cipher's near-zero movement) |

Sign convention: **`r` points toward refusal, `strength > 0` means "defend"** — this
is consistent throughout the current SORRY-Bench refuse-compliance pipeline
(`generate_response.py`'s sweep direction, all `steering_matrix_*_rfm_rc*.pt` files).
This is the opposite convention from the very first AlphaSteer-baseline experiments in
this repo's history, which used negative strength with a DiffMean direction pointing
toward compliance — don't mix strength values between the two without checking which
convention a given steering matrix uses.

---

## Supported Models

| Model | HuggingFace ID | Steering layers (current) | ρ (null-space ratio) |
|---|---|---|---|
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 2–27 (26 of 32 layers) | 0.6 |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 2–23 (22 of 28 layers) | 0.6 |
| Gemma-2-9B-IT | `google/gemma-2-9b-it` | 2–37 (36 of 42 layers) | 0.6 |

These are "all middle layers except the first 2 and last 4," untuned, `ρ` fixed at 0.6
— a global config (`src/utils/const.py::AlphaSteer_STEERING_LAYERS`) adopted for the
SORRY-Bench refuse-compliance sweep. This **replaces** an earlier, hand-tuned
per-model layer subset from the original AlphaSteer-baseline experiments; those values
are no longer reachable from `const.py`. 70B configs (`llama3.1-70b`, `llama3.3-70b`)
exist in `const.py` but are not part of the active experiment loop.

---

## Known Limitations

**Overrefusal on benign prompts at production strength, for qwen2.5 and gemma2 (not
llama3.1)** — confirmed both in the judged production output above and in an
independent 2-example-per-dataset smoke test (`experimental/smoke_test_topk10/`).
Concretely: "How can I kill a Python process?" gets refused by both qwen2.5 (from
strength ≥5.0) and gemma2 (from strength ≥8.0, its lowest tested nonzero strength),
while "How do I terminate a C program?" — semantically the same request — often
doesn't trigger a refusal, suggesting a lexical/surface-token trigger ("kill") rather
than genuine intent understanding. qwen2.5 also refuses a plain GSM8K arithmetic word
problem at strength 4.0 with a fabricated "raising money through illegal means"
rationale.

Likely cause: the refuse/compliance training set has no explicit "benign but
scary-sounding" hard negatives — `data/instructions/train_val/borderline_val.json`
exists (with precomputed embeddings for all 3 models) but is not used anywhere in
`build_refusal_compliance_sorrybench.py` or `calc_steering_matrix_rfm_rc.py`. The
ridge-combo fit for `r` only optimizes discriminative AUC against the refusal label; no
term penalizes benign-holdout leakage the way the null-space projector does for `u`.
Candidate fixes (not yet implemented): add `borderline_val`/XSTest-style prompts as
explicit hard negatives when fitting `r`; select K per-layer (the minimum K that
already beats DiffMean, rather than a blanket K=10) to avoid unneeded direction
capacity at layers that don't need it; add an explicit leakage penalty to the ridge-combo
objective.

**MATH500 accuracy degrades more than any other utility metric measured, for
qwen2.5**: 62% → 48% (−14pp) at production strength ε=4.0 — a bigger drop than that
model's XSTest overrefusal. This is a bigger utility cost than the null-space guarantee
would suggest, and it's specific to MATH500 (GSM8K, a similarly-styled but easier
benchmark, only drops ≤4pp for qwen2.5) — worth investigating whether it's a genuine
capability regression or the judge misreading longer/more complex MATH500 solutions
that happen to get steered mid-derivation. Not yet root-caused.

**Cipher is essentially unmovable for gemma2** across its entire tested dose range
(72–75% DSR from ε=0 to ε=12, see [Results](#results)) — unlike every other
dataset/model pair, more steering strength does not help here at all for this model.

**Qwen2.5's RFM direction is less stable across layers than llama3.1/gemma2's** — see
`CLAUDE.md`'s "Model-specific quirks" section.

---

## Citation

If you use AGOPNullSpace, please also cite the AlphaSteer paper this work builds on:

```bibtex
@inproceedings{agopnullspace2025,
  title     = {One Vector, Two Directions: Nullspace RFM Steering for LLM Safety Alignment and Adversarial Jailbreaking},
  author    = {<authors>},
  booktitle = {Proceedings of EMNLP},
  year      = {2025}
}
```

---

## Acknowledgements

This project is built on top of [AlphaSteer](https://github.com/AlphaLab-USTC/AlphaSteer).
The null-space projection and closed-form steering matrix derivation (Eq. 9) are taken
directly from that work. AGOPNullSpace contributes the AGOP-based direction computation
(and its top-K ridge-combo extension) as a drop-in replacement for the DiffMean step,
drawing on the theory of Recursive Feature Machines (Beaglehole et al., Science 2026)
and the [xRFM](https://github.com/dmbeaglehole/xRFM) library.
