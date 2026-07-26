# AGOPNullSpace — Project Context (for a new Claude Code session)

This file is meant to let a **fresh session with zero prior context** understand this
repository at the depth needed to work on it safely — architecture, the actual math
being implemented (not just what the paper claims), file layout, naming conventions,
and the operational constraints (shared GPUs, non-determinism, etc.) that have bitten
this project before. `README.md` is the paper-facing pitch and is **partly aspirational
/ out of sync with the actual code** (see §8) — treat this file, not the README, as the
map of what's really here.

For a chronological log of bugs found/fixed and why, see `CHANGELOG.md`.

---

## 1. What this project is

**AGOPNullSpace (AGOPN)** is an EMNLP submission on LLM safety steering. It is a direct
extension of **AlphaSteer** (ICLR 2026): given a target LLM, compute a per-layer
**concept direction** `r` that separates "should refuse" from "should comply" activation
space, then add a **null-space-constrained, rank-1 gated steering vector** to the
model's residual stream at inference time so that malicious prompts get pushed toward
refusal while benign prompts are (provably, up to numerical leakage) left untouched.

The paper's core contribution over AlphaSteer: replace AlphaSteer's **DiffMean**
direction (`r = mean(H_comply) − mean(H_refuse)`) with a direction learned by a
**Recursive Feature Machine (RFM)** — the top eigenvector of the **Average Gradient
Outer Product (AGOP)** matrix of a kernel regression fit to distinguish the two classes.
This captures nonlinear/encoding-obfuscated boundaries (e.g. Cipher-encoded prompts)
that a linear mean-difference direction misses. The null-space projection machinery
(which direction gets *applied* to activations, and how utility preservation is
guaranteed) is taken from AlphaSteer essentially unchanged.

A second, actively-evolving axis of the project (see §4) replaces AlphaSteer's own
source of "refused vs complied" labels (an externally-labeled harmful/harmless dataset)
with the model's **own actual refusal behavior** on SORRY-Bench prompts, and replaces
the old unbounded linear gate with a bounded sigmoid/clip gate.

Models in scope: **Llama-3.1-8B-Instruct**, **Qwen2.5-7B-Instruct**, **Gemma-2-9B-IT**
(a few 70B configs exist in `const.py` but are not part of the active experiment loop).

---

## 2. The math

### 2.1 AGOP / RFM concept direction

Given two activation sets `H_pos` (label y=1) and `H_neg` (label y=0) at a fixed layer,
`d`-dimensional each:

```
For t = 1 … T:
    K_M(x,z) = exp(−(x−z)ᵀ M (x−z) / L)        # Laplace kernel, metric M (M₀ = I)
    α        = (K_M + λI)⁻¹ y                     # kernel ridge regression
    G_t      = (1/n) Σᵢ ∇f(xᵢ) ∇f(xᵢ)ᵀ          # Average Gradient Outer Product
    M_{t+1}  = G_t / ‖G_t‖_F                      # metric update (Deep Neural Feature Ansatz)

r = top_eigenvector(M_T)
```

Implementation: `src/rfm_refusal_vector.py` via the `xrfm` library's `RFM` class
(`kernel="l2_high_dim"`). Practical details that matter (see file docstring, "R1"–"R8"
for the full list of fixes vs an earlier, subtly-wrong v2):

- `tuning_metric="auc"` must be passed into the `RFM` constructor itself, not just used
  externally to pick the best grid-search cell — otherwise the *internal* early-stop/
  best-iteration selection inside `RFM.fit()` optimizes a different metric (MSE) than
  the outer loop's model-selection criterion (AUC), and you silently keep the AGOP of a
  model that isn't actually the best one by your own stated criterion.
- `center_grads` (whether the AGOP is a centered or raw/uncentered gradient covariance)
  is swept over `{True, False}`, not hardcoded — AGOP as defined by the Deep Neural
  Feature Ansatz is *uncentered*; for a binary probe the mean gradient direction *is*
  the discriminative signal, so centering can delete exactly the thing being extracted.
- The train/val split must be **stratified AND shuffled** (`sklearn.train_test_split`).
  An earlier version took the first `n` positive/negative rows positionally, which for
  one dataset meant the validation positives were 100% one sub-category (AdvBench) with
  zero jailbreak examples — val AUC then measured nothing about the harder distribution
  the paper claims robustness on.
- One fixed `torch.Generator` permutation is reused across every layer's benign
  subsample — otherwise each layer trains on a different random benign subset, and
  cross-layer comparisons (which layers "work best") aren't apples-to-apples.
- Sign calibration: `r ← r · sign(Pearson(X_train @ r, y_train))`, computed on the
  **train** split only (not train+val) — orients `r` to point toward the malicious/
  refusal class as defined by the label convention in use (see §2.3).
- `top_eigenvectors()` uses `torch.lobpcg` (fast, O(d²k)) with an `eps·I` regularizer
  and falls back to a full `torch.linalg.eigh` if lobpcg fails to converge or returns
  non-finite values — lobpcg is known to fail on ill-conditioned matrices, and the
  original code called it unguarded.

A linear-probe fallback (`train_linear_probe_on_concept`, ridge regression, AGOP of a
linear predictor `f(z)=zᵀβ` is the rank-1 matrix `ββᵀ`) exists for when `xrfm` isn't
installed or as a fast baseline; not part of the main pipeline.

### 2.2 Null-space projection and the closed-form steering solution

This is **AlphaSteer's Eq. 9**, reused as-is. Given malicious activations `H_m`
`(N×d)`, a concept direction `r` `(d,)`, and a null-space projector `P̂ = ÛÛᵀ` built
from the **lowest** eigenvectors of the benign activation covariance (i.e. the
subspace where benign activations have the *least* energy):

```
Δ* = R H_mᵀ P̂ᵀ (P̂ H_m H_mᵀ P̂ᵀ + α P̂ P̂ᵀ)⁺        # R = 1_N rᵀ (rank-1 target)
```

`Δ* H_b ≈ 0` (the steering matrix does ~nothing to benign activations) *by
construction*, independent of which `r` was plugged in — the null-space projector is
what buys the utility-preservation guarantee, the AGOP direction only changes *which
way* malicious activations get pushed.

**This repo's actual implementation reformulates that d×d problem into a k×k one**,
where `k = null-space rank ≪ d` (`src/utils/steering_utils.py`, "v4", read the module
docstring — it documents a real numerical bug found and fixed along the way: an
assertion `P² ≈ P` was failing with rel-error `2.3e-3` on GPU, traced to **TF32**
matmul (10-bit mantissa) being enabled by default on Ampere+ GPUs, not to a data bug —
`disable_tf32()` must be called at the top of every steering-matrix script). Because
the target `R = 1_N rᵀ` is rank-1, the whole system reduces to a rank-1 steering
factorization `M = u rᵀ`, so `Mᵀh = (uᵀh)·r` — i.e. steering is *always* an additive
push in the fixed direction `r`, scaled by a *scalar gate* `uᵀh` that depends on the
current activation. With `Q ∈ R^{d×k}` an orthonormal basis of the null space (not the
full `d×d` projector `P = QQᵀ`) and `Y = H_m Q ∈ R^{N×k}`:

```
XᵀX + λPᵀP = Q (YᵀY + λI_k) Qᵀ
⇒ w = (YᵀY + λI_k)⁻¹ Yᵀ1_N            # k×k Cholesky solve — SPD, no pseudo-inverse
⇒ u = Q w                              # u ∈ range(Q) BY CONSTRUCTION, not by cancellation
```

Why this matters beyond speed (`cholesky(k²)` in milliseconds vs `pinv(d²)` in
seconds, `cond ~ 1e20` in d-space vs `~2.7e3` in k-space): the old d×d path needed
`Δ*H_b ≈ 0` to hold via near-cancellation in floating point; the k-space path makes
`u ∈ range(Q)` an algebraic identity (measured `‖u − QQᵀu‖/‖u‖ ≈ 2.4e-15`), so the
utility-preservation guarantee no longer depends on how much numerical error the GPU's
matmul happens to introduce. `null_space_basis_l()` returns `Q` (not `P`); prefer
`cal_steering_factors_q()` over the legacy `cal_tilde_delta_with_regularization_l()` d×d
path, which is kept only for cross-checking.

**Important vector-orientation subtlety** (flagged in the module docstring as
"chỗ manuscript sai" — a place the manuscript itself has the convention backwards):
`P` is the **left** factor of `Δ̃`, not the right factor. The guarantee holds for
`h' = h + α·Mᵀh` (equivalently `h + α·(uᵀh)·r`, which is what the code actually
computes), **not** for `h' = h + α·Mh` as the manuscript's Eq.(1) literally states.

### 2.3 Inference-time formula and sign convention

At inference, per steered layer:

```
gate  = σ(a · (uᵀh_last − 0.5))     # sigmoid, slope a = gate_slope (default 10.0)
        or clip(uᵀh_last, 0, 1)      # alternative: hard clip instead of sigmoid
h'    = h + strength · gate · r
```

applied **only at the last non-padding token position of the prompt during prefill**
(`hidden_states.shape[1] > 1`), then left for causal attention to propagate through
subsequent decode steps via the KV cache. This is the **`rank1_gate_v1`** format (see
§5). An older **dense legacy format** also exists (`steering_matrix` as a full `[d,d]`
matrix per layer, applied as `h' = h + strength·(h_lastᵀ M)`) for the original
DIM/plain-RFM/HH-trained matrices — both formats are handled by the same model classes
(`should_apply_dense_steering` vs `should_apply_gate_steering` branches).

Sign convention actually used throughout the current codebase (SORRY-Bench
refuse-compliance pipeline, §4): **y=1 = refusal, y=0 = compliance**, so `r` points
toward refusal and **`strength > 0` means "defend"** — this is consistent across
`README.md`'s stated convention, `steering_matrix_*_dim.pt`/`*_rfm.pt`, and
`generate_response.py`'s sweep direction. (Note: `rfm_refusal_vector.py`'s own
docstring, written for a different/older calling context, describes the opposite
raw convention `y=1=malicious` — the sign is calibrated *after* the fact via
`sign(Pearson(...))` against whatever `y` was passed in by the caller, so the caller's
label convention is what actually determines which way `r` ends up pointing. Always
check the calling script's `y=1=...` comment, not this file's docstring, to know which
way a given saved vector points.)

---

## 3. Architecture: how a forward pass actually gets steered

`src/AlphaSteerModel/{AlphaLlama,AlphaQwen,AlphaGemma}.py` each define an
`Alpha<Model>ForCausalLM` that **subclasses and fully re-implements** (copy-pasted and
patched, not wrapped) the corresponding HuggingFace `<Model>ForCausalLM` /
`<Model>Model` / `<Model>DecoderLayer` classes, injecting the steering-vector
computation directly into `DecoderLayer.forward()` before the residual/attention/MLP
computation. This is **structurally different** from the alternative approach used in
the user's own `inference_sample.ipynb`, which instead **monkey-patches**
`layer.forward` to inject steering into `hidden_states` and then calls the *original,
unmodified* `original_forward(*args, **kwargs)` — i.e. it never re-implements any HF
internals, so it can never drift out of sync with a HF version bump or contain a
transcription bug in the copied logic.

**`inference_sample.ipynb` is treated as the trusted ground-truth reference** whenever
the two approaches disagree (this is how the bug in `CHANGELOG.md` §3 was found and
proven correct) — if `AlphaGemma.py`/`AlphaLlama.py`/`AlphaQwen.py` and the monkey-patch
disagree on a generation, assume the copy-pasted class has the bug until proven
otherwise, not the notebook.

All three model classes share `get_last_valid_token_index()` from
`src/utils/mask_utils.py` to find the last non-padding prompt token position from
either a 2D or 4D attention mask (identical algebraic logic to the notebook's own
`_last_token_idx()` — verified to agree, see `CHANGELOG.md`). This function had a real
bug (`CHANGELOG.md` §3) affecting Gemma2 specifically because Gemma2 defaults to
`HybridCache`, whose fixed-size buffer allocation makes the mask's last dimension
diverge from the current hidden-states length during prefill — Llama/Qwen default to
`DynamicCache` and were never affected.

`set_steering_parameters()` on every layer/model class is **state-preserving**: a
strength-only call (`u`/`r` args omitted) must not wipe previously-set `u`/`r`/
`steering_matrix` to `None` — the per-strength sweep loop in `generate_response.py`
relies on this to change only `strength` between iterations without reloading the
model or resetting the concept vectors. (An earlier bug in a since-removed
`NaiveSteerModel` class did wipe state this way — see `calc_steering_matrix_rfm_rc_no_
nullspace.py`'s docstring for why the no-nullspace ablation was moved onto the same
`AlphaSteerModel` classes instead of keeping a separate naive model family.)

`src/generate_response.py` is the single production entry point for generating model
responses. It reads one YAML config (`--config_path`), loads the right model class
(dense-matrix / rank1-gate / unsteered, decided by which of `steering_matrix_path` vs
none is present in the config), then loops over a comma-joined `strength` list,
re-calling `set_steering_parameters(strength=...)` and `model.generate(...)` once per
strength value, appending each strength's response under a `response_strength:{s}` key
in the (resumable, incrementally-saved) output JSON. `torch._dynamo.config.disable =
True` is set at import time — **do not remove this** — Gemma2's HybridCache gets
auto-compiled via dynamo/CUDAGraphs on some transformers versions, and a captured
CUDAGraph freezes whatever steering strength was active at capture time and just
replays it on every subsequent call regardless of the new strength passed in,
*silently* (see the comment block at the top of the file for the exact symptom: every
`response_strength` column for a row came out byte-identical).

---

## 4. The SORRY-Bench refuse-compliance pipeline (the active experiment axis)

This is a from-scratch replacement of AlphaSteer's original concept-vector training
data (`justinphan3110/harmful_harmless_instructions`, an externally-labeled dataset) —
motivation: it doesn't reflect what the *target model itself* actually refuses vs
complies with, which is what a "refusal direction" should arguably be trained on. It
also introduces a bounded nonlinear gate in place of the old unbounded linear one.

**Stage 1 — build the refuse-compliance dataset**
(`src/build_refusal_compliance_sorrybench.py`): for each model, load SORRY-Bench
(`sorry-bench/sorry-bench-202503`) prompts, generate a response (greedy,
`max_new_tokens=128`), judge each response as refusal (y=1) or compliance (y=0), save
text + label. Two dataset sizes:
  - default: `prompt_style=='base'` only → 440 rows (44 categories × 10).
  - `--full`: all 21 linguistic-mutation `prompt_style` variants (base + 20, e.g.
    slang, role_play, ascii, morse, caesar, multiple translations) → 9,240 rows.
Two judge strictness levels (`evaluation/jailbreak.py` vs
`evaluation/hard_refusal_judge.py`, see §6) — plain vs `--hard_refusal` — because the
harm-content jailbreak judge's "reject" can fire without the response ever using actual
refusal language (only 16% of `refusal_compliance_full` rows were decided by strict
phrase match; the rest came from an LLM judge whose criterion is "not harmful", not
"looks like a refusal"), which is too loose a criterion for training a *refusal*
concept vector specifically. Output: `data/embeddings/<model>/refusal_compliance[_full]
[_hard_refusal]/sorrybench_rc_dataset.json` (+ a `sorrybench_rc_report.json` with
per-category/per-style refusal-rate breakdowns) — text + labels only; checkpointed
every N rows so a crash resumes instead of restarting. Activation extraction is a
**separate** phase (`src/extract_refusal_compliance_embeddings.py`, run after this
finishes) — an earlier version extracted `output_hidden_states=True` activations in
the same generation pass, which for the long ascii/morse `--full` variants (up to
~8,300 tokens) OOM'd hard enough to lose all progress since nothing was checkpointed;
decoupling means a crash in one phase never discards the other's completed work.
Gemma2 specifically runs with `attn_implementation="eager"` (dodges a known SDPA fused-
kernel alignment bug on certain batch/seq-length combos) at `batch_size=1` and a
2048-token cap (eager attention materializes the full `O(seq_len²)` matrix in fp32,
which OOM'd a 9B model on 24GB even at 4096 tokens/batch=1).

**Stage 2 — new concept-vector calc scripts**, both consuming Stage 1's output:
  - `src/calc_steering_matrix_rfm_rc.py` (nullspace + gate): trains `r` on the
    refuse-compliance set (`H_malicious=refusal acts, H_benign=compliance acts`, i.e.
    `y=1=refusal`), keeps AlphaSteer's own D_m/D_b (14k benign + 2k malicious, via
    `load_alphasteer_embeddings`) for the **gate `u` and null-space `Q`** — only the
    source of `r` changed, not the gate-fitting data. Saves the new `rank1_gate_v1`
    dict format (§5) plus a companion `<save_path>_r.pt` plain `[L,d]` tensor of `r`
    alone, so the no-nullspace ablation below can reuse the *identical* `r` without
    an independent (and therefore slightly different, due to RFM's grid-search
    stochasticity) refit.
  - `src/calc_steering_matrix_rfm_rc_no_nullspace.py` (ablation): pure additive
    `h' = h + strength·r` at **every** token position, no gate, no null-space, no
    regression — the literal `Ã_{l,i}(X) = A_{l,i}(X) + ε·v_l` formula from the paper.
    Saves the same `rank1_gate_v1` format with `u=None` per layer so the *same* model
    classes' `should_apply_nogate_steering` branch handles it — this replaced an
    earlier separate `NaiveSteerModel` family that had a real state-wiping bug (see
    §3) and an uninitialized-tensor crash; one shared, correct code path was judged
    better than patching a second one.

**Stage 3 — model class changes**: added `u`/`r`/`gate_type`/`gate_slope` fields
alongside the legacy `steering_matrix` in all three `Alpha<Model>DecoderLayer`
classes; `set_steering_parameters()` accepts either the legacy dense tensor or the new
per-layer factors; `generate_response.py` branches on
`isinstance(loaded, dict) and loaded.get("format")=="rank1_gate_v1"` when loading a
`--steering_matrix_path` to decide which code path to route through.

**Stage 4 — layer/ratio config** (`src/utils/const.py`): `AlphaSteer_STEERING_LAYERS`
for llama3.1/qwen2.5/gemma2 changed to **all middle layers except the first 2 and last
4** (untuned, `ρ=0.6` fixed) for this sweep — `llama3.1: range(2,28)`,
`qwen2.5: range(2,24)`, `gemma2: range(2,38)` (out of 32/28/42 total layers). **This is
a global mutation** — any old DIM/plain-RFM/HH config re-run after this change uses
these new layers/ratio, not the hand-tuned per-model subsets the README's "Supported
Models" table describes (e.g. Gemma2's README-documented `6, 8, 10–16, 18, 22` no
longer reflects what `const.py` actually contains). 70B entries are untouched.

**Stage 5 — config generation**: per-model, per-nullspace-variant config directories
under `config/` (see §5 for the naming scheme) generated from
`scripts_claude/generate_configs_rc*.sh`, restricted to 7 attack + 3 utility datasets
(no `alpaca_eval`).

**Stage 6 — execution**: steering-matrix computation jobs (up to 6 in parallel across
GPUs 0–6), then the generation sweep proper via `scripts_claude/run_generation_rc*.sh`
orchestration scripts (one GPU queue per script invocation, sequential within a queue,
parallel across queues) — see `CHANGELOG.md` for the specific runs executed and their
outcomes.

---

## 5. Naming conventions (config dirs, steering-matrix files)

Steering matrix filenames (`data/steering_matrix/steering_matrix_<model>_rfm_rc*.pt`):

| suffix | meaning |
|---|---|
| `_rc.pt` | base 440-row refuse-compliance set (`prompt_style=='base'` only) |
| `_rc_full.pt` | full 9,240-row set (all 21 `prompt_style` linguistic variants) |
| `_rc_full_hard_refusal.pt` | full set + strict hard-refusal judge (not the harm-content jailbreak judge) |
| `..._no_nullspace.pt` | the Stage-2 ablation script's output (pure additive, `u=None`) |
| `..._r.pt` | companion plain `[L,d]` tensor of `r` alone, for the no-nullspace script's `--refusal_vectors_path` |
| `..._meta.json` | per-layer diagnostics: benign leakage, gate selectivity, RFM probe metadata |

Config directories (`config/<model>_<variant>_rfm/`):

| suffix | meaning |
|---|---|
| `rc_ns` | refuse-compliance, **n**ull-**s**pace (gated), base 440-row set |
| `rc_nons` | refuse-compliance, **no** **n**ull-**s**pace (pure additive ablation) |
| `rc_ns_full` / `rc_nons_full` | same, but trained on the full 9,240-row set (wider strength range, up to 11.0) |
| `rc_hr` | refuse-compliance, hard-refusal judge, uses the `_full_hard_refusal.pt` matrix — this is the variant the Gemma2 bug in `CHANGELOG.md` was found in |
| `rc_hr2` / `rc_hr3` (qwen2.5 only) | extra high-strength sweep extensions on top of `rc_hr` (e.g. 7.0–11.5) |
| (no suffix, e.g. `gemma2_rfm`, `gemma2_dim_rfm`, `gemma2_agopn_rfm`) | older DIM / plain-RFM / HH-trained runs, predating the refuse-compliance pipeline |

Response filenames: `data/responses/<model>/<dataset>_<model>_rfm_results_<model>_agopn_<model>_<variant>.json`,
one `response_strength:{s}` key per swept strength value.

---

## 6. Evaluation

- `evaluation/jailbreak.py`: `judge_refusal_strmatch()` — fast phrase-match against a
  fixed `REFUSAL_PHRASES` list, checked first; `evaluate_jailbreak()` — LLM-judge
  fallback (gpt-4o-mini) returning `"reject"`/`"jailbreak"` for anything the phrase
  match doesn't catch. This is the harm-content judge used for the original attack
  datasets (aim, autodan, cipher, gcg, jailbroken, pair, renellm).
- `evaluation/hard_refusal_judge.py`: stricter judge for the refuse-compliance
  *training* data specifically — same phrase list, but the LLM fallback
  (`evaluate_hard_refusal()`) asks specifically "is this an explicit, unhedged hard
  refusal" rather than "is this harmful", to avoid training a refusal-concept vector
  on responses that merely weren't harmful without actually reading as a refusal.
  Prefers `OPENROUTER_API_KEY` over `OPENAI_API_KEY` — a prior run blew through
  OpenAI's daily gpt-4o-mini request cap partway through an 8-way-sharded, 9,236-row
  ×3-model judging pass and silently defaulted thousands of rows to
  `"not_hard_refusal"`; OpenRouter is a separate quota.
- `evaluation/xstest.py`: utility-preservation judge (full-compliance rate on
  safe-but-scary-sounding prompts).
- `evaluation/summarize_results.py` + `evaluation/TheLastJudgment.sh`: post-process all
  per-dataset eval outputs into the paper's Table 1/Table 2 format for one strength
  value at a time.
- Defense Success Rate (DSR) = fraction of attack-dataset responses judged `"reject"`.

---

## 7. Operational constraints (learned the hard way this project)

- **8 GPUs total, RTX 3090 24GB each. GPU 7 is reserved for the user's own interactive
  work** (`inference_sample.ipynb`) — never launch a job on GPU 7. GPUs 0–6 are fair
  game, **one process per GPU at a time** (batch_size=1 Gemma2 jobs at
  `max_new_tokens` up to 4096 can each use most of a card).
- **Never overwrite an existing output file** — rename/move to `.tmp/` first. This
  matters doubly here because `generate_response.py` itself treats an existing
  `output_file` as resumable state (loads and appends to it rather than starting
  fresh) — if you want a truly clean regeneration, the old file must be moved out of
  the way first, not just because of the "don't destroy work" policy but because the
  script's own resume logic will otherwise silently reuse stale rows.
- **bf16 generation on GPU is not run-to-run deterministic**, even with greedy decoding
  (`do_sample=False`) and a fixed seed — confirmed empirically (`CHANGELOG.md` §6):
  rerunning the *identical* code/config/GPU can still flip a handful of rows,
  especially on prompts that push the model toward a genuinely close decision (e.g.
  jailbreak-attack prompts designed to sit near the refusal/compliance boundary).
  `README.md` itself notes this for the baseline table ("non-determinism in GCG suffix
  generation across runs"). **Don't conclude "these two runs differ ⇒ the code
  changed something" without first ruling out this baseline noise** — compare strength
  0.0 (unsteered) rows as a noise floor if the two runs used different code.
- **Zombie GPU memory**: `nvidia-smi` can show a PID holding many GB of VRAM when that
  PID no longer exists in `/proc` (a crashed process that never tore down its CUDA
  context cleanly). This isn't fixable by killing a PID that's already gone; just pick
  a different free GPU rather than trying to reclaim it.
- **`ps aux | grep <pattern>`** can false-negative because the COMMAND column gets
  silently truncated — cross-check via `/proc/<pid>/cmdline` before concluding a
  process isn't running.
- Config files generated via `sed`-based scripts are a recurring source of bugs:
  base YAML files without a trailing newline + `sed -i "/^key:/d"` +
  `printf 'key: val\n' >> file` can concatenate the new key onto the same physical
  line as the previous last line, and a later `sed -i "s|^otherkey:.*|...|"` with a
  greedy `.*` then silently swallows it. **Always `grep -c "^steering_matrix_path:"`
  (or whichever key matters) across every generated config file before launching a
  run** — this exact bug caused the Gemma2 incident in `CHANGELOG.md` and recurred
  once more for a llama3.1 batch before being caught proactively.

---

## 8. Where `README.md` is stale

The README's "Repository Structure" section describes files that **do not exist** in
this repo (`alphafm_run.py`, `src/agop_core.py`, `src/null_space.py`,
`src/activation_collector.py`, `configs/` at top level, `eval/calc_dsr.ipynb`,
`scripts/gen_tables.py`) — that structure appears to be a planned/aspirational
refactor that was never carried out; the actual implementation is the
`src/rfm_refusal_vector.py` + `src/utils/steering_utils.py` + `src/calc_steering_
matrix_rfm*.py` + `src/AlphaSteerModel/` set of files described in §2–§3 above. The
"Supported Models / Steering layers" table is also stale post-Stage-4 (§4) —
`const.py` is the source of truth for current layer/ratio config, not the README.
The DSR tables in the README reflect the **original DIM/HH-based pipeline**, not the
newer SORRY-Bench refuse-compliance runs (§4), which as of this writing have not yet
been fully evaluated/tabulated.

Treat the README as historical/paper-narrative context, and this file + the actual
source as the operational truth.
