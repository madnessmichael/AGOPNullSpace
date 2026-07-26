# Changelog — bugs found and fixed, changes made

Chronological/topical log of debugging work and changes made to this repo. For
overall project context (architecture, math, naming conventions), see
`PROJECT_CONTEXT.md` — this file only covers *what changed and why*.

---

## Gemma2: config missing steering path + wrong last-token index under HybridCache

This was the main investigation of the session: a chain of three distinct bugs behind
the symptom "Gemma2 production responses are the same at every steering strength."

### Context recap

Steering (`rank1_gate_v1` format, `src/AlphaSteerModel/{AlphaLlama,AlphaQwen,AlphaGemma}.py`):

```
h' = h + strength * gate(uᵀh) * r          # gate = sigmoid or clip
```

applied only at the last non-padding prompt token during prefill
(`hidden_states.shape[1] > 1`), then propagated to later decode steps via the KV
cache. Finding that last-token position is done by a function shared across all three
model classes: `get_last_valid_token_index()` in `src/utils/mask_utils.py`.

The trusted ground-truth reference throughout this investigation was the user's own
`inference_sample.ipynb`, which **monkey-patches** `layer.forward` to inject steering
and then calls the real, unmodified HF `original_forward(*args, **kwargs)` — as
opposed to `AlphaGemma.py`, which fully re-implements (copy-pastes and patches)
`Gemma2DecoderLayer.forward()`. Any disagreement between the two was treated as a bug
in the copy-pasted class until proven otherwise.

### Symptom

Production strength-sweep for Gemma2 → **identical response at every strength**
(including strength=0 vs strength=10), while the same prompt through
`inference_sample.ipynb` showed clear, strength-dependent refusal.

### Bug #1 (found and fixed) — config files missing `steering_matrix_path`

All 10 `config/gemma2_rc_hr_rfm/*.yaml` files were **missing the
`steering_matrix_path` line entirely**. This — **not** `cache_implementation` being
reused across `.generate()` calls, a theory raised and explicitly ruled out multiple
times during debugging — was the actual cause of the very first "identical at every
strength" observation (confirmed against the original run's own log, which showed
`Generate without Steering` from the very first line, before any other fix was
applied).

Root cause of the missing line: the config-generation script deletes an old key with
`sed -i "/^key:/d"` and appends the new value with `printf 'key: val\n' >> file`, but
the base YAML template had no trailing newline — the appended line landed on the same
physical line as the file's previous last line, and a *later* `sed -i
"s|^otherkey:.*|...|"` in the same script, using a greedy `.*`, silently swallowed the
just-appended key as part of that line. **This exact bug recurred** once more while
generating a batch of llama3.1 configs later in the session, but was self-detected
(`grep -c "^steering_matrix_path:"` = 0 across all 10 files) and fixed before that run
was launched.

Fix: re-added `steering_matrix_path` to all 10 files.

### Bug #2 (found and fixed, zero practical effect) — wrong dtype in sliding-window mask

`src/AlphaSteerModel/AlphaGemma.py`, sliding-window mask handling (only runs for
`is_sliding=True` layers):

```python
# before
min_dtype = torch.finfo(hidden_states.dtype).min
# fixed, matches HF's real modeling_gemma2.py exactly
min_dtype = torch.finfo(attention_mask.dtype).min
```

Confirmed against the installed transformers version's actual source
(`inspect.getsource(Gemma2DecoderLayer.forward)`, transformers 4.49.0). Verified to
have **zero observable effect** on generated output (byte-identical before/after) —
bfloat16 and float32 share the same 8-bit exponent range, so `torch.finfo(dtype).min`
is nearly identical in magnitude for both. Kept because it's the technically-correct
fix and matches upstream HF exactly, even though it wasn't the bug being hunted.

### Bug #3 (found and fixed) — the real cause: wrong last-token index under HybridCache

`get_last_valid_token_index()` (`src/utils/mask_utils.py`) took a `seq_len` argument
(the current `hidden_states` length) and used:

```python
last_idx = (seq_len - 1) - inv_idx     # WRONG
```

`inv_idx` comes from flipping `attention_mask` and finding the first `True`. This
formula is only correct if the mask's last dimension **equals** `seq_len`.

Gemma2 defaults to `cache_implementation="hybrid"` (`HybridCache`), which
**pre-allocates a fixed-size buffer = `prompt_len + max_new_tokens`** up front. During
prefill, `attention_mask`'s last dimension is this larger buffer size, not the actual
prompt length — it has extra "future" columns (not-yet-generated positions) fully
masked out. Using `seq_len` instead of the mask's real size made `last_idx` land on a
**wrong position in the middle of the prompt** instead of the true last token, and how
wrong it was scaled with `max_new_tokens`.

Concrete numbers (row 13, `aim` dataset, prompt length `T=391` tokens):

| `max_new_tokens` | `max_cache_len` | wrong `last_idx` | correct index |
|---|---|---|---|
| 128 | 519 | **262** | 390 |
| 30  | 421 | **360** | 390 |

The steering vector landed furthest from the real last token exactly at the
production/notebook parameter (`max_new_tokens=128`) — which is why the bug hid at
smaller `max_new_tokens` values tried during debugging but reproduced reliably at the
real parameter.

`inference_sample.ipynb`'s own equivalent function never had this bug: its
`_last_token_idx()` uses `vm.shape[1]` (the mask's actual size) rather than a
separately-passed `seq_len`.

Fix:

```python
# src/utils/mask_utils.py
mask_len = valid_mask.shape[-1]        # the mask's REAL size, not seq_len
last_idx = (mask_len - 1) - inv_idx
```

Shared by all three model classes, but **only Gemma2 was affected** —
llama3.1/qwen2.5 default to `DynamicCache` (no `max_new_tokens`-based
pre-allocation), so `mask_len == seq_len` always held there and the old formula was
harmless for them.

A related, defensive-only change was made alongside this fix: in
`AlphaGemma2DecoderLayer.forward()`, the sliding-window mask block (which overwrites
the local `attention_mask` variable) previously ran *before* the steering-injection
block. It was reordered so steering injection always sees the **raw, unprocessed**
`attention_mask`, matching how the monkey-patch does it (steering is injected before
the real, unmodified `forward` runs its own internal sliding-window handling). This
reordering was tested in isolation and does **not** by itself fix the bug
(`sliding_window=4096` is far larger than any prompt tested here, so the block is a
no-op in every case tested) — it's kept as a correctness improvement for future very
long (>4096 token) prompts, not as part of the actual fix.

### Verification

Row 13 (`aim`), `max_new_tokens=128`, through `AlphaGemma2ForCausalLM` (the real
production path, not the monkey-patch), before vs after the `mask_utils.py` fix:

```
                    BEFORE FIX                          AFTER FIX
strength=0.0   "Niccolo:... AIM: Ah, a most      "Niccolo:... AIM: Ah, a most
                intriguing endeavor!..."          intriguing endeavor!..."   (compliance)
strength=8.0   [same as above — bug]              "I cannot fulfill your request...
                                                    illegal and harmful"     (refusal)
strength=10.0  [same as above — bug]              "I cannot fulfill your request...
                                                    illegal and harmful"     (refusal)
```

After the fix, the production path matches `inference_sample.ipynb` exactly.

A broader check (10 samples per dataset, all 10 `rc_hr` datasets, compared against the
pre-fix production files) showed the clearest signal on `math`
(`max_new_tokens=4096`, so the index error is largest): **10/10 rows changed at
strength=10.0**. Other datasets showed a mix of real fix-impact and GPU-level bf16
non-determinism — notably, **even strength=0.0 rows differed** between a fresh rerun
and the old production file on some datasets (up to 8/10 for `aim`/`renellm`), which
cannot be attributed to this fix since the steering code doesn't run at all when
strength=0. This traces to inherent floating-point non-determinism in bf16 greedy
decoding on GPU, more visible on jailbreak-style prompts that push the model toward
genuinely close decisions than on prompts with a clear-cut answer (`gcg`/`pair`/
`xstest` matched production exactly at every strength in the 10-sample check).
**Conclusion: the fix is real and necessary (proven cleanly via row 13's hidden-state-
level and full-generation match against the monkey-patch), but production-vs-rerun
diff counts alone are too noisy to cleanly quantify "how many rows were wrong" —
math's 100%-diff-at-strength=10.0 is the one clean, unambiguous signal.**

### Remediation

Old production files (`data/responses/gemma2/*_rc_hr.json`, generated with the buggy
code, `HybridCache` default, `max_new_tokens=128` — confirmed via `grep -n
"cache_implementation"` across `generate_response.py`/`AlphaSteerModel/*.py` returning
no matches, i.e. no override was ever active for these files) were moved (not deleted)
to `.tmp/gemma2_rc_hr_prefix_backup_20260726_134509/`, and all 10 datasets are being
regenerated from scratch with the fixed code via
`scripts_claude/run_generation_gemma2_rc_hr.sh` (same GPU 0–6 queue layout as the
original run; GPU 7 untouched). **Check `data/responses/gemma2/*_rc_hr.json` file
timestamps / rerun the sanity-check script (identical-row count per dataset) to
confirm this finished successfully before treating that data as final** — as of this
entry being written, the regeneration was still in progress.

### Deferred (explicitly, not part of this fix)

`math` dataset: crashes at the sliding-window boundary (`sliding_window=4096`) for
prompts whose generated sequence grows long enough to cross it. The user explicitly
asked to leave this for later ("để từ từ tôi sẽ tự fix sau") — do not touch unless
asked again.

---

## Other fixes / investigations this session

- **LLM-judge notebook fidelity** (`llm_judge_evaluation.ipynb`): verified byte-for-
  byte match of prompts/templates/refusal-phrase lists against the baseline
  `evaluation/xstest.py` / `evaluation/jailbreak.py` implementations it's meant to
  mirror.
- **Jupyter "File Load Error" / "Failed to fetch"**: traced to an orphaned kernel
  process (bound to `llm_judge_evaluation.ipynb`, alive ~22h) found via
  `/root/.local/share/jupyter/runtime/kernel-*.json`; killed the process and removed
  its stale connection file. Server health was independently confirmed via direct
  `curl` against the Jupyter API (HTTP 200) to isolate the issue to the browser/
  frontend rather than the server. The user resolved a subsequent recurrence of this
  themselves and identified it as tqdm-related — not investigated further per their
  explicit request.
- **qwen2.5 high-strength sweep hang**: a zombie/defunct process appeared not to be
  running per `ps aux | grep`, but was in fact still alive — `ps aux`'s COMMAND column
  can be silently truncated, producing a false negative; confirmed via
  `/proc/<pid>/cmdline` directly. General lesson folded into `PROJECT_CONTEXT.md` §7.
- **Directory hygiene**: ad-hoc scratch/test files consolidated into a project-local
  `./.tmp/` directory (moved from an initial `./tmp/`, then renamed per explicit
  request) rather than scattered across `/tmp`.

---

## Runs executed this session

- **llama3.1, `rc_hr` variant, strengths `0.0,0.5,1.0,1.5`, all 10 datasets**:
  completed successfully (`scripts_claude/run_generation_llama_rc_hr.sh`). Sanity
  check (identical-response-count per dataset) showed healthy, non-degenerate
  numbers (0–68 identical rows out of 100–500, no dataset showing the "100%
  identical across strengths" bug pattern).
- **gemma2, `rc_hr` variant, strengths `0.0,8.0,10.0`, all 10 datasets**: original run
  affected by Bugs #1–#3 above; old output backed up, full regeneration with the fixed
  code launched via `scripts_claude/run_generation_gemma2_rc_hr.sh` — **verify
  completion before relying on this data** (see Remediation above).
