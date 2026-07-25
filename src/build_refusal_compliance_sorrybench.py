"""
build_refusal_compliance_sorrybench.py
=======================================
Stage 1a of the SORRY-Bench refuse-compliance pipeline: GENERATE + JUDGE ONLY.

v2: this used to also extract final-token activations in the same pass
(output_hidden_states=True on every batch). That materializes hidden_states
for every layer x every token position before we ever slice out the last
one -- for --full's ascii/morse variants (up to ~8,300 tokens for qwen/gemma
tokenizers vs ~400 for base), that spiked memory hard enough to OOM and, since
nothing was checkpointed, lose every row judged so far. Activation extraction
is now a fully separate phase (extract_refusal_compliance_embeddings.py) that
runs against the finished text dataset this script produces -- decoupling the
two means each phase's memory profile only has to account for its own job,
and a crash in one never throws away the other's completed work.

Two dataset modes:
  --full not set (default): prompt_style=='base' only -- 440 rows
      (44 categories x 10), output -> refusal_compliance/
  --full: all 21 prompt_style variants (base + 20 linguistic mutations) --
      9,240 rows total (44 categories x 10 x 21), output -> refusal_compliance_full/

Length-tiered batching: SORRY-Bench prompt length is extremely skewed (80.7%
of --full rows are <250 tokens, but ascii/morse run to 3,000-8,300). Rows are
grouped into fixed tiers and each tier gets its own batch size, so short rows
run in big fast batches and long outliers run alone instead of dragging a
whole batch's memory footprint up to their size.

Label convention: y=1=refusal, y=0=compliance, so the RFM/AGOP concept vector
trained on this data (Stage 2, via the separate extract script + calc_steering
_matrix_rfm_rc*.py) points toward refusal -- keeps ``strength>0 == defend``
consistent with the rest of the repo (README, *_dim.pt, *_rfm.pt).

Checkpointing: progress is saved every --checkpoint_every rows to
<output_dir>/_checkpoint.json (text only now, so this is cheap) -- rerunning
the same command after a crash resumes instead of starting over.

Usage:
    python src/build_refusal_compliance_sorrybench.py \
        --model_name llama3.1 --device cuda:0

    python src/build_refusal_compliance_sorrybench.py \
        --model_name llama3.1 --device cuda:0 --full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import torch
import torch._dynamo
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# Gemma2's default generation_config selects a "hybrid" cache, which recent
# transformers versions auto-compile via torch.dynamo/cudagraphs -- that
# compiled path OOM'd on a 24GB card even at small batch size. Force eager
# execution for all models here (this script never needs generation speed).
torch._dynamo.config.disable = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation"))

from utils.const import MODELS_DICT                    # noqa: E402
from jailbreak import judge_refusal_strmatch, evaluate_jailbreak  # noqa: E402
from hard_refusal_judge import judge_hard_refusal_strmatch, evaluate_hard_refusal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAME = "sorry-bench/sorry-bench-202503"

# base + 20 linguistic mutations -- one jsonl file per variant, 440 rows each
# (44 categories x 10), 9,240 rows total across all 21.
DATASET_FILES_FULL = [
    "question.jsonl",  # base
    "question_question.jsonl",
    "question_slang.jsonl",
    "question_uncommon_dialects.jsonl",
    "question_technical_terms.jsonl",
    "question_role_play.jsonl",
    "question_misspellings.jsonl",
    "question_logical_appeal.jsonl",
    "question_authority_endorsement.jsonl",
    "question_misrepresentation.jsonl",
    "question_evidence-based_persuasion.jsonl",
    "question_expert_endorsement.jsonl",
    "question_ascii.jsonl",
    "question_caesar.jsonl",
    "question_morse.jsonl",
    "question_atbash.jsonl",
    "question_translate-ml.jsonl",
    "question_translate-ta.jsonl",
    "question_translate-mr.jsonl",
    "question_translate-zh-cn.jsonl",
    "question_translate-fr.jsonl",
]

# (min_tokens, max_tokens, batch_size) -- measured across all 3 tokenizers:
# 80.7% of --full rows are <250 tok, ascii/morse run to ~8,300 (qwen/gemma).
# Generation-only (no hidden_states retained) so these batch sizes are far
# less conservative than the old combined script needed.
DEFAULT_TIERS = [
    (0, 250, 32),
    (250, 1000, 8),
    (1000, 3000, 2),
    (3000, 10**9, 1),
]


def _load_jsonl(fname: str, token: str) -> list[dict]:
    path = hf_hub_download(repo_id=DATASET_NAME, filename=fname,
                            repo_type="dataset", token=token)
    with open(path) as f:
        return [json.loads(line) for line in f]


def _drop_null_turns(rows: list[dict], source: str) -> list[dict]:
    """
    A handful of rows in the upstream dataset have turns[0]=None (data gap,
    not our bug -- e.g. question_id 158/161/167/405 in question_question.jsonl).
    Llama's chat template silently coerces None -> the literal string "None"
    (bogus prompt, not a crash); Qwen's template raises a Jinja2 TypeError on
    it. Either way the row carries no real content, so drop it for every
    model rather than let it silently pollute the dataset for some and crash
    for others.
    """
    good = [r for r in rows if isinstance(r["turns"][0], str) and r["turns"][0].strip()]
    n_dropped = len(rows) - len(good)
    if n_dropped:
        dropped_ids = [(r["question_id"], r["prompt_style"]) for r in rows
                       if not (isinstance(r["turns"][0], str) and r["turns"][0].strip())]
        logger.warning("  %s: dropped %d row(s) with null/empty turns[0]: %s",
                       source, n_dropped, dropped_ids)
    return good


def load_sorrybench_base(token: str) -> list[dict]:
    rows = _load_jsonl("question.jsonl", token)
    rows = [r for r in rows if r["prompt_style"] == "base"]
    if len(rows) != 440:
        logger.warning("Expected 440 base prompts, got %d", len(rows))
    return _drop_null_turns(rows, "base")


def load_sorrybench_full(token: str) -> list[dict]:
    """All 21 prompt_style variants concatenated -- 9,240 rows (minus any
    dropped null-content rows, see _drop_null_turns)."""
    rows = []
    for fname in DATASET_FILES_FULL:
        file_rows = _load_jsonl(fname, token)
        logger.info("  %s: %d rows (prompt_style=%s)", fname, len(file_rows),
                    file_rows[0]["prompt_style"] if file_rows else "?")
        rows.extend(file_rows)
    if len(rows) != 9240:
        logger.warning("Expected 9240 rows across 21 variants, got %d", len(rows))
    return _drop_null_turns(rows, "full")


def make_tiered_batches(lengths: list[int], tiers: list[tuple[int, int, int]]) -> list[list[int]]:
    """
    Assign each index to the first tier whose [lo, hi) contains its length,
    then chunk each tier's indices (sorted ascending by length, so a batch's
    padding waste stays low) into that tier's fixed batch size.
    """
    tier_buckets: list[list[int]] = [[] for _ in tiers]
    for idx, length in enumerate(lengths):
        for t, (lo, hi, _bs) in enumerate(tiers):
            if lo <= length < hi:
                tier_buckets[t].append(idx)
                break
        else:
            tier_buckets[-1].append(idx)  # longer than every tier -> last (smallest-batch) tier

    batches = []
    for (lo, hi, bs), idxs in zip(tiers, tier_buckets):
        idxs = sorted(idxs, key=lambda i: lengths[i])
        for i in range(0, len(idxs), bs):
            batches.append(idxs[i:i + bs])
        if idxs:
            logger.info("  tier [%d,%d) bs=%d: %d rows -> %d batches",
                        lo, hi, bs, len(idxs), (len(idxs) + bs - 1) // bs)
    return batches


def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, "_checkpoint.json")


def _load_checkpoint(output_dir: str):
    path = _checkpoint_path(output_dir)
    if os.path.exists(path):
        with open(path) as f:
            ckpt = json.load(f)
        logger.info("Resuming from checkpoint: %d rows already done -> %s",
                    len(ckpt["records"]), path)
        return ckpt["records"], ckpt["counts"]
    return [], {"n_strmatch": 0, "n_llm": 0, "n_judge_error": 0}


def _save_checkpoint(output_dir: str, records: list[dict], counts: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tmp = _checkpoint_path(output_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"records": records, "counts": counts}, f, ensure_ascii=False)
    os.replace(tmp, _checkpoint_path(output_dir))  # atomic -- no half-written checkpoint on crash


@torch.no_grad()
def run_model(model_name: str, args: argparse.Namespace, token: str, output_dir: str):
    model_class, _config_class, model_id = MODELS_DICT[model_name]
    logger.info("Loading %s (%s) on %s...", model_name, model_id, args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Gemma2's sliding-window attention occasionally hits a known PyTorch SDPA
    # kernel bug ("p.attn_bias_ptr is not correctly aligned") for certain
    # batch/seq-length combinations under the fused memory-efficient backend.
    # Eager attention is slower but sidesteps the fused kernel entirely.
    extra_kwargs = {"attn_implementation": "eager"} if model_name == "gemma2" else {}
    model = model_class.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=args.device, token=token,
        **extra_kwargs,
    )
    model.eval()

    rows = load_sorrybench_full(token) if args.full else load_sorrybench_base(token)
    if args.num_shards > 1:
        rows = rows[args.shard_idx::args.num_shards]  # interleaved -> balanced category/style mix per shard
        logger.info("Shard %d/%d: %d rows assigned", args.shard_idx, args.num_shards, len(rows))

    records, counts = _load_checkpoint(output_dir)
    n_strmatch, n_llm, n_judge_error = counts["n_strmatch"], counts["n_llm"], counts["n_judge_error"]
    done_keys = {(r["question_id"], r.get("prompt_style", "base")) for r in records}
    if done_keys:
        rows = [r for r in rows if (r["question_id"], r.get("prompt_style", "base")) not in done_keys]
        logger.info("%d rows remaining after resume", len(rows))

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["turns"][0]}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in rows
    ]
    lengths = [len(tokenizer(p)["input_ids"]) for p in formatted]
    tiers = args.tiers
    if model_name == "gemma2":
        # Eager attention (below) is O(seq_len^2) in fp32 -- even capped at
        # 4096 tokens, a batch of 2 such sequences still OOM'd a 9B model on
        # 24GB. Force batch=1 across every tier (same tier boundaries as
        # every other shard, just batch_size=1) so all 8 shards use the same
        # tier config, only ever one sequence per batch.
        tiers = [(lo, hi, 1) for lo, hi, _bs in tiers]
        logger.info("gemma2: forcing batch=1 for all tiers (eager attention memory) -> %s", tiers)
    batches = make_tiered_batches(lengths, tiers)
    logger.info("%d rows -> %d tiered batches", len(rows), len(batches))

    rows_done_since_ckpt = 0

    # gemma2 runs eager attention (see model load above) to dodge an SDPA
    # alignment bug -- eager materializes the full seq_len^2 attention matrix
    # in fp32, so the ~8,300-token ascii/morse outliers OOM even at batch=1.
    # Cap length there; llama3.1/qwen2.5 keep SDPA (O(n) memory) and don't
    # need this.
    # 4096 still OOM'd at batch=1 (process already at 22.87/23.56 GiB just
    # from one eager-attention forward pass on a 9B model) -- 2048 halves
    # the O(n^2) attention-matrix memory again.
    max_len = 2048 if model_name == "gemma2" else 10000

    for batch_i, idxs in enumerate(batches):
        batch_rows = [rows[i] for i in idxs]
        batch_prompts = [formatted[i] for i in idxs]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len).to(model.device)
        T = enc["input_ids"].shape[1]

        gen_out = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        responses = tokenizer.batch_decode(gen_out[:, T:], skip_special_tokens=True)
        del gen_out, enc
        torch.cuda.empty_cache()

        for row, resp in zip(batch_rows, responses):
            if args.hard_refusal:
                # Stricter criterion: y=1 only for an explicit, unhedged hard
                # refusal (phrase match or LLM-confirmed semantic equivalent),
                # not merely "the harm-content judge said reject" -- see
                # evaluation/hard_refusal_judge.py docstring for why.
                if not resp or resp.strip() in ("", "\n"):
                    verdict = "hard_refusal"
                elif judge_hard_refusal_strmatch(resp):
                    verdict = "hard_refusal"
                    n_strmatch += 1
                else:
                    try:
                        verdict = evaluate_hard_refusal(resp, model=args.judge_model)
                        n_llm += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning("judge failed for qid=%s: %s -> defaulting to not_hard_refusal",
                                       row["question_id"], e)
                        # Opposite default from the non-strict path below: an
                        # unconfirmed hard refusal must NOT count as one, so a
                        # judge failure falls to the not-hard-refusal side.
                        verdict = "not_hard_refusal"
                        n_judge_error += 1
                label = 1 if verdict == "hard_refusal" else 0  # y=1=hard refusal, y=0=everything else
            else:
                if not resp or resp.strip() in ("", "\n"):
                    verdict = "reject"
                elif judge_refusal_strmatch(resp):
                    verdict = "reject"
                    n_strmatch += 1
                else:
                    try:
                        verdict = evaluate_jailbreak(resp, model=args.judge_model)
                        n_llm += 1
                    except Exception as e:  # noqa: BLE001
                        logger.warning("judge failed for qid=%s: %s -> defaulting to reject",
                                       row["question_id"], e)
                        verdict = "reject"
                        n_judge_error += 1

                label = 1 if verdict == "reject" else 0  # y=1=refusal, y=0=compliance
            records.append({
                "question_id": row["question_id"], "category": row["category"],
                "prompt_style": row.get("prompt_style", "base"),
                "prompt": row["turns"][0], "response": resp, "verdict": verdict,
                "label": label,
            })

        rows_done_since_ckpt += len(idxs)
        logger.info("[%s] %d judged (strmatch=%d, llm=%d, errors=%d) [batch %d/%d, bs=%d, len<=%d]",
                    model_name, len(records), n_strmatch, n_llm, n_judge_error,
                    batch_i + 1, len(batches), len(idxs), T)

        if rows_done_since_ckpt >= args.checkpoint_every:
            _save_checkpoint(output_dir, records,
                             {"n_strmatch": n_strmatch, "n_llm": n_llm, "n_judge_error": n_judge_error})
            rows_done_since_ckpt = 0
            logger.info("  checkpoint saved (%d rows total)", len(records))

    _save_checkpoint(output_dir, records,
                     {"n_strmatch": n_strmatch, "n_llm": n_llm, "n_judge_error": n_judge_error})
    return records, {"strmatch": n_strmatch, "llm_fallback": n_llm, "judge_errors": n_judge_error}


def save_and_report(model_name: str, records: list[dict], judge_breakdown: dict, out_dir: str) -> dict:
    refusal = [r for r in records if r["label"] == 1]
    compliance = [r for r in records if r["label"] == 0]

    os.makedirs(out_dir, exist_ok=True)

    # Text dataset only -- activations are extracted separately (Stage 1b,
    # extract_refusal_compliance_embeddings.py) against this file.
    dataset_path = os.path.join(out_dir, "sorrybench_rc_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    cat_stats = defaultdict(lambda: {"n": 0, "n_refusal": 0})
    for r in records:
        cat_stats[r["category"]]["n"] += 1
        cat_stats[r["category"]]["n_refusal"] += r["label"]
    category_breakdown = {
        cat: {"n": v["n"], "n_refusal": v["n_refusal"], "refusal_rate": v["n_refusal"] / v["n"]}
        for cat, v in sorted(cat_stats.items(), key=lambda kv: int(kv[0]))
    }

    style_stats = defaultdict(lambda: {"n": 0, "n_refusal": 0})
    for r in records:
        style = r.get("prompt_style", "base")
        style_stats[style]["n"] += 1
        style_stats[style]["n_refusal"] += r["label"]
    prompt_style_breakdown = {
        style: {"n": v["n"], "n_refusal": v["n_refusal"], "refusal_rate": v["n_refusal"] / v["n"]}
        for style, v in sorted(style_stats.items())
    }

    report = {
        "model_name": model_name,
        "dataset": DATASET_NAME,
        "n_total": len(records),
        "n_refusal": len(refusal),
        "n_compliance": len(compliance),
        "refusal_rate": len(refusal) / len(records) if records else 0.0,
        "prompt_style_breakdown": prompt_style_breakdown,
        "judge_breakdown": judge_breakdown,
        "category_breakdown": category_breakdown,
        "samples": [
            {"question_id": r["question_id"], "category": r["category"], "prompt": r["prompt"],
             "response": r["response"][:300], "verdict": r["verdict"]}
            for r in records[:6]
        ],
    }
    report_path = os.path.join(out_dir, "sorrybench_rc_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved refusal=%d compliance=%d -> %s", len(refusal), len(compliance), dataset_path)
    logger.info("Report -> %s", report_path)

    ckpt = _checkpoint_path(out_dir)
    if os.path.exists(ckpt):
        os.remove(ckpt)
        logger.info("Removed checkpoint (run completed successfully)")
    return report


def _parse_tiers(spec: str) -> list[tuple[int, int, int]]:
    """'0:250:32,250:1000:8,1000:3000:2,3000:-1:1' -> DEFAULT_TIERS-shaped list."""
    tiers = []
    for part in spec.split(","):
        lo, hi, bs = part.split(":")
        tiers.append((int(lo), 10**9 if hi == "-1" else int(hi), int(bs)))
    return tiers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, choices=["llama3.1", "qwen2.5", "gemma2"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tiers", type=_parse_tiers,
                   default=DEFAULT_TIERS,
                   help="Comma-separated lo:hi:batch_size tiers by token length, e.g. "
                        "'0:250:32,250:1000:8,1000:3000:2,3000:-1:1' (default). Tune the "
                        "batch sizes down per-model if you still see OOM (gemma2 especially).")
    p.add_argument("--checkpoint_every", type=int, default=500,
                   help="Save partial progress every N rows so a crash doesn't lose "
                        "everything done so far; rerunning the same command resumes.")
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--full", action="store_true",
                   help="Use all 21 prompt_style variants (9,240 rows) instead of "
                        "just prompt_style=='base' (440 rows). Output goes to "
                        "refusal_compliance_full/ instead of refusal_compliance/.")
    p.add_argument("--hard_refusal", action="store_true",
                   help="Use the stricter hard-refusal judge (evaluation/hard_refusal_judge.py) "
                        "instead of the harm-content jailbreak judge -- y=1 only for an explicit, "
                        "unhedged refusal (phrase match or LLM-confirmed equivalent), not just "
                        "'the jailbreak judge said reject'. Output subdir gets a _hard_refusal "
                        "suffix, e.g. refusal_compliance_full_hard_refusal/.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split rows across N independent processes (e.g. one per GPU) to "
                        "parallelize a single model's build across multiple GPUs -- models "
                        "still run one at a time, but each model's own run no longer has to "
                        "sit on a single GPU. Each shard gets its own subdir/checkpoint; run "
                        "merge_shards() (or --merge) once all --num_shards processes finish.")
    p.add_argument("--shard_idx", type=int, default=0, help="0-indexed shard this process handles.")
    p.add_argument("--merge", action="store_true",
                   help="Instead of running generation, merge all --num_shards shard "
                        "directories under the output dir into the final dataset+report.")
    return p.parse_args()


def merge_shards(model_name: str, output_dir: str, num_shards: int) -> None:
    all_records = []
    all_counts = {"strmatch": 0, "llm_fallback": 0, "judge_errors": 0}
    for shard_idx in range(num_shards):
        shard_dir = os.path.join(output_dir, f"shard_{shard_idx}")
        dataset_path = os.path.join(shard_dir, "sorrybench_rc_dataset.json")
        report_path = os.path.join(shard_dir, "sorrybench_rc_report.json")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Shard {shard_idx} not finished yet -- missing {dataset_path}")
        with open(dataset_path) as f:
            all_records.extend(json.load(f))
        with open(report_path) as f:
            r = json.load(f)["judge_breakdown"]
            all_counts["strmatch"] += r["strmatch"]
            all_counts["llm_fallback"] += r["llm_fallback"]
            all_counts["judge_errors"] += r["judge_errors"]
    logger.info("Merged %d rows from %d shards", len(all_records), num_shards)
    save_and_report(model_name, all_records, all_counts, output_dir)


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    args = parse_args()
    default_subdir = "refusal_compliance_full" if args.full else "refusal_compliance"
    if args.hard_refusal:
        default_subdir += "_hard_refusal"
    base_output_dir = args.output_dir or f"data/embeddings/{args.model_name}/{default_subdir}"

    if args.merge:
        merge_shards(args.model_name, base_output_dir, args.num_shards)
    else:
        output_dir = (os.path.join(base_output_dir, f"shard_{args.shard_idx}")
                      if args.num_shards > 1 else base_output_dir)
        records, judge_breakdown = run_model(args.model_name, args, hf_token, output_dir)
        save_and_report(args.model_name, records, judge_breakdown, output_dir)
