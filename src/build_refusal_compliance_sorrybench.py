"""
build_refusal_compliance_sorrybench.py
=======================================
Stage 1 of the SORRY-Bench refuse-compliance pipeline.

Runs the UNSTEERED base model on SORRY-Bench prompts
(sorry-bench/sorry-bench-202503), judges each response as refusal or
compliance, and saves final-token activations split by label. Replaces
justinphan3110/harmful_harmless_instructions as the concept-vector training
data for calc_steering_matrix_rfm_rc*.py.

Two modes:
  --full not set (default): prompt_style=='base' only -- 440 rows
      (44 categories x 10), output -> refusal_compliance/
  --full: all 21 prompt_style variants (base + 20 linguistic mutations:
      question, slang, uncommon_dialects, technical_terms, role_play,
      misspellings, logical_appeal, authority_endorsement, misrepresentation,
      evidence_based_persuasion, expert_endorsement, ascii, caesar, morse,
      atbash, translate_{ml,ta,mr,zh_cn,fr}) -- 9,240 rows total
      (44 categories x 10 x 21), output -> refusal_compliance_full/

Label convention: y=1=refusal, y=0=compliance, so the RFM/AGOP concept vector
trained on this data (Stage 2) points toward refusal -- keeps ``strength>0
== defend`` consistent with the rest of the repo (README, *_dim.pt, *_rfm.pt).

Usage:
    python src/build_refusal_compliance_sorrybench.py \
        --model_name llama3.1 --device cuda:0 --batch_size 8

    python src/build_refusal_compliance_sorrybench.py \
        --model_name llama3.1 --device cuda:0 --batch_size 8 --full
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
from utils.mask_utils import get_last_valid_token_index  # noqa: E402
from jailbreak import judge_refusal_strmatch, evaluate_jailbreak  # noqa: E402

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


@torch.no_grad()
def run_model(model_name: str, args: argparse.Namespace, token: str):
    model_class, _config_class, model_id = MODELS_DICT[model_name]
    logger.info("Loading %s (%s) on %s...", model_name, model_id, args.device)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = model_class.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=args.device, token=token,
    )
    model.eval()

    rows = load_sorrybench_full(token) if args.full else load_sorrybench_base(token)
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["turns"][0]}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in rows
    ]

    records = []
    n_strmatch = n_llm = n_judge_error = 0

    for i in range(0, len(formatted), args.batch_size):
        batch_rows = rows[i:i + args.batch_size]
        batch_prompts = formatted[i:i + args.batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                         truncation=True).to(model.device)
        B, T = enc["input_ids"].shape

        # -- forward-only pass: final-token activation per decoder layer --
        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states[1:]  # drop embedding layer
        last_idx = get_last_valid_token_index(
            enc["attention_mask"], seq_len=T, batch_size=B, device=model.device)
        batch_idx = torch.arange(B, device=model.device)
        layer_acts = torch.stack(
            [h[batch_idx, last_idx, :].float().cpu() for h in hidden_states], dim=1,
        )  # [B, L, d]
        del out, hidden_states

        # -- generation pass: response text for judging --
        gen_out = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        responses = tokenizer.batch_decode(gen_out[:, T:], skip_special_tokens=True)
        del gen_out
        torch.cuda.empty_cache()

        for row, resp, act in zip(batch_rows, responses, layer_acts):
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
                "label": label, "activation": act,
            })

        logger.info("[%s] %d/%d judged (strmatch=%d, llm=%d, errors=%d)",
                    model_name, len(records), len(rows), n_strmatch, n_llm, n_judge_error)

    return records, {"strmatch": n_strmatch, "llm_fallback": n_llm, "judge_errors": n_judge_error}


def save_and_report(model_name: str, records: list[dict], judge_breakdown: dict, out_dir: str) -> dict:
    refusal = [r for r in records if r["label"] == 1]
    compliance = [r for r in records if r["label"] == 0]

    os.makedirs(out_dir, exist_ok=True)
    torch.save(torch.stack([r["activation"] for r in refusal]),
               os.path.join(out_dir, "embeds_refusal.pt"))
    torch.save(torch.stack([r["activation"] for r in compliance]),
               os.path.join(out_dir, "embeds_compliance.pt"))

    # Full raw dataset (all rows, text included) -- previously only embeddings
    # + a 6-example slice in the report were persisted; the actual generated
    # text for all 440 prompts only existed in-process memory and was lost
    # once the run exited. Save it so the dataset is independently auditable
    # without re-running generation.
    dataset_path = os.path.join(out_dir, "sorrybench_rc_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump([
            {"question_id": r["question_id"], "category": r["category"],
             "prompt_style": r.get("prompt_style", "base"),
             "prompt": r["prompt"], "response": r["response"],
             "verdict": r["verdict"], "label": r["label"]}
            for r in records
        ], f, indent=2, ensure_ascii=False)

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
    logger.info("Saved refusal=%d compliance=%d -> %s", len(refusal), len(compliance), out_dir)
    logger.info("Full dataset -> %s", dataset_path)
    logger.info("Report -> %s", report_path)
    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, choices=["llama3.1", "qwen2.5", "gemma2"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--judge_model", default="gpt-4o-mini")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--full", action="store_true",
                   help="Use all 21 prompt_style variants (9,240 rows) instead of "
                        "just prompt_style=='base' (440 rows). Output goes to "
                        "refusal_compliance_full/ instead of refusal_compliance/.")
    return p.parse_args()


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    args = parse_args()
    default_subdir = "refusal_compliance_full" if args.full else "refusal_compliance"
    output_dir = args.output_dir or f"data/embeddings/{args.model_name}/{default_subdir}"

    records, judge_breakdown = run_model(args.model_name, args, hf_token)
    save_and_report(args.model_name, records, judge_breakdown, output_dir)
