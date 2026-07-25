"""
extract_refusal_compliance_embeddings.py
==========================================
Stage 1b of the SORRY-Bench refuse-compliance pipeline: ACTIVATIONS ONLY.

Reads the finished text dataset from build_refusal_compliance_sorrybench.py
(sorrybench_rc_dataset.json: prompt/response/verdict/label, no generation
here) and extracts the final-valid-token activation per decoder layer for
each prompt, split by label into embeds_refusal.pt / embeds_compliance.pt --
the same output calc_steering_matrix_rfm_rc*.py already expects.

Kept as a separate phase from generation on purpose: output_hidden_states=True
materializes hidden_states for every layer x every token position before we
slice out just the last one, which is the single biggest memory cost in this
whole pipeline. Decoupling it from generation means each phase's memory
profile only has to account for its own job, and a crash in one never
throws away the other's completed work. Same length-tiered batching, gemma2
safety settings (eager attention, max_length=2048, forced batch=1 -- these
were needed to get the build phase through gemma2's ascii/morse outliers
without OOM, same O(n^2)-attention-memory reasoning applies here), and
checkpoint/resume support as the build script.

Sharding (--num_shards/--shard_idx/--merge): same pattern as the build
script -- split one model's rows across N GPUs in parallel, then merge.

Usage:
    python src/extract_refusal_compliance_embeddings.py \
        --model_name llama3.1 --device cuda:0 --full
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import torch._dynamo
from dotenv import load_dotenv
from transformers import AutoTokenizer

torch._dynamo.config.disable = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.const import MODELS_DICT                     # noqa: E402
from utils.mask_utils import get_last_valid_token_index  # noqa: E402
from build_refusal_compliance_sorrybench import make_tiered_batches, DEFAULT_TIERS, _parse_tiers  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, "_extract_checkpoint.pt")


def _load_checkpoint(output_dir: str):
    path = _checkpoint_path(output_dir)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        logger.info("Resuming extraction from checkpoint: %d rows already done -> %s",
                    len(ckpt["done_keys"]), path)
        return ckpt["done_keys"], ckpt["acts_refusal"], ckpt["acts_compliance"]
    return set(), [], []


def _save_checkpoint(output_dir: str, done_keys: set, acts_refusal: list, acts_compliance: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tmp = _checkpoint_path(output_dir) + ".tmp"
    torch.save({"done_keys": done_keys, "acts_refusal": acts_refusal,
               "acts_compliance": acts_compliance}, tmp)
    os.replace(tmp, _checkpoint_path(output_dir))  # atomic


@torch.no_grad()
def extract(model_name: str, args: argparse.Namespace, token: str, data_dir: str, output_dir: str):
    model_class, _config_class, model_id = MODELS_DICT[model_name]

    dataset_path = os.path.join(data_dir, "sorrybench_rc_dataset.json")
    with open(dataset_path) as f:
        records = json.load(f)
    logger.info("Loaded %d rows from %s", len(records), dataset_path)

    if args.num_shards > 1:
        records = records[args.shard_idx::args.num_shards]
        logger.info("Shard %d/%d: %d rows assigned", args.shard_idx, args.num_shards, len(records))

    done_keys, acts_refusal, acts_compliance = _load_checkpoint(output_dir)
    pending = [r for r in records
               if (r["question_id"], r.get("prompt_style", "base")) not in done_keys]
    logger.info("%d rows remaining after resume", len(pending))

    if not pending:
        logger.info("Nothing to do -- all rows already extracted.")
        return acts_refusal, acts_compliance

    logger.info("Loading %s (%s) on %s...", model_name, model_id, args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Same gemma2 safety settings as build_refusal_compliance_sorrybench.py:
    # eager attention dodges a known SDPA alignment bug on gemma2's sliding-
    # window attention, but eager is O(seq_len^2) in fp32, so it also needs
    # the tighter length cap + forced batch=1 below to avoid OOM on a 9B
    # model at 24GB (proven necessary in the build phase's ascii/morse tail).
    extra_kwargs = {"attn_implementation": "eager"} if model_name == "gemma2" else {}
    model = model_class.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=args.device, token=token,
        **extra_kwargs,
    )
    model.eval()

    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        for r in pending
    ]
    lengths = [len(tokenizer(p)["input_ids"]) for p in formatted]
    tiers = args.tiers
    max_len = 10000
    if model_name == "gemma2":
        tiers = [(lo, hi, 1) for lo, hi, _bs in tiers]
        max_len = 2048
        logger.info("gemma2: forcing batch=1 for all tiers + max_length=2048 (eager attention memory)")
    batches = make_tiered_batches(lengths, tiers)
    logger.info("%d rows -> %d tiered batches", len(pending), len(batches))

    rows_done_since_ckpt = 0

    for batch_i, idxs in enumerate(batches):
        batch_rows = [pending[i] for i in idxs]
        batch_prompts = [formatted[i] for i in idxs]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(model.device)
        B, T = enc["input_ids"].shape

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states[1:]  # drop embedding layer
        last_idx = get_last_valid_token_index(
            enc["attention_mask"], seq_len=T, batch_size=B, device=model.device)
        batch_idx = torch.arange(B, device=model.device)
        layer_acts = torch.stack(
            [h[batch_idx, last_idx, :].float().cpu() for h in hidden_states], dim=1,
        )  # [B, L, d]
        del out, hidden_states, enc
        torch.cuda.empty_cache()

        for row, act in zip(batch_rows, layer_acts):
            (acts_refusal if row["label"] == 1 else acts_compliance).append(act)
            done_keys.add((row["question_id"], row.get("prompt_style", "base")))

        rows_done_since_ckpt += len(idxs)
        logger.info("[%s] %d/%d extracted [batch %d/%d, bs=%d, len<=%d]",
                    model_name, len(acts_refusal) + len(acts_compliance), len(records),
                    batch_i + 1, len(batches), len(idxs), T)

        if rows_done_since_ckpt >= args.checkpoint_every:
            _save_checkpoint(output_dir, done_keys, acts_refusal, acts_compliance)
            rows_done_since_ckpt = 0
            logger.info("  checkpoint saved (%d refusal + %d compliance)",
                        len(acts_refusal), len(acts_compliance))

    _save_checkpoint(output_dir, done_keys, acts_refusal, acts_compliance)
    return acts_refusal, acts_compliance


def save(output_dir: str, acts_refusal: list, acts_compliance: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(torch.stack(acts_refusal), os.path.join(output_dir, "embeds_refusal.pt"))
    torch.save(torch.stack(acts_compliance), os.path.join(output_dir, "embeds_compliance.pt"))
    logger.info("Saved refusal=%d compliance=%d -> %s", len(acts_refusal), len(acts_compliance), output_dir)

    ckpt = _checkpoint_path(output_dir)
    if os.path.exists(ckpt):
        os.remove(ckpt)
        logger.info("Removed extraction checkpoint (run completed successfully)")


def merge_shards(output_dir: str, num_shards: int) -> None:
    acts_refusal, acts_compliance = [], []
    for shard_idx in range(num_shards):
        shard_dir = os.path.join(output_dir, f"shard_{shard_idx}")
        r_path = os.path.join(shard_dir, "embeds_refusal.pt")
        c_path = os.path.join(shard_dir, "embeds_compliance.pt")
        if not (os.path.exists(r_path) and os.path.exists(c_path)):
            raise FileNotFoundError(f"Shard {shard_idx} not finished yet -- missing {r_path}/{c_path}")
        acts_refusal.append(torch.load(r_path, map_location="cpu"))
        acts_compliance.append(torch.load(c_path, map_location="cpu"))
    refusal = torch.cat(acts_refusal, dim=0)
    compliance = torch.cat(acts_compliance, dim=0)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(refusal, os.path.join(output_dir, "embeds_refusal.pt"))
    torch.save(compliance, os.path.join(output_dir, "embeds_compliance.pt"))
    logger.info("Merged %d shards -> refusal=%s compliance=%s -> %s",
                num_shards, tuple(refusal.shape), tuple(compliance.shape), output_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, choices=["llama3.1", "qwen2.5", "gemma2"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tiers", type=_parse_tiers, default=DEFAULT_TIERS,
                   help="Same tier format as build_refusal_compliance_sorrybench.py.")
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--data_dir", default=None,
                   help="Dir containing sorrybench_rc_dataset.json. Default matches "
                        "build_refusal_compliance_sorrybench.py's output_dir convention.")
    p.add_argument("--output_dir", default=None, help="Default: same as --data_dir.")
    p.add_argument("--full", action="store_true")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split rows across N independent processes (e.g. one per GPU).")
    p.add_argument("--shard_idx", type=int, default=0)
    p.add_argument("--merge", action="store_true",
                   help="Instead of extracting, merge all --num_shards shard dirs' "
                        "embeds_{refusal,compliance}.pt into the final combined file.")
    return p.parse_args()


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    args = parse_args()
    default_subdir = "refusal_compliance_full" if args.full else "refusal_compliance"
    base_data_dir = args.data_dir or f"data/embeddings/{args.model_name}/{default_subdir}"
    base_output_dir = args.output_dir or base_data_dir

    if args.merge:
        merge_shards(base_output_dir, args.num_shards)
    else:
        data_dir = base_data_dir  # dataset json always lives at the base dir, not per-shard
        output_dir = (os.path.join(base_output_dir, f"shard_{args.shard_idx}")
                      if args.num_shards > 1 else base_output_dir)
        acts_refusal, acts_compliance = extract(args.model_name, args, hf_token, data_dir, output_dir)
        save(output_dir, acts_refusal, acts_compliance)
