"""
extract_rc_style_sample.py
============================
Extracts a stratified per-prompt_style sample of activations from
sorrybench_rc_dataset.json, WITH metadata (prompt_style, category, label)
preserved alongside each row -- something the production
extract_refusal_compliance_embeddings.py pipeline does NOT give you, because
it shards rows across GPUs and reorders them into length-tiered batches
before saving, so the final embeds_refusal.pt / embeds_compliance.pt cannot
be mapped back to which SORRY-Bench prompt_style/category each row came
from.

This script exists specifically to test the "heterogeneous jailbreak"
claim: is RFM's advantage over DIM concentrated in the harder/rarer
prompt_style families (persuasion techniques, character-level encodings)
that DIM's simple mean-difference (dominated by the numerically largest,
easiest-to-separate style family) might underserve?

21 prompt_style families in refusal_compliance_full_hard_refusal, refusal
rate varies wildly: natural-language styles (base, role_play, slang, ...)
are 64-78% refusal; persuasion styles (logical_appeal, misrepresentation,
evidence-based_persuasion, authority/expert_endorsement) are 6-17%;
character encodings (caesar, morse, atbash, ascii) are 0-5%.

Usage:
    python evaluation/extract_rc_style_sample.py \
        --model_name meta-llama/Llama-3.1-8B-Instruct \
        --rc_json data/embeddings/llama3.1/refusal_compliance_full_hard_refusal/sorrybench_rc_dataset.json \
        --out_file data/embeddings/llama3.1/rc_style_sample.pt \
        --n_per_style 100 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils.embedding_utils import EmbeddingExtractor  # noqa: E402

# Styles whose prompts run long (character-level encodings) get a small
# batch size to avoid the OOM seen extracting cipher/autodan attack prompts
# at batch_size=16 (see logs/extract_attack/*.log this session).
LONG_STYLES = {"ascii", "morse", "caesar", "atbash"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, help="HF model id (not the repo alias)")
    p.add_argument("--rc_json", required=True)
    p.add_argument("--out_file", required=True,
                   help="With --only_style, this is treated as a directory to hold "
                        "one file per style; with --merge, the directory to read from.")
    p.add_argument("--n_per_style", type=int, default=100)
    p.add_argument("--batch_size_short", type=int, default=16)
    p.add_argument("--batch_size_long", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2706)
    p.add_argument("--only_style", default=None,
                   help="Process a single prompt_style and exit (fresh CUDA context per "
                        "style avoids cross-style allocator fragmentation/OOM seen when "
                        "looping over all 21 styles in one long-lived process).")
    p.add_argument("--merge", action="store_true",
                   help="Instead of extracting, merge all per-style files under --out_file "
                        "(as a directory) into a single combined .pt (out_file + '.pt').")
    return p.parse_args()


def _sample_records(rc_json: str, n_per_style: int, seed: int):
    random.seed(seed)
    with open(rc_json) as f:
        records = json.load(f)
    by_style = defaultdict(list)
    for r in records:
        by_style[r["prompt_style"]].append(r)
    sampled_by_style = {}
    for style, rows in sorted(by_style.items()):
        random.shuffle(rows)
        sampled_by_style[style] = rows[:n_per_style]
    return sampled_by_style


def main():
    args = parse_args()

    if args.merge:
        style_dir = args.out_file
        files = sorted(f for f in os.listdir(style_dir) if f.endswith(".pt"))
        all_H, all_style, all_category, all_label, all_qid = [], [], [], [], []
        for fname in files:
            d = torch.load(os.path.join(style_dir, fname), map_location="cpu")
            all_H.append(d["H"])
            all_style.extend(d["prompt_style"])
            all_category.extend(d["category"])
            all_label.extend(d["label"])
            all_qid.extend(d["question_id"])
        H_all = torch.cat(all_H, dim=0)
        payload = {"H": H_all, "prompt_style": all_style, "category": all_category,
                   "label": all_label, "question_id": all_qid}
        merged_path = style_dir.rstrip("/") + ".pt"
        torch.save(payload, merged_path)
        print(f"Merged {len(files)} style files -> {H_all.shape} -> {merged_path}")
        return

    sampled_by_style = _sample_records(args.rc_json, args.n_per_style, args.seed)

    if args.only_style:
        styles_to_run = [args.only_style]
        os.makedirs(args.out_file, exist_ok=True)
    else:
        styles_to_run = list(sampled_by_style.keys())

    extractor = EmbeddingExtractor(args.model_name, device=args.device)
    layers = list(range(extractor.num_layers))

    for style in styles_to_run:
        rows = sampled_by_style[style]
        bs = args.batch_size_long if style in LONG_STYLES else args.batch_size_short
        print(f"Extracting {len(rows)} rows for style={style} (batch_size={bs})...")
        prompts = [r["prompt"] for r in rows]
        H = extractor.extract_embeddings(prompts, batch_size=bs, layers=layers)
        payload = {
            "H": H,
            "prompt_style": [r["prompt_style"] for r in rows],
            "category": [r["category"] for r in rows],
            "label": [r["label"] for r in rows],
            "question_id": [r["question_id"] for r in rows],
        }
        if args.only_style:
            out_path = os.path.join(args.out_file, f"{style}.pt")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.out_file)) or ".", exist_ok=True)
            out_path = args.out_file if len(styles_to_run) == 1 else args.out_file + f".{style}.pt"
        torch.save(payload, out_path)
        print(f"Saved {H.shape} -> {out_path}")


if __name__ == "__main__":
    main()
