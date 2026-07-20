"""
prepare_harmful_harmless_instruction_embeddings.py
===================================================
Extract final-token hidden-state activations from:
    justinphan3110/harmful_harmless_instructions

Output files expected by rfm_refusal_vector_hh_v5.py and
calc_steering_matrix_rfm_hh_v5.py:
    embeds_harmful_harmless_instructions.pt
    labels_harmful_harmless_instructions.pt

Label mapping from the HF dataset:
    True/1  = harmless
    False/0 = harmful
"""

import os
import argparse
import logging
from typing import List

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "justinphan3110/harmful_harmless_instructions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def flatten_hh_dataset(split: str = "train"):
    ds = load_dataset(DATASET_NAME)
    if split not in ds:
        logger.warning("Split %s not found. Falling back to train.", split)
        split = "train"
    rows = ds[split]

    prompts: List[str] = []
    labels: List[int] = []
    for row in rows:
        sentences = row["sentence"]
        row_labels = row["label"]
        if not isinstance(sentences, (list, tuple)):
            sentences = [sentences]
        if not isinstance(row_labels, (list, tuple)):
            row_labels = [row_labels]
        if len(sentences) != len(row_labels):
            raise ValueError(f"sentence/label length mismatch in row: {row}")
        prompts.extend([str(x) for x in sentences])
        labels.extend([int(bool(x)) for x in row_labels])

    logger.info(
        "Loaded %s split=%s: %d prompts, harmless=%d, harmful=%d",
        DATASET_NAME, split, len(prompts), sum(labels), len(labels) - sum(labels),
    )
    return prompts, torch.tensor(labels, dtype=torch.int64)


def apply_chat_template(tokenizer, prompts: List[str]) -> List[str]:
    formatted = []
    for prompt in prompts:
        chat = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        except TypeError:
            text = tokenizer.apply_chat_template(chat, tokenize=False)
        formatted.append(text)
    return formatted


@torch.no_grad()
def extract_final_token_activations(model, tokenizer, prompts: List[str], batch_size: int, device: str):
    model.eval()
    all_batches = []

    loader = DataLoader(prompts, batch_size=batch_size, shuffle=False)
    for batch_prompts in loader:
        enc = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[1:]  # remove embedding layer

        # final non-padding token position for each sequence
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        batch_layers = []
        for h in hidden_states:
            # h: [B, T, d]
            final_h = h[torch.arange(h.shape[0], device=device), last_idx, :]
            batch_layers.append(final_h.detach().cpu().float())
        # [B, num_layers, d]
        all_batches.append(torch.stack(batch_layers, dim=1))

    return torch.cat(all_batches, dim=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--save_separated", action="store_true", help="Also save embeds_hh_harmful.pt and embeds_hh_harmless.pt")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device != "cpu" else None,
        trust_remote_code=args.trust_remote_code,
    )
    if args.device == "cpu":
        model.to("cpu")

    prompts, labels = flatten_hh_dataset(args.split)
    prompts = apply_chat_template(tokenizer, prompts)
    H = extract_final_token_activations(model, tokenizer, prompts, args.batch_size, args.device)

    embeds_path = os.path.join(args.output_dir, "embeds_harmful_harmless_instructions.pt")
    labels_path = os.path.join(args.output_dir, "labels_harmful_harmless_instructions.pt")
    torch.save(H, embeds_path)
    torch.save(labels, labels_path)
    logger.info("Saved embeddings: %s shape=%s", embeds_path, tuple(H.shape))
    logger.info("Saved labels: %s shape=%s", labels_path, tuple(labels.shape))

    if args.save_separated:
        harmful = H[~labels.bool()]
        harmless = H[labels.bool()]
        torch.save(harmful, os.path.join(args.output_dir, "embeds_hh_harmful.pt"))
        torch.save(harmless, os.path.join(args.output_dir, "embeds_hh_harmless.pt"))
        logger.info("Saved separated harmful=%s harmless=%s", tuple(harmful.shape), tuple(harmless.shape))


if __name__ == "__main__":
    main()