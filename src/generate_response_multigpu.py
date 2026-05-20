"""
generate_response_multigpu.py
─────────────────────────────────────────────────────────────────────────────
Multi-GPU inference script for AlphaSteer / NullSpace steering experiments.
Supports 8× RTX 3090 (or any multi-GPU setup) via HuggingFace device_map="auto".

Usage:
    python src/generate_response_multigpu.py --config_path config/llama3.3-70b_rfm/xstest.yaml

Expected YAML fields:
    model_name            : key into MODELS_DICT / AlphaSteer_MODELS_DICT
    input_file            : path to JSON list of prompt dicts
    output_file           : path to write results  (optional; defaults to input_file)
    batch_size            : int  (recommend 1 for 70B on 8×3090)
    max_new_tokens        : int
    prompt_column         : str  key in each prompt dict
    file_rename           : bool (append timestamp to output filename)
    strength              : comma-separated floats, e.g. "-1.0,-0.5,0.0,0.5,1.0"

  One of:
    steering_matrix_path  : path to .pt file  →  AlphaSteer (null-space matrix)
    steering_vector_path  : path to .pt file  →  Naive vector steering
    (neither)             : plain generation without any steering

  GPU settings (optional — override defaults):
    num_gpus              : int,  default 8
    vram_per_gpu_gib      : int,  default 22   (leave 2 GiB buffer on each 24 GiB card)
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import time
import json
import logging
import argparse
import datetime

# ── disable torch.compile before any torch import ────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch._dynamo
torch._dynamo.config.disable = True          # incompatible with multi-device pipeline

import numpy as np
import yaml
from jinja2 import Template
from transformers import AutoTokenizer

# ── local imports (adjust sys.path if running from repo root) ─────────────────
# Assumes script lives in  <repo>/src/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.const import (
    AlphaSteer_MODELS_DICT,
    AlphaSteer_STEERING_LAYERS,
    Steer_MODELS_DICT,
    MODELS_DICT,
)

# ── reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── math/GSM prompt template ──────────────────────────────────────────────────
_MATH_TEMPLATE = Template(
    "Please solve this problem, and put your final answer within \\boxed{}\n"
    "This is the problem:\n"
    "{{prompt}}\n"
    "Please remember to put your final answer within \\boxed{}\n"
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU AlphaSteer inference")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to YAML config file")
    return parser.parse_args()


def build_max_memory(num_gpus: int, vram_per_gpu_gib: int) -> dict:
    """
    Returns a max_memory dict for HuggingFace from_pretrained.
    Keeps a small CPU buffer so the model head / embedding can overflow if needed.
    """
    mem = {i: f"{vram_per_gpu_gib}GiB" for i in range(num_gpus)}
    mem["cpu"] = "60GiB"   # overflow buffer – adjust if RAM is limited
    return mem


def get_input_device(model) -> torch.device:
    """Return the device that holds the embedding / first layer of the model."""
    return next(model.parameters()).device


def format_prompts(prompts: list, prompt_column: str,
                   tokenizer, is_math: bool) -> list:
    """Apply chat template (+ optional math wrapper) to raw prompt list."""
    formatted = []
    for p in prompts:
        text = p[prompt_column]
        if is_math:
            text = _MATH_TEMPLATE.render(prompt=text)
        message = {"role": "user", "content": text}
        formatted.append(
            tokenizer.apply_chat_template(
                [message], tokenize=False, add_generation_prompt=True
            )
        )
    return formatted


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(args):
    """
    Decide steering mode, load model with device_map='auto' across all GPUs,
    return (model, tokenizer, steering_matrix_or_vector, steering_layers).
    """
    num_gpus        = getattr(args, "num_gpus", 8)
    vram_per_gpu    = getattr(args, "vram_per_gpu_gib", 22)
    max_memory      = build_max_memory(num_gpus, vram_per_gpu)

    # ── choose model class + steering artifact ────────────────────────────────
    if hasattr(args, "steering_matrix_path"):
        if not os.path.exists(args.steering_matrix_path):
            raise FileNotFoundError(f"steering_matrix_path not found: {args.steering_matrix_path}")
        model_class, config_class, model_id = AlphaSteer_MODELS_DICT[args.model_name]
        steering_artifact = torch.load(args.steering_matrix_path, map_location="cpu").to(torch.bfloat16)
        steering_layers   = AlphaSteer_STEERING_LAYERS[args.model_name]
        mode_label        = "NullSpace (AlphaSteer) matrix"

    elif hasattr(args, "steering_vector_path"):
        if not os.path.exists(args.steering_vector_path):
            raise FileNotFoundError(f"steering_vector_path not found: {args.steering_vector_path}")
        model_class, config_class, model_id = Steer_MODELS_DICT[args.model_name]
        steering_artifact = torch.load(args.steering_vector_path, map_location="cpu").to(torch.bfloat16)
        steering_layers   = list(range(steering_artifact.shape[0]))
        mode_label        = "Naive vector steering"

    else:
        model_class, config_class, model_id = MODELS_DICT[args.model_name]
        steering_artifact = None
        steering_layers   = None
        mode_label        = "No steering (plain generation)"

    logger.info(f"Steering mode : {mode_label}")
    logger.info(f"Model id      : {model_id}")
    logger.info(f"Max memory    : {max_memory}")

    # ── model config (to read hidden_size, num_hidden_layers) ─────────────────
    hf_config   = config_class.from_pretrained(model_id)
    num_layers  = hf_config.num_hidden_layers

    # ── tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token      = tokenizer.eos_token
    tokenizer.padding_side   = "left"   # required for left-padded batch generation

    # ── load model across all GPUs ────────────────────────────────────────────
    logger.info("Loading model weights — this may take a few minutes for 70B …")
    t0 = time.time()
    model = model_class.from_pretrained(
        model_id,
        device_map="auto",          # HuggingFace/accelerate handles layer placement
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
    )
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    # Log device map for transparency
    if hasattr(model, "hf_device_map"):
        device_counts: dict = {}
        for dev in model.hf_device_map.values():
            device_counts[str(dev)] = device_counts.get(str(dev), 0) + 1
        logger.info(f"Layer distribution across devices: {device_counts}")

    # ── initial steering setup (strength=0 placeholder) ──────────────────────
    strength_zeros = [0.0] * num_layers
    if steering_artifact is not None:
        model.set_steering_parameters(steering_artifact, strength=strength_zeros)
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token    = tokenizer.pad_token

    return model, tokenizer, steering_artifact, steering_layers, num_layers


# ─────────────────────────────────────────────────────────────────────────────
#  Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args, model, tokenizer, steering_artifact,
                  steering_layers, num_layers, prompts, formatted_prompts):

    # ── strength schedule ─────────────────────────────────────────────────────
    if hasattr(args, "strength") and args.strength:
        const_strength_list = [float(s) for s in str(args.strength).split(",") if s.strip()]
    else:
        const_strength_list = [0.0]
    logger.info(f"Strength schedule ({len(const_strength_list)} values): {const_strength_list}")

    # ── output file path ──────────────────────────────────────────────────────
    output_file = getattr(args, "output_file", None) or args.input_file
    if getattr(args, "file_rename", False):
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_file.replace(".json", f"_{ts}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    logger.info(f"Output file   : {output_file}")

    # Resume from existing output if present
    if os.path.exists(output_file):
        logger.info("Resuming from existing output file.")
        with open(output_file, "r") as fh:
            prompts = json.load(fh)

    total_batches = (len(formatted_prompts) + args.batch_size - 1) // args.batch_size
    input_device  = get_input_device(model)
    logger.info(f"Input device  : {input_device}")
    logger.info(f"Total prompts : {len(formatted_prompts)}")
    logger.info(f"Batch size    : {args.batch_size}")
    logger.info(f"Total batches : {total_batches}")

    # ── per-strength outer loop ───────────────────────────────────────────────
    completed_strengths: list = []
    last_batch_idx = 0

    try:
        for const_strength in const_strength_list:

            # Update steering strength for this sweep
            if steering_layers is not None:
                strength = [0.0] * num_layers
                for layer in steering_layers:
                    strength[layer] = const_strength
                model.set_steering_parameters(strength=strength)
                logger.info(f"[Strength={const_strength}] Steering applied to layers: {steering_layers}")
            else:
                logger.info(f"[Strength={const_strength}] No steering layers — plain generation.")

            response_key = f"response_strength:{const_strength}"

            # ── batch loop ────────────────────────────────────────────────────
            for batch_start in range(0, len(formatted_prompts), args.batch_size):
                batch_end      = min(batch_start + args.batch_size, len(formatted_prompts))
                batch_prompts  = formatted_prompts[batch_start:batch_end]
                last_batch_idx = batch_start // args.batch_size + 1

                # Tokenise
                batch_inputs = tokenizer(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(input_device)

                input_lengths      = [len(ids) for ids in batch_inputs["input_ids"]]
                batch_input_ids    = batch_inputs["input_ids"]
                batch_attention_mask = batch_inputs["attention_mask"]

                t_start = time.time()

                with torch.no_grad():
                    batch_outputs = model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        num_return_sequences=1,
                        do_sample=False,
                        temperature=None,       # must be None when do_sample=False
                        top_p=None,             # same
                    )

                elapsed     = time.time() - t_start
                actual_bs   = batch_end - batch_start
                per_example = elapsed / actual_bs

                # Decode and store
                for j, output in enumerate(batch_outputs):
                    generated_part = output[input_lengths[j]:]
                    response = tokenizer.decode(generated_part, skip_special_tokens=True)
                    prompts[batch_start + j][response_key] = response

                # Free GPU memory
                del batch_outputs, batch_input_ids, batch_attention_mask, batch_inputs
                torch.cuda.empty_cache()

                logger.info(
                    f"[s={const_strength}] Batch {last_batch_idx}/{total_batches} | "
                    f"items {batch_start+1}–{batch_end}/{len(formatted_prompts)} | "
                    f"{elapsed:.1f}s total | {per_example:.2f}s/example"
                )

            # Save after each full strength sweep
            with open(output_file, "w") as fh:
                json.dump(prompts, fh, indent=4, ensure_ascii=False)
            logger.info(f"[s={const_strength}] Saved → {output_file}")
            completed_strengths.append(const_strength)

    except Exception as exc:
        logger.error(f"Exception at strength={const_strength}, batch={last_batch_idx}: {exc}")
        logger.error(f"Completed strengths so far: {completed_strengths}")
        # Emergency save
        with open(output_file, "w") as fh:
            json.dump(prompts, fh, indent=4, ensure_ascii=False)
        logger.info(f"Emergency checkpoint saved → {output_file}")
        raise

    logger.info(f"All done. Results in: {output_file}")
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    config = load_config(args.config_path)
    for key, value in config.items():
        setattr(args, key, value)
    logger.info(f"Config: {vars(args)}")

    # ── load model ────────────────────────────────────────────────────────────
    model, tokenizer, steering_artifact, steering_layers, num_layers = \
        load_model_and_tokenizer(args)

    # ── load prompts ──────────────────────────────────────────────────────────
    with open(args.input_file, "r") as fh:
        prompts = json.load(fh)
    logger.info(f"Loaded {len(prompts)} prompts from {args.input_file}")

    # ── detect dataset type ───────────────────────────────────────────────────
    is_math = any(kw in args.input_file for kw in ("gsm8k", "math"))
    if is_math:
        logger.info("Math/GSM dataset detected — applying \\boxed{} template.")

    # ── format prompts ────────────────────────────────────────────────────────
    formatted_prompts = format_prompts(
        prompts, args.prompt_column, tokenizer, is_math
    )

    # ── run inference ─────────────────────────────────────────────────────────
    run_inference(
        args, model, tokenizer,
        steering_artifact, steering_layers, num_layers,
        prompts, formatted_prompts,
    )


if __name__ == "__main__":
    main()
