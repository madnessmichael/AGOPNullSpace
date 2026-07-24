from transformers import LlamaConfig, Qwen2Config, Gemma2Config
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, Gemma2ForCausalLM
from AlphaSteerModel import *
from NaiveSteerModel import *

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

__all__ = [
    "MODELS_DICT", "AlphaSteer_MODELS_DICT", "Steer_MODELS_DICT",
    "AlphaSteer_STEERING_LAYERS", "AlphaSteer_CALCULATION_CONFIG",
]

MODELS_DICT = {
    "llama3.1": (LlamaForCausalLM, LlamaConfig, "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5": (Qwen2ForCausalLM, Qwen2Config, "Qwen/Qwen2.5-7B-Instruct"),
    "gemma2": (Gemma2ForCausalLM, Gemma2Config, "google/gemma-2-9b-it"),
    # ── NEW ──
    "llama3.1-70b": (LlamaForCausalLM, LlamaConfig, "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"),
    "llama3.3-70b": (LlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"),

}

AlphaSteer_MODELS_DICT = {
    "llama3.1": (AlphaLlamaForCausalLM, LlamaConfig, "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5": (AlphaQwen2ForCausalLM, Qwen2Config, "Qwen/Qwen2.5-7B-Instruct"),
    "gemma2": (AlphaGemma2ForCausalLM, Gemma2Config, "google/gemma-2-9b-it"),
    # ── NEW ──
    "llama3.1-70b": (AlphaLlamaForCausalLM, LlamaConfig, "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"),
    "llama3.3-70b": (AlphaLlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"),
}

Steer_MODELS_DICT = {
    "llama3.1": (SteerLlamaForCausalLM, LlamaConfig, "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5": (SteerQwen2ForCausalLM, Qwen2Config, "Qwen/Qwen2.5-7B-Instruct"),
    "gemma2": (SteerGemma2ForCausalLM, Gemma2Config, "google/gemma-2-9b-it"),
    # ── NEW ──
    "llama3.1-70b": (SteerLlamaForCausalLM, LlamaConfig, "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"),
    "llama3.3-70b": (SteerLlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"),
}

# NOTE: llama3.1/qwen2.5/gemma2 entries below were changed to "all middle layers
# except the first 2 and last 4" (untuned, ratio fixed at 0.6) for the SORRY-Bench
# refuse-compliance sweep. This is a global change -- any old DIM/plain-RFM/HH run
# re-executed after this edit will use these new layers/ratio too, not the
# hand-tuned originals. num_hidden_layers: llama3.1=32, qwen2.5=28, gemma2=42.
AlphaSteer_STEERING_LAYERS = {
    "llama3.1": list(range(2, 32 - 4)),
    "qwen2.5": list(range(2, 28 - 4)),
    "gemma2": list(range(2, 42 - 4)),
    # ── NEW (tỷ lệ ~35-65% tổng layers, vùng middle-upper) ──
    "llama3.1-70b": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
                     70],
    "llama3.3-70b": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
                     70],
}

# llama3.1/qwen2.5/gemma2 below: rho=0.6 fixed across the new SORRY-Bench-sweep
# layer range (see AlphaSteer_STEERING_LAYERS above) -- untuned, replaces the
# hand-tuned per-layer ratios that were curated for the original 10/15/11-layer
# subsets. The previous hand-tuned values are no longer reachable from here;
# see git history if they're needed again.
AlphaSteer_CALCULATION_CONFIG = {
    "llama3.1": [(l, 0.6) for l in AlphaSteer_STEERING_LAYERS["llama3.1"]],
    "qwen2.5": [(l, 0.6) for l in AlphaSteer_STEERING_LAYERS["qwen2.5"]],
    "gemma2": [(l, 0.6) for l in AlphaSteer_STEERING_LAYERS["gemma2"]],
    # ── NEW (strength 0.5 mặc định — tune lại sau khi có DSR) ──
    "llama3.1-70b": [(20, 0.5), (22, 0.5), (24, 0.5), (26, 0.5), (28, 0.5), (30, 0.5), (32, 0.5), (34, 0.5), (36, 0.5),
                     (38, 0.5), (40, 0.5), (42, 0.5), (44, 0.5), (46, 0.5), (48, 0.5), (50, 0.5), (52, 0.5), (54, 0.5),
                     (56, 0.5), (58, 0.5), (60, 0.5), (62, 0.5), (64, 0.5), (66, 0.5), (68, 0.5), (70, 0.5)],
    "llama3.3-70b": [(20, 0.5), (22, 0.5), (24, 0.5), (26, 0.5), (28, 0.5), (30, 0.5), (32, 0.5), (34, 0.5), (36, 0.5),
                     (38, 0.5), (40, 0.5), (42, 0.5), (44, 0.5), (46, 0.5), (48, 0.5), (50, 0.5), (52, 0.5), (54, 0.5),
                     (56, 0.5), (58, 0.5), (60, 0.5), (62, 0.5), (64, 0.5), (66, 0.5), (68, 0.5), (70, 0.5)],
}
