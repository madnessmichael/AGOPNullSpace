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
    "llama3.1":  (LlamaForCausalLM,  LlamaConfig,  "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5":   (Qwen2ForCausalLM,  Qwen2Config,  "Qwen/Qwen2.5-7B-Instruct"),
    "gemma2":    (Gemma2ForCausalLM, Gemma2Config, "google/gemma-2-9b-it"),
    # ── NEW ──
    "llama3.3-70b":        (LlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"),
    "llama3.1-8b-unsloth": (LlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"),
    "qwen3-32b":           (Qwen2ForCausalLM, Qwen2Config, "unsloth/Qwen3-32B-unsloth-bnb-4bit"),
    "gpt-oss-120b":        (LlamaForCausalLM, LlamaConfig, "unsloth/gpt-oss-120b-unsloth-bnb-4bit"),
}

AlphaSteer_MODELS_DICT = {
    "llama3.1":  (AlphaLlamaForCausalLM,  LlamaConfig,  "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5":   (AlphaQwen2ForCausalLM,  Qwen2Config,  "Qwen/Qwen2.5-7B-Instruct"),
    "gemma2":    (AlphaGemma2ForCausalLM, Gemma2Config, "google/gemma-2-9b-it"),
    # ── NEW ──
    "llama3.3-70b":        (AlphaLlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"),
    "llama3.1-8b-unsloth": (AlphaLlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"),
    "qwen3-32b":           (AlphaQwen2ForCausalLM, Qwen2Config, "unsloth/Qwen3-32B-unsloth-bnb-4bit"),
    "gpt-oss-120b":        (AlphaLlamaForCausalLM, LlamaConfig, "unsloth/gpt-oss-120b-unsloth-bnb-4bit"),
}

Steer_MODELS_DICT = {
    "llama3.1":  (SteerLlamaForCausalLM,  LlamaConfig,  "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5":   (SteerQwen2ForCausalLM,  Qwen2Config,  "Qwen/Qwen2.5-7B-Instruct"),
    "gemma2":    (SteerGemma2ForCausalLM, Gemma2Config, "google/gemma-2-9b-it"),
    # ── NEW ──
    "llama3.3-70b":        (SteerLlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"),
    "llama3.1-8b-unsloth": (SteerLlamaForCausalLM, LlamaConfig, "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"),
    "qwen3-32b":           (SteerQwen2ForCausalLM, Qwen2Config, "unsloth/Qwen3-32B-unsloth-bnb-4bit"),
    "gpt-oss-120b":        (SteerLlamaForCausalLM, LlamaConfig, "unsloth/gpt-oss-120b-unsloth-bnb-4bit"),
}

AlphaSteer_STEERING_LAYERS = {
    "llama3.1": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
    "qwen2.5":  [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20],
    "gemma2":   [6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 22],
    # ── NEW (tỷ lệ ~35-65% tổng layers, vùng middle-upper) ──
    "llama3.3-70b":        [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],  # 80 layers total
    "llama3.1-8b-unsloth": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],            # 32 layers, giống llama3.1
    "qwen3-32b":           [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],  # 64 layers total
    "gpt-oss-120b":        [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],          # 126 layers total
}

AlphaSteer_CALCULATION_CONFIG = {
    "llama3.1": [(8,0.6),(9,0.6),(10,0.6),(11,0.6),(12,0.4),(13,0.5),(14,0.6),(16,0.6),(18,0.6),(19,0.6)],
    "qwen2.5":  [(5,0.6),(6,0.6),(7,0.6),(8,0.6),(9,0.5),(10,0.6),(11,0.5),(12,0.5),(13,0.5),(14,0.3),(15,0.3),(16,0.5),(18,0.5),(19,0.6)],
    "gemma2":   [(6,0.5),(8,0.4),(10,0.6),(11,0.6),(12,0.6),(13,0.6),(14,0.6),(15,0.6),(16,0.6),(18,0.6),(22,0.5)],
    # ── NEW (strength 0.5 mặc định — tune lại sau khi có DSR) ──
    "llama3.3-70b":        [(l, 0.5) for l in [28,30,32,34,36,38,40,42,44,46,48,50]],
    "llama3.1-8b-unsloth": [(8,0.6),(9,0.6),(10,0.6),(11,0.6),(12,0.4),(13,0.5),(14,0.6),(16,0.6),(18,0.6),(19,0.6)],
    "qwen3-32b":           [(l, 0.5) for l in [16,18,20,22,24,26,28,30,32,34,36,38]],
    "gpt-oss-120b":        [(l, 0.5) for l in [40,45,50,55,60,65,70,75,80,85]],
}