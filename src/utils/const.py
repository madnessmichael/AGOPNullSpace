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

AlphaSteer_STEERING_LAYERS = {
    "llama3.1": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
    "qwen2.5": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20],
    "gemma2": [6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 22],
    # ── NEW (tỷ lệ ~35-65% tổng layers, vùng middle-upper) ──
    "llama3.1-70b": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
                     70],
    "llama3.3-70b": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
                     70],
}

AlphaSteer_CALCULATION_CONFIG = {
    "llama3.1": [(8, 0.6), (9, 0.6), (10, 0.6), (11, 0.6), (12, 0.4), (13, 0.5), (14, 0.6), (16, 0.6), (18, 0.6),
                 (19, 0.6)],

    # "llama3.1": [(8, 0.1), (9, 0.1), (10, 0.1), (11, 0.1), (12, 0.1), (13, 0.1), (14, 0.1), (16, 0.1), (18, 0.1),
    #              (19, 0.1)],

    # "llama3.1": [(8, 0.3), (9, 0.3), (10, 0.3), (11, 0.3), (12, 0.3), (13, 0.3), (14, 0.3), (16, 0.3), (18, 0.3),
    #              (19, 0.3)],
    #
    # "llama3.1": [(8, 0.5), (9, 0.5), (10, 0.5), (11, 0.5), (12, 0.5), (13, 0.5), (14, 0.5), (16, 0.5), (18, 0.5),
    #              (19, 0.5)],
    #
    # "llama3.1": [(8, 0.7), (9, 0.7), (10, 0.7), (11, 0.7), (12, 0.7), (13, 0.7), (14, 0.7), (16, 0.7), (18, 0.7),
    #              (19, 0.7)],
    #
    # "llama3.1": [(8, 0.9), (9, 0.9), (10, 0.9), (11, 0.9), (12, 0.9), (13, 0.9), (14, 0.9), (16, 0.9), (18, 0.9),
    #              (19, 0.9)],

    # "llama3.1": [(8, 0.99), (9, 0.99), (10, 0.99), (11, 0.99), (12, 0.99), (13, 0.99), (14, 0.99), (16, 0.99), (18, 0.99),
    #              (19, 0.99)],


    # "llama3.1": [(8, 0.0), (9, 0.0), (10, 0.0), (11, 0.0), (12, 0.0), (13, 0.0), (14, 0.0), (16, 0.0), (18, 0.0),
    #              (19, 0.0)],
    #
    # "llama3.1": [(8, 0.2), (9, 0.2), (10, 0.2), (11, 0.2), (12, 0.2), (13, 0.2), (14, 0.2), (16, 0.2), (18, 0.2),
    #              (19, 0.2)],
    #
    # "llama3.1": [(8, 0.4), (9, 0.4), (10, 0.4), (11, 0.4), (12, 0.4), (13, 0.4), (14, 0.4), (16, 0.4), (18, 0.4),
    #              (19, 0.4)],
    #
    # "llama3.1": [(8, 0.8), (9, 0.8), (10, 0.8), (11, 0.8), (12, 0.8), (13, 0.8), (14, 0.8), (16, 0.8), (18, 0.8),
    #              (19, 0.8)],
    #
    # "llama3.1": [(8, 1.0), (9, 1.0), (10, 1.0), (11, 1.0), (12, 1.0), (13, 1.0), (14, 1.0), (16, 1.0), (18, 1.0),
    #              (19, 1.0)],




    "qwen2.5": [(5, 0.6), (6, 0.6), (7, 0.6), (8, 0.6), (9, 0.5), (10, 0.6), (11, 0.5), (12, 0.5), (13, 0.5), (14, 0.3),
                (15, 0.3), (16, 0.5), (18, 0.5), (19, 0.6)],
    "gemma2": [(6, 0.5), (8, 0.4), (10, 0.6), (11, 0.6), (12, 0.6), (13, 0.6), (14, 0.6), (15, 0.6), (16, 0.6),
               (18, 0.6), (22, 0.5)],
    # ── NEW (strength 0.5 mặc định — tune lại sau khi có DSR) ──
    "llama3.1-70b": [(20, 0.5), (22, 0.5), (24, 0.5), (26, 0.5), (28, 0.5), (30, 0.5), (32, 0.5), (34, 0.5), (36, 0.5),
                     (38, 0.5), (40, 0.5), (42, 0.5), (44, 0.5), (46, 0.5), (48, 0.5), (50, 0.5), (52, 0.5), (54, 0.5),
                     (56, 0.5), (58, 0.5), (60, 0.5), (62, 0.5), (64, 0.5), (66, 0.5), (68, 0.5), (70, 0.5)],
    "llama3.3-70b": [(20, 0.5), (22, 0.5), (24, 0.5), (26, 0.5), (28, 0.5), (30, 0.5), (32, 0.5), (34, 0.5), (36, 0.5),
                     (38, 0.5), (40, 0.5), (42, 0.5), (44, 0.5), (46, 0.5), (48, 0.5), (50, 0.5), (52, 0.5), (54, 0.5),
                     (56, 0.5), (58, 0.5), (60, 0.5), (62, 0.5), (64, 0.5), (66, 0.5), (68, 0.5), (70, 0.5)],
}
