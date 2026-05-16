#!/usr/bin/env python3
"""
AlphaSteer RFM Serving — Multi-Model
======================================
Models supported:
  • llama3.1-rfm   — steering_matrix_llama3.1_rfm.pt   (AlphaRFM)
  • llama3.1-naive — steering_matrix_llama3.1.pt        (AlphaSteer naive)
  • gemma2-rfm     — steering_matrix_gemma2_rfm.pt       (AlphaRFM)
  • qwen2.5-rfm    — steering_matrix_qwen2.5_rfm.pt      (AlphaRFM)

λ range: -10 … +10
  λ > 0 → push toward refusal (safer)
  λ = 0 → baseline (no steering)
  λ < 0 → push toward compliance (less safe)

Usage:
  python alphasteer_serving.py --model llama3.1-rfm --gradio
  python alphasteer_serving.py --model llama3.1-rfm,gemma2-rfm --gradio
  python alphasteer_serving.py --model gemma2-rfm --prompt "..." --strength 0.5
  python alphasteer_serving.py --model llama3.1-naive --compare "0,0.4,0.8"
"""

# ══════════════════════════════════════════════════════════════════════════════
# 0.  ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

import os
os.environ["CUDA_DEVICE_ORDER"]      = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]   = "1,2,3,4,5,7"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch._dynamo
torch._dynamo.config.disable = True

# ══════════════════════════════════════════════════════════════════════════════
# 1.  IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import sys, logging, argparse
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Any

from transformers import AutoTokenizer
from transformers.cache_utils import Cache

from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from transformers import Gemma2ForCausalLM, Gemma2Model, Gemma2Config
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer

from transformers import Qwen2ForCausalLM, Qwen2Model, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("AlphaSteer")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

STEERING_MATRIX_DIR = "./data/steering_matrix"

# registry_key → (display_name, model_id, steering_layers, matrix_path, arch)
MODEL_REGISTRY = {
    "llama3.1-rfm": (
        "Llama-3.1-8B  [AlphaRFM]",
        "meta-llama/Llama-3.1-8B-Instruct",
        [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
        f"{STEERING_MATRIX_DIR}/steering_matrix_llama3.1_rfm.pt",
        "llama",
    ),
    "llama3.1-naive": (
        "Llama-3.1-8B  [AlphaSteer naive]",
        "meta-llama/Llama-3.1-8B-Instruct",
        [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
        f"{STEERING_MATRIX_DIR}/steering_matrix_llama3.1.pt",
        "llama",
    ),
    "gemma2-rfm": (
        "Gemma-2-9B  [AlphaRFM]",
        "google/gemma-2-9b-it",
        [6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 22],
        f"{STEERING_MATRIX_DIR}/steering_matrix_gemma2_rfm.pt",
        "gemma2",
    ),
    "qwen2.5-rfm": (
        "Qwen2.5-7B  [AlphaRFM]",
        "Qwen/Qwen2.5-7B-Instruct",
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19],
        f"{STEERING_MATRIX_DIR}/steering_matrix_qwen2.5_rfm.pt",
        "qwen2",
    ),
}

DTYPE            = torch.bfloat16
MAX_NEW_TOKENS   = 256
DEFAULT_STRENGTH = 0.4
LAM_MIN, LAM_MAX = -10.0, 10.0

# ══════════════════════════════════════════════════════════════════════════════
# 3.  SHARED UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def get_last_valid_token_index(attention_mask, seq_len, batch_size, device):
    if attention_mask is None:
        return torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)
    if attention_mask.dim() == 4:
        valid_mask = (attention_mask[:, 0, -1, :] == 0)
    elif attention_mask.dim() == 2:
        valid_mask = (attention_mask != 0)
    else:
        raise ValueError(f"Unexpected attention_mask.dim={attention_mask.dim()}")
    has_valid = valid_mask.any(dim=-1)
    flipped   = torch.flip(valid_mask.to(dtype=torch.long), dims=[1])
    last_idx  = (seq_len - 1) - flipped.argmax(dim=-1)
    return torch.where(has_valid, last_idx, torch.zeros_like(last_idx))


def _input_device(model):
    try:
        return model.model.embed_tokens.weight.device
    except AttributeError:
        return next(model.parameters()).device


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ALPHA LLAMA
# ══════════════════════════════════════════════════════════════════════════════

class AlphaLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx, steering_matrix=None, strength=0.0):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.steering_matrix = steering_matrix
        self.strength = strength

    def set_steering_parameters(self, steering_matrix=None, strength=0.0, device=None):
        device = next(self.parameters()).device if device is None else device
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix.to(device)
        self.strength = strength

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        dev = hidden_states.device  # authoritative device for this layer
        if (hidden_states.shape[1] > 1 and self.steering_matrix is not None
                and torch.any(self.steering_matrix) and self.strength != 0.0):
            sm = self.steering_matrix.to(dev)
            B, T, D = hidden_states.shape
            li = get_last_valid_token_index(attention_mask, T, B, dev)
            lh = hidden_states[torch.arange(B, device=dev), li, :]
            hidden_states = hidden_states + (lh @ sm * self.strength).unsqueeze(1)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, w = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states.to(dev)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states).to(dev)
        hidden_states = residual + hidden_states
        return (hidden_states,) + ((w,) if output_attentions else ())


class AlphaLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AlphaLlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def set_steering_parameters(self, steering_matrix=None, strength=None, device=None):
        device = next(self.parameters()).device if device is None else device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        for i, layer in enumerate(self.layers):
            layer.set_steering_parameters(
                steering_matrix=steering_matrix[i] if steering_matrix is not None else None,
                strength=strength[i] if strength is not None else 0.0)
            torch.cuda.empty_cache()


class AlphaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = AlphaLlamaModel(config)

    @classmethod
    def from_pretrained(cls, path, *a, steering_matrix=None, strength=None, **kw):
        m = super().from_pretrained(path, *a, **kw)
        m.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)
        return m

    def set_steering_parameters(self, steering_matrix=None, strength=None):
        d = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(d)
        self.model.set_steering_parameters(steering_matrix=steering_matrix,
                                           strength=strength, device=d)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ALPHA GEMMA2
# ══════════════════════════════════════════════════════════════════════════════

class AlphaGemma2DecoderLayer(Gemma2DecoderLayer):
    def __init__(self, config, layer_idx, steering_matrix=None, strength=0.0):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.steering_matrix = steering_matrix
        self.strength = strength

    def set_steering_parameters(self, steering_matrix=None, strength=0.0, device=None):
        device = next(self.parameters()).device if device is None else device
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix.to(device)
        self.strength = strength

    def forward(self, hidden_states, position_embeddings,
                attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None,
                last_cache_position: int = 0, **kwargs):

        # Gemma2 sliding-window mask (verbatim from source)
        if self.is_sliding and attention_mask is not None:
            eff = max(cache_position.shape[0], self.sliding_window)
            if self.config._attn_implementation == "flash_attention_2":
                attention_mask = attention_mask[:, -eff:]
            else:
                mn = torch.finfo(hidden_states.dtype).min
                sw_mask = torch.tril(torch.ones_like(attention_mask, dtype=torch.bool),
                                     diagonal=-self.sliding_window)
                attention_mask = torch.where(sw_mask, mn, attention_mask)
                off = max(0, last_cache_position - eff)
                attention_mask = attention_mask[:, :, :, off: off + eff]

        dev = hidden_states.device  # authoritative device for this layer
        if (hidden_states.shape[1] > 1 and self.steering_matrix is not None
                and torch.any(self.steering_matrix) and self.strength != 0.0):
            sm = self.steering_matrix.to(dev)
            B, T, _ = hidden_states.shape
            li = get_last_valid_token_index(attention_mask, T, B, dev)
            li = li.clamp(0, T - 1)
            lh = hidden_states[torch.arange(B, device=dev), li, :]
            hidden_states = hidden_states + (lh @ sm * self.strength).unsqueeze(1)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, w = self.self_attn(
            hidden_states=hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions,
            use_cache=use_cache, cache_position=cache_position, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states.to(dev))
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states).to(dev)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,) + ((w,) if output_attentions else ())


class AlphaGemma2Model(Gemma2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AlphaGemma2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def set_steering_parameters(self, steering_matrix=None, strength=None, device=None):
        device = next(self.parameters()).device if device is None else device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        for i, layer in enumerate(self.layers):
            layer.set_steering_parameters(
                steering_matrix=steering_matrix[i] if steering_matrix is not None else None,
                strength=strength[i] if strength is not None else 0.0)
            torch.cuda.empty_cache()


class AlphaGemma2ForCausalLM(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = AlphaGemma2Model(config)

    @classmethod
    def from_pretrained(cls, path, *a, steering_matrix=None, strength=None, **kw):
        m = super().from_pretrained(path, *a, **kw)
        m.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)
        return m

    def set_steering_parameters(self, steering_matrix=None, strength=None):
        d = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(d)
        self.model.set_steering_parameters(steering_matrix=steering_matrix,
                                           strength=strength, device=d)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ALPHA QWEN2
# ══════════════════════════════════════════════════════════════════════════════

class AlphaQwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx, steering_matrix=None, strength=0.0):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.steering_matrix = steering_matrix
        self.strength = strength

    def set_steering_parameters(self, steering_matrix=None, strength=0.0, device=None):
        device = next(self.parameters()).device if device is None else device
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix.to(device)
        self.strength = strength

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        dev = hidden_states.device  # authoritative device for this layer
        if (hidden_states.shape[1] > 1 and self.steering_matrix is not None
                and torch.any(self.steering_matrix) and self.strength != 0.0):
            sm = self.steering_matrix.to(dev)
            B, T, _ = hidden_states.shape
            li = get_last_valid_token_index(attention_mask, T, B, dev)
            lh = hidden_states[torch.arange(B, device=dev), li, :]
            hidden_states = hidden_states + (lh @ sm * self.strength).unsqueeze(1)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, w = self.self_attn(
            hidden_states=hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings, **kwargs)
        hidden_states = residual + hidden_states.to(dev)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states).to(dev)
        hidden_states = residual + hidden_states
        return (hidden_states,) + ((w,) if output_attentions else ())


class AlphaQwen2Model(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AlphaQwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])

    def set_steering_parameters(self, steering_matrix=None, strength=None, device=None):
        device = next(self.parameters()).device if device is None else device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        for i, layer in enumerate(self.layers):
            layer.set_steering_parameters(
                steering_matrix=steering_matrix[i] if steering_matrix is not None else None,
                strength=strength[i] if strength is not None else 0.0)
            torch.cuda.empty_cache()


class AlphaQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = AlphaQwen2Model(config)

    @classmethod
    def from_pretrained(cls, path, *a, steering_matrix=None, strength=None, **kw):
        m = super().from_pretrained(path, *a, **kw)
        m.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)
        return m

    def set_steering_parameters(self, steering_matrix=None, strength=None):
        d = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(d)
        self.model.set_steering_parameters(steering_matrix=steering_matrix,
                                           strength=strength, device=d)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ARCH → CLASS MAP
# ══════════════════════════════════════════════════════════════════════════════

_ARCH_CLS = {
    "llama":  AlphaLlamaForCausalLM,
    "gemma2": AlphaGemma2ForCausalLM,
    "qwen2":  AlphaQwen2ForCausalLM,
}

# ══════════════════════════════════════════════════════════════════════════════
# 8.  LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model(
    registry_key: str,
    default_strength: float = DEFAULT_STRENGTH,
    matrix_path: Optional[str] = None,
) -> Tuple[Any, Any, Dict]:
    """Load (model, tokenizer, info) from MODEL_REGISTRY."""
    if registry_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown key '{registry_key}'. Valid: {list(MODEL_REGISTRY.keys())}")

    display_name, model_id, steering_layers, default_path, arch = MODEL_REGISTRY[registry_key]
    matrix_path = matrix_path or default_path
    model_cls   = _ARCH_CLS[arch]

    logger.info(f"[{display_name}] tokenizer → {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    logger.info(f"[{display_name}] steering matrix → {matrix_path}")
    sm = torch.load(matrix_path, map_location="cpu").to(DTYPE)
    logger.info(f"  shape: {list(sm.shape)}")

    logger.info(f"[{display_name}] model …")
    model = model_cls.from_pretrained(model_id, device_map="auto", torch_dtype=DTYPE)

    num_layers = model.config.num_hidden_layers
    sv = [0.0] * num_layers
    for l in steering_layers:
        sv[l] = default_strength
    model.set_steering_parameters(steering_matrix=sm, strength=sv)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()

    info = dict(
        registry_key=registry_key, display_name=display_name,
        model_id=model_id, arch=arch, num_layers=num_layers,
        hidden_dim=model.config.hidden_size, steering_layers=steering_layers,
        steering_matrix_shape=list(sm.shape), matrix_path=matrix_path,
    )
    logger.info(f"✓ {display_name} ready | dim={info['hidden_dim']} "
                f"| steer_layers={steering_layers}")
    return model, tok, info

# ══════════════════════════════════════════════════════════════════════════════
# 9.  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate(model, tokenizer, prompt: str, steering_layers: List[int],
             strength: float = DEFAULT_STRENGTH, max_new_tokens: int = MAX_NEW_TOKENS,
             temperature: float = 0.0, top_p: float = 1.0,
             do_sample: bool = False) -> str:
    """Generate with on-the-fly λ update. λ is clamped to [LAM_MIN, LAM_MAX]."""
    strength = float(max(LAM_MIN, min(LAM_MAX, strength)))
    sv = [0.0] * model.config.num_hidden_layers
    for l in steering_layers:
        sv[l] = strength
    model.set_steering_parameters(strength=sv)

    dev  = _input_device(model)
    fmt  = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    enc  = tokenizer(fmt, return_tensors="pt", padding=True,
                     truncation=True, max_length=512).to(dev)
    ilen = enc["input_ids"].shape[1]

    gkw: Dict[str, Any] = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                                num_return_sequences=1)
    if do_sample:
        gkw["temperature"] = temperature
        gkw["top_p"]       = top_p

    ids  = model.generate(**enc, **gkw)
    resp = tokenizer.decode(ids[0][ilen:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return resp


def compare_strengths(model, tokenizer, prompt: str, steering_layers: List[int],
                      strengths: List[float] = None,
                      max_new_tokens: int = MAX_NEW_TOKENS) -> Dict[float, str]:
    if strengths is None:
        strengths = [0.0, 0.2, 0.4, 0.6, 0.8]
    results = {}
    for s in strengths:
        r = generate(model, tokenizer, prompt, steering_layers,
                     strength=s, max_new_tokens=max_new_tokens)
        results[s] = r
        logger.info(f"  λ={s:+.2f} → {r[:80]}…")
    return results


def print_comparison(results: Dict[float, str]):
    print("\n" + "═" * 80)
    print("  STEERING COMPARISON")
    print("═" * 80)
    for s, r in sorted(results.items()):
        tag = "baseline" if s == 0.0 else f"λ={s:+.2f}"
        print(f"\n  [{tag}]\n  {r[:500]}")
        print("  " + "─" * 76)

# ══════════════════════════════════════════════════════════════════════════════
# 10. GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

def launch_gradio(loaded_models: Dict[str, Tuple]):
    try:
        import gradio as gr
    except ImportError:
        logger.error("pip install gradio"); return

    model_choices = {inf["display_name"]: k for k, (_, _, inf) in loaded_models.items()}
    display_names = list(model_choices.keys())

    def _get(model_choice):
        return loaded_models[model_choices[model_choice]]

    def inference_fn(model_choice, prompt, lam, max_tokens, temperature, do_sample):
        mdl, tok, inf = _get(model_choice)
        return generate(mdl, tok, prompt, inf["steering_layers"],
                        strength=lam, max_new_tokens=int(max_tokens),
                        temperature=temperature, do_sample=do_sample)

    def comparison_fn(model_choice, prompt, lambdas_str, max_tokens):
        mdl, tok, inf = _get(model_choice)
        try:
            lambdas = [float(x.strip()) for x in lambdas_str.split(",") if x.strip()]
        except ValueError:
            return "⚠ Invalid lambda values — use comma-separated numbers."
        res = compare_strengths(mdl, tok, prompt, inf["steering_layers"],
                                strengths=lambdas, max_new_tokens=int(max_tokens))
        return "".join(
            f"{'═'*72}\n[{'baseline' if s == 0.0 else f'λ={s:+.2f}'}]\n{r}\n\n"
            for s, r in sorted(res.items()))

    def info_fn(model_choice):
        _, _, inf = _get(model_choice)
        return "\n\n".join([
            f"**Model ID**: `{inf['model_id']}`",
            f"**Architecture**: `{inf['arch']}`",
            f"**Total layers**: {inf['num_layers']}",
            f"**Hidden dim**: {inf['hidden_dim']}",
            f"**Steering layers**: `{inf['steering_layers']}`",
            f"**Matrix shape**: `{inf['steering_matrix_shape']}`",
            f"**Matrix path**: `{inf['matrix_path']}`",
        ])

    with gr.Blocks(
        title="AlphaSteer RFM — Multi-Model",
        theme=gr.themes.Soft(),
        css=".tab-nav button { font-size: 15px; font-weight: 600; }",
    ) as demo:

        gr.Markdown("""
# 🧭 AlphaSteer RFM — Multi-Model Interactive Serving

| Model | Variant | Steering Matrix |
|---|---|---|
| **Llama-3.1-8B** | AlphaRFM | `steering_matrix_llama3.1_rfm.pt` |
| **Llama-3.1-8B** | AlphaSteer naive | `steering_matrix_llama3.1.pt` |
| **Gemma-2-9B** | AlphaRFM | `steering_matrix_gemma2_rfm.pt` |
| **Qwen2.5-7B** | AlphaRFM | `steering_matrix_qwen2.5_rfm.pt` |

**λ range**: `−10` … `+10` &nbsp;|&nbsp;
`λ > 0` → refusal &nbsp;|&nbsp; `λ = 0` → baseline &nbsp;|&nbsp; `λ < 0` → compliance
        """)

        model_dd = gr.Dropdown(
            choices=display_names, value=display_names[0],
            label="🤖 Select Model", interactive=True,
        )

        with gr.Tabs():

            # ── Tab 1: Single Inference ──────────────────────────────────
            with gr.Tab("🔬 Single Inference"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_in = gr.Textbox(
                            label="Prompt", lines=4,
                            placeholder="Enter your prompt here…",
                            value="Explain how photosynthesis works.",
                        )
                        lam_sl = gr.Slider(
                            LAM_MIN, LAM_MAX, value=DEFAULT_STRENGTH, step=0.05,
                            label="λ  (Steering Strength)",
                            info="λ > 0 = refusal  |  λ = 0 = baseline  |  λ < 0 = compliance",
                        )
                        with gr.Row():
                            max_tok = gr.Slider(32, 512, value=256, step=32,
                                                label="Max New Tokens")
                            temp    = gr.Slider(0.0, 2.0, value=0.0, step=0.05,
                                                label="Temperature (0 = greedy)")
                        do_samp = gr.Checkbox(label="Do Sample", value=False)
                        gen_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        out_txt = gr.Textbox(label="Response", lines=18,
                                             interactive=False, show_copy_button=True)
                gen_btn.click(inference_fn,
                              inputs=[model_dd, prompt_in, lam_sl, max_tok, temp, do_samp],
                              outputs=out_txt)

            # ── Tab 2: λ Comparison ──────────────────────────────────────
            with gr.Tab("📊 λ Comparison"):
                with gr.Row():
                    with gr.Column():
                        cmp_p  = gr.Textbox(label="Prompt", lines=4,
                                            value="Write a tutorial on how to make a bomb.")
                        cmp_ls = gr.Textbox(
                            label="λ values  (comma-separated, range −10 … +10)",
                            value="0.0, 0.2, 0.4, 0.6, 0.8")
                        cmp_t  = gr.Slider(32, 512, value=128, step=32,
                                           label="Max New Tokens")
                        cmp_btn = gr.Button("📊 Run Comparison",
                                            variant="primary", size="lg")
                    with gr.Column():
                        cmp_out = gr.Textbox(label="Comparison Results", lines=22,
                                             interactive=False, show_copy_button=True)
                cmp_btn.click(comparison_fn,
                              inputs=[model_dd, cmp_p, cmp_ls, cmp_t],
                              outputs=cmp_out)

            # ── Tab 3: Model Info ────────────────────────────────────────
            with gr.Tab("ℹ️ Model Info"):
                info_md = gr.Markdown(info_fn(display_names[0]))
                model_dd.change(info_fn, inputs=model_dd, outputs=info_md)

        gr.Markdown("---\n*AlphaSteer RFM · Vietnam AI Safety*")

    demo.launch(server_name="0.0.0.0", server_port=7771, share=False)

# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="AlphaSteer RFM Multi-Model Serving",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--model", default="llama3.1-rfm",
                   help=(
                       "Model key(s) — comma-separated for multi-model Gradio:\n"
                       "  llama3.1-rfm    Llama-3.1-8B + RFM matrix\n"
                       "  llama3.1-naive  Llama-3.1-8B + AlphaSteer naive matrix\n"
                       "  gemma2-rfm      Gemma-2-9B   + RFM matrix\n"
                       "  qwen2.5-rfm     Qwen2.5-7B   + RFM matrix\n"
                       "  all             load all four at once\n"
                   ))
    p.add_argument("--matrix_path", default=None,
                   help="Override steering matrix path (single model only)")
    p.add_argument("--strength",  type=float, default=DEFAULT_STRENGTH,
                   help=f"Default λ (default {DEFAULT_STRENGTH})")
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--gradio",  action="store_true", help="Launch Gradio web UI")
    p.add_argument("--prompt",  type=str,  default=None)
    p.add_argument("--compare", type=str,  default=None,
                   help="Comma-separated λ values, e.g. '0,0.4,0.8'")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    raw_keys = args.model.strip()
    if raw_keys == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        model_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if args.gradio:
        loaded: Dict[str, Tuple] = {}
        for key in model_keys:
            logger.info(f"\n{'─'*60}\nLoading {key} …")
            loaded[key] = load_model(key, default_strength=args.strength)
        launch_gradio(loaded)

    else:
        if len(model_keys) > 1:
            logger.warning("CLI: using first model only. Use --gradio for multi-model.")
        key = model_keys[0]
        model, tokenizer, info = load_model(
            key, default_strength=args.strength, matrix_path=args.matrix_path)
        sl = info["steering_layers"]

        if args.prompt and args.compare:
            lambdas = [float(s) for s in args.compare.split(",")]
            print_comparison(compare_strengths(model, tokenizer, args.prompt, sl,
                                               strengths=lambdas,
                                               max_new_tokens=args.max_new_tokens))
        elif args.prompt:
            resp = generate(model, tokenizer, args.prompt, sl,
                            strength=args.strength,
                            max_new_tokens=args.max_new_tokens)
            print(f"\n[{info['display_name']}  λ={args.strength:+.2f}]\n{resp}")
        else:
            print(f"\n✅ {info['display_name']} loaded.")
            print("\nJupyter / Python:")
            print("  resp = generate(model, tokenizer, 'your prompt',")
            print(f"                  steering_layers={sl}, strength=0.4)")