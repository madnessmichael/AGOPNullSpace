import os
import logging
import datetime

import torch.nn as nn
import torch
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Add these imports
from typing import Optional, Tuple, Union, List, Dict
from transformers.cache_utils import Cache
from utils.mask_utils import get_last_valid_token_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Only CausalLM can be called from outside
__all__ = [
    'AlphaLlamaForCausalLM',
    ]


class AlphaLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig,
                 layer_idx: int,
                 steering_matrix: Optional[torch.Tensor] = None,
                 strength: float = 0.0
                 ):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

        # GIỮ NGUYÊN thiết bị gốc, không ép buộc bằng next(parameters()) ở hàm dựng
        self.steering_matrix = steering_matrix
        self.strength = strength

    def set_steering_parameters(self,
        steering_matrix: Optional[torch.Tensor]=None,
        strength: float = 0.0,
        device: Optional[torch.device]=None):

        # Loại bỏ bẫy ép device sớm tại đây để thích ứng linh hoạt với Accelerate
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix
        self.strength = strength

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        should_apply_steering = (
                hidden_states.shape[1] > 1
                and self.steering_matrix is not None
                and torch.any(self.steering_matrix)
                and self.strength != 0.0
        )

        if should_apply_steering:
            dev = hidden_states.device
            steer_matrix_dev = self.steering_matrix.to(dev)

            B, T, D = hidden_states.shape

            last_idx = get_last_valid_token_index(
                attention_mask=attention_mask,
                seq_len=T,
                batch_size=B,
                device=dev,
            )

            batch_idx = torch.arange(B, device=dev)
            last_hidden = hidden_states[batch_idx, last_idx, :]

            steering_vector = last_hidden @ steer_matrix_dev * self.strength

            steering_vector = steering_vector.unsqueeze(1)
            hidden_states = hidden_states + steering_vector

        residual = hidden_states  # resid_pre

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # FIX: Ép residual về đúng GPU của hidden_states sau khi đi qua Attention
        hidden_states = residual.to(hidden_states.device) + hidden_states
        residual = hidden_states  # resid_mid

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # FIX: Ép residual về đúng GPU của hidden_states sau khi đi qua MLP
        hidden_states = residual.to(hidden_states.device) + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

    def fnn_output(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states  # resid_pre

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # FIX: Ép residual về đúng thiết bị
        hidden_states = residual.to(hidden_states.device) + hidden_states
        # Ở fnn_output chúng ta chỉ cần trả về output của mlp nên không cộng residual nữa

        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        return mlp_output

class AlphaLlamaModel(LlamaModel):

    _no_split_modules = ["AlphaLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # Replace the layers with our custom Alpha layers, keeping everything else unchanged
        self.layers = nn.ModuleList(
            [AlphaLlamaDecoderLayer(
                config=config,
                layer_idx=layer_idx,
            )
                for layer_idx in range(config.num_hidden_layers)]
        )

    def set_steering_parameters(
        self,
        steering_matrix: Optional[torch.Tensor]=None,
        strength: Optional[list[float]] = None,
        device: Optional[torch.device] = None):

        # Không tự ý ép .to(device) cho toàn bộ khối ma trận lớn ở cấp độ Model toàn cục
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_matrix = None
            if steering_matrix is not None:
                layer_steering_matrix = steering_matrix[layer_idx]

            layer.set_steering_parameters(
                steering_matrix=layer_steering_matrix,
                strength=strength[layer_idx] if strength is not None else 0.0
            )

        self.print_steering_parameters()

    def print_steering_parameters(self):
        logger.info("Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<20}{'Steering Matrix (First Element)'}")
        logger.info("="*60)
        for layer_idx, layer in enumerate(self.layers):
            strength_val = str(layer.strength)
            if layer.steering_matrix is not None:
                # Đọc phần tử đầu an toàn bất kể ma trận đang ở CPU hay GPU
                steering_matrix_str = f"{layer.steering_matrix[0, 0].item():.4f} ({layer.steering_matrix.device})"
            else:
                steering_matrix_str = "None"
            logger.info(f"{layer_idx:<10}{strength_val:<20}{steering_matrix_str}")


class AlphaLlamaForCausalLM(LlamaForCausalLM):
    _no_split_modules = ["AlphaLlamaDecoderLayer"]

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = AlphaLlamaModel(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        steering_matrix: Optional[torch.Tensor] = None,
                        strength: Optional[list[float]] = None,
                        **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)
        return model

    def set_steering_parameters(
            self,
            steering_matrix: Optional[torch.Tensor]=None,
            strength: Optional[list[float]] = None):

        # Giữ nguyên luồng dữ liệu ma trận truyền xuống mà không can thiệp ép cứng GPU tại đây
        self.model.set_steering_parameters(
            steering_matrix=steering_matrix,
            strength=strength,
            device=None
        )