import os
import logging
import datetime

import torch.nn as nn
import torch
from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Add these imports
from typing import Optional, Tuple, Union, List, Dict #, Unpack
from transformers.cache_utils import Cache
# from transformers.models.llama.modeling_llama import FlashAttentionKwargs
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
        # self.steering_vector = None
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            self.steering_matrix = steering_matrix.to(device)
        else:
            self.steering_matrix = None
        self.strength = strength
        
    def set_steering_parameters(self, 
        steering_matrix: Optional[torch.Tensor]=None, 
        strength: float = 0.0,
        device: Optional[torch.device]=None):
        
        device = next(self.parameters()).device if device is None else device
        
        if steering_matrix is not None and torch.any(steering_matrix):
            self.steering_matrix = steering_matrix.to(device)
        self.strength = strength
        # self.steering_vector = None

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
        # **kwargs: Unpack[FlashAttentionKwargs],
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Use attention_mask to find the last non-PAD token for each sample,
        then compute the steering_vector based on the hidden state at that position.
        This approach is robust to both left and right padding.
        """
        # Ensure steering_matrix is on the same device as hidden_states
        # Only apply steering on initial input
        should_apply_steering = (
            hidden_states.shape[1] > 1
            and self.steering_matrix is not None
            and torch.any(self.steering_matrix)
            and self.strength != 0.0
        )
        
        if should_apply_steering:

            # Only apply steering once during input processing
            if self.steering_matrix.device != hidden_states.device:
                self.steering_matrix = self.steering_matrix.to(hidden_states.device)
            
            B, T, D = hidden_states.shape
            device = hidden_states.device

            # Use the unified function to acquire the index of the last non-pad token
            last_idx = get_last_valid_token_index(
                attention_mask=attention_mask,
                seq_len=T,
                batch_size=B,
                device=device,
            )  # (B,)

            # Construct indices to extract the last valid hidden state from each sample in hidden_states
            # hidden_states: (B, T, D)
            # last_hidden: (B, D)

            batch_idx = torch.arange(B, device=hidden_states.device)
            last_hidden = hidden_states[batch_idx, last_idx, :]  # (B, D)
            steering_vector = last_hidden @ self.steering_matrix * self.strength  # (B, D)

            # Reshape to match hidden_states dimensions and move to the same device
            steering_vector = steering_vector.unsqueeze(1)
            # Apply steering by adding the steering vector to hidden states
            hidden_states = hidden_states + steering_vector
            
        residual = hidden_states # resid_pre - save for residual connection

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
        
        hidden_states = residual + hidden_states
        residual = hidden_states # resid_mid - save after attention residual
        
        # Normalize hidden states after attention, then pass through MLP
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # resid_post - final residual connection
        hidden_states = residual + hidden_states

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
        """
        Forward pass through the layer, but stop at the MLP output (before residual connection).
        """
        residual = hidden_states # resid pre
        
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

        hidden_states = residual + hidden_states
        residual = hidden_states  # Update residual
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        mlp_output = self.mlp(hidden_states)

        return mlp_output

class AlphaLlamaModel(LlamaModel):
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
        device = next(self.parameters()).device if device is None else device
        
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        
        for layer_idx, layer in enumerate(self.layers):
            layer_steering_matrix = None
            if steering_matrix is not None:
                layer_steering_matrix = steering_matrix[layer_idx]
                
            layer.set_steering_parameters(
                steering_matrix=layer_steering_matrix, 
                strength=strength[layer_idx] if strength is not None else 0.0
            )
            torch.cuda.empty_cache()
        
        self.print_steering_parameters()
        
    def print_steering_parameters(self):
        logger.info("Steering Parameters:")
        logger.info(f"{'Layer':<10}{'Strength':<20}{'Steering Matrix (First Element)'}")
        logger.info("="*60)
        for layer_idx, layer in enumerate(self.layers):
            # Ensure strength is a string or formattable type
            strength_val = str(layer.strength)
            
            if layer.steering_matrix is not None:
                steering_matrix_str = layer.steering_matrix[0, 0]
            else:
                steering_matrix_str = "None"
            logger.info(f"{layer_idx:<10}{strength_val:<20}{steering_matrix_str}")
    


class AlphaLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = AlphaLlamaModel(config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, 
                        steering_matrix: Optional[torch.Tensor] = None,
                        strength: Optional[list[float]] = None,
                        **kwargs):
        # Call the parent class's from_pretrained method to load the model
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_steering_parameters(steering_matrix=steering_matrix, strength=strength)
        return model

    def set_steering_parameters(
            self, 
            steering_matrix: Optional[torch.Tensor]=None, 
            strength: Optional[list[float]] = None):
        
        device = next(self.parameters()).device
        if steering_matrix is not None:
            steering_matrix = steering_matrix.to(device)
        self.model.set_steering_parameters(
            steering_matrix=steering_matrix, 
            strength=strength,
            device=device
        )
        