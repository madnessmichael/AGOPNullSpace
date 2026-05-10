import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # Using GPU 1
import pandas as pd
import torch
import numpy as np
import time
import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

torch.manual_seed(42)
np.random.seed(42)

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

class EmbeddingExtractor:
    def __init__(self, model_name_or_path, device=None):
        self.device = device if device is not None \
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {model_name_or_path}")

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.num_layers = self.config.num_hidden_layers

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── bnb-4bit: dùng BitsAndBytesConfig + device_map="auto" ──────────────
        if "bnb-4bit" in model_name_or_path:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",          # multi-GPU cho 70B/120B
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=self.device,
                torch_dtype=torch.float16,
            )

        # ── input_device: GPU đầu tiên chứa embedding layer ────────────────────
        # Dùng next(parameters()).device thay vì self.device
        # vì khi device_map="auto", self.device có thể không khớp với GPU thật
        self.input_device = next(self.model.parameters()).device

        logger.info(f"Model loaded successfully, input_device={self.input_device}")
        logger.info(f"Number of layers: {self.num_layers}")

    def extract_embeddings(self, prompts, batch_size, layers):
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [message], tokenize=False, add_generation_prompt=True
            )
            for message in messages
        ]

        resid_pre_cache = {i: [] for i in layers}

        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = formatted_prompts[i:i + batch_size]
            batch_inputs = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.input_device)  # ← FIX: dùng input_device, không phải self.device

            with torch.no_grad():
                outputs = self.model(**batch_inputs, output_hidden_states=True)

            for layer_idx in layers:
                resid_pre_cache[layer_idx].append(
                    outputs.hidden_states[layer_idx][:, -1, :].detach().to("cpu")
                )

            outputs = None
            torch.cuda.empty_cache()

        resid_pre_benign_embs = {
            layer: torch.cat(resid_pre_cache[layer], dim=0)
            for layer in layers
        }
        logger.info(
            f"resid_pre_benign_embs[{layers[0]}].shape: {resid_pre_benign_embs[layers[0]].shape}"
        )

        H = torch.stack(list(resid_pre_benign_embs.values()), dim=1)
        logger.info(f"H's shape: {H.shape}")
        return H
    # def __init__(self, model_name_or_path, device=None):
    #     """
    #     Initialize the embedding extractor with a model.
    #
    #     Parameters:
    #     - model_name_or_path: The name or path of the model to load
    #     """
    #     self.device = device if device is not None \
    #         else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     logger.info(f"Loading model from {model_name_or_path}")
    #
    #     self.config = AutoConfig.from_pretrained(model_name_or_path)
    #     self.num_layers = self.config.num_hidden_layers
    #
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #     self.tokenizer.padding_side = "left"
    #     self.tokenizer.pad_token = self.tokenizer.eos_token
    #
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         model_name_or_path,
    #         device_map="auto",
    #         torch_dtype=torch.float16,
    #     )
    #     self.input_device = next(self.model.parameters()).device
    #     logger.info(f"Model loaded successfully to {self.device}")
    #     logger.info(f"Number of layers: {self.num_layers}")
    # def extract_embeddings(self, prompts, batch_size, layers):
    #     """
    #     Extract embeddings from the model for given prompts.
    #
    #     Parameters:
    #     - prompts: List of prompts to extract embeddings from
    #     - batch_size: Batch size for processing
    #     - layers: List of layer indices to extract embeddings from
    #
    #     Returns:
    #     - H: Tensor of shape [num_prompts, num_layers, hidden_dim]
    #     """
    #     messages = [{"role": "user", "content": prompt} for prompt in prompts]
    #     formatted_prompts = [
    #         self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    #         for message in messages
    #     ]
    #
    #     resid_pre_cache = {i: [] for i in layers}
    #
    #     for i in tqdm(range(0, len(prompts), batch_size)):
    #
    #         batch_prompts = formatted_prompts[i:i+batch_size]
    #         batch_inputs = self.tokenizer(
    #             batch_prompts,
    #             padding=True,
    #             truncation=True,
    #             return_tensors="pt"
    #         ).to(self.input_device)
    #
    #         with torch.no_grad():
    #             outputs = self.model(**batch_inputs, output_hidden_states=True)
    #
    #         for layer_idx in layers:
    #             resid_pre_cache[layer_idx].append(
    #                 outputs.hidden_states[layer_idx][:, -1, :].detach().to('cpu'))
    #
    #         outputs = None
    #         torch.cuda.empty_cache()
    #
    #     resid_pre_benign_embs = {
    #         layer: torch.cat(resid_pre_cache[layer], dim=0)
    #         for layer in layers}
    #     logger.info(f"resid_pre_benign_embs[{layers[0]}].shape: {resid_pre_benign_embs[layers[0]].shape}")
    #
    #     H = torch.stack(list(resid_pre_benign_embs.values()), dim=1)
    #     logger.info(f"H's shape: {H.shape}")
    #     return H