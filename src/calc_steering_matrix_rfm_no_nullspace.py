"""
calc_steering_matrix_rfm_no_nullspace.py
=========================================
Phiên bản RFM KHÔNG dùng NullSpace projection.

So với calc_steering_matrix_rfm.py (có NullSpace):
  - BỎ: null_space_projection_l → P_layer
  - BỎ: cal_tilde_delta_with_regularization_l (cần P_layer)
  - THAY: steering_matrix[layer] = refusal_vector ⊗ refusal_vector  (rank-1)
           hoặc dùng trực tiếp cal_steering_matrix_l(I, tilde_delta_direct)

Logic mới tại mỗi layer:
    r  = refusal_vectors[layer]          # (d,)
    # tilde_delta = giải least-squares H_harmful @ M ≈ r (không chiếu null-space)
    # Cách đơn giản nhất: M = rᵀ / ||H_harmful||² * H_harmful  (ridge regression, P=I)
    tilde_delta_layer = cal_tilde_delta_with_regularization_l(
        H_harmful[:, layer, :],
        P=I_d,                    # identity — không chiếu null-space
        refusal_vec=r,
        lambda_reg=...,
    )
    steering_matrix[layer] = cal_steering_matrix_l(I_d, tilde_delta_layer)

Cách dùng:
    python src/calc_steering_matrix_rfm_no_nullspace.py \\
        --model_name llama3.1 \\
        --embedding_dir data/embeddings/llama3.1 \\
        --device cuda \\
        --save_path data/steering_matrix/steering_matrix_llama3.1_rfm_no_nullspace.pt \\
        --rfm_method rfm
"""

import os
import gc
import sys
import time
import logging
import argparse

import numpy as np
import torch

torch.manual_seed(42)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils.const import AlphaSteer_CALCULATION_CONFIG
from utils.steering_utils import (
    cal_tilde_delta_with_regularization_l,
    cal_steering_matrix_l,
)
from rfm_refusal_vector import (
    load_alphasteer_embeddings,
    compute_rfm_refusal_vectors,
    STEERING_LAYERS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",    required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--device",        default="cuda")
    p.add_argument("--save_path",     required=True)
    p.add_argument("--rfm_method",    default="rfm", choices=["linear", "rfm"])
    p.add_argument("--rfm_iters",     type=int,   default=3)
    p.add_argument("--lambda_reg",    type=float, default=10.0)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]
    embeds_dir = args.embedding_dir

    # ── 1. Load embeddings lên CPU (layer-by-layer strategy) ────────────────
    logger.info("Loading benign embeddings to CPU...")
    H_benign_train_10000 = torch.load(
        f"{embeds_dir}/embeds_benign_train.pt", map_location="cpu").float()
    H_coconot_pref = torch.load(
        f"{embeds_dir}/embeds_coconot_pref.pt", map_location="cpu").float()
    H_coconot_original = torch.load(
        f"{embeds_dir}/embeds_coconot_original.pt", map_location="cpu").float()

    indices_borderline = torch.randperm(
        H_coconot_original.size(0))[:4000 - H_coconot_pref.size(0)]
    H_benign_train = torch.cat([
        H_benign_train_10000,
        H_coconot_original[indices_borderline],
        H_coconot_pref,
    ], dim=0)  # kept on CPU

    del H_benign_train_10000, H_coconot_pref, H_coconot_original
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("H_benign_train shape: %s", tuple(H_benign_train.shape))

    logger.info("Loading harmful embeddings to CPU...")
    H_harmful_train_1000 = torch.load(
        f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location="cpu").float()
    H_jailbreak_train_full = torch.load(
        f"{embeds_dir}/embeds_jailbreak_train.pt", map_location="cpu").float()

    indices = torch.randperm(H_jailbreak_train_full.size(0))[:1000]
    H_jailbreak_train = H_jailbreak_train_full[indices]
    H_harmful_train = torch.cat([H_harmful_train_1000, H_jailbreak_train], dim=0)

    del H_harmful_train_1000, H_jailbreak_train_full, H_jailbreak_train
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("H_harmful_train shape: %s", tuple(H_harmful_train.shape))

    # ── 2. Compute RFM refusal vectors ──────────────────────────────────────
    logger.info("Computing RFM refusal vectors (method=%s)...", args.rfm_method)
    H_refusal_cpu, H_compliant_cpu = load_alphasteer_embeddings(embeds_dir)

    num_total_layers = H_benign_train.shape[1]
    d_model          = H_benign_train.shape[2]

    refusal_vectors_np = compute_rfm_refusal_vectors(
        H_refusal=H_refusal_cpu,
        H_compliant=H_compliant_cpu,
        layers=[layer for layer, _ in layers_ratio_list],
        num_total_layers=num_total_layers,
        method=args.rfm_method,
        rfm_iters=args.rfm_iters,
        device=args.device,
    )
    del H_refusal_cpu, H_compliant_cpu
    gc.collect()
    torch.cuda.empty_cache()

    refusal_vectors = torch.tensor(
        refusal_vectors_np, dtype=torch.float32).to(device)
    logger.info("refusal_vectors shape: %s", tuple(refusal_vectors.shape))

    # ── 3. Steering matrix — NO NullSpace projection ─────────────────────────
    # Thay vì P = null_space_projection_l(H_benign), dùng P = Identity.
    # cal_tilde_delta_with_regularization_l(H_harmful, I, r) giải:
    #   min_M  ||H_harmful @ M - r||² + λ||M||²
    # Kết quả: M = (HᵀH + λI)⁻¹ Hᵀ r  (ridge regression thuần)
    # cal_steering_matrix_l(I, tilde_delta) → steering = tilde_delta (no masking)

    steering_matrix = torch.zeros(
        num_total_layers, d_model, d_model, device="cpu")

    I_d = torch.eye(d_model, device=device)  # identity — thay cho P_layer

    for layer, _ in layers_ratio_list:   # ratio bỏ qua vì không cần NullSpace
        logger.info("Processing layer=%d (no nullspace)", layer)

        harmful_layer = H_harmful_train[:, layer, :].to(device)  # (N, d)

        # Ridge regression: no projection, P = Identity
        tilde_delta_layer = cal_tilde_delta_with_regularization_l(
            harmful_layer,
            I_d,
            refusal_vectors[layer],
            lambda_reg=args.lambda_reg,
            device=args.device,
        )
        logger.info("  tilde_delta_norm=%.4f",
                    torch.norm(tilde_delta_layer).item())

        # Steering matrix = I @ tilde_delta = tilde_delta (no nullspace masking)
        steering_matrix_layer = cal_steering_matrix_l(
            I_d, tilde_delta_layer, device=args.device)
        steering_matrix[layer] = steering_matrix_layer.cpu()
        logger.info("  steering_matrix_norm=%.4f",
                    torch.norm(steering_matrix_layer).item())

        del harmful_layer, tilde_delta_layer, steering_matrix_layer
        torch.cuda.empty_cache()

    # ── 4. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(steering_matrix, args.save_path)   # float32
    logger.info("Saved → %s", args.save_path)
    logger.info("Total time: %.1fs", time.time() - t0)

    del H_benign_train, H_harmful_train, steering_matrix, I_d
    gc.collect()
    torch.cuda.empty_cache()