"""
calc_steering_matrix_rfm.py  (v4 — fixed data balance)
=======================================================
Drop-in replacement cho src/calc_steering_matrix.py của AlphaSteer.

THAY ĐỔI SO VỚI v2:
  - Dùng load_balanced_embeddings() thay vì load_alphasteer_embeddings()
    → Fix imbalanced 2000 harmful vs 14000 harmless của v2
    → Downsample harmless về đúng N_harmful (1:1 balance)

  - Thêm args --n_jailbreak và --balance_ratio để control balance

Mọi logic khác (null-space P̂, regression Hm, steering matrix) giống gốc.

Cách dùng:
    python calc_steering_matrix_rfm_v4.py \\
        --model_name llama3.1 \\
        --embedding_dir data/embeddings/llama3.1 \\
        --device cuda \\
        --save_path data/steering_matrix/steering_matrix_llama3.1_rfm_v4.pt \\
        --rfm_method linear \\
        --balance_ratio 1.0
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
    null_space_projection_l,
    cal_tilde_delta_with_regularization_l,
    cal_steering_matrix_l,
)

from rfm_refusal_vector_v4 import (
    load_balanced_embeddings,       # FIX: balanced thay vì imbalanced
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
    p.add_argument("--model_name",     required=True)
    p.add_argument("--embedding_dir",  required=True)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--save_path",      required=True)
    p.add_argument("--rfm_method",     default="rfm", choices=["linear", "rfm"],
                   help="rfm = full RFM-AGOP (recommended); linear = logistic probe (ablation)")
    p.add_argument("--rfm_iters",      type=int,   default=3)
    p.add_argument("--lambda_reg",     type=float, default=10.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--n_jailbreak",    type=int,   default=1000,
                   help="Số jailbreak samples cho harmful side")
    p.add_argument("--balance_ratio",  type=float, default=1.0,
                   help="Tỉ lệ harmless/harmful (1.0 = 1:1)")
    return p.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]
    embeds_dir = args.embedding_dir

    # ── 1. Load benign embeddings cho null-space P̂ (GIỐNG GỐC) ─────────────
    # Null-space dùng TOÀN BỘ benign data (~14000) — không balance ở đây
    # vì P̂ cần cover rộng nhất có thể activation space của benign prompts
    logger.info("Loading benign embeddings for null-space P̂ (full, unbalanced)...")
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
    ], dim=0)

    del H_benign_train_10000, H_coconot_pref, H_coconot_original
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("H_benign_train (null-space): %s", tuple(H_benign_train.shape))

    # ── 2. Load harmful embeddings cho Hm regression (GIỐNG GỐC) ────────────
    # Hm dùng TOÀN BỘ harmful data (~2000) — không cần balance ở đây
    # vì regression cần coverage rộng của malicious activation space
    logger.info("Loading harmful embeddings for regression Hm (full)...")
    H_harmful_train_1000 = torch.load(
        f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location="cpu").float()
    H_jailbreak_train_full = torch.load(
        f"{embeds_dir}/embeds_jailbreak_train.pt", map_location="cpu").float()

    indices = torch.randperm(H_jailbreak_train_full.size(0))[:args.n_jailbreak]
    H_jailbreak_train = H_jailbreak_train_full[indices]
    H_harmful_train = torch.cat([H_harmful_train_1000, H_jailbreak_train], dim=0)

    del H_harmful_train_1000, H_jailbreak_train_full, H_jailbreak_train
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("H_harmful_train (regression Hm): %s", tuple(H_harmful_train.shape))

    # ── 3. Load BALANCED harmful/harmless cho RFM refusal direction (FIX) ────
    #
    # ĐÂY LÀ ĐIỂM THAY ĐỔI CHÍNH SO VỚI v2:
    #
    # v2 dùng load_alphasteer_embeddings():
    #   H_refusal:   2000 harmful  (imbalanced)
    #   H_compliant: ~14000 benign (imbalanced)
    #   → DIM/RFM học direction phân tách DOMAIN thay vì REFUSAL concept
    #
    # v4 dùng load_balanced_embeddings():
    #   H_harmful:  N harmful samples
    #   H_harmless: N harmless samples  ← downsample về cùng N
    #   → DIM/RFM học direction phân tách HARMFUL vs HARMLESS đúng nghĩa
    #
    # Tại sao dùng data riêng cho refusal vector (bước 3) thay vì
    # reuse data từ bước 1 và 2?
    #   - Bước 1 (null-space): cần NHIỀU benign để cover activation space rộng
    #   - Bước 2 (regression): cần NHIỀU harmful để cover malicious space
    #   - Bước 3 (refusal direction): cần BALANCE để không bị dominated
    #   → Ba mục đích khác nhau → data strategy khác nhau
    #
    logger.info(
        "Loading BALANCED harmful/harmless for RFM refusal direction "
        "(balance_ratio=%.1f)...", args.balance_ratio
    )
    H_rfm_harmful, H_rfm_harmless = load_balanced_embeddings(
        embedding_dir=embeds_dir,
        n_jailbreak=args.n_jailbreak,
        balance_ratio=args.balance_ratio,
        seed=args.seed,
    )
    logger.info(
        "RFM data — harmful: %d, harmless: %d",
        H_rfm_harmful.shape[0], H_rfm_harmless.shape[0],
    )

    num_total_layers = H_benign_train.shape[1]
    d_model          = H_benign_train.shape[2]

    # ── 4. Compute RFM refusal vectors từ balanced data ──────────────────────
    logger.info("Computing RFM refusal vectors (method=%s)...", args.rfm_method)
    refusal_vectors_np = compute_rfm_refusal_vectors(
        H_harmful=H_rfm_harmful,
        H_harmless=H_rfm_harmless,
        layers=[layer for layer, _ in layers_ratio_list],
        num_total_layers=num_total_layers,
        method=args.rfm_method,
        rfm_iters=args.rfm_iters,
        device=args.device,
    )
    del H_rfm_harmful, H_rfm_harmless
    gc.collect()
    torch.cuda.empty_cache()

    refusal_vectors = torch.tensor(
        refusal_vectors_np, dtype=torch.float32).to(device)
    logger.info(
        "refusal_vectors: %s  dtype=%s",
        tuple(refusal_vectors.shape), refusal_vectors.dtype,
    )

    # ── 5. Null-space + regression + steering matrix (GIỐNG GỐC) ─────────────
    P               = torch.zeros(num_total_layers, d_model, d_model, device="cpu")
    tilde_delta     = torch.zeros(num_total_layers, d_model, d_model, device="cpu")
    steering_matrix = torch.zeros(num_total_layers, d_model, d_model, device="cpu")

    for layer, ratio in layers_ratio_list:
        logger.info("layer=%d  null_ratio=%.1f", layer, ratio)

        # Null-space từ FULL benign data (không balance — đúng theo gốc)
        P_layer = null_space_projection_l(
            H_benign_train[:, layer, :].to(args.device),
            abs_nullspace_ratio=ratio,
        )
        P[layer] = P_layer
        logger.info("  P_norm=%.4f", torch.norm(P_layer).item())

        # Regression: map FULL harmful Hm qua P̂ → refusal direction r (RFM)
        tilde_delta_layer = cal_tilde_delta_with_regularization_l(
            H_harmful_train[:, layer, :].to(args.device),
            P_layer,
            refusal_vectors[layer],
            lambda_reg=args.lambda_reg,
            device=args.device,
        )
        tilde_delta[layer] = tilde_delta_layer
        logger.info("  tilde_delta_norm=%.4f", torch.norm(tilde_delta_layer).item())

        steering_matrix_layer = cal_steering_matrix_l(
            P_layer, tilde_delta_layer, device=args.device)
        steering_matrix[layer] = steering_matrix_layer
        logger.info(
            "  steering_matrix_norm=%.4f",
            torch.norm(steering_matrix_layer).item(),
        )

        torch.cuda.empty_cache()

    # ── 6. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save(steering_matrix, args.save_path)
    logger.info("Saved → %s", args.save_path)
    logger.info("Total time: %.1fs", time.time() - t0)

    # Cleanup
    H_benign_train  = None
    H_harmful_train = None
    P               = None
    tilde_delta     = None
    steering_matrix = None
    gc.collect()
    torch.cuda.empty_cache()