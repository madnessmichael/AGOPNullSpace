"""
calc_steering_matrix_rfm.py  (v5 — harmful_harmless_instructions)
=======================================================
Drop-in replacement cho src/calc_steering_matrix.py của AlphaSteer, dùng activations từ justinphan3110/harmful_harmless_instructions.

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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils.const import AlphaSteer_CALCULATION_CONFIG
from utils.steering_utils import (
    null_space_projection_l,
    cal_tilde_delta_with_regularization_l,
    cal_steering_matrix_l,
)

from rfm_refusal_vector_hh_v5 import (
    DATASET_NAME,
    load_harmful_harmless_instruction_embeddings,
    balance_harmful_harmless_embeddings,
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
    p.add_argument("--embedding_dir",  default=None,
                   help="Optional dir containing HH activation files")
    p.add_argument("--hh_embeddings_path", default=None,
                   help="Combined activation tensor [N, num_layers, d_model] extracted from harmful_harmless_instructions")
    p.add_argument("--hh_labels_path", default=None,
                   help="Labels for combined activations; HF mapping: True=harmless, False=harmful")
    p.add_argument("--hh_harmful_path", default=None,
                   help="Optional pre-separated harmful activation tensor [N_harmful, num_layers, d_model]")
    p.add_argument("--hh_harmless_path", default=None,
                   help="Optional pre-separated harmless activation tensor [N_harmless, num_layers, d_model]")
    p.add_argument("--device",         default="cuda")
    p.add_argument("--save_path",      required=True)
    p.add_argument("--rfm_method",     default="rfm", choices=["linear", "rfm"],
                   help="rfm = full RFM-AGOP; linear = logistic-probe AGOP ablation")
    p.add_argument("--rfm_iters",      type=int,   default=3)
    p.add_argument("--lambda_reg",     type=float, default=10.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--balance_ratio",  type=float, default=1.0,
                   help="harmless/harmful ratio for RFM direction only; 1.0 = balanced")
    return p.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]

    # ── 1. Load harmful_harmless_instructions activations ───────────────────
    # Đây là thay đổi chính so với v4:
    #   OLD data:
    #     null-space: embeds_benign_train + embeds_coconot_*
    #     regression: embeds_harmful_train_1000 + embeds_jailbreak_train
    #     RFM vector: balanced harmful/jailbreak vs benign/coconot
    #
    #   NEW data:
    #     all three components come from justinphan3110/harmful_harmless_instructions.
    #     label=False/0 -> harmful
    #     label=True/1  -> harmless
    #
    # Note: this script expects activations already extracted from the raw HF text.
    # It does not run the LLM forward pass itself.
    logger.info("Loading activations from %s", DATASET_NAME)
    if args.embedding_dir and not (
        args.hh_embeddings_path or args.hh_labels_path or args.hh_harmful_path or args.hh_harmless_path
    ):
        args.hh_embeddings_path = os.path.join(
            args.embedding_dir, "embeds_harmful_harmless_instructions.pt"
        )
        args.hh_labels_path = os.path.join(
            args.embedding_dir, "labels_harmful_harmless_instructions.pt"
        )

    H_harmful_all, H_harmless_all = load_harmful_harmless_instruction_embeddings(
        embeddings_path=args.hh_embeddings_path,
        labels_path=args.hh_labels_path,
        harmful_path=args.hh_harmful_path,
        harmless_path=args.hh_harmless_path,
        balance=False,
        seed=args.seed,
    )

    # ── 2. Data role mapping in AlphaSteer matrix computation ────────────────
    # Null-space P̂: use harmless activations from the HF dataset.
    # Regression Hm: use harmful activations from the HF dataset.
    # RFM direction r: use balanced harmful/harmless activations from the same HF dataset.
    H_benign_train = H_harmless_all
    H_harmful_train = H_harmful_all
    H_rfm_harmful, H_rfm_harmless = balance_harmful_harmless_embeddings(
        H_harmful_all,
        H_harmless_all,
        balance_ratio=args.balance_ratio,
        seed=args.seed,
    )

    logger.info("H_harmless_train/null-space: %s", tuple(H_benign_train.shape))
    logger.info("H_harmful_train/regression:  %s", tuple(H_harmful_train.shape))
    logger.info(
        "RFM data: harmful=%d, harmless=%d, balance_ratio=%.2f",
        H_rfm_harmful.shape[0], H_rfm_harmless.shape[0], args.balance_ratio,
    )

    num_total_layers = H_benign_train.shape[1]
    d_model = H_benign_train.shape[2]

    # ── 3. Compute RFM refusal vectors từ HF harmful/harmless data ───────────
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

    refusal_vectors = torch.tensor(refusal_vectors_np, dtype=torch.float32).to(device)
    logger.info("refusal_vectors: %s dtype=%s", tuple(refusal_vectors.shape), refusal_vectors.dtype)

    # ── 4. Null-space + regression + steering matrix ────────────────────────
    P               = torch.zeros(num_total_layers, d_model, d_model, device="cpu")
    tilde_delta     = torch.zeros(num_total_layers, d_model, d_model, device="cpu")
    steering_matrix = torch.zeros(num_total_layers, d_model, d_model, device="cpu")

    for layer, ratio in layers_ratio_list:
        logger.info("layer=%d  null_ratio=%.1f", layer, ratio)

        P_layer = null_space_projection_l(
            H_benign_train[:, layer, :].to(args.device),
            abs_nullspace_ratio=ratio,
        )
        P[layer] = P_layer
        logger.info("  P_norm=%.4f", torch.norm(P_layer).item())

        tilde_delta_layer = cal_tilde_delta_with_regularization_l(
            H_harmful_train[:, layer, :].to(args.device),
            P_layer,
            refusal_vectors[layer],
            lambda_reg=args.lambda_reg,
            device=args.device,
        )
        tilde_delta[layer] = tilde_delta_layer
        logger.info("  tilde_delta_norm=%.4f", torch.norm(tilde_delta_layer).item())

        steering_matrix_layer = cal_steering_matrix_l(P_layer, tilde_delta_layer, device=args.device)
        steering_matrix[layer] = steering_matrix_layer
        logger.info("  steering_matrix_norm=%.4f", torch.norm(steering_matrix_layer).item())

        torch.cuda.empty_cache()

    # ── 5. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save(steering_matrix, args.save_path)
    logger.info("Saved → %s", args.save_path)
    logger.info("Total time: %.1fs", time.time() - t0)

    H_benign_train = None
    H_harmful_train = None
    P = None
    tilde_delta = None
    steering_matrix = None
    gc.collect()
    torch.cuda.empty_cache()
