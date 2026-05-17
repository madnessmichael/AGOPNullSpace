"""
calc_steering_matrix_rfm.py  (v2 — fixed)
==========================================
Drop-in replacement cho src/calc_steering_matrix.py của AlphaSteer.
Thay đổi DUY NHẤT: compute refusal vector bằng RFM thay vì load pkl DIM.
Mọi logic khác (null-space, regression, matrix) giống hệt gốc dòng 3894-3982.

BUG ĐÃ FIX so với v1:
  C1. v1 load embeddings sang CPU rồi gọi cal_P với wrapper phức tạp.
      Gốc load thẳng sang device (map_location=device) và gọi trực tiếp.
      → Fixed: load to device như gốc, gọi null_space_projection_l trực tiếp.
  C2. v1 dùng cal_tilde_delta_with_regularization (batch version).
      Gốc dùng cal_tilde_delta_with_regularization_l (per-layer, dùng trong loop).
      → Fixed: dùng per-layer call giống gốc để nhất quán.
  C3. v1 save .to(torch.bfloat16). Gốc save full float32.
      → Fixed: save float32 như gốc dòng 3976.
  C4. v1 import null_space_projection_l trong vòng for loop.
      → Fixed: import một lần ở đầu file.
  C5. v1 tạo P tensor [num_layers,d,d] với zeros rồi điền — logic đúng
      nhưng tốn VRAM hơn. Gốc giữ P_layer riêng lẻ.
      → Fixed: theo đúng pattern của gốc (P tensor initialize rồi fill).
  C6. v1 thiếu torch.manual_seed.
      → Fixed: thêm seed.
  C7. v1 thiếu cleanup memory (H_benign_train = None; torch.cuda.empty_cache()).
      → Fixed: thêm cleanup sau mỗi bước như gốc dòng 3980-3982.

Cách dùng:
    python src/calc_steering_matrix_rfm.py \\
        --model_name llama3.1 \\
        --embedding_dir data/embeddings/llama3.1 \\
        --device cuda \\
        --save_path data/steering_matrix/steering_matrix_llama3.1_rfm.pt \\
        --rfm_method linear
"""

import os
import glob

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,7"  # Using GPU 1
import argparse
import sys
import argparse
import logging
import time

import torch
torch.manual_seed(42)  # FIX C6

import pickle
import numpy as np

# AlphaSteer src phải trong path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils.const import AlphaSteer_CALCULATION_CONFIG
from utils.steering_utils import (
    null_space_projection_l,            # FIX C4: import ở đây, không trong loop
    cal_tilde_delta_with_regularization_l,  # FIX C2: per-layer version
    cal_steering_matrix_l,
)

# rfm_refusal_vector.py phải cùng thư mục hoặc trong PYTHONPATH
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
    p.add_argument("--model_name",     required=True)
    p.add_argument("--embedding_dir",  required=True)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--save_path",      required=True)
    p.add_argument("--rfm_method",     default="linear", choices=["linear", "rfm"])
    p.add_argument("--rfm_iters",      type=int, default=3)
    p.add_argument("--lambda_reg",     type=float, default=10.0)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]  # [(layer, ratio), ...]
    embeds_dir = args.embedding_dir

    # ── 1. Load embeddings — giống gốc dòng 3905-3929 ───────────────────────
    # FIX C1: load trực tiếp sang device như gốc (map_location=device)
    logger.info("Loading embeddings...")

    H_benign_train_10000 = torch.load(
        f"{embeds_dir}/embeds_benign_train.pt", map_location=device).float()
    H_coconot_pref = torch.load(
        f"{embeds_dir}/embeds_coconot_pref.pt", map_location=device).float()
    H_coconot_original = torch.load(
        f"{embeds_dir}/embeds_coconot_original.pt", map_location=device).float()

    indices_borderline = torch.randperm(H_coconot_original.size(0))[:4000 - H_coconot_pref.size(0)]
    H_benign_train = torch.cat([
        H_benign_train_10000,
        H_coconot_original[indices_borderline],
        H_coconot_pref
    ], dim=0).to(device)

    logger.info("H_benign_train shape: %s", tuple(H_benign_train.shape))
    torch.cuda.empty_cache()

    H_harmful_train_1000 = torch.load(
        f"{embeds_dir}/embeds_harmful_train_1000.pt", map_location="cpu").float()
    H_jailbreak_train_full = torch.load(
        f"{embeds_dir}/embeds_jailbreak_train.pt", map_location="cpu").float()

    indices = torch.randperm(H_jailbreak_train_full.size(0))[:1000]
    H_jailbreak_train = H_jailbreak_train_full[indices]
    H_harmful_train = torch.cat([H_harmful_train_1000, H_jailbreak_train], dim=0)

    # Cleanup như gốc dòng 3928-3929
    H_harmful_train_1000 = None
    H_jailbreak_train_full = None
    torch.cuda.empty_cache()
    logger.info("H_harmful_train shape: %s", tuple(H_harmful_train.shape))

    # ── 2. Compute RFM refusal vectors ──────────────────────────────────────
    # *** ĐIỂM THAY ĐỔI DUY NHẤT so với gốc ***
    # Gốc: refusal_vectors = pickle.load(open(RV_PATH)) → DIM
    # Mới: compute_rfm_refusal_vectors()                → RFM
    #
    # Note: load_alphasteer_embeddings() dùng CPU để tránh double-load VRAM.
    # RFM computation bên trong sẽ move sang device khi cần.
    logger.info("Computing RFM refusal vectors (method=%s)...", args.rfm_method)

    H_refusal_cpu, H_compliant_cpu = load_alphasteer_embeddings(embeds_dir)
    num_total_layers = H_benign_train.shape[1]
    d_model = H_benign_train.shape[2]

    refusal_vectors_np = compute_rfm_refusal_vectors(
        H_refusal=H_refusal_cpu,
        H_compliant=H_compliant_cpu,
        layers=[layer for layer, _ in layers_ratio_list],
        num_total_layers=num_total_layers,
        method=args.rfm_method,
        rfm_iters=args.rfm_iters,
        device=args.device,
    )
    H_refusal_cpu = None
    H_compliant_cpu = None
    torch.cuda.empty_cache()

    refusal_vectors = torch.tensor(refusal_vectors_np, dtype=torch.float32).to(device)
    logger.info("refusal_vectors shape: %s  dtype: %s", tuple(refusal_vectors.shape), refusal_vectors.dtype)

    # ── 3. Null-space + regression + steering matrix — giống gốc dòng 3941-3977
    # FIX C5: init theo đúng gốc
    P              = torch.zeros(num_total_layers, d_model, d_model, device="cpu")
    tilde_delta    = torch.zeros(num_total_layers, d_model, d_model, device="cpu")
    steering_matrix = torch.zeros(num_total_layers, d_model, d_model, device="cpu")

    for layer, ratio in layers_ratio_list:
        logger.info("layer=%d  null_ratio=%.1f", layer, ratio)

        # Null-space projection (giống gốc dòng 3953)
        P_layer = null_space_projection_l(
            H_benign_train[:, layer, :].to(args.device),
            abs_nullspace_ratio=ratio
        )
        P[layer] = P_layer
        logger.info("  P_norm=%.4f", torch.norm(P_layer).item())

        # FIX C2: per-layer call giống gốc dòng 3959-3960
        tilde_delta_layer = cal_tilde_delta_with_regularization_l(
            H_harmful_train[:, layer, :].to(args.device),
            P_layer,
            refusal_vectors[layer],
            lambda_reg=args.lambda_reg,
            device=args.device,
        )
        tilde_delta[layer] = tilde_delta_layer
        logger.info("  tilde_delta_norm=%.4f", torch.norm(tilde_delta_layer).item())

        # Steering matrix (giống gốc dòng 3967-3968)
        steering_matrix_layer = cal_steering_matrix_l(P_layer, tilde_delta_layer, device=args.device)
        steering_matrix[layer] = steering_matrix_layer
        logger.info("  steering_matrix_norm=%.4f", torch.norm(steering_matrix_layer).item())


        # del h_benign_l, h_harmful_l, ref_vector_l, P_layer, tilde_delta_layer, steering_matrix_layer
        torch.cuda.empty_cache()

    # ── 4. Save — FIX C3: float32 như gốc dòng 3976 ────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(steering_matrix, args.save_path)  # float32, không convert bfloat16
    logger.info("Saved → %s", args.save_path)
    logger.info("Total time: %.1fs", time.time() - t0)

    # FIX C7: cleanup memory như gốc dòng 3980-3982
    H_benign_train = None
    H_harmful_train = None
    P = None
    tilde_delta = None
    steering_matrix = None
    torch.cuda.empty_cache()