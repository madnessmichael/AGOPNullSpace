"""
calc_steering_matrix_rfm_hh.py  (v3)
=================================
Drop-in replacement cho src/calc_steering_matrix.py của AlphaSteer.
Thay đổi duy nhất về mặt phương pháp: refusal vector từ RFM/AGOP thay vì DIM.

THAY ĐỔI SO VỚI v2:
  D1. v2 load embeddings HAI LẦN: một lần thủ công ở bước 1 (H_benign_train /
      H_harmful_train), một lần nữa qua load_alphasteer_embeddings() ở bước 2.
      Cả hai đều gọi torch.randperm trên RNG global ⇒ subset jailbreak dùng cho
      RFM KHÁC subset dùng cho regression. Manuscript nói "the full set fits Δ̃"
      — không nhất quán, và tốn gấp đôi RAM (~15 GB thừa ở 8B).
      → Fix: load MỘT lần, dùng chung cho cả RFM lẫn regression.

  D2. v2 cấp phát P / tilde_delta / steering_matrix mỗi cái (L, d, d) float32
      trên CPU. Ở 70B: 80 × 8192² × 4 B = 21.5 GB MỖI tensor = 64 GB RAM,
      rồi torch.save ghi file 21.5 GB dù chỉ 26 layer khác 0.
      → Fix: P và tilde_delta là biến tạm trong vòng lặp (không lưu mảng);
        steering_matrix lưu dạng dict {layer: (d,d)} (mặc định) → 70B còn ~7 GB.
        --save_format dense vẫn có nếu cần tương thích ngược tuyệt đối.

  D3. `P[layer] = P_layer` gán cross-device (CUDA → CPU) ngầm.
      → Fix: .cpu() tường minh.

  D4. Ghi lại metadata (ρ per-layer, bandwidth/reg/center_grads được chọn,
      leakage benign held-out) vào sidecar JSON — chính là các con số manuscript
      đang ghi sai (ρ=0.6 đồng nhất, T∈{1,2,5,10}, N_m=2720, N_b=14900).

  D5. Thêm --holdout_benign: tách một phần D_b ra KHÔNG dùng để fit P, để đo
      leakage thật. Bài đang claim "provable near-zero by construction" mà
      không có số liệu held-out nào.

Cách dùng:
    python src/calc_steering_matrix_rfm_hh.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --device cuda \
        --save_path data/steering_matrix/steering_matrix_llama3.1_agopn.pt \
        --probe rfm --rfm_iters 5 --lambda_reg 10.0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils.const import AlphaSteer_CALCULATION_CONFIG          # noqa: E402
from utils.steering_utils import (                              # noqa: E402
    disable_tf32,
    null_space_basis_l,
    cal_steering_factors_q,
    benign_leakage,
    steering_signal_stats,
)
from rfm_refusal_vector import (                                # noqa: E402
    load_alphasteer_embeddings,
    compute_refusal_vectors,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



DATASET_NAME = "justinphan3110/harmful_harmless_instructions"

def _assert_activation_shape(name: str, H: torch.Tensor) -> None:
    if H.ndim != 3:
        raise ValueError(
            f"{name} must have shape [N, num_layers, d_model], got {tuple(H.shape)}. "
            "This script expects pre-extracted model activations, not raw text."
        )


def _load_tensor_or_dict(path: str):
    """Load .pt/.pth tensor or dict saved by an activation-extraction script."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, dict):
        for key in ["embeddings", "hidden_states", "activations", "states", "H"]:
            if key in obj:
                val = obj[key]
                if isinstance(val, torch.Tensor):
                    return val.float()
                if isinstance(val, np.ndarray):
                    return torch.from_numpy(val).float()
        raise KeyError(
            f"{path} is a dict, but no embedding key found. "
            "Expected one of: embeddings, hidden_states, activations, states, H."
        )
    raise TypeError(f"Unsupported object type in {path}: {type(obj)}")


def _load_labels(path: str) -> torch.Tensor:
    """
    Load labels for justinphan3110/harmful_harmless_instructions.

    HF convention for this dataset:
      label=True  -> harmless
      label=False -> harmful

    We convert labels to int64 after flattening nested pair labels.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["labels", "label", "y", "targets"]:
            if key in obj:
                obj = obj[key]
                break
    labels = torch.as_tensor(obj)
    labels = labels.reshape(-1).to(torch.int64)
    return labels

def balance_harmful_harmless_embeddings(
    H_harmful: torch.Tensor,
    H_harmless: torch.Tensor,
    balance_ratio: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Balance data for training the RFM/logistic concept direction.

    balance_ratio = harmless / harmful after sampling.
    With 1.0, both sides have exactly min-compatible size.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_harmful = H_harmful.shape[0]
    n_harmless_target = round(n_harmful * balance_ratio)

    # If harmless is the limiting class, downsample harmful to match it.
    if H_harmless.shape[0] < n_harmless_target:
        n_harmful_target = max(1, round(H_harmless.shape[0] / balance_ratio))
        idx_h = torch.randperm(H_harmful.shape[0], generator=gen)[:n_harmful_target]
        idx_s = torch.arange(H_harmless.shape[0])
    else:
        idx_h = torch.arange(H_harmful.shape[0])
        idx_s = torch.randperm(H_harmless.shape[0], generator=gen)[:n_harmless_target]

    Hh = H_harmful[idx_h].contiguous().float()
    Hs = H_harmless[idx_s].contiguous().float()
    logger.info(
        "Balanced HH data for RFM — harmful=%d, harmless=%d, ratio=%.3f",
        Hh.shape[0], Hs.shape[0], Hs.shape[0] / max(1, Hh.shape[0]),
    )
    return Hh, Hs

def load_harmful_harmless_instruction_embeddings(
    embeddings_path: str = None,
    labels_path: str = None,
    harmful_path: str = None,
    harmless_path: str = None,
    balance: bool = False,
    balance_ratio: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Load activations extracted from HF dataset:
        justinphan3110/harmful_harmless_instructions

    Supported formats:

    1) Already separated files:
        --hh_harmful_path  embeds_hh_harmful.pt
        --hh_harmless_path embeds_hh_harmless.pt

       Each tensor must be [N, num_layers, d_model].

    2) Combined activations + labels:
        --hh_embeddings_path embeds_harmful_harmless_instructions.pt
        --hh_labels_path     labels_harmful_harmless_instructions.pt

       embeddings: [N, num_layers, d_model]
       labels:     [N] or nested [num_pairs, 2]

       Label mapping follows the HF dataset examples:
         True/1  = harmless
         False/0 = harmful

    Returns:
        H_harmful:  [N_harmful,  num_layers, d_model] float32 CPU
        H_harmless: [N_harmless, num_layers, d_model] float32 CPU
    """
    if harmful_path and harmless_path:
        logger.info("Loading separated harmful/harmless embeddings from HF dataset...")
        H_harmful = _load_tensor_or_dict(harmful_path)
        H_harmless = _load_tensor_or_dict(harmless_path)
    else:
        if not embeddings_path or not labels_path:
            raise ValueError(
                "Provide either (--hh_harmful_path and --hh_harmless_path) "
                "or (--hh_embeddings_path and --hh_labels_path)."
            )
        logger.info("Loading combined HH embeddings: %s", embeddings_path)
        H = _load_tensor_or_dict(embeddings_path)
        y = _load_labels(labels_path)
        _assert_activation_shape("HH combined embeddings", H)
        if H.shape[0] != y.numel():
            raise ValueError(
                f"Embedding/label length mismatch: embeddings N={H.shape[0]}, labels N={y.numel()}. "
                "If your HF dataset rows are pairs, flatten both sentence and label before activation extraction."
            )

        # HF label=True means harmless; label=False means harmful.
        harmless_mask = y.bool()
        harmful_mask = ~harmless_mask
        H_harmful = H[harmful_mask].contiguous().float()
        H_harmless = H[harmless_mask].contiguous().float()

    _assert_activation_shape("H_harmful", H_harmful)
    _assert_activation_shape("H_harmless", H_harmless)
    if H_harmful.shape[1:] != H_harmless.shape[1:]:
        raise ValueError(
            f"Shape mismatch after split: harmful={tuple(H_harmful.shape)}, "
            f"harmless={tuple(H_harmless.shape)}"
        )

    logger.info(
        "Loaded %s activations — harmful=%s, harmless=%s",
        DATASET_NAME, tuple(H_harmful.shape), tuple(H_harmless.shape),
    )

    if balance:
        H_harmful, H_harmless = balance_harmful_harmless_embeddings(
            H_harmful, H_harmless, balance_ratio=balance_ratio, seed=seed
        )

    return H_harmful, H_harmless


# Backward-compatible name, but now intentionally points to the HF HH dataset loader.
def load_balanced_embeddings(
    embedding_dir: str = None,
    n_jailbreak: int = 1000,  # unused; kept to avoid breaking old imports
    balance_ratio: float = 1.0,
    seed: int = 42,
    embeddings_path: str = None,
    labels_path: str = None,
    harmful_path: str = None,
    harmless_path: str = None,
) -> tuple:
    """
    Compatibility wrapper.

    v4 used AlphaSteer files:
      embeds_harmful_train_1000.pt, embeds_jailbreak_train.pt,
      embeds_benign_train.pt, embeds_coconot_*.pt

    v5 uses only activations from:
      justinphan3110/harmful_harmless_instructions

    If embedding_dir is provided and explicit paths are omitted, defaults are:
      {embedding_dir}/embeds_harmful_harmless_instructions.pt
      {embedding_dir}/labels_harmful_harmless_instructions.pt
    """
    if embedding_dir and not (embeddings_path or labels_path or harmful_path or harmless_path):
        embeddings_path = os.path.join(embedding_dir, "embeds_harmful_harmless_instructions.pt")
        labels_path = os.path.join(embedding_dir, "labels_harmful_harmless_instructions.pt")

    return load_harmful_harmless_instruction_embeddings(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        harmful_path=harmful_path,
        harmless_path=harmless_path,
        balance=True,
        balance_ratio=balance_ratio,
        seed=seed,
    )

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--save_path", required=True)
    p.add_argument("--device", default="cuda")
    # RFM / concept vector
    p.add_argument("--probe", default="rfm", choices=["rfm", "linear"])
    p.add_argument("--rfm_iters", type=int, default=5)
    p.add_argument("--tuning_metric", default="auc", choices=["auc", "acc", "f1", "mse"])
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--include_math", action="store_true",
                   help="Thêm 900 mẫu MATH vào D_b để khớp con số 14,900 trong manuscript. "
                        "Mặc định TẮT — khớp AlphaSteer gốc, D_b = 14,000.")
    # Regression
    p.add_argument("--lambda_reg", type=float, default=10.0)
    # Numerics
    p.add_argument("--nullspace_dtype", default="float32", choices=["float32", "float64"],
                   help="dtype cho SVD null-space. float64 chính xác hơn nhưng chậm trên "
                        "GPU consumer (1/64 rate); SVD chỉ chạy 1 lần/layer nên float64 "
                        "thường vẫn chấp nhận được. TF32 đã tắt sẵn nên float32 thường đủ.")
    # Diagnostics
    p.add_argument("--holdout_benign", type=int, default=1000,
                   help="Số mẫu benign giữ lại KHÔNG dùng fit P, để đo leakage.")
    # IO
    p.add_argument("--save_format", default="dense", choices=["dense", "rank1", "sparse"],
                   help="dense (MẶC ĐỊNH): tensor [L,d,d] float32 GIỐNG HỆT AlphaSteer/DIM. "
                        "So sánh trực tiếp với *_dim.pt và ma trận cũ. rank1/sparse chỉ dùng "
                        "khi cần tiết kiệm bộ nhớ và bạn chủ động chọn.")
    p.add_argument("--seed", type=int, default=2706)
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    disable_tf32()   # BẮT BUỘC: null-space dựa vào Qᵀh_b ≈ 0; TF32 phá phép triệt tiêu

    device = torch.device(args.device)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]   # [(layer, ρ), ...]
    layers = [l for l, _ in layers_ratio_list]
    ratios = {l: r for l, r in layers_ratio_list}

    # ══ 1. Load embeddings — MỘT LẦN DUY NHẤT (D1) ═══════════════════════════
    # v2 load 2 lần với 2 permutation khác nhau ⇒ RFM và regression thấy 2 tập
    # jailbreak khác nhau. Ở đây H_malicious/H_benign được dùng cho CẢ HAI.
    H_malicious, H_benign = load_alphasteer_embeddings(
        args.embedding_dir, seed=args.seed, include_math=args.include_math)

    num_total_layers = H_benign.shape[1]
    d_model = H_benign.shape[2]

    # D5: tách benign held-out để đo leakage (không dùng fit P)
    g = torch.Generator().manual_seed(args.seed + 1)
    perm = torch.randperm(H_benign.shape[0], generator=g)
    n_ho = min(args.holdout_benign, H_benign.shape[0] // 10)
    ho_idx, fit_idx = perm[:n_ho], perm[n_ho:]
    H_benign_ho = H_benign[ho_idx]
    H_benign_fit = H_benign[fit_idx]
    logger.info("D_b: %d fit / %d held-out | D_m: %d | L=%d d=%d",
                len(fit_idx), len(ho_idx), H_malicious.shape[0], num_total_layers, d_model)

    # ══ 2. Concept vectors qua RFM/AGOP ══════════════════════════════════════
    # *** ĐIỂM THAY ĐỔI DUY NHẤT so với AlphaSteer gốc ***
    #   Gốc: refusal_vectors = pickle.load(RV_PATH)     → DIM
    #   Mới: compute_refusal_vectors(...)               → AGOP top eigenvector
    logger.info("Computing %s concept vectors (metric=%s, iters=%d)...",
                args.probe.upper(), args.tuning_metric, args.rfm_iters)


    H_harmful_all, H_harmless_all = load_harmful_harmless_instruction_embeddings(
        embeddings_path=os.path.join( args.embedding_dir, "harmful_harmless_instructions/embeds_harmful_harmless_instructions.pt"),
        labels_path=os.path.join(args.embedding_dir, "harmful_harmless_instructions/labels_harmful_harmless_instructions.pt"),
        harmful_path=os.path.join( args.embedding_dir, "harmful_harmless_instructions/embeds_hh_harmful.pt"),
        harmless_path=os.path.join( args.embedding_dir, "harmful_harmless_instructions/embeds_hh_harmless.pt"),
        balance=False,
        seed=args.seed,
    )

    H_rfm_harmful, H_rfm_harmless = balance_harmful_harmless_embeddings(
        H_harmful_all,
        H_harmless_all,
        balance_ratio=1.0,
        seed=args.seed,
    )

    num_total_layers = H_harmless_all.shape[1]
    d_model = H_harmless_all.shape[2]


    # refusal_vectors_np, probe_meta = compute_refusal_vectors(
    #     H_malicious=H_malicious,
    #     H_benign=H_benign_fit,          # cùng tập với tập fit P
    #     layers=layers,
    #     num_total_layers=num_total_layers,
    #     probe=args.probe,
    #     rfm_iters=args.rfm_iters,
    #     n_components=args.n_components,
    #     tuning_metric=args.tuning_metric,
    #     max_per_class=args.max_per_class,
    #     seed=args.seed,
    #     device=args.device,
    # )


    refusal_vectors_np, probe_meta = compute_refusal_vectors(
        H_malicious=H_rfm_harmful,
        H_benign=H_rfm_harmless,          # cùng tập với tập fit P
        layers=layers,
        num_total_layers=num_total_layers,
        probe=args.probe,
        rfm_iters=args.rfm_iters,
        n_components=args.n_components,
        tuning_metric=args.tuning_metric,
        max_per_class=args.max_per_class,
        seed=args.seed,
        device=args.device,
    )

    del H_rfm_harmful, H_rfm_harmless


    refusal_vectors = torch.from_numpy(refusal_vectors_np).float()
    logger.info("refusal_vectors %s dtype=%s  (||r||=1 với mọi layer)",
                tuple(refusal_vectors.shape), refusal_vectors.dtype)

    # ══ 3. Null space + regression + steering factors ════════════════════════
    # D2/D6: KHÔNG cấp phát mảng (L,d,d) nào. M luôn là RANK-1 (M = u⊗r, xem
    # block giải thích ở đầu steering_utils.py) nên ta chỉ lưu (u, r).
    factors: dict[int, dict] = {}
    diagnostics: dict[str, dict] = {}

    for layer in layers:
        ratio = ratios[layer]
        logger.info("=== layer %d (ρ=%.2f) ===", layer, ratio)

        h_b = H_benign_fit[:, layer, :].to(device).float()
        # V4: trả về CƠ SỞ Q (d, k), KHÔNG dựng P (d, d). Tránh luôn assert P²=P
        # từng fail giả trên GPU có TF32.
        ns_dtype = torch.float64 if args.nullspace_dtype == "float64" else None
        Q_layer = null_space_basis_l(h_b, abs_nullspace_ratio=ratio, dtype=ns_dtype)
        Q_layer = Q_layer.float()   # hạ về fp32 cho phần regression (đủ, và nhanh)
        del h_b

        # D5: leakage benign held-out. ||P h|| = ||Qᵀh|| (Q trực chuẩn) ⇒ chỉ cần Q.
        leak = benign_leakage(H_benign_ho[:, layer, :], Q_layer)
        logger.info("  benign held-out leakage ||Qᵀh||/||h||: mean=%.4f p95=%.4f max=%.4f",
                    leak["leakage_mean"], leak["leakage_p95"], leak["leakage_max"])

        h_m = H_malicious[:, layer, :].to(device).float()
        # Giải trong không gian k chiều: SPD ⇒ Cholesky, KHÔNG pinv của cond ~1e20
        u, r = cal_steering_factors_q(
            h_m, Q_layer, refusal_vectors[layer].to(device),
            lambda_reg=args.lambda_reg, device=args.device)
        del h_m, Q_layer

        factors[layer] = {"u": u.detach().cpu(), "r": r.detach().cpu()}   # D3

        # Gate uᵀh chính là "prompt classifier" ẩn của phương pháp.
        st_m = steering_signal_stats(H_malicious[:1000, layer, :], u, r)
        st_b = steering_signal_stats(H_benign_ho[:, layer, :], u, r)
        sel = st_m["mean_signal_norm"] / max(st_b["mean_signal_norm"], 1e-12)
        logger.info("  gate uᵀh: malicious=%.4f±%.4f  benign_holdout=%.4f±%.4f  "
                    "→ selectivity=%.1fx",
                    st_m["gate_mean"], st_m["gate_std"],
                    st_b["gate_mean"], st_b["gate_std"], sel)

        diagnostics[str(layer)] = {
            "rho": ratio,
            "k_nullspace": int(round(ratio * d_model)),
            "u_norm": float(u.norm()),
            "benign_holdout_leakage": leak,
            "gate_malicious": st_m,
            "gate_benign_holdout": st_b,
            "selectivity_ratio": sel,
            "probe": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                      for k, v in probe_meta[layer].items()},
        }

        del u, r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══ 4. Save ══════════════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    if args.save_format == "dense":
        # MẶC ĐỊNH: tensor [L, d, d] float32 — GIỐNG HỆT AlphaSteer/DIM.
        # Load được bằng MỌI code cũ; so sánh trực tiếp với *_dim.pt bằng
        # torch.load rồi trừ/norm/svd như bình thường.
        dense = torch.zeros(num_total_layers, d_model, d_model, dtype=torch.float32)
        for layer, f in factors.items():
            dense[layer] = torch.outer(f["u"], f["r"])   # M = P Δ̃ = u rᵀ
        torch.save(dense, args.save_path)                # KHÔNG cast bfloat16
    elif args.save_format == "rank1":
        # Tùy chọn tiết kiệm bộ nhớ (70B: 7.0 GB → 1.7 MB). CHỈ dùng khi bạn
        # chủ động chọn và AlphaSteerModel v3 đọc được format này.
        torch.save({
            "format": "rank1_v1", "num_layers": num_total_layers,
            "d_model": d_model, "layers": layers, "factors": factors,
        }, args.save_path)
    else:  # sparse
        torch.save({
            "format": "sparse_v1", "num_layers": num_total_layers,
            "d_model": d_model, "layers": layers,
            "matrices": {l: torch.outer(f["u"], f["r"]) for l, f in factors.items()},
        }, args.save_path)

    meta = {
        "model_name": args.model_name,
        "probe": args.probe,
        "rfm_iters": args.rfm_iters,
        "tuning_metric": args.tuning_metric,
        "lambda_reg": args.lambda_reg,
        "seed": args.seed,
        # Các con số này phải được copy nguyên văn vào manuscript:
        "N_malicious": int(H_malicious.shape[0]),     # = 2000, KHÔNG phải 2720
        "N_benign_total": int(H_benign.shape[0]),     # = 14000 (include_math=False)
        "N_benign_fit_P": int(len(fit_idx)),
        "N_benign_holdout": int(len(ho_idx)),
        "include_math_in_Db": args.include_math,
        "rho_per_layer": {str(l): ratios[l] for l in layers},   # KHÔNG đồng nhất 0.6
        "steering_rank": 1,   # M = u⊗r — inference là O(d), KHÔNG phải O(d²)
        "layers": layers,
        "per_layer": diagnostics,
    }
    meta_path = os.path.splitext(args.save_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved → %s (format=%s)", args.save_path, args.save_format)

    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()