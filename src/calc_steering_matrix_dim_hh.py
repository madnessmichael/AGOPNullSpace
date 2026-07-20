"""
calc_steering_matrix_dim_hh.py  (v5-dim)
========================================
Biến thể của calc_steering_matrix_rfm.py v5:
  - GIỮ NGUYÊN: null-space P từ benign AlphaSteer, gate u fit trên H_malicious
    AlphaSteer, regression Q-space, format output [L,d,d].
  - THAY DUY NHẤT: concept vector r từ DIM trên harmful_harmless_instructions
    (mean(harmful) − mean(harmless), normalized) thay vì RFM/AGOP.

VÌ SAO ABLATION NÀY QUAN TRỌNG
------------------------------
So 4 ô của bảng 2×2 {data: AlphaSteer-mix | HH-paired} × {estimator: DIM | RFM}:
  - RFM-mix   : bản chính hiện tại
  - DIM-mix   : AlphaSteer gốc (RV_refusal.pkl — nhưng chú ý: đó là DIM trên
                refusal/comply RESPONSES, không phải harmful/benign prompts;
                nếu muốn ô này đúng nghĩa, chạy DIM trên D_m/D_b của AlphaSteer)
  - RFM-HH    : bản v5 của bạn
  - DIM-HH    : SCRIPT NÀY
Nếu DIM-HH ≈ RFM-HH ⇒ lợi ích đến từ DATA PAIRING, không phải AGOP.
Nếu RFM-HH > DIM-HH rõ rệt ⇒ AGOP thật sự đóng góp ngay cả khi covariance
đã được pairing kiểm soát — bằng chứng mạnh hơn nhiều cho claim của manuscript.

LƯU Ý SIGN CONVENTION
---------------------
r = mean(HARMFUL) − mean(HARMLESS)  ⇒ r trỏ về lớp harmful, GIỐNG quy ước sign
của rfm_refusal_vector.py (corr(Z@r, y_malicious) > 0). Steering +α đẩy về
hướng harmful-detection ⇒ kích refusal. KHÔNG đảo dấu.

CÁCH DÙNG
---------
python src/calc_steering_matrix_dim_hh.py \
    --model_name llama3.1 \
    --embedding_dir data/embeddings/llama3.1 \
    --save_path data/steering_matrix/steering_matrix_llama3.1_dim_experiment_hh.pt \
    --lambda_reg 10.0 --holdout_benign 1000 --device cuda \
    --ref_vectors_path data/steering_matrix/steering_matrix_llama3.1_agopn_rfm_meta.json  # optional cos-check

Yêu cầu embeddings HH đã extract sẵn tại:
    {embedding_dir}/harmful_harmless_instructions/embeds_hh_harmful.pt
    {embedding_dir}/harmful_harmless_instructions/embeds_hh_harmless.pt
hoặc cặp combined + labels (như v5).

⚠ NHẮC LẠI 3 BUG EXTRACTION phải fix TRƯỚC khi tin kết quả script này
(chúng nằm ở prepare_harmful_harmless_instruction_embeddings.py, không ở đây):
  E1. hidden_states[1:] lệch 1 layer so với EmbeddingExtractor của repo
      (repo dùng hidden_states[layer_idx] trực tiếp, KHÔNG bỏ embedding layer).
  E2. last_idx = attention_mask.sum(1)-1 SAI với padding_side="left"
      (left padding ⇒ token cuối luôn ở vị trí -1; dùng h[:, -1, :]).
  E3. add_generation_prompt=False khác repo (repo dùng True) ⇒ lệch phân phối.
Cách sửa gọn nhất: extract HH bằng chính utils.embedding_utils.EmbeddingExtractor.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.const import AlphaSteer_CALCULATION_CONFIG          # noqa: E402
from utils.steering_utils import (                              # noqa: E402
    disable_tf32,
    null_space_basis_l,
    cal_steering_factors_q,
    benign_leakage,
    steering_signal_stats,
)
from rfm_refusal_vector import load_alphasteer_embeddings       # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATASET_NAME = "justinphan3110/harmful_harmless_instructions"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD HH EMBEDDINGS (rút gọn từ v5 — một nhánh rõ ràng, fail sớm nếu thiếu)
# ══════════════════════════════════════════════════════════════════════════════

def _load_tensor(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("embeddings", "hidden_states", "activations", "H"):
            if key in obj:
                obj = obj[key]
                break
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path}: không đọc được tensor, got {type(obj)}")
    return obj.float()


def load_hh_embeddings(embedding_dir: str,
                       harmful_path: str = None,
                       harmless_path: str = None,
                       embeddings_path: str = None,
                       labels_path: str = None):
    """
    Ưu tiên cặp separated; nếu không có, dùng combined+labels.
    CHỌN MỘT nhánh và fail rõ ràng — không im lặng chuyển nguồn (khác v5,
    vốn nhận cả 4 path và lặng lẽ ưu tiên separated).
    Trả (H_harmful, H_harmless): mỗi cái [N, num_layers, d] float32 CPU.
    """
    hh_dir = os.path.join(embedding_dir, "harmful_harmless_instructions")
    harmful_path = harmful_path or os.path.join(hh_dir, "embeds_hh_harmful.pt")
    harmless_path = harmless_path or os.path.join(hh_dir, "embeds_hh_harmless.pt")

    if os.path.exists(harmful_path) and os.path.exists(harmless_path):
        logger.info("HH source: SEPARATED files")
        Hh = _load_tensor(harmful_path)
        Hs = _load_tensor(harmless_path)
    else:
        embeddings_path = embeddings_path or os.path.join(
            hh_dir, "embeds_harmful_harmless_instructions.pt")
        labels_path = labels_path or os.path.join(
            hh_dir, "labels_harmful_harmless_instructions.pt")
        if not (os.path.exists(embeddings_path) and os.path.exists(labels_path)):
            raise FileNotFoundError(
                f"Không tìm thấy HH embeddings. Đã thử:\n"
                f"  separated: {harmful_path} / {harmless_path}\n"
                f"  combined : {embeddings_path} / {labels_path}\n"
                f"Chạy extraction trước (khuyến nghị dùng EmbeddingExtractor của repo).")
        logger.info("HH source: COMBINED + labels")
        H = _load_tensor(embeddings_path)
        y = torch.as_tensor(torch.load(labels_path, map_location="cpu")).reshape(-1).bool()
        if H.shape[0] != y.numel():
            raise ValueError(f"embeddings N={H.shape[0]} != labels N={y.numel()}")
        # HF convention: label=True ⇒ harmless, False ⇒ harmful
        Hh, Hs = H[~y].contiguous(), H[y].contiguous()

    for name, t in (("H_harmful", Hh), ("H_harmless", Hs)):
        if t.ndim != 3:
            raise ValueError(f"{name} phải là [N, L, d], got {tuple(t.shape)}")
    if Hh.shape[1:] != Hs.shape[1:]:
        raise ValueError(f"Shape mismatch: {tuple(Hh.shape)} vs {tuple(Hs.shape)}")
    logger.info("HH loaded: harmful=%s harmless=%s", tuple(Hh.shape), tuple(Hs.shape))
    return Hh, Hs


# ══════════════════════════════════════════════════════════════════════════════
# DIM CONCEPT VECTORS — thay thế duy nhất so với v5
# ══════════════════════════════════════════════════════════════════════════════

def compute_dim_vectors_hh(H_harmful: torch.Tensor,
                           H_harmless: torch.Tensor,
                           layers,
                           num_total_layers: int) -> tuple[np.ndarray, dict]:
    """
    r[l] = normalize( mean(H_harmful[:,l,:]) − mean(H_harmless[:,l,:]) ).

    KHÔNG cần balance: DIM là hiệu hai trung bình lớp — kích thước lớp không
    ảnh hưởng ước lượng (khác probe, vốn nhạy class imbalance). Ta dùng TOÀN BỘ
    mẫu mỗi lớp để giảm variance của mean.

    Trả (num_total_layers, d) float32; layer không tính = 0 (khớp format pkl DIM
    của AlphaSteer / output của compute_refusal_vectors).
    """
    d = H_harmful.shape[2]
    rv = np.zeros((num_total_layers, d), dtype=np.float32)
    meta = {}
    for l in layers:
        mh = H_harmful[:, l, :].mean(0)
        ms = H_harmless[:, l, :].mean(0)
        r = mh - ms                                   # trỏ về lớp HARMFUL (khớp sign RFM)
        raw_norm = float(r.norm())
        r = r / r.norm().clamp(min=1e-8)
        rv[l] = r.numpy()
        # separation chuẩn hóa: khoảng cách 2 mean so với độ phân tán trong lớp —
        # con số này cho biết DIM "dễ" đến mức nào ở layer đó (HH pairing thường cho
        # giá trị cao; nếu thấp bất thường ⇒ nghi extraction sai layer, xem E1).
        pooled_std = float(0.5 * (H_harmful[:, l, :].std(0).mean()
                                  + H_harmless[:, l, :].std(0).mean()))
        meta[l] = {
            "probe": "dim_hh",
            "raw_diff_norm": raw_norm,
            "separation_ratio": raw_norm / max(pooled_std * (d ** 0.5), 1e-8),
            "n_harmful": int(H_harmful.shape[0]),
            "n_harmless": int(H_harmless.shape[0]),
        }
        logger.info("  layer %d: ||Δmean||=%.4f  separation=%.3f",
                    l, raw_norm, meta[l]["separation_ratio"])
    return rv, meta


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC: cos với refusal vectors tham chiếu (RFM-HH / RFM-mix / DIM AlphaSteer)
# ══════════════════════════════════════════════════════════════════════════════

def cos_vs_reference(rv_new: np.ndarray, ref_path: str, layers) -> dict:
    """
    ref_path nhận: .pkl (dict{'refusal_vectors':...} hoặc array — format repo),
    hoặc .pt (tensor [L,d]). Trả {layer: cos}.
    Diễn giải: cos cao ⇒ DIM-HH ≈ hướng tham chiếu ⇒ thí nghiệm sẽ tái tạo
    kết quả cũ; cos thấp ở Qwen/Gemma là nơi đáng chạy eval nhất.
    """
    if ref_path.endswith(".pkl"):
        with open(ref_path, "rb") as f:
            obj = pickle.load(f)
        ref = obj["refusal_vectors"] if isinstance(obj, dict) else obj
        ref = np.asarray(ref, dtype=np.float32)
    else:
        ref = torch.load(ref_path, map_location="cpu").float().numpy()
    out = {}
    for l in layers:
        a, b = rv_new[l], ref[l]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        out[l] = float(a @ b / (na * nb)) if na > 0 and nb > 0 else None
        logger.info("  cos(r_dim_hh, r_ref)[layer %d] = %s",
                    l, f"{out[l]:+.4f}" if out[l] is not None else "N/A (ref zero)")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — cấu trúc GIỐNG HỆT v5 từ bước 3 trở đi
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--embedding_dir", required=True,
                   help="data/embeddings/<model> — chứa cả AlphaSteer embeddings "
                        "lẫn thư mục con harmful_harmless_instructions/")
    p.add_argument("--save_path", required=True,
                   help="KHUYẾN NGHỊ naming: steering_matrix_<model>_dim_experiment_hh.pt "
                        "để không đè kết quả chính thức.")
    p.add_argument("--device", default="cuda")
    # HH embedding overrides (mặc định tự dò trong embedding_dir/harmful_harmless_instructions/)
    p.add_argument("--hh_harmful_path", default=None)
    p.add_argument("--hh_harmless_path", default=None)
    p.add_argument("--hh_embeddings_path", default=None)
    p.add_argument("--hh_labels_path", default=None)
    # Regression (giống v5)
    p.add_argument("--lambda_reg", type=float, default=10.0)
    p.add_argument("--nullspace_dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--holdout_benign", type=int, default=1000)
    p.add_argument("--save_format", default="dense", choices=["dense", "rank1", "sparse"])
    # Diagnostic
    p.add_argument("--ref_vectors_path", default=None,
                   help="(tùy chọn) .pkl/.pt refusal vectors tham chiếu để in cos per layer. "
                        "Vd: data/refusal_vectors/RV/llama3.1_RV_refusal.pkl (DIM AlphaSteer) "
                        "hoặc file r của bản RFM-HH.")
    p.add_argument("--seed", type=int, default=2706)
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    disable_tf32()

    device = torch.device(args.device)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]
    layers = [l for l, _ in layers_ratio_list]
    ratios = {l: r for l, r in layers_ratio_list}

    # ══ 1. AlphaSteer embeddings — cho P (benign) và gate u (malicious) ══════
    H_malicious, H_benign = load_alphasteer_embeddings(
        args.embedding_dir, seed=args.seed, include_math=False)
    num_total_layers = H_benign.shape[1]
    d_model = H_benign.shape[2]

    g = torch.Generator().manual_seed(args.seed + 1)
    perm = torch.randperm(H_benign.shape[0], generator=g)
    n_ho = min(args.holdout_benign, H_benign.shape[0] // 10)
    ho_idx, fit_idx = perm[:n_ho], perm[n_ho:]
    H_benign_ho, H_benign_fit = H_benign[ho_idx], H_benign[fit_idx]
    logger.info("D_b: %d fit / %d held-out | D_m: %d | L=%d d=%d",
                len(fit_idx), len(ho_idx), H_malicious.shape[0], num_total_layers, d_model)

    # ══ 2. Concept vectors — DIM trên HH (điểm thay đổi duy nhất so với v5) ══
    Hh, Hs = load_hh_embeddings(
        args.embedding_dir,
        harmful_path=args.hh_harmful_path, harmless_path=args.hh_harmless_path,
        embeddings_path=args.hh_embeddings_path, labels_path=args.hh_labels_path)

    if Hh.shape[1] == num_total_layers + 1:
        # Extraction mới giữ đủ 33 hidden_states (embedding + 32 layer output);
        # repo chỉ dùng index 0..num_total-1. Quy ước index ĐÃ khớp → cắt đuôi.
        logger.info("HH có %d entries (num_total+1): quy ước H[:,l,:]=hidden_states[l] "
                    "đã khớp repo, cắt bỏ entry cuối (output layer cuối, repo không dùng).",
                    Hh.shape[1])
        Hh = Hh[:, :num_total_layers, :]
        Hs = Hs[:, :num_total_layers, :]
    elif Hh.shape[1] != num_total_layers:
        raise ValueError(
            f"HH embeddings có {Hh.shape[1]} layers, AlphaSteer có {num_total_layers}. "
            f"Không khớp và không phải num_total+1 → kiểm tra lại extraction/convention.")
    if Hh.shape[2] != d_model:
        raise ValueError(f"d mismatch: HH d={Hh.shape[2]} vs AlphaSteer d={d_model}")

    rv_np, probe_meta = compute_dim_vectors_hh(Hh, Hs, layers, num_total_layers)
    refusal_vectors = torch.from_numpy(rv_np).float()
    n_hh_harmful, n_hh_harmless = int(Hh.shape[0]), int(Hs.shape[0])
    del Hh, Hs

    cos_ref = None
    if args.ref_vectors_path:
        logger.info("── cos với reference: %s ──", args.ref_vectors_path)
        cos_ref = cos_vs_reference(rv_np, args.ref_vectors_path, layers)

    # ══ 3. Null space + gate — NGUYÊN VĂN v5 ═════════════════════════════════
    factors: dict[int, dict] = {}
    diagnostics: dict[str, dict] = {}

    for layer in layers:
        ratio = ratios[layer]
        logger.info("=== layer %d (ρ=%.2f) ===", layer, ratio)

        h_b = H_benign_fit[:, layer, :].to(device).float()
        ns_dtype = torch.float64 if args.nullspace_dtype == "float64" else None
        Q_layer = null_space_basis_l(h_b, abs_nullspace_ratio=ratio, dtype=ns_dtype).float()
        del h_b

        leak = benign_leakage(H_benign_ho[:, layer, :], Q_layer)
        logger.info("  benign held-out leakage: mean=%.4f p95=%.4f",
                    leak["leakage_mean"], leak["leakage_p95"])

        h_m = H_malicious[:, layer, :].to(device).float()
        u, r = cal_steering_factors_q(
            h_m, Q_layer, refusal_vectors[layer].to(device),
            lambda_reg=args.lambda_reg, device=args.device)
        del h_m, Q_layer

        factors[layer] = {"u": u.detach().cpu(), "r": r.detach().cpu()}

        st_m = steering_signal_stats(H_malicious[:1000, layer, :], u, r)
        st_b = steering_signal_stats(H_benign_ho[:, layer, :], u, r)
        sel = st_m["mean_signal_norm"] / max(st_b["mean_signal_norm"], 1e-12)
        logger.info("  gate uᵀh: mal=%.4f±%.4f  ben=%.4f±%.4f  → sel=%.1fx",
                    st_m["gate_mean"], st_m["gate_std"],
                    st_b["gate_mean"], st_b["gate_std"], sel)

        diagnostics[str(layer)] = {
            "rho": ratio,
            "u_norm": float(u.norm()),
            "benign_holdout_leakage": leak,
            "gate_malicious": st_m,
            "gate_benign_holdout": st_b,
            "selectivity_ratio": sel,
            "concept": probe_meta[layer],
            "cos_vs_reference": (cos_ref.get(layer) if cos_ref else None),
        }
        del u, r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══ 4. Save — giống v5 ═══════════════════════════════════════════════════
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    if args.save_format == "dense":
        dense = torch.zeros(num_total_layers, d_model, d_model, dtype=torch.float32)
        for layer, f in factors.items():
            dense[layer] = torch.outer(f["u"], f["r"])
        torch.save(dense, args.save_path)
    elif args.save_format == "rank1":
        torch.save({"format": "rank1_v1", "num_layers": num_total_layers,
                    "d_model": d_model, "layers": layers, "factors": factors},
                   args.save_path)
    else:
        torch.save({"format": "sparse_v1", "num_layers": num_total_layers,
                    "d_model": d_model, "layers": layers,
                    "matrices": {l: torch.outer(f["u"], f["r"]) for l, f in factors.items()}},
                   args.save_path)

    # Meta: GHI RÕ nguồn concept (thứ v5 đang thiếu)
    meta = {
        "model_name": args.model_name,
        "probe": "dim",
        "concept_source": DATASET_NAME,                 # ← phân biệt với mọi bản khác
        "concept_data": {"n_harmful": n_hh_harmful, "n_harmless": n_hh_harmless},
        "gate_data": "alphasteer_malicious",            # u fit trên D_m AlphaSteer
        "nullspace_data": "alphasteer_benign",          # P từ D_b AlphaSteer
        "lambda_reg": args.lambda_reg,
        "seed": args.seed,
        "N_malicious": int(H_malicious.shape[0]),
        "N_benign_total": int(H_benign.shape[0]),
        "N_benign_holdout": int(n_ho),
        "rho_per_layer": {str(l): ratios[l] for l in layers},
        "steering_rank": 1,
        "layers": layers,
        "ref_vectors_path": args.ref_vectors_path,
        "per_layer": diagnostics,
    }
    meta_path = os.path.splitext(args.save_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved → %s (format=%s)", args.save_path, args.save_format)
    logger.info("Meta  → %s", meta_path)
    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()