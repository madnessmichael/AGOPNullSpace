"""
calc_steering_matrix_rfm_no_nullspace.py  (v4 — concept vector từ HH dataset)
==============================================================================
ABLATION: AGOPNs KHÔNG có null-space projection.

╔══════════════════════════════════════════════════════════════════════════════╗
║ THAY ĐỔI SO VỚI v3 (ĐIỂM DUY NHẤT): concept vector r                          ║
║ ------------------------------------------------------------------------    ║
║ v3  : r tính từ RFM/AGOP trên AlphaSteer D_m (malicious/jailbreak) vs D_b     ║
║       (benign/alpaca/coconot) — cùng nguồn dùng để fit null-space/gate.       ║
║ v4  : r tính từ RFM/AGOP trên justinphan3110/harmful_harmless_instructions,   ║
║       Y HỆT calc_steering_matrix_rfm_hh.py (cùng loader, cùng balance,        ║
║       cùng convention). KHÔNG đổi gì khác.                                    ║
║                                                                              ║
║ VÌ SAO GIỮ NGUYÊN PHẦN CÒN LẠI:                                               ║
║   H_malicious/H_benign (AlphaSteer) vẫn giữ đúng vai trò cũ của chúng:        ║
║     - ablation "identity": h_m = H_malicious dùng để fit gate u qua ridge     ║
║       (P = I, KHÔNG null-space) — đây là biến số ablation đang cô lập,        ║
║       không phải nguồn của r.                                                ║
║     - ablation "vector"  : H_malicious/H_benign_ho CHỈ dùng để đo diagnostic  ║
║       (perturb_rel_*, mean_proj_*), không dùng để tính r.                     ║
║   Nói cách khác: script này đổi DATA của r giống hệt bản rfm_hh, nhưng vẫn    ║
║   đo ablation KHÔNG-null-space trên cùng một cặp D_m/D_b AlphaSteer như v3,   ║
║   để so sánh trực tiếp được với bản null-space HH (calc_steering_matrix_     ║
║   rfm_hh.py) — hai bản chỉ khác đúng một biến: null-space có/không.           ║
║                                                                              ║
║ ⚠ H1 (SỬA THÊM, KHÁC rfm_hh.py — xem review trước đó):                       ║
║   rfm_hh.py sau khi load HH tensors thì GHI ĐÈ num_total_layers/d_model       ║
║   bằng shape của HH, không validate. Nếu HH bị lệch layer do bug E1 (xem      ║
║   calc_steering_matrix_dim_hh.py), việc ghi đè này làm layer-indexing của     ║
║   toàn bộ steering matrix lệch so với AlphaSteer mà KHÔNG có cảnh báo nào.    ║
║   Ở đây: num_total_layers/d_model LUÔN lấy từ AlphaSteer (nguồn dùng để       ║
║   fit gate/dense tensor); HH bị validate/cắt cho KHỚP AlphaSteer, không       ║
║   phải ngược lại. Nếu lệch quá +1 layer → raise, không âm thầm dùng.          ║
║   Hướng cắt (+1 → bỏ đầu hay cuối) VẪN CHƯA được verify thực nghiệm — xem     ║
║   log cảnh báo bên dưới và cross-check bằng --ref_vectors_path /              ║
║   separation_ratio trước khi tin kết quả nếu nhánh này được kích hoạt.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════════════════╗
║ NHẮC LẠI VẤN ĐỀ GỐC CỦA v1/v2 (vẫn còn nguyên giá trị, KHÔNG đổi ở v4):       ║
║                                                                              ║
║   Manuscript §Baselines: "AGOPNs (w/o NullSpace) ... FIXED ADDITIVE VECTOR   ║
║   without null-space projection"                              ⇒  h + α·r     ║
║   File gốc thực sự làm: tilde_delta = ridge(H_m → r) với P = I ⇒ h + α·Mᵀh    ║
║   (input-dependent) — hai ablation khác nhau, trả lời hai câu hỏi khác nhau: ║
║     • P = I    → cô lập ĐÚNG một biến: null-space constraint.                 ║
║     • h + α·r  → bỏ CẢ null-space LẪN regression/adaptivity.                  ║
║   ⇒ Script vẫn implement CẢ HAI (--ablation identity | vector).              ║
╚══════════════════════════════════════════════════════════════════════════════╝

CÁC LỖI KHÁC ĐÃ SỬA SO VỚI FILE GỐC (giữ nguyên từ v3):

  A1. [BUG] CUDA_VISIBLE_DEVICES đặt sau import torch/manual_seed → bỏ hẳn,
      dùng biến môi trường ở shell.
  A2. [BUG, lãng phí] Không load AlphaSteer embeddings hai lần.
  A3. [BUG] Không dùng hai randperm khác nhau trên RNG global cho cùng một
      tập — load_alphasteer_embeddings() dùng generator riêng, seed cố định.
  A4. [DEAD] method=args.rfm_method chết vì compute_rfm_refusal_vectors v2
      luôn chạy RFM — đã bỏ tham số chết, --probe thực sự có tác dụng.
  A5. [PERF] Không dựng I_d (d,d) rồi nhân H@I_d — dùng thẳng Cholesky ridge.
  A6. [DOCSTRING SAI] Đã sửa mô tả cho khớp code thực tế.
  A7. [MEMORY] dense [L,d,d] chỉ dùng khi --save_format dense; có rank1/sparse.
  A8. [KHOA HỌC] λ riêng cho ablation identity (mặc định 1e4, KHÔNG dùng
      chung 10.0 với bản null-space vì phổ HᵀH lớn hơn nhiều bậc so với
      PHᵀHP đã bị cắt phổ).
  A9. [THIẾU] Diagnostic ||h@M|| / gate benign-vs-malicious, xuất JSON.
  A10. M vẫn RANK-1 (u⊗r) ở nhánh identity — công thức chỉ khác đúng cách
       tính u (ridge toàn không gian d, so với Q-space null-space).

Cách dùng:
    # Ablation chính (cô lập null-space): P = I, giữ regression, r từ HH
    CUDA_VISIBLE_DEVICES=0 python src/calc_steering_matrix_rfm_no_nullspace.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --save_path data/steering_matrix/llama3.1_agopn_hh_noNS.pt \
        --ablation identity --lambda_reg 1e4 --device cuda
    # → generate.py với --steering_matrix_path (AlphaSteer_MODELS_DICT)

    # Naive-steering baseline: h + α·r, r từ HH
    CUDA_VISIBLE_DEVICES=0 python src/calc_steering_matrix_rfm_no_nullspace.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --save_path data/steering_vector/llama3.1_agop_hh_vector.pt \
        --ablation vector --device cuda
    # → generate.py với --steering_vector_path (Steer_MODELS_DICT)

    # KHUYẾN NGHỊ MẠNH: tái dùng r đã tính từ bản null-space HH (rfm_hh.py) để
    # ablation chỉ khác bản full ĐÚNG MỘT BIẾN (null-space có/không), thay vì
    # tính lại r (có thể lệch nhẹ do RFM là stochastic ở center_grads/bw sweep):
        --refusal_vectors_path data/steering_matrix/steering_matrix_llama3.1_agopn_hh_meta.json  # (hoặc .pkl nếu bạn xuất riêng r)

    embedding_dir cần có CẢ hai:
        {embedding_dir}/embeds_benign_train.pt, embeds_coconot_*.pt,
        embeds_harmful_train_1000.pt, embeds_jailbreak_train.pt   (AlphaSteer, cho gate/diagnostics)
        {embedding_dir}/harmful_harmless_instructions/embeds_hh_harmful.pt (+harmless)  (HH, cho r)
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.const import AlphaSteer_CALCULATION_CONFIG          # noqa: E402
from utils.steering_utils import (                              # noqa: E402
    disable_tf32,
    cal_steering_factors_ridge_l,
    steering_signal_stats,
)
from rfm_refusal_vector import (                                # noqa: E402
    load_alphasteer_embeddings,
    compute_refusal_vectors,
)
# H1: tái dùng ĐÚNG loader/balance mà calc_steering_matrix_rfm_hh.py dùng, để
# concept vector r ở đây tính theo CÙNG quy trình — không viết lại logic HH
# một lần nữa ở nơi khác rồi lệch nhau.
from calc_steering_matrix_rfm_hh import (                        # noqa: E402
    load_harmful_harmless_instruction_embeddings,
    balance_harmful_harmless_embeddings,
    DATASET_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--embedding_dir", required=True,
                   help="Chứa cả AlphaSteer embeddings (gate/diagnostics) lẫn "
                        "harmful_harmless_instructions/ (nguồn của r).")
    p.add_argument("--save_path", required=True)
    p.add_argument("--device", default="cuda")

    p.add_argument("--ablation", default="identity", choices=["identity", "vector"],
                   help="identity: P=I, giữ regression (ablation của RIÊNG null-space). "
                        "vector: h + α·r, naive additive steering (bỏ cả regression).")

    # Concept vector (HH) — phải GIỐNG HỆT bản full (rfm_hh.py) để ablation hợp lệ
    p.add_argument("--probe", default="rfm", choices=["rfm", "linear"])
    p.add_argument("--rfm_iters", type=int, default=5)
    p.add_argument("--tuning_metric", default="auc", choices=["auc", "acc", "f1", "mse"])
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--include_math", action="store_true",
                   help="Chỉ ảnh hưởng D_b của AlphaSteer (gate/diagnostics), KHÔNG "
                        "ảnh hưởng nguồn HH của r.")

    # HH overrides (mặc định tự dò trong embedding_dir/harmful_harmless_instructions/,
    # giống rfm_hh.py)
    p.add_argument("--hh_harmful_path", default=None)
    p.add_argument("--hh_harmless_path", default=None)
    p.add_argument("--hh_embeddings_path", default=None)
    p.add_argument("--hh_labels_path", default=None)

    # A8: λ RIÊNG cho ablation identity — KHÔNG dùng chung 10.0 với bản null-space
    p.add_argument("--lambda_reg", type=float, default=1e4,
                   help="Chỉ dùng cho --ablation identity. Mặc định 1e4 (KHÔNG phải 10.0 "
                        "như bản null-space) vì phổ của HᵀH lớn hơn nhiều bậc so với PHᵀHP. "
                        "NÊN sweep: --lambda_reg 1e2,1e3,1e4,1e5")

    p.add_argument("--refusal_vectors_path", default=None,
                   help="Tái dùng r đã tính từ bản full HH (rfm_hh.py) — KHUYẾN NGHỊ MẠNH: "
                        "ablation phải khác bản full ĐÚNG MỘT biến (null-space). Bỏ trống "
                        "⇒ tính lại r từ HH tại đây (tốn thời gian, và RFM có thể lệch nhẹ "
                        "nếu seed/search-space khác bản full).")

    p.add_argument("--holdout_benign", type=int, default=1000)
    p.add_argument("--save_format", default="dense", choices=["dense", "rank1", "sparse"],
                   help="dense (mặc định): [L,d,d] giống AlphaSteer/DIM để so sánh trực tiếp.")
    p.add_argument("--seed", type=int, default=42,
                   help="⚠ Phải KHỚP seed dùng khi chạy calc_steering_matrix_rfm_hh.py "
                        "(bản đó default=2706) nếu muốn so sánh 2×2 hợp lệ — seed quyết "
                        "định subset jailbreak/coconot của D_m/D_b VÀ split train/val của "
                        "RFM trên HH. Truyền tường minh, đừng dựa vào default.")
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    disable_tf32()   # nhất quán với bản full (dù ablation P=I ít nhạy hơn)

    device = torch.device(args.device)
    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]
    layers = [l for l, _ in layers_ratio_list]     # ρ bị bỏ qua — đó là điểm của ablation

    # ══ 1. AlphaSteer embeddings — CHỈ cho gate (identity) / diagnostics (cả hai) ══
    # KHÔNG dùng để tính r nữa (khác v3). num_total_layers/d_model lấy từ ĐÂY
    # và LÀ NGUỒN THAM CHIẾU DUY NHẤT cho layer-indexing của toàn bộ script (H1).
    H_malicious, H_benign = load_alphasteer_embeddings(
        args.embedding_dir, seed=args.seed, include_math=args.include_math)
    num_total_layers = H_malicious.shape[1]
    d_model = H_malicious.shape[2]

    g = torch.Generator().manual_seed(args.seed + 1)
    perm = torch.randperm(H_benign.shape[0], generator=g)
    n_ho = min(args.holdout_benign, H_benign.shape[0] // 10)
    H_benign_ho = H_benign[perm[:n_ho]]            # held-out: chỉ để ĐO, không fit
    H_benign_fit = H_benign[perm[n_ho:]]           # không dùng fit gì trong ablation này;
                                                    # giữ lại để tương thích diagnostics/tái sử dụng

    logger.info("ABLATION = %s | D_m=%d D_b=%d (holdout=%d) L=%d d=%d  [AlphaSteer, gate/diag]",
                args.ablation, H_malicious.shape[0], H_benign.shape[0],
                n_ho, num_total_layers, d_model)

    # ══ 2. Concept vectors — từ HH dataset, Y HỆT calc_steering_matrix_rfm_hh.py ══
    concept_source = None
    concept_data = None

    if args.refusal_vectors_path:
        obj = torch.load(args.refusal_vectors_path, map_location="cpu") \
            if args.refusal_vectors_path.endswith(".pt") else None
        if obj is None:
            import pickle
            with open(args.refusal_vectors_path, "rb") as f:
                obj = pickle.load(f)
        rv_np = obj["refusal_vectors"] if isinstance(obj, dict) else obj
        refusal_vectors = torch.as_tensor(np.asarray(rv_np), dtype=torch.float32)
        probe_meta = {l: {"source": args.refusal_vectors_path} for l in layers}
        concept_source = args.refusal_vectors_path
        logger.info("Nạp lại refusal vectors từ %s — ablation khác bản full ĐÚNG 1 biến ✓",
                    args.refusal_vectors_path)
        if refusal_vectors.shape[0] != num_total_layers or refusal_vectors.shape[1] != d_model:
            raise ValueError(
                f"refusal_vectors_path có shape {tuple(refusal_vectors.shape)}, kỳ vọng "
                f"({num_total_layers}, {d_model}) khớp AlphaSteer d/layers hiện tại.")
    else:
        logger.warning("Không có --refusal_vectors_path ⇒ tính lại r TỪ HH DATASET. Đảm bảo "
                       "--seed, --probe, --rfm_iters, --tuning_metric giống hệt bản full "
                       "(calc_steering_matrix_rfm_hh.py) để ablation chỉ khác đúng null-space.")

        H_harmful_all, H_harmless_all = load_harmful_harmless_instruction_embeddings(
            embeddings_path=args.hh_embeddings_path or os.path.join(
                args.embedding_dir, "harmful_harmless_instructions",
                "embeds_harmful_harmless_instructions.pt"),
            labels_path=args.hh_labels_path or os.path.join(
                args.embedding_dir, "harmful_harmless_instructions",
                "labels_harmful_harmless_instructions.pt"),
            harmful_path=args.hh_harmful_path or os.path.join(
                args.embedding_dir, "harmful_harmless_instructions", "embeds_hh_harmful.pt"),
            harmless_path=args.hh_harmless_path or os.path.join(
                args.embedding_dir, "harmful_harmless_instructions", "embeds_hh_harmless.pt"),
            balance=False,
            seed=args.seed,
        )

        # H1: validate TRƯỚC khi dùng — KHÔNG ghi đè num_total_layers/d_model bằng
        # shape của HH (khác rfm_hh.py). AlphaSteer là nguồn tham chiếu vì dense
        # tensor / gate ridge ở phần 3 dùng num_total_layers/d_model của AlphaSteer.
        if H_harmless_all.shape[1] == num_total_layers + 1:
            logger.warning(
                "  HH có %d layer-entries (= AlphaSteer + 1) → cắt bỏ 1 entry để khớp "
                "convention AlphaSteer (nghi do bug E1, xem calc_steering_matrix_dim_hh.py). "
                "⚠ HƯỚNG CẮT (đầu/cuối) CHƯA verify thực nghiệm — cross-check bằng "
                "separation_ratio per layer hoặc cos với r của rfm_hh.py trước khi tin.",
                H_harmless_all.shape[1])
            H_harmful_all = H_harmful_all[:, :num_total_layers, :]
            H_harmless_all = H_harmless_all[:, :num_total_layers, :]
        elif H_harmless_all.shape[1] != num_total_layers:
            raise ValueError(
                f"HH có {H_harmless_all.shape[1]} layers, AlphaSteer có {num_total_layers}. "
                f"Không khớp và không phải +1 → kiểm tra lại extraction/convention trước khi "
                f"chạy tiếp (KHÔNG âm thầm ghi đè num_total_layers như rfm_hh.py đang làm).")
        if H_harmless_all.shape[2] != d_model:
            raise ValueError(
                f"d mismatch: HH d={H_harmless_all.shape[2]} vs AlphaSteer d={d_model}.")

        H_rfm_harmful, H_rfm_harmless = balance_harmful_harmless_embeddings(
            H_harmful_all, H_harmless_all, balance_ratio=1.0, seed=args.seed,
        )
        n_hh_harmful, n_hh_harmless = int(H_harmful_all.shape[0]), int(H_harmless_all.shape[0])
        del H_harmful_all, H_harmless_all

        logger.info("Computing %s concept vectors từ HH (metric=%s, iters=%d)...",
                    args.probe.upper(), args.tuning_metric, args.rfm_iters)
        rv_np, probe_meta = compute_refusal_vectors(
            H_malicious=H_rfm_harmful,
            H_benign=H_rfm_harmless,
            layers=layers,
            num_total_layers=num_total_layers,     # từ AlphaSteer, đã validate ở trên
            probe=args.probe, rfm_iters=args.rfm_iters,
            n_components=args.n_components, tuning_metric=args.tuning_metric,
            max_per_class=args.max_per_class, seed=args.seed, device=args.device,
        )
        del H_rfm_harmful, H_rfm_harmless

        refusal_vectors = torch.from_numpy(rv_np).float()
        concept_source = DATASET_NAME
        concept_data = {"n_harmful": n_hh_harmful, "n_harmless": n_hh_harmless}
        logger.info("refusal_vectors %s dtype=%s từ %s (n_harmful=%d n_harmless=%d)",
                    tuple(refusal_vectors.shape), refusal_vectors.dtype,
                    DATASET_NAME, n_hh_harmful, n_hh_harmless)

    # ══ 3a. ABLATION "vector": h + α·r  (naive additive) ═════════════════════
    if args.ablation == "vector":
        # KHÔNG có ma trận, KHÔNG có regression. Xuất [L, d] float32 cho
        # Steer_MODELS_DICT / NaiveSteerModel. Diagnostics vẫn đo trên AlphaSteer
        # D_m/D_b_holdout (đúng phân phối eval, không phải HH).
        diagnostics = {}
        for layer in layers:
            r = refusal_vectors[layer].to(device)
            hb = H_benign_ho[:, layer, :].to(device).float()
            hm = H_malicious[:1000, layer, :].to(device).float()
            diagnostics[str(layer)] = {
                "r_norm": float(r.norm()),
                "perturb_rel_benign": float((r.norm() / hb.norm(dim=1)).mean()),
                "perturb_rel_malicious": float((r.norm() / hm.norm(dim=1)).mean()),
                "selectivity_ratio": 1.0,   # theo định nghĩa: cùng vector cho mọi input
                "mean_proj_benign": float((hb @ r).mean()),
                "mean_proj_malicious": float((hm @ r).mean()),
            }
            logger.info("layer %d: ||r||=%.3f | α·r so với ||h||: benign %.2f%% / "
                        "malicious %.2f%% | selectivity = 1.00x (theo định nghĩa)",
                        layer, float(r.norm()),
                        100 * diagnostics[str(layer)]["perturb_rel_benign"],
                        100 * diagnostics[str(layer)]["perturb_rel_malicious"])

        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        torch.save(refusal_vectors, args.save_path)     # (L, d) float32
        _save_meta(args, num_total_layers, d_model, layers, H_malicious, H_benign,
                   n_ho, diagnostics, probe_meta, concept_source, concept_data,
                   extra={"selectivity_note":
                   "Vector ablation không chọn lọc theo định nghĩa: mọi prompt nhận cùng α·r."})
        logger.info("Saved vector → %s  shape=%s", args.save_path,
                    tuple(refusal_vectors.shape))
        logger.info("Total time: %.1fs", time.time() - t0)
        return

    # ══ 3b. ABLATION "identity": P = I, giữ regression trên AlphaSteer D_m ═══
    # M vẫn RANK-1 (u⊗r): h' = h + α·(uᵀh)·r. Khác bản null-space ĐÚNG một chỗ:
    #     u = P·A⁺·Xᵀ1  (full, Q-space)   vs   u = (HᵀH+λI)⁻¹Hᵀ1  (ablation, ridge)
    # r ở cả hai bản giờ đều từ HH — biến duy nhất còn lại là null-space có/không.
    factors: dict[int, dict] = {}
    diagnostics: dict[str, dict] = {}

    for layer in layers:
        logger.info("=== layer %d (ablation: P = I) ===", layer)
        r = refusal_vectors[layer].to(device)
        h_m = H_malicious[:, layer, :].to(device).float()

        u, r = cal_steering_factors_ridge_l(h_m, r, lambda_reg=args.lambda_reg,
                                            device=args.device)
        factors[layer] = {"u": u.detach().cpu(), "r": r.detach().cpu()}

        st_m = steering_signal_stats(h_m, u, r)
        st_b = steering_signal_stats(H_benign_ho[:, layer, :], u, r)
        sel = st_m["mean_signal_norm"] / max(st_b["mean_signal_norm"], 1e-12)
        logger.info("  gate uᵀh: malicious=%.4f±%.4f  benign_holdout=%.4f±%.4f  "
                    "→ selectivity=%.2fx", st_m["gate_mean"], st_m["gate_std"],
                    st_b["gate_mean"], st_b["gate_std"], sel)

        diagnostics[str(layer)] = {
            "lambda_reg": args.lambda_reg,
            "u_norm": float(u.norm()),
            "gate_malicious": st_m,
            "gate_benign_holdout": st_b,
            "selectivity_ratio": sel,
        }
        del h_m, u, r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ══ 4. Save ══════════════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    if args.save_format == "dense":
        dense = torch.zeros(num_total_layers, d_model, d_model, dtype=torch.float32)
        for layer, f in factors.items():
            dense[layer] = torch.outer(f["u"], f["r"])
        torch.save(dense, args.save_path)             # [L,d,d] giống AlphaSteer
    elif args.save_format == "rank1":
        torch.save({"format": "rank1_v1", "num_layers": num_total_layers,
                    "d_model": d_model, "layers": layers, "factors": factors},
                   args.save_path)
    else:  # sparse
        torch.save({"format": "sparse_v1", "num_layers": num_total_layers,
                    "d_model": d_model, "layers": layers,
                    "matrices": {l: torch.outer(f["u"], f["r"]) for l, f in factors.items()}},
                   args.save_path)

    _save_meta(args, num_total_layers, d_model, layers, H_malicious, H_benign,
               n_ho, diagnostics, probe_meta, concept_source, concept_data)

    sel = [v["selectivity_ratio"] for v in diagnostics.values()]
    logger.info("Selectivity trung bình (malicious/benign) = %.2fx "
                "— so với bản null-space; nếu ≈1 thì benign bị steer y hệt malicious.",
                float(np.mean(sel)))
    logger.info("Saved → %s", args.save_path)
    logger.info("Total time: %.1fs", time.time() - t0)


def _save_meta(args, num_total_layers, d_model, layers, H_malicious, H_benign,
               n_ho, diagnostics, probe_meta, concept_source, concept_data, extra=None):
    meta = {
        "ablation": args.ablation,
        "model_name": args.model_name,
        "probe": args.probe,
        "rfm_iters": args.rfm_iters,
        "tuning_metric": args.tuning_metric,
        "lambda_reg": args.lambda_reg if args.ablation == "identity" else None,
        # H1: ghi rõ nguồn r — giờ mặc định là HH, không phải AlphaSteer.
        "concept_source": concept_source,
        "concept_data": concept_data,
        "gate_data": "alphasteer_malicious" if args.ablation == "identity" else None,
        "nullspace_data": None,   # ablation: không có null-space
        "refusal_vectors_path": args.refusal_vectors_path,
        "seed": args.seed,
        "N_malicious": int(H_malicious.shape[0]),      # 2000 — KHÔNG phải 2720
        "N_benign_total": int(H_benign.shape[0]),      # 14000 — KHÔNG phải 14900
        "N_benign_holdout": int(n_ho),
        "num_layers": num_total_layers,
        "d_model": d_model,
        "layers": layers,
        "nullspace": False,
        "per_layer": diagnostics,
        "probe_meta": {str(k): {kk: (vv if isinstance(vv, (int, float, str, bool)) else str(vv))
                                for kk, vv in v.items()} for k, v in probe_meta.items()},
    }
    if extra:
        meta.update(extra)
    path = os.path.splitext(args.save_path)[0] + "_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Meta  → %s", path)


if __name__ == "__main__":
    main()