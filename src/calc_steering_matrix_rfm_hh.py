"""
calc_steering_matrix_rfm_hh.py  (v4)
=================================
Drop-in replacement cho src/calc_steering_matrix.py của AlphaSteer.
Thay đổi duy nhất về mặt phương pháp: refusal vector từ RFM/AGOP thay vì DIM,
tính trên justinphan3110/harmful_harmless_instructions.

THAY ĐỔI SO VỚI v2 → v3 (giữ nguyên, xem chi tiết bên dưới): D1-D5.

THAY ĐỔI MỚI Ở v4 (phát hiện khi viết calc_steering_matrix_dim_hh.py và
calc_steering_matrix_rfm_no_nullspace.py — cả hai đều cần import lại logic ở
đây, và việc đó lộ ra các chỗ sau):

  D6. [BUG NGHIÊM TRỌNG] v3 làm:
          num_total_layers = H_harmless_all.shape[1]
          d_model           = H_harmless_all.shape[2]
      NGAY SAU khi đã dùng num_total_layers (từ H_benign của AlphaSteer, bước 1)
      để KHÔNG làm gì trong bước đó — nhưng biến này được DÙNG LẠI ở bước 4 để
      cấp phát `dense = torch.zeros(num_total_layers, d_model, d_model)`. Nếu
      HH embeddings bị lệch 1 layer so với AlphaSteer (rất có thể, xem 3 bug
      E1-E3 trong calc_steering_matrix_dim_hh.py — ví dụ hidden_states có/không
      bao gồm embedding layer), việc ghi đè này khiến `dense` được cấp phát với
      SỐ LAYER SAI so với model thật, mà KHÔNG có bất kỳ cảnh báo/lỗi nào — chỉ
      lộ ra sau này dưới dạng shape mismatch khó hiểu ở generate.py, hoặc tệ hơn,
      không lộ ra gì cả nếu con số tình cờ tương thích.
      → Fix: num_total_layers/d_model LUÔN lấy từ AlphaSteer (H_benign, bước 1)
        — đây là nguồn dùng để fit null-space P và cấp phát dense tensor, nên
        phải là nguồn tham chiếu duy nhất. HH tensors được VALIDATE và CẮT cho
        khớp AlphaSteer, không phải ngược lại. Lệch quá ±1 layer → raise ngay,
        không âm thầm dùng.
        (Hướng cắt +1 — bỏ layer đầu hay cuối — VẪN CHƯA verify thực nghiệm;
        xem cảnh báo log bên dưới và cross-check bằng cos_vs_reference/
        separation_ratio trước khi tin kết quả nếu nhánh này được kích hoạt.)

  D7. [BUG] `load_harmful_harmless_instruction_embeddings` được gọi với CẢ BỐN
      path (harmful/harmless VÀ embeddings/labels) luôn luôn non-None (dựng
      bằng os.path.join). Điều kiện chọn nhánh trong hàm là `if harmful_path
      and harmless_path:` — chỉ kiểm tra CHUỖI có rỗng hay không, KHÔNG kiểm
      tra file có tồn tại. Hậu quả: nhánh "combined embeddings + labels" KHÔNG
      BAO GIỜ được dùng trong thực tế dù bạn truyền --hh_embeddings_path/
      --hh_labels_path hợp lệ, vì nhánh separated luôn được chọn trước và
      crash bằng FileNotFoundError khó hiểu nếu embeds_hh_harmful.pt không tồn
      tại — thay vì fallback rõ ràng hoặc thông báo hữu ích.
      → Fix: kiểm tra os.path.exists() cho cả hai cặp, chọn nhánh theo cái nào
        THỰC SỰ có trên đĩa (giống pattern load_hh_embeddings() của
        calc_steering_matrix_dim_hh.py), raise với thông báo liệt kê đủ 4 path
        đã thử nếu không cặp nào tồn tại.

  D8. [THIẾU] meta.json không ghi concept_source/concept_data — không phân
      biệt được (khi đọc lại meta) rằng r ở đây đến từ HH chứ không phải từ
      AlphaSteer D_m/D_b như DIM gốc. → thêm "concept_source": DATASET_NAME,
      "concept_data": {n_harmful, n_harmless}, và "gate_data"/"nullspace_data"
      để nhất quán với meta của calc_steering_matrix_dim_hh.py.

  D9. [KHOA HỌC] seed default của file này là 2706, của
      calc_steering_matrix_dim_hh.py là 42. Cả hai default đều truyền vào
      load_alphasteer_embeddings(seed=...), quyết định subset jailbreak/coconot
      của D_m/D_b VÀ benign holdout split. Nếu so sánh DIM-HH vs RFM-HH (ô 2×2
      trong docstring của calc_steering_matrix_dim_hh.py) mà không truyền
      --seed tường minh giống nhau ở cả hai lệnh gọi, hai bản dùng hai tập D_m/
      D_b khác nhau → so sánh không còn hợp lệ. → cảnh báo rõ trong help text.

  D10. Thêm --hh_harmful_path/--hh_harmless_path/--hh_embeddings_path/
       --hh_labels_path override args, khớp giao diện của
       calc_steering_matrix_dim_hh.py và calc_steering_matrix_rfm_no_nullspace.py.

Cách dùng:
    python src/calc_steering_matrix_rfm_hh.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --device cuda \
        --save_path data/steering_matrix/steering_matrix_llama3.1_agopn.pt \
        --probe rfm --rfm_iters 5 --lambda_reg 10.0 --seed 2706
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

    D7: chọn nhánh theo file NÀO THỰC SỰ TỒN TẠI trên đĩa, không phải theo
    chuỗi path có rỗng hay không. Trước đây `harmful_path`/`harmless_path`
    luôn được truyền vào (dựng sẵn bằng os.path.join ở call site) nên nhánh
    separated luôn được chọn, kể cả khi file không tồn tại và người dùng thực
    ra có sẵn combined format — kết quả là FileNotFoundError khó hiểu thay vì
    fallback đúng nhánh hoặc thông báo rõ ràng.

    Returns:
        H_harmful:  [N_harmful,  num_layers, d_model] float32 CPU
        H_harmless: [N_harmless, num_layers, d_model] float32 CPU
    """
    have_separated = bool(harmful_path) and bool(harmless_path) and \
        os.path.exists(harmful_path) and os.path.exists(harmless_path)
    have_combined = bool(embeddings_path) and bool(labels_path) and \
        os.path.exists(embeddings_path) and os.path.exists(labels_path)

    if have_separated:
        logger.info("HH source: SEPARATED files (%s / %s)", harmful_path, harmless_path)
        H_harmful = _load_tensor_or_dict(harmful_path)
        H_harmless = _load_tensor_or_dict(harmless_path)
    elif have_combined:
        logger.info("HH source: COMBINED + labels (%s)", embeddings_path)
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
    else:
        raise FileNotFoundError(
            "Không tìm thấy HH embeddings. Đã thử:\n"
            f"  separated: {harmful_path} / {harmless_path}\n"
            f"  combined : {embeddings_path} / {labels_path}\n"
            "Chạy prepare_harmful_harmless_instruction_embeddings.py trước "
            "(--save_separated để có cả hai định dạng)."
        )

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
    # D10: HH overrides — khớp giao diện của calc_steering_matrix_dim_hh.py /
    # calc_steering_matrix_rfm_no_nullspace.py. Mặc định None ⇒ tự dò trong
    # {embedding_dir}/harmful_harmless_instructions/.
    p.add_argument("--hh_harmful_path", default=None)
    p.add_argument("--hh_harmless_path", default=None)
    p.add_argument("--hh_embeddings_path", default=None)
    p.add_argument("--hh_labels_path", default=None)
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
    p.add_argument("--seed", type=int, default=2706,
                   help="⚠ D9: calc_steering_matrix_dim_hh.py default=42, file này "
                        "default=2706. Seed quyết định subset jailbreak/coconot của D_m/D_b "
                        "VÀ benign holdout split (qua load_alphasteer_embeddings). Nếu so "
                        "sánh DIM-HH vs RFM-HH, truyền --seed TƯỜNG MINH giống nhau ở cả hai "
                        "lệnh gọi, đừng dựa vào default của từng file.")
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
    # jailbreak khác nhau. Ở đây H_malicious/H_benign được dùng cho gate/null-space.
    H_malicious, H_benign = load_alphasteer_embeddings(
        args.embedding_dir, seed=args.seed, include_math=args.include_math)

    # D6: num_total_layers/d_model LUÔN lấy từ AlphaSteer — đây là nguồn tham
    # chiếu DUY NHẤT cho layer-indexing của null-space P, gate u, và dense
    # tensor ở bước 4. KHÔNG bị ghi đè bởi shape của HH ở bước 2 nữa.
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

    # ══ 2. Concept vectors qua RFM/AGOP trên HH dataset ═══════════════════════
    # *** ĐIỂM THAY ĐỔI DUY NHẤT so với AlphaSteer gốc ***
    #   Gốc: refusal_vectors = pickle.load(RV_PATH)          → DIM
    #   Đây: compute_refusal_vectors(...) trên HH             → AGOP top eigenvector
    logger.info("Computing %s concept vectors (metric=%s, iters=%d)...",
                args.probe.upper(), args.tuning_metric, args.rfm_iters)

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

    # D6: validate/cắt HH cho KHỚP num_total_layers/d_model của AlphaSteer,
    # KHÔNG ghi đè num_total_layers/d_model bằng shape của HH như v3.
    if H_harmless_all.shape[1] == num_total_layers + 1:
        logger.warning(
            "HH có %d layer-entries (= AlphaSteer + 1) → cắt bỏ 1 entry để khớp convention "
            "AlphaSteer (nghi do bug E1, xem calc_steering_matrix_dim_hh.py). ⚠ HƯỚNG CẮT "
            "(đầu hay cuối) CHƯA verify thực nghiệm — cross-check bằng separation_ratio "
            "per-layer hoặc cos với r của một bản DIM-HH/RFM-HH đã biết đúng, trước khi tin "
            "kết quả.", H_harmless_all.shape[1])
        H_harmful_all = H_harmful_all[:, :num_total_layers, :]
        H_harmless_all = H_harmless_all[:, :num_total_layers, :]
    elif H_harmless_all.shape[1] != num_total_layers:
        raise ValueError(
            f"HH có {H_harmless_all.shape[1]} layers, AlphaSteer có {num_total_layers}. "
            f"Không khớp và không phải +1 → kiểm tra lại extraction/convention trước khi "
            f"chạy tiếp.")
    if H_harmless_all.shape[2] != d_model:
        raise ValueError(f"d mismatch: HH d={H_harmless_all.shape[2]} vs AlphaSteer d={d_model}.")

    # Ghi lại số liệu HH GỐC (trước balance) cho meta — khớp convention của
    # calc_steering_matrix_dim_hh.py.
    n_hh_harmful, n_hh_harmless = int(H_harmful_all.shape[0]), int(H_harmless_all.shape[0])

    H_rfm_harmful, H_rfm_harmless = balance_harmful_harmless_embeddings(
        H_harmful_all,
        H_harmless_all,
        balance_ratio=1.0,
        seed=args.seed,
    )
    del H_harmful_all, H_harmless_all

    refusal_vectors_np, probe_meta = compute_refusal_vectors(
        H_malicious=H_rfm_harmful,
        H_benign=H_rfm_harmless,
        layers=layers,
        num_total_layers=num_total_layers,   # từ AlphaSteer, đã validate ở trên
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
    factors: dict[int, dict] = {}
    diagnostics: dict[str, dict] = {}

    for layer in layers:
        ratio = ratios[layer]
        logger.info("=== layer %d (ρ=%.2f) ===", layer, ratio)

        h_b = H_benign_fit[:, layer, :].to(device).float()
        ns_dtype = torch.float64 if args.nullspace_dtype == "float64" else None
        Q_layer = null_space_basis_l(h_b, abs_nullspace_ratio=ratio, dtype=ns_dtype)
        Q_layer = Q_layer.float()
        del h_b

        leak = benign_leakage(H_benign_ho[:, layer, :], Q_layer)
        logger.info("  benign held-out leakage ||Qᵀh||/||h||: mean=%.4f p95=%.4f max=%.4f",
                    leak["leakage_mean"], leak["leakage_p95"], leak["leakage_max"])

        h_m = H_malicious[:, layer, :].to(device).float()
        u, r = cal_steering_factors_q(
            h_m, Q_layer, refusal_vectors[layer].to(device),
            lambda_reg=args.lambda_reg, device=args.device)
        del h_m, Q_layer

        factors[layer] = {"u": u.detach().cpu(), "r": r.detach().cpu()}

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
        dense = torch.zeros(num_total_layers, d_model, d_model, dtype=torch.float32)
        for layer, f in factors.items():
            dense[layer] = torch.outer(f["u"], f["r"])   # M = P Δ̃ = u rᵀ
        torch.save(dense, args.save_path)
    elif args.save_format == "rank1":
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
        # D8: ghi rõ nguồn r — HH, không phải AlphaSteer D_m/D_b.
        "concept_source": DATASET_NAME,
        "concept_data": {"n_harmful": n_hh_harmful, "n_harmless": n_hh_harmless},
        "gate_data": "alphasteer_malicious",     # u fit trên D_m AlphaSteer
        "nullspace_data": "alphasteer_benign",   # P từ D_b AlphaSteer
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
    logger.info("Meta  → %s", meta_path)
    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()