"""
rfm_refusal_vector.py  (v3 — aligned with xRFM + neural_controllers)
====================================================================
Trích xuất AGOP concept vector cho AlphaSteer/AGOPNs pipeline.

Bản này được viết lại để KHỚP CHÍNH XÁC với reference implementation:
  - xRFM                : xrfm/rfm_src/recursive_feature_machine.py
  - neural_controllers  : direction_utils.py::train_rfm_probe_on_concept
                          control_toolkits.py::RFMToolkit._compute_directions
                          control_toolkits.py::RFMToolkit._compute_signs

THAY ĐỔI SO VỚI v2 (và lý do):
  R1. RFM(...) THIẾU `tuning_metric`  → mặc định 'mse'.
      Hậu quả: best_iter / early_stop / best_M / agop_best_model đều được
      chọn theo MSE, trong khi vòng ngoài lại chọn theo AUC → AGOP lấy ra
      KHÔNG phải AGOP của model AUC-best.
      → Fix: truyền tuning_metric='auc' vào constructor như reference.

  R2. `center_grads=True` bị hard-code.
      Reference search {True, False}. Ngoài ra AGOP theo định nghĩa
      (Deep Neural Feature Ansatz) là UNCENTERED gradient covariance;
      với binary probe, gradient trung bình CHÍNH LÀ hướng phân biệt, nên
      centering xoá đúng thành phần ta cần.
      → Fix: đưa center_grads vào search space (mặc định [True, False]).

  R3. Split "stratified" nhưng KHÔNG shuffle: `pos_idx[:nv_pos]` lấy 400 mẫu
      ĐẦU của H_refusal = toàn bộ AdvBench, không có jailbreak nào.
      → Fix: dùng sklearn train_test_split(stratify=y, shuffle=True).

  R4. `randperm` gọi lại mỗi layer → mỗi layer dùng benign subset khác nhau.
      → Fix: 1 permutation cố định (torch.Generator có seed), tái dùng cho
        mọi layer.

  R5. `torch.lobpcg` gọi trần, không regularization/fallback; nếu fail hoặc
      nếu tất cả RFM fit fail (`best_model is None`) → AttributeError khó hiểu
      (fallback đã bị comment mất).
      → Fix: top_eigenvectors() theo pattern xrfm/rfm_src/utils.py::
        get_top_eigenvector (eps*I + fallback eigh); raise RuntimeError rõ ràng.

  R6. Tham số `method` bị bỏ qua hoàn toàn (luôn chạy RFM dù truyền "linear").
      → Fix: bỏ hẳn tham số chết; `--probe {rfm,linear}` thực sự có tác dụng.

  R7. Sign calibrate trên toàn bộ X (train+val).
      → Fix: calibrate trên TRAIN split, dùng pearson_corr +
        project_onto_direction đúng như RFMToolkit._compute_signs.

  R8. `deepcopy(model)` mỗi khi có model tốt hơn → tốn VRAM vô ích.
      → Fix: chỉ clone `agop_best_model` (thứ duy nhất được dùng về sau).

GHI CHÚ VỀ NGỮ NGHĨA (quan trọng cho manuscript):
  y = 1 gán cho D_m (malicious/jailbreak), y = 0 cho D_b (benign).
  Sign được calibrate sao cho corr(Z @ r, y) > 0, tức r trỏ về phía lớp
  MALICIOUS. Đây là "harmfulness direction", KHÔNG phải "refusal direction"
  theo nghĩa của AlphaSteer (vốn lấy D_r = prompt model thực sự đã từ chối).
  Hàm trả về r với quy ước này; nếu cần hướng refusal thật, xem
  `--dr_from_responses` (chưa implement — cần chạy model để lọc refusal).
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from xrfm import RFM

try:
    from utils.steering_utils import disable_tf32
except Exception:   # chạy standalone ngoài repo
    def disable_tf32(verbose=True):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers — sao chép ngữ nghĩa từ neural_controllers/direction_utils.py
# ══════════════════════════════════════════════════════════════════════════════

def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """direction_utils.py::pearson_corr (nguyên văn)."""
    assert x.shape == y.shape
    x = x.float() + 0.0
    y = y.float() + 0.0
    xc = x - x.mean()
    yc = y - y.mean()
    num = torch.sum(xc * yc)
    den = torch.sqrt(torch.sum(xc ** 2) * torch.sum(yc ** 2))
    return num / den.clamp(min=1e-12)


def project_onto_direction(tensors: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """direction_utils.py::project_onto_direction — chỉ là `tensors @ direction`."""
    assert tensors.dim() == 2
    assert tensors.shape[1] == direction.shape[0]
    return tensors @ direction.to(device=tensors.device, dtype=tensors.dtype)


def compute_prediction_metrics(preds, labels) -> Dict[str, float]:
    """Rút gọn direction_utils.py::compute_prediction_metrics cho binary."""
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = labels.reshape(-1, 1) if labels.ndim == 1 else labels
    preds = preds.reshape(labels.shape)
    auc = roc_auc_score(labels, preds)
    mse = float(np.mean((preds - labels) ** 2))
    acc = float(np.mean((preds >= 0.5) == (labels >= 0.5)) * 100)
    return {"auc": float(auc), "mse": mse, "acc": acc}


def top_eigenvectors(M: torch.Tensor, k: int = 1, eps: float = 1e-6
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k eigenpairs của ma trận đối xứng PSD.

    Theo pattern của xrfm/rfm_src/utils.py::get_top_eigenvector:
    lobpcg (nhanh, O(d^2 k)) với regularization eps*I, fallback eigh nếu fail.
    Code cũ gọi `torch.lobpcg(agop, k=1)` trần — chính là pattern mà xRFM
    cố tình bọc lại vì lobpcg hay fail trên ma trận ill-conditioned.

    Returns
    -------
    S : (k,)   eigenvalues giảm dần
    U : (d, k) eigenvectors (cột), đã chuẩn hoá
    """
    d = M.shape[0]
    M = 0.5 * (M + M.T)  # ép đối xứng: AGOP về lý thuyết đối xứng, thực tế lệch ~1e-7

    try:
        M_reg = M + eps * torch.eye(d, device=M.device, dtype=M.dtype)
        S, U = torch.lobpcg(M_reg, k=k, largest=True)
        if torch.isfinite(S).all() and torch.isfinite(U).all():
            return S, U
        logger.warning("  lobpcg trả về NaN/Inf → fallback eigh")
    except Exception as e:  # noqa: BLE001
        logger.warning("  lobpcg failed (%s) → fallback eigh", e)

    evals, evecs = torch.linalg.eigh(M)          # tăng dần
    S = evals.flip(0)[:k]
    U = evecs.flip(1)[:, :k]
    return S, U


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — RFM probe  (mirror: direction_utils.py::train_rfm_probe_on_concept)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SEARCH_SPACE = {
    "regs": [1e-3, 1e-2],
    "bws": [1.0, 10.0, 100.0],
    "center_grads": [True, False],   # R2: reference search cả hai
}


def train_rfm_probe_on_concept(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    rfm_iters: int = 5,
    n_components: int = 1,
    tuning_metric: str = "auc",
    search_space: Optional[dict] = None,
    device: str = "cuda",
) -> Tuple[torch.Tensor, dict]:
    """
    Grid-search RFM probe, trả về AGOP của model tốt nhất.

    Khác v2 ở 3 điểm, tất cả đều để khớp reference:
      1. `tuning_metric` được truyền vào RFM(...) → internal early-stop /
         best_iter / agop_best_model đều nhất quán với metric vòng ngoài.
      2. `center_grads` nằm trong search space.
      3. Chỉ clone AGOP thay vì deepcopy cả model.

    Returns
    -------
    best_agop : (d, d) trên `device`
    info      : dict metadata (best_score/bw/reg/center_grads/n_fits_ok)
    """
    if search_space is None:
        search_space = DEFAULT_SEARCH_SPACE

    maximize = tuning_metric in ("f1", "auc", "acc", "top_agop_vectors_ols_auc")
    best_score = float("-inf") if maximize else float("inf")
    best_agop = None
    best_cfg: Dict[str, object] = {}
    n_ok = 0
    n_fail = 0

    for reg in search_space["regs"]:
        for bw in search_space["bws"]:
            for center_grads in search_space["center_grads"]:
                try:
                    model = RFM(
                        kernel="l2_high_dim",
                        bandwidth=bw,
                        tuning_metric=tuning_metric,   # ← R1
                        device=device,
                        verbose=False,
                    )
                    model.fit(
                        (train_X, train_y),
                        (val_X, val_y),
                        reg=reg,
                        iters=rfm_iters,
                        center_grads=center_grads,     # ← R2
                        early_stop_rfm=True,
                        get_agop_best_model=True,
                        top_k=n_components,
                    )

                    preds = model.predict(val_X)
                    val_score = compute_prediction_metrics(preds, val_y)[tuning_metric]
                    n_ok += 1

                    improved = (val_score > best_score) if maximize else (val_score < best_score)
                    if improved:
                        best_score = val_score
                        # R8: chỉ giữ AGOP, không deepcopy cả RFM (centers + M + weights)
                        best_agop = model.agop_best_model.detach().clone()
                        best_cfg = {
                            "reg": reg,
                            "bandwidth": bw,
                            "center_grads": center_grads,
                            "best_iter": int(model.best_iter),
                        }

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:  # noqa: BLE001
                    n_fail += 1
                    logger.warning(
                        "  RFM fit failed (bw=%.1f reg=%.0e center_grads=%s): %s",
                        bw, reg, center_grads, e,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # R5: fail rõ ràng thay vì AttributeError trên None
    if best_agop is None:
        raise RuntimeError(
            f"Tất cả {n_fail} cấu hình RFM đều fail — không có AGOP nào để trích xuất. "
            f"Kiểm tra VRAM, dtype (phải float32), và shape của y (phải [N,1])."
        )

    logger.info(
        "  RFM best %s=%.4f | bw=%.1f reg=%.0e center_grads=%s best_iter=%d (%d/%d fits ok)",
        tuning_metric, best_score, best_cfg["bandwidth"], best_cfg["reg"],
        best_cfg["center_grads"], best_cfg["best_iter"], n_ok, n_ok + n_fail,
    )
    best_cfg["score"] = best_score
    best_cfg["tuning_metric"] = tuning_metric
    return best_agop, best_cfg


def train_linear_probe_on_concept(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, dict]:
    """
    Ridge probe làm baseline (mirror direction_utils.py::train_linear_probe_on_concept:
    cùng reg_search_space, cùng linear_solve, cùng tuning bằng AUC).

    AGOP của một predictor tuyến tính f(z) = z@beta là beta·beta^T (rank-1),
    nên top eigenvector = beta/||beta||. Trả về AGOP để dùng chung downstream.
    """
    reg_space = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1]
    X = train_X.to(device).float()
    y = train_y.to(device).float()
    Xv = val_X.to(device).float()

    n, d = X.shape
    best_beta, best_score, best_reg = None, float("-inf"), None
    XtX = X.T @ X
    Xty = X.T @ y
    eye = torch.eye(d, device=device, dtype=X.dtype)
    for reg in reg_space:
        beta = torch.linalg.solve(XtX + reg * n * eye, Xty)   # (d, 1)
        score = compute_prediction_metrics(Xv @ beta, val_y)["auc"]
        if score > best_score:
            best_score, best_beta, best_reg = score, beta, reg

    beta = best_beta.squeeze(-1)
    logger.info("  Linear probe best auc=%.4f (reg=%.0e)", best_score, best_reg)
    agop = torch.outer(beta, beta)
    return agop, {"score": best_score, "reg": best_reg, "probe": "linear"}


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — Concept vector cho 1 layer
# ══════════════════════════════════════════════════════════════════════════════

def compute_concept_vector_layer(
    H_pos: torch.Tensor,
    H_neg: torch.Tensor,
    probe: str = "rfm",
    rfm_iters: int = 5,
    n_components: int = 1,
    tuning_metric: str = "auc",
    val_ratio: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Parameters
    ----------
    H_pos : (n_pos, d) activation của D_m (malicious), label y = 1
    H_neg : (n_neg, d) activation của D_b (benign),    label y = 0
            Hai tensor này đã được cân bằng/subsample ở tầng gọi.

    Returns
    -------
    agop       : (d, d) trên CPU
    components : (n_components, d) trên CPU, đã calibrate dấu
    info       : dict
    """
    dev = torch.device(device)
    n_pos, d = H_pos.shape
    n_neg = H_neg.shape[0]

    X = torch.cat([H_pos, H_neg], dim=0).float()
    y = torch.cat([
        torch.ones(n_pos, 1),
        torch.zeros(n_neg, 1),
    ], dim=0).float()

    # ── R3: stratified split THỰC SỰ ngẫu nhiên ──────────────────────────────
    # v2 dùng pos_idx[:nv_pos] → val positive toàn bộ là AdvBench, không có
    # jailbreak nào (vì H_pos = cat([harmful_1000, jailbreak_1000]) và H_pos
    # không bị shuffle khi len(H_pos) == n_min). Val AUC khi đó không đo được
    # năng lực trên jailbreak — đúng distribution mà bài báo tuyên bố mạnh.
    idx = np.arange(len(X))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=y.squeeze(-1).numpy(),
    )
    tr_idx = torch.from_numpy(tr_idx)
    va_idx = torch.from_numpy(va_idx)

    train_X, train_y = X[tr_idx].to(dev), y[tr_idx].to(dev)
    val_X, val_y = X[va_idx].to(dev), y[va_idx].to(dev)

    logger.info(
        "  train %s / val %s  (pos ratio: train=%.3f val=%.3f)",
        tuple(train_X.shape), tuple(val_X.shape),
        train_y.mean().item(), val_y.mean().item(),
    )

    if probe == "rfm":
        agop, info = train_rfm_probe_on_concept(
            train_X, train_y, val_X, val_y,
            rfm_iters=rfm_iters, n_components=n_components,
            tuning_metric=tuning_metric, device=device,
        )
    elif probe == "linear":
        agop, info = train_linear_probe_on_concept(train_X, train_y, val_X, val_y, device=device)
    else:
        raise ValueError(f"probe phải là 'rfm' hoặc 'linear', nhận: {probe}")

    # ── Top eigenvectors (RFMToolkit._compute_directions: components = U.T) ──
    S, U = top_eigenvectors(agop, k=n_components)
    components = U.T.contiguous()                      # (k, d)
    logger.info("  AGOP top-%d eigenvalues: %s", n_components,
                np.array2string(S.detach().cpu().numpy(), precision=4))

    # ── R7: sign calibrate trên TRAIN split (mirror _compute_signs) ──────────
    for c in range(n_components):
        proj = project_onto_direction(train_X, components[c])
        sign = 2 * (pearson_corr(train_y.squeeze(-1), proj) > 0).float() - 1
        components[c] = components[c] * sign
        info[f"corr_c{c}"] = float(pearson_corr(
            train_y.squeeze(-1), project_onto_direction(train_X, components[c])))

    logger.info("  corr(Z_train @ r, y_train) = %+.4f  (>0 ⇒ r trỏ về lớp malicious)",
                info["corr_c0"])

    del X, y, train_X, train_y, val_X, val_y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return agop.detach().cpu(), components.detach().cpu(), info


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — Per-layer loop
# ══════════════════════════════════════════════════════════════════════════════

def _balanced_indices(n_pos: int, n_neg: int, seed: int, max_per_class: Optional[int] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    R4: sinh MỘT permutation cố định dùng chung cho MỌI layer.
    v2 gọi randperm bên trong vòng lặp layer → mỗi layer dùng benign subset
    khác nhau, khiến concept vector giữa các layer không so sánh được.
    """
    g = torch.Generator().manual_seed(seed)
    n = min(n_pos, n_neg)
    if max_per_class is not None:
        n = min(n, max_per_class)
    pos_sel = torch.randperm(n_pos, generator=g)[:n]
    neg_sel = torch.randperm(n_neg, generator=g)[:n]
    return pos_sel, neg_sel


def compute_refusal_vectors(
    H_malicious: torch.Tensor,      # (N_m, L, d) CPU
    H_benign: torch.Tensor,         # (N_b, L, d) CPU
    layers: Sequence[int],
    num_total_layers: int,
    probe: str = "rfm",
    rfm_iters: int = 5,
    n_components: int = 1,
    tuning_metric: str = "auc",
    balance: bool = True,
    max_per_class: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[np.ndarray, dict]:
    """
    Returns
    -------
    refusal_vectors : (num_total_layers, d) float32 — layer không tính = zero
                      (giữ đúng format pkl DIM của AlphaSteer)
    meta            : dict[layer] -> info
    """
    assert H_malicious.dim() == 3 and H_benign.dim() == 3
    assert H_malicious.shape[1] == H_benign.shape[1] == num_total_layers
    d = H_malicious.shape[2]

    refusal_vectors = np.zeros((num_total_layers, d), dtype=np.float32)
    meta: Dict[int, dict] = {}

    if balance:
        pos_sel, neg_sel = _balanced_indices(
            H_malicious.shape[0], H_benign.shape[0], seed=seed, max_per_class=max_per_class)
        logger.info("Balanced: %d pos / %d neg (permutation cố định cho mọi layer)",
                    len(pos_sel), len(neg_sel))
    else:
        pos_sel = torch.arange(H_malicious.shape[0])
        neg_sel = torch.arange(H_benign.shape[0])

    for layer_idx in layers:
        logger.info("=== Layer %d ===", layer_idx)
        h_pos = H_malicious[pos_sel, layer_idx, :].float()
        h_neg = H_benign[neg_sel, layer_idx, :].float()

        _, components, info = compute_concept_vector_layer(
            h_pos, h_neg,
            probe=probe, rfm_iters=rfm_iters, n_components=n_components,
            tuning_metric=tuning_metric, seed=seed, device=device,
        )

        r = components[0]
        r = r / r.norm().clamp(min=1e-8)   # lobpcg đã trả unit vector; ép lại cho chắc
        refusal_vectors[layer_idx] = r.numpy()
        meta[layer_idx] = info
        logger.info("  ||r[%d]|| = %.6f", layer_idx, float(np.linalg.norm(refusal_vectors[layer_idx])))

    return refusal_vectors, meta


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — Load embeddings
# ══════════════════════════════════════════════════════════════════════════════

def load_alphasteer_embeddings(embedding_dir: str, seed: int = 42, include_math: bool = False):
    """
    Load D_m (malicious) và D_b (benign) theo đúng calc_steering_matrix.py gốc.

    QUAN TRỌNG — số liệu thật (đối chiếu manuscript):
      D_m = 1000 (harmful_train) + 1000 (jailbreak_train subsample) = 2,000
            → KHÔNG phải 2,720 như manuscript ghi (2,720 = 720 + 2,000, đếm trùng)
      D_b = 10,000 (alpaca) + 4,000 (coconot) = 14,000
            → KHÔNG phải 14,900; 900 mẫu MATH KHÔNG nằm trong D_b (đúng với gốc).
              Bật include_math=True nếu muốn khớp con số 14,900 trong bài.

    Dùng torch.Generator có seed → reproducible và ĐỘC LẬP với RNG global,
    nên gọi hàm này nhiều lần luôn cho cùng subset (v2 dùng torch.randperm
    global nên hai lần gọi cho hai subset khác nhau).
    """
    logger.info("Loading embeddings from %s", embedding_dir)
    g = torch.Generator().manual_seed(seed)

    def _load(fname: str) -> torch.Tensor:
        path = os.path.join(embedding_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return torch.load(path, map_location="cpu").float()   # luôn float32

    # ── D_b (benign / compliant) ────────────────────────────────────────────
    H_alpaca = _load("embeds_benign_train.pt")            # 10,000
    H_coconot_pref = _load("embeds_coconot_pref.pt")
    H_coconot_orig = _load("embeds_coconot_original.pt")
    n_borderline = 4000 - H_coconot_pref.size(0)
    idx_b = torch.randperm(H_coconot_orig.size(0), generator=g)[:n_borderline]
    parts = [H_alpaca, H_coconot_orig[idx_b], H_coconot_pref]
    if include_math:
        parts.append(_load("embeds_math_train.pt"))       # 900
    H_benign = torch.cat(parts, dim=0)

    # ── D_m (malicious) ─────────────────────────────────────────────────────
    H_harmful = _load("embeds_harmful_train_1000.pt")     # 1,000
    H_jb_full = _load("embeds_jailbreak_train.pt")
    idx_jb = torch.randperm(H_jb_full.size(0), generator=g)[:1000]
    H_malicious = torch.cat([H_harmful, H_jb_full[idx_jb]], dim=0)

    logger.info("D_m (malicious) %s", tuple(H_malicious.shape))
    logger.info("D_b (benign)    %s", tuple(H_benign.shape))
    return H_malicious, H_benign


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--layers", required=True,
                   help="Comma-separated, vd '8,9,10,11,12,13,14,16,18,19'")
    # R6: tham số này giờ THỰC SỰ có tác dụng (v2 bỏ qua hoàn toàn `method`)
    p.add_argument("--probe", default="rfm", choices=["rfm", "linear"])
    p.add_argument("--rfm_iters", type=int, default=5,
                   help="Reference dùng hyperparams['rfm_iters']; manuscript ghi T∈{1,2,5,10}")
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--tuning_metric", default="auc", choices=["auc", "acc", "f1", "mse"])
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--include_math", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    disable_tf32()   # RFM/AGOP cũng nhạy với TF32 khi d lớn

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    H_m, H_b = load_alphasteer_embeddings(args.embedding_dir, seed=args.seed,
                                          include_math=args.include_math)
    num_total_layers = H_m.shape[1]

    logger.info("layers=%s d=%d probe=%s metric=%s device=%s",
                layers, H_m.shape[2], args.probe, args.tuning_metric, args.device)

    rv, meta = compute_refusal_vectors(
        H_malicious=H_m, H_benign=H_b,
        layers=layers, num_total_layers=num_total_layers,
        probe=args.probe, rfm_iters=args.rfm_iters,
        n_components=args.n_components, tuning_metric=args.tuning_metric,
        max_per_class=args.max_per_class, seed=args.seed, device=args.device,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump({"refusal_vectors": rv, "meta": meta, "layers": layers}, f)
    logger.info("Saved → %s  shape=%s", args.save_path, rv.shape)