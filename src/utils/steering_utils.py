"""
steering_utils.py  (v4)
=======================
Null-space + regularized regression cho AlphaSteer/AGOPNs.

╔══════════════════════════════════════════════════════════════════════════════╗
║ V4 SỬA LỖI "P không idempotent" TẬN GỐC — bằng cách KHÔNG BAO GIỜ DỰNG P.    ║
║                                                                              ║
║ Triệu chứng (v3): assert P² = P fail với rel_err = 2.31e-3 ở d=4096/CUDA.    ║
║ Đo được: fp32 thuần (LAPACK CPU) cho ~1e-6; TF32 (mantissa 10 bit) cho       ║
║ ~7e-4..2e-3. ⇒ thủ phạm là TF32 bật sẵn trên Ampere/Blackwell, KHÔNG phải    ║
║ dữ liệu hỏng. Assert của v3 vừa quá chặt vừa đo sai chỗ. Lỗi của tôi.        ║
║                                                                              ║
║ Nhưng vá assert chỉ là chữa triệu chứng. Vấn đề THẬT: cả pipeline đang giải  ║
║ một bài toán k chiều bằng cách nhúng nó vào không gian d chiều, tạo ra một   ║
║ ma trận SUY BIẾN nhân tạo rồi phải dùng pseudo-inverse để gỡ:                ║
║     cond(A)         trong không gian d = 3.0e+20   ← suy biến, buộc pinv     ║
║     cond(YᵀY + λI)  trong không gian k = 2.7e+03   ← SPD, Cholesky là đủ     ║
║                                                                              ║
║ ĐẠI SỐ MỚI. Với Q ∈ R^{d×k} trực chuẩn, P = QQᵀ, Y = H_m Q ∈ R^{N×k}:        ║
║     XᵀX + λPᵀP = Q (YᵀY + λI_k) Qᵀ                                           ║
║     ⇒ A⁺        = Q (YᵀY + λI_k)⁻¹ Qᵀ            (Q trực chuẩn)              ║
║     ⇒ v = A⁺Xᵀ1 = Q (YᵀY + λI_k)⁻¹ Yᵀ1_N                                     ║
║     ⇒ u = Pv    = Q w,   w = (YᵀY + λI_k)⁻¹ Yᵀ1_N                            ║
║                                                                              ║
║ Kiểm chứng: ||u_cũ − u_mới|| / ||u_cũ|| = 1.7e-12  (verify_qspace.py)        ║
║                                                                              ║
║ LỢI ÍCH:                                                                     ║
║  1. KHÔNG dựng P (d×d) ⇒ không còn gì để "không idempotent". Assert biến mất.║
║  2. u = Qw ⇒ u ∈ range(Q) THEO CẤU TRÚC, không phụ thuộc sai số số học       ║
║     (đo được ||u − QQᵀu||/||u|| = 2.4e-15). Guarantee utility trở thành      ║
║     tính chất đại số, không còn là "hy vọng số học".                         ║
║  3. YᵀY + λI_k là SPD khi λ>0 ⇒ Cholesky. KHÔNG pinv, KHÔNG eigh của d×d.    ║
║  4. Nhanh & nhẹ hơn nhiều: pinv(4096²) vài giây → cholesky(2457²) vài ms.    ║
║  5. Leakage: ||P h_b|| = ||QQᵀh_b|| = ||Qᵀh_b|| (Q trực chuẩn) — không cần P.║
║                                                                              ║
║ P chỉ còn tồn tại trong null_space_projection_l() cho tương thích ngược.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

CẤU TRÚC RANK-1 (giữ từ v3):
    Target R = 1_N rᵀ là rank-1 ⇒ Δ̃ = v rᵀ, M = P Δ̃ = u rᵀ, và  Mᵀh = (uᵀh)·r
    AGOPNs = additive steering với CỔNG TUYẾN TÍNH uᵀh. Hướng luôn là r.
    Null-space làm ĐÚNG MỘT việc: ép u ∈ range(Q) ⇒ uᵀh_b = wᵀ(Qᵀh_b) ≈ 0.
    Inference O(d), không phải O(d²). Lưu trữ 70B: 7.0 GB → 1.7 MB.

QUY ƯỚC VECTOR (chỗ manuscript sai — xem verify_math.py):
    P là LEFT factor của Δ̃, không phải right factor.
    ⇒ h' = h + α·Mᵀh  (= h + α·hM, đúng như code inference) CÓ guarantee.
      h' = h + α·M h   (Eq.(1) của manuscript)              KHÔNG có guarantee.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "disable_tf32", "check_orthonormal",
    "null_space_basis_l", "null_space_projection_l", "cal_P",
    "cal_steering_factors_q", "cal_steering_factors_ridge_l",
    "benign_leakage", "steering_signal_stats",
    "pinv_apply", "cal_tilde_delta_l", "cal_tilde_delta",
    "cal_tilde_delta_with_regularization_l", "cal_tilde_delta_with_regularization",
    "cal_tilde_delta_ridge_l", "cal_steering_matrix_l", "cal_steering_matrix",
]


# ══════════════════════════════════════════════════════════════════════════════
# TF32
# ══════════════════════════════════════════════════════════════════════════════

def disable_tf32(verbose: bool = True) -> None:
    """
    Tắt TF32. GỌI Ở ĐẦU MỌI SCRIPT tính steering.

    TF32 (Ampere/Ada/Hopper/Blackwell) chỉ có 10 bit mantissa (eps ≈ 4.9e-4) so
    với 23 bit của fp32 (eps ≈ 1.2e-7). Toàn bộ phương pháp dựa vào Qᵀh_b ≈ 0 —
    một phép TRIỆT TIÊU tổng d = 4096 số hạng. Sai số TF32 hoàn toàn có thể lớn
    hơn chính đại lượng ta muốn đo.

    Đo được (Q trực chuẩn chính xác, kiểm tra ||P²−P||):
        fp32 thuần : 2.5e-07
        TF32       : 7.1e-04     ← ~2800x tệ hơn
    """
    for setter in (
        lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", False),
        lambda: setattr(torch.backends.cudnn, "allow_tf32", False),
        lambda: setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee"),  # torch>=2.9
        lambda: torch.set_float32_matmul_precision("highest"),
    ):
        try:
            setter()
        except (AttributeError, RuntimeError, TypeError):
            pass
    if verbose:
        logger.info("TF32 disabled (fp32 IEEE). Bắt buộc: null-space dựa vào triệt tiêu "
                    "Qᵀh_b ≈ 0; TF32 (10-bit mantissa) sẽ phá nó.")


# ══════════════════════════════════════════════════════════════════════════════
# Null space — trả về CƠ SỞ Q, không phải projector P
# ══════════════════════════════════════════════════════════════════════════════

def check_orthonormal(Q: torch.Tensor, name: str = "Q",
                      warn_tol: float = 1e-4, fail_tol: float = 1e-2) -> float:
    """
    Đo ||QᵀQ − I_k||_F / √k — đại lượng ĐÚNG cần kiểm tra.
    (v3 kiểm tra P² = P trên một sub-block ngẫu nhiên: vừa gián tiếp, vừa có
     mẫu số xấu, vừa đặt ngưỡng theo cảm tính.)

    Ngưỡng tham chiếu, đo thực nghiệm (d=1024..2048, κ≈4200):
        fp64        : ~3e-15
        fp32 (CPU)  : ~2e-6
        fp32 + TF32 : ~7e-4 .. 2e-3     ← gọi disable_tf32()

    CẢNH BÁO, KHÔNG assert: sai số số học ở đây không làm hỏng phương pháp, nó
    chỉ làm bẩn phép đo. Thứ thực sự quan trọng là ||Qᵀh_b||, đo bằng
    benign_leakage().
    """
    k = Q.shape[1]
    err = float((Q.T @ Q - torch.eye(k, device=Q.device, dtype=Q.dtype)).norm() / (k ** 0.5))
    if err > fail_tol:
        raise RuntimeError(
            f"{name} mất trực chuẩn nghiêm trọng: ||{name}ᵀ{name}−I||/√k = {err:.2e} "
            f"> {fail_tol:.0e}. Nhiều khả năng SVD đã fail. Thử --nullspace_dtype float64.")
    if err > warn_tol:
        logger.warning(
            "  %s hơi lệch trực chuẩn: %.2e (fp32 thuần kỳ vọng ~2e-6). Nếu ~1e-3 thì "
            "TF32 đang bật → gọi disable_tf32(). KHÔNG chặn chạy tiếp.", name, err)
    return err


def null_space_basis_l(
    A: torch.Tensor,
    min_null_space_ratio: float = 0.1,
    abs_nullspace_ratio: float = 0.0,
    dtype: Optional[torch.dtype] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Cơ sở TRỰC CHUẨN Q (d, k) của null space xấp xỉ của A (N, d).

    S1: SVD trực tiếp trên A thay vì trên AᵀA — lập AᵀA bình phương condition
    number, mà ta lại cần đúng phần trị riêng NHỎ NHẤT (vùng hỏng nhất).

    Parameters
    ----------
    abs_nullspace_ratio : > 0 ⇒ lấy floor(d·ρ) hướng có singular value nhỏ nhất
                          (đường AlphaSteer thực dùng, ρ ∈ [0.4, 0.6])
    dtype : None (giữ dtype của A) hoặc torch.float64.
            fp64 trên GPU consumer rất chậm (1/64 rate) — cân nhắc A.cpu().double().
    """
    orig_device = A.device
    if dtype is not None and dtype != A.dtype:
        A = A.to(dtype)
    N, d = A.shape

    if N >= d:
        _, S, Vh = torch.linalg.svd(A, full_matrices=False)   # Vh (d,d), S giảm dần
    else:
        G = A.T @ A
        evals, evecs = torch.linalg.eigh(0.5 * (G + G.T))
        S = evals.flip(0).clamp(min=0).sqrt()
        Vh = evecs.flip(1).T

    if abs_nullspace_ratio > 0:
        num = int(d * abs_nullspace_ratio)
    else:
        rcond = torch.finfo(S.dtype).eps * max(N, d)
        num = int(torch.sum(S < torch.amax(S) * rcond).item())
        if num / d < min_null_space_ratio:
            num = int(d * min_null_space_ratio)
    num = max(1, min(num, d))

    Q = Vh[-num:, :].T.conj().contiguous()                    # (d, num)

    if verbose:
        energy = (S[-num:] ** 2).sum() / (S ** 2).sum()
        logger.info("  null space: k=%d/%d (ρ=%.3f) | σ_max=%.3e σ_min=%.3e | "
                    "benign energy trong subspace=%.3e",
                    num, d, num / d, S[0].item(), S[-1].item(), energy.item())
    check_orthonormal(Q, "Q")
    return Q.to(device=orig_device)


def null_space_projection_l(A, min_null_space_ratio=0.1, abs_nullspace_ratio=0.0,
                            dtype=None, verbose=True, check=None) -> torch.Tensor:
    """
    P = QQᵀ (d, d). CHỈ giữ cho tương thích ngược / đối chiếu.

    Đường khuyến nghị: null_space_basis_l() + cal_steering_factors_q() — không
    dựng P, không pinv, và u ∈ range(Q) theo cấu trúc.

    (`check` bị bỏ qua: v3 assert P²=P ở đây và fail giả trên GPU có TF32.)
    """
    Q = null_space_basis_l(A, min_null_space_ratio, abs_nullspace_ratio,
                           dtype=dtype, verbose=verbose)
    return Q @ Q.T


def cal_P(H_b, layers: Sequence[int], min_nullspace_ratio=0.1,
          abs_nullspace_ratio=0.0, device="cuda:0") -> torch.Tensor:
    return torch.stack([
        null_space_projection_l(H_b[:, l, :].to(device),
                                min_null_space_ratio=min_nullspace_ratio,
                                abs_nullspace_ratio=abs_nullspace_ratio)
        for l in layers], dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# Steering factors — ĐƯỜNG CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def cal_steering_factors_q(
    H_h_layer: torch.Tensor,
    Q: torch.Tensor,
    refusal_vector: torch.Tensor,
    lambda_reg: float,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Nghiệm ĐÚNG của Eq.(7), tính hoàn toàn trong không gian null k chiều.

        Y = H_m Q  ∈ R^{N×k}
        w = (YᵀY + λ I_k)⁻¹ Yᵀ 1_N        ← SPD ⇒ Cholesky, không pinv
        u = Q w                            ← u ∈ range(Q) theo CẤU TRÚC
        M = u rᵀ,   Mᵀh = (uᵀh)·r

    Tương đương chính xác với đường cũ (P + pinv d×d): rel err 1.7e-12.

    Returns
    -------
    u : (d,) gate vector — uᵀh_b = wᵀ(Qᵀh_b) ≈ 0
    r : (d,) concept direction
    """
    H = H_h_layer.to(device).float()
    Q = Q.to(device).float()
    r = refusal_vector.to(device).float()
    k = Q.shape[1]

    Y = H @ Q                                              # (N, k)
    G = Y.T @ Y
    G.diagonal().add_(lambda_reg)                          # + λ I_k
    G = 0.5 * (G + G.T)
    rhs = Y.sum(dim=0).unsqueeze(1)                        # Yᵀ1_N, (k, 1)

    try:
        L = torch.linalg.cholesky(G)                       # SPD khi λ > 0
        w = torch.cholesky_solve(rhs, L).squeeze(1)
    except Exception as e:                                 # noqa: BLE001
        logger.warning("  cholesky failed (%s) → lstsq", e)
        w = torch.linalg.lstsq(G, rhs).solution.squeeze(1)

    u = Q @ w                                              # (d,)

    with torch.no_grad():
        gate = H @ u
        rmse = float((Y @ w - 1.0).pow(2).mean().sqrt())
        in_range = float((u - Q @ (Q.T @ u)).norm() / u.norm().clamp(min=1e-12))
    logger.info("  [Q-space] k=%d ||u||=%.4f | gate uᵀh_m = %.4f ± %.4f "
                "(mục tiêu 1.0, rmse=%.4f) | u ngoài range(Q): %.1e",
                k, u.norm().item(), gate.mean().item(), gate.std().item(), rmse, in_range)
    return u, r


def cal_steering_factors_ridge_l(
    H_h_layer: torch.Tensor,
    refusal_vector: torch.Tensor,
    lambda_reg: float,
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    ABLATION P = I:  u = (HᵀH + λI_d)⁻¹ Hᵀ1_N.

    Cùng dạng rank-1 h' = h + α(uᵀh)r; khác bản full ĐÚNG MỘT biến: u KHÔNG bị
    ép nằm trong range(Q).

    ⚠ λ KHÔNG dùng chung được với bản null-space: ở đó regularizer λ‖PΔ̃‖² chỉ
    phạt trong range(P) và phổ đã bị cắt; ở đây phổ HᵀH giữ nguyên. Đo được
    λ=10 cho λ/λ_max ≈ 4e-4 (≈ không regularize). Phải sweep λ riêng, nếu không
    "utility collapse" một phần do under-regularization chứ không do bỏ P.
    """
    H = H_h_layer.to(device).float()
    r = refusal_vector.to(device).float()

    A = H.T @ H
    lam_max = float(torch.linalg.eigvalsh(A)[-1])
    A.diagonal().add_(lambda_reg)
    A = 0.5 * (A + A.T)
    rhs = H.sum(dim=0).unsqueeze(1)
    try:
        u = torch.cholesky_solve(rhs, torch.linalg.cholesky(A)).squeeze(1)
    except Exception as e:                                 # noqa: BLE001
        logger.warning("  cholesky failed (%s) → pinv_apply", e)
        u = pinv_apply(A, rhs.squeeze(1))

    logger.info("  [P=I] λ=%.3g λ_max(HᵀH)=%.3g λ/λ_max=%.2e ||u||=%.4f",
                lambda_reg, lam_max, lambda_reg / max(lam_max, 1e-30), u.norm().item())
    return u, r


# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def benign_leakage(H_b_layer: torch.Tensor, Q_or_P: torch.Tensor,
                   max_n: int = 4000) -> dict:
    """
    ||P h_b|| / ||h_b||. Nhận Q (d, k) HOẶC P (d, d) — norm như nhau vì
    ||QQᵀh|| = ||Qᵀh|| khi Q trực chuẩn, nên KHÔNG cần dựng P.

    Con số này nên vào bài THAY CHO chữ "provable": với rank(H_b) = d, null space
    THẬT là {0}; cái ta giữ là low-energy subspace ⇒ leakage nhỏ nhưng KHÁC 0,
    và chỉ được kiểm chứng bằng dữ liệu, không phải chứng minh.
    Luôn đo trên benign HELD-OUT.
    """
    H = H_b_layer[:max_n].to(Q_or_P.device).float()
    ratio = (H @ Q_or_P).norm(dim=1) / H.norm(dim=1).clamp(min=1e-12)
    return {
        "leakage_mean": float(ratio.mean()),
        "leakage_p95": float(ratio.quantile(0.95)),
        "leakage_max": float(ratio.max()),
    }


def steering_signal_stats(H_layer: torch.Tensor, u: torch.Tensor, r: torch.Tensor,
                          max_n: int = 2000) -> dict:
    """
    Thống kê gate = uᵀh — "prompt classifier" ẩn của phương pháp.

    Bản null-space: gate(benign) ≈ 0, gate(malicious) ≈ 1.
    Bản P=I: gate(benign) không bị ràng buộc ⇒ benign cũng bị đẩy về r
             ⇒ over-refusal ⇒ XSTest CR sập.
    Đây là bằng chứng định lượng cho ablation — manuscript hiện không có.
    """
    H = H_layer[:max_n].to(u.device).float()
    gate = H @ u
    signal = gate.abs() * r.norm()
    return {
        "gate_mean": float(gate.mean()),
        "gate_abs_mean": float(gate.abs().mean()),
        "gate_std": float(gate.std()),
        "mean_signal_norm": float(signal.mean()),
        "mean_relative_norm": float((signal / H.norm(dim=1).clamp(min=1e-9)).mean()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY — đường ma trận d×d. Chỉ để đối chiếu với v2: chậm hơn ~1000x và cần
# pinv của ma trận cond ~1e20. Đường chính là cal_steering_factors_q.
# ══════════════════════════════════════════════════════════════════════════════

def pinv_apply(A: torch.Tensor, x: torch.Tensor, rcond: float = 1e-10) -> torch.Tensor:
    """A⁺x cho A đối xứng, không dựng A⁺."""
    A = 0.5 * (A + A.T)
    evals, Qe = torch.linalg.eigh(A)
    tol = rcond * evals.abs().max()
    inv = torch.where(evals.abs() > tol, 1.0 / evals, torch.zeros_like(evals))
    return Qe @ (inv * (Qe.T @ x))


def _report_reconstruction(X, tilde_delta, r, tag=""):
    with torch.no_grad():
        result = X @ tilde_delta
        R = r.unsqueeze(0).expand_as(result)
        rel = float(torch.linalg.norm(result - R) / torch.linalg.norm(R))
        cos = float(torch.nn.functional.cosine_similarity(result, R, dim=1).mean())
    logger.info("  %sreconstruction: rel_err=%.4f mean_cos=%.4f ||r||=%.4f",
                tag, rel, cos, r.norm().item())
    return {"rel_err": rel, "mean_cos": cos}


def cal_tilde_delta_with_regularization_l(H_h_layer, P_layer, refusal_vector,
                                          lambda_reg, device="cuda:0"):
    """Δ̃ = (PHᵀHP + λP)⁺ PHᵀR, (d, d) — MA TRẬN (docstring v2 ghi "vector" là sai)."""
    H = H_h_layer.to(device).float()
    P = P_layer.to(device).float()
    r = refusal_vector.to(device).float()
    X = H @ P
    XtX = X.T @ X
    A = 0.5 * ((XtX + lambda_reg * (P.T @ P)) + (XtX + lambda_reg * (P.T @ P)).T)
    # S7: b = Xᵀ(1rᵀ) = (Xᵀ1)rᵀ = outer(X.sum(0), r) — chính xác (rel err ~4e-16)
    b = torch.outer(X.sum(dim=0), r)
    tilde_delta = torch.linalg.pinv(A, hermitian=True) @ b
    _report_reconstruction(X, tilde_delta, r)
    return tilde_delta


def cal_tilde_delta_ridge_l(H_h_layer, refusal_vector, lambda_reg, device="cuda:0"):
    u, r = cal_steering_factors_ridge_l(H_h_layer, refusal_vector, lambda_reg, device)
    return torch.outer(u, r)


def cal_tilde_delta_l(H_h_layer, P_layer, refusal_vector, device="cuda:0"):
    H = H_h_layer.to(device).float()
    P = P_layer.to(device).float()
    r = refusal_vector.to(device).float()
    X = H @ P
    td = torch.linalg.pinv(X) @ r.repeat(X.shape[0], 1)
    _report_reconstruction(X, td, r)
    return td


def cal_tilde_delta_with_regularization(H_h, P, refusal_vectors, layers,
                                        lambda_reg=1e-5, device="cuda:0"):
    return torch.stack([cal_tilde_delta_with_regularization_l(
        H_h[:, l, :], P[l], refusal_vectors[l], lambda_reg, device) for l in layers], dim=0)


def cal_tilde_delta(H_h, P, refusal_vectors, layers, device="cuda:0"):
    return torch.stack([cal_tilde_delta_l(
        H_h[:, l, :], P[l], refusal_vectors[l], device) for l in layers], dim=0)


def cal_steering_matrix_l(P_layer, tilde_delta_layer, device="cuda:0"):
    """M = PΔ̃. Lý thuyết: PΔ̃ = Δ̃ vì range(Δ̃) ⊆ range(P)."""
    return P_layer.to(device).float() @ tilde_delta_layer.to(device).float()


def cal_steering_matrix(P, tilde_delta, layers, device="cuda:0"):
    return torch.stack([cal_steering_matrix_l(P[l], tilde_delta[l], device)
                        for l in layers], dim=0)