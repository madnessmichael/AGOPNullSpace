"""
rfm_refusal_vector.py  (v2 — fixed)
=====================================
Thay thế DIM bằng RFM để tạo refusal vector trong AlphaSteer pipeline.

BUG ĐÃ FIX so với v1:
  B1. compute_agop_linear dùng sklearn/numpy → luôn CPU dù pass device="cuda".
      → Fixed: GPU-native logistic via torch LBFGS + standardize_gpu.
  B2. compute_agop_rfm: val split random có thể all-pos hoặc all-neg.
      → Fixed: stratified split theo label.
  B3. compute_rfm_refusal_vectors: device bị ignore trong nhánh "linear".
      → Fixed: giờ compute_agop_linear nhận device và chạy đúng trên GPU.
  B4. load_alphasteer_embeddings: thiếu .float() → dtype mismatch với bfloat16.
      → Fixed: luôn cast sang float32 sau khi load.
  B5. main block: thiếu seed → không reproducible.
      → Fixed: thêm --seed arg và set torch + numpy seed.

THIẾU SÓT CÒN LẠI (data):
  - H_refusal = H_malicious là approximation cho Dr.
    Dr thực sự = prompts mà model actually refused sau khi pass qua model.
    AlphaSteer gốc extract Dr/Dc riêng từ 720 malicious prompts.
  - H_math không được include trong H_benign (đúng với gốc dòng 3905-3913).
"""

import os
import glob
import argparse
import pickle
import argparse
import logging
import numpy as np
import torch
from copy import deepcopy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

STEERING_LAYERS = {
    "llama3.1": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
    "qwen2.5":  [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19],
    "gemma2":   [6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 22],
    "llama3.3-70b": [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],
    "llama3.1-8b-unsloth": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
    "qwen3-32b": [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    "gpt-oss-120b": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
}


# ── GPU-native standardization ────────────────────────────────────────────────

def standardize_gpu(X: torch.Tensor):
    """Z-score trên GPU. Returns (X_scaled, mean_, std_)."""
    mean_ = X.mean(dim=0)
    std_  = X.std(dim=0).clamp(min=1e-8)
    return (X - mean_) / std_, mean_, std_


# ── GPU-native logistic regression via L-BFGS ────────────────────────────────

def _logistic_loss(w, X, y, C=1.0):
    logits = X @ w
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction='mean'
    )
    return loss + (0.5 / C) * (w @ w)


def train_logistic_gpu(X: torch.Tensor, y: torch.Tensor,
                       C: float = 1.0, max_iter: int = 500) -> torch.Tensor:
    """
    Binary logistic regression trên GPU bằng L-BFGS.
    X: [N,d] float32 on device, y: [N] float32 on device.
    Returns w: [d] on device.
    """
    w = torch.zeros(X.shape[1], dtype=torch.float32, device=X.device,
                    requires_grad=True)
    opt = torch.optim.LBFGS(
        [w], lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-6, tolerance_change=1e-6,
        history_size=10, line_search_fn='strong_wolfe'
    )
    def closure():
        opt.zero_grad()
        loss = _logistic_loss(w, X, y, C=C)
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — AGOP functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_agop_linear(H_pos: torch.Tensor,
                        H_neg: torch.Tensor,
                        device: str = "cpu") -> tuple:
    """
    Refusal direction = top eigenvector của AGOP(w·wᵀ) từ logistic probe.
    Toàn bộ chạy trên `device` (GPU nếu chỉ định).

    Returns:
        agop : [d,d] trên CPU  (để tránh OOM khi lưu nhiều layers)
        r    : [d]   trên CPU
    """
    dev = torch.device(device)
    N_pos, d = H_pos.shape

    X = torch.cat([H_pos, H_neg], dim=0).float().to(dev)
    y = torch.cat([
        torch.ones(N_pos, device=dev),
        torch.zeros(H_neg.shape[0], device=dev)
    ])

    X_sc, mean_, std_ = standardize_gpu(X)

    logger.info("  GPU logistic: %d pos + %d neg, d=%d, device=%s",
                N_pos, H_neg.shape[0], d, dev)

    # Nhỏ hyperparameter search
    best_acc, best_C, best_w_sc = -1.0, 1.0, None
    for C in [0.1, 1.0, 10.0]:
        w_sc = train_logistic_gpu(X_sc, y, C=C)
        acc  = ((X_sc @ w_sc > 0).float() == y).float().mean().item()
        if acc > best_acc:
            best_acc, best_C, best_w_sc = acc, C, w_sc.clone()

    logger.info("  Best C=%.1f  train_acc=%.4f", best_C, best_acc)

    # Unscale
    w_orig = (best_w_sc / std_).cpu()
    r = w_orig / w_orig.norm().clamp(min=1e-8)
    agop = torch.outer(w_orig, w_orig)  # rank-1, CPU

    # Clear GPU cache
    del X, y, X_sc, mean_, std_, best_w_sc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return agop, r


def compute_agop_rfm(H_pos: torch.Tensor,
                     H_neg: torch.Tensor,
                     rfm_iters: int = 3,
                     device: str = "cpu") -> tuple:
    """
    Tính AGOP bằng full RFM. Cần: pip install xrfm
    Fallback sang linear nếu xrfm không có.

    Returns:
        agop : [d,d] CPU
        r    : [d]   CPU
    """
    try:
        from xrfm import RFM
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("xrfm not installed. Falling back to linear AGOP.")
        return compute_agop_linear(H_pos, H_neg, device)

    dev = torch.device(device)
    N_pos, d = H_pos.shape
    N_neg = H_neg.shape[0]

    X = torch.cat([H_pos, H_neg], dim=0).float().to(dev)
    y = torch.cat([
        torch.ones(N_pos, 1, device=dev),
        torch.zeros(N_neg, 1, device=dev)
    ])

    # Stratified split (FIX B2)
    pos_idx = (y.squeeze() == 1).nonzero(as_tuple=True)[0]
    neg_idx = (y.squeeze() == 0).nonzero(as_tuple=True)[0]
    nv_pos  = max(1, int(0.2 * len(pos_idx)))
    nv_neg  = max(1, int(0.2 * len(neg_idx)))
    val_idx   = torch.cat([pos_idx[:nv_pos], neg_idx[:nv_neg]])
    train_idx = torch.cat([pos_idx[nv_pos:], neg_idx[nv_neg:]])

    Xtr, ytr = X[train_idx], y[train_idx]
    Xvl, yvl = X[val_idx],   y[val_idx]

    best_model, best_auc = None, -1.0
    for bw in [1.0, 10.0, 100.0]:
        for reg in [1e-3, 1e-2]:
            try:
                m = RFM(kernel='l2_high_dim', bandwidth=bw, device=device)
                m.fit((Xtr, ytr), (Xvl, yvl),
                      reg=reg, iters=rfm_iters,
                      center_grads=True, early_stop_rfm=True,
                      get_agop_best_model=True, top_k=1)
                preds = m.predict(Xvl).cpu().numpy()
                auc = roc_auc_score(yvl.cpu().numpy(), preds)
                if auc > best_auc:
                    best_auc = auc
                    best_model = deepcopy(m)
            except Exception as e:
                logger.warning("  RFM bw=%.1f reg=%.0e failed: %s", bw, reg, e)

    if best_model is None:
        logger.warning("  All RFM fits failed. Falling back to linear.")
        return compute_agop_linear(H_pos, H_neg, device)

    logger.info("  Best RFM AUC=%.4f", best_auc)
    agop = best_model.agop_best_model.cpu()

    S, U = torch.lobpcg(agop, k=1)
    r = U[:, 0]
    proj = X.cpu() @ r
    if torch.corrcoef(torch.stack([proj, y.squeeze().cpu()]))[0, 1] < 0:
        r = -r

    del X, y, Xtr, ytr, Xvl, yvl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return agop, r.cpu()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — Per-layer computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_rfm_refusal_vectors(
    H_refusal:        torch.Tensor,
    H_compliant:      torch.Tensor,
    layers:           list,
    num_total_layers: int,
    method:           str = "linear",
    rfm_iters:        int = 3,
    device:           str = "cpu",
    layer_to_local: dict = None,  # thêm tham số này
) -> np.ndarray:
    """
    Compute RFM refusal direction cho mỗi layer.

    Output: numpy [num_total_layers, d] float32 — giống DIM pkl của AlphaSteer.
    Layers không compute = zero vector.
    """
    d = H_refusal.shape[2]
    refusal_vectors = np.zeros((num_total_layers, d), dtype=np.float32)

    for layer_idx in layers:
        logger.info("=== Layer %d ===", layer_idx)

        local_idx = layer_to_local[layer_idx] if layer_to_local else layer_idx  # thêm dòng này

        h_pos = H_refusal[:, local_idx, :].float()   # CPU
        h_neg = H_compliant[:, local_idx, :].float()  # CPU

        # Balance
        n_min = min(len(h_pos), len(h_neg))
        if len(h_pos) > n_min:
            h_pos = h_pos[torch.randperm(len(h_pos))[:n_min]]
        if len(h_neg) > n_min:
            h_neg = h_neg[torch.randperm(len(h_neg))[:n_min]]

        if method == "linear":
            _, r = compute_agop_linear(h_pos, h_neg, device=device)  # FIX B3
        elif method == "rfm":
            _, r = compute_agop_rfm(h_pos, h_neg, rfm_iters=rfm_iters, device=device)
        else:
            raise ValueError(f"Unknown method: {method}")

        refusal_vectors[local_idx] = r.float().numpy()
        logger.info("  r[%d] norm=%.6f", layer_idx, np.linalg.norm(refusal_vectors[local_idx]))

    return refusal_vectors


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — Load data + main
# ══════════════════════════════════════════════════════════════════════════════

def load_alphasteer_embeddings(embedding_dir: str):
    """Load theo đúng calc_steering_matrix.py gốc (dòng 3905-3926)."""
    logger.info("Loading embeddings from %s", embedding_dir)

    def _load(fname):
        t = torch.load(os.path.join(embedding_dir, fname), map_location="cpu")
        return t.float()  # FIX B4: luôn float32

    H_benign_10k   = _load("embeds_benign_train.pt")
    H_coconot_pref = _load("embeds_coconot_pref.pt")
    H_coconot_orig = _load("embeds_coconot_original.pt")
    idx_b = torch.randperm(H_coconot_orig.size(0))[:4000 - H_coconot_pref.size(0)]
    H_compliant = torch.cat([H_benign_10k, H_coconot_orig[idx_b], H_coconot_pref], dim=0)

    H_harmful = _load("embeds_harmful_train_1000.pt")
    H_jailbreak = _load("embeds_jailbreak_train.pt")
    idx_jb = torch.randperm(H_jailbreak.size(0))[:1000]
    H_refusal = torch.cat([H_harmful, H_jailbreak[idx_jb]], dim=0)

    logger.info("H_refusal   %s (proxy Dr)", tuple(H_refusal.shape))
    logger.info("H_compliant %s (proxy Dc)", tuple(H_compliant.shape))
    return H_refusal, H_compliant


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_dir", required=True)
    p.add_argument("--model_name", required=True, choices=["llama3.1", "qwen2.5", "gemma2"])
    p.add_argument("--method", default="linear", choices=["linear", "rfm"])
    p.add_argument("--rfm_iters", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_path", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)   # FIX B5
    np.random.seed(args.seed)

    H_refusal, H_compliant = load_alphasteer_embeddings(args.embedding_dir)
    num_total_layers = H_refusal.shape[1]
    d = H_refusal.shape[2]
    layers = STEERING_LAYERS[args.model_name]

    logger.info("model=%s layers=%s d=%d method=%s device=%s",
                args.model_name, layers, d, args.method, args.device)

    rv = compute_rfm_refusal_vectors(
        H_refusal=H_refusal, H_compliant=H_compliant,
        layers=layers, num_total_layers=num_total_layers,
        method=args.method, rfm_iters=args.rfm_iters, device=args.device,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(rv, f)

    logger.info("Saved → %s  shape=%s", args.save_path, rv.shape)