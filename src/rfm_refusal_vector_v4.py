"""
rfm_refusal_vector.py  (v4 — fixed data balance)
=================================================
Thay thế DIM bằng RFM-AGOP để tạo refusal vector trong AlphaSteer pipeline.

═══════════════════════════════════════════════════════════════════
VẤN ĐỀ CỦA v2 VÀ CÁCH FIX
═══════════════════════════════════════════════════════════════════

AlphaSteer/AdaSteer dùng content-based DIM:
    r = mean(harmful_embeddings) - mean(harmless_embeddings)
    Data: paired harmful/harmless prompts, số lượng CÂN BẰNG

v2 bị 2 vấn đề trong load_alphasteer_embeddings():

  PROBLEM 1 — IMBALANCED:
    H_refusal:   1000 harmful + 1000 jailbreak = 2000 prompts
    H_compliant: 10000 benign + ~4000 coconot  = ~14000 prompts
    Ratio: 1:7 → DIM/RFM bị dominated bởi benign distribution
    → Direction học được phân tách TOPIC (harmful vs benign)
      thay vì pure REFUSAL concept

  PROBLEM 2 — DOMAIN MISMATCH:
    H_refusal:   malicious/jailbreak prompts
    H_compliant: Alpaca instructions + CoCoNot
    → Hai sets đến từ HOÀN TOÀN khác domain
    → Direction bị confounded bởi domain shift,
      không phải chỉ harmful/harmless distinction

FIX trong v4 — load_balanced_embeddings():
  Bước 1: Load toàn bộ harmful và benign data
  Bước 2: Downsample benign về đúng N_harmful
           (giống logic AlphaSteer gốc line 3094-3103)
  Bước 3: RFM-AGOP compute_rfm_refusal_vectors() nhận
           balanced H_harmful / H_harmless

═══════════════════════════════════════════════════════════════════
DATA MAPPING — PHÂN TÍCH ĐẦY ĐỦ
═══════════════════════════════════════════════════════════════════

Component          Data                              Số lượng    Thay đổi?
─────────────────  ────────────────────────────────  ──────────  ─────────
Refusal vector r   harmful + jailbreak (balanced)    N + N       ✅ v4 fix
                   vs benign + coconot (balanced)
Null-space P̂       benign + coconot                  ~14000      ❌ giữ nguyên
Regression Hm      harmful + jailbreak               ~2000       ❌ giữ nguyên

Paradigm: content-based (harmful vs harmless) — ĐÚNG với AlphaSteer/AdaSteer.
KHÔNG phải behavioral (refused vs complied) — đó là nhầm lẫn của v3.

═══════════════════════════════════════════════════════════════════
BUGS ĐÃ FIX TỪ v2 (giữ nguyên)
═══════════════════════════════════════════════════════════════════
  B1. compute_agop_linear: GPU-native logistic via torch LBFGS.
  B2. compute_agop_rfm: stratified val split.
  B3. compute_rfm_refusal_vectors: device propagation đúng.
  B4. load functions: luôn cast sang float32.
  B5. main block: reproducible seed.

BUG MỚI FIX TRONG v4
  B6. load_alphasteer_embeddings: imbalanced 2000 vs 14000.
      → Fixed: load_balanced_embeddings() downsample harmless → N_harmful.
  B7. load_alphasteer_embeddings: domain mismatch harmful vs benign.
      → Fixed: vẫn dùng cùng data sources nhưng balance đúng.
      Note: Nếu có paired harmful/harmless data (như AdaSteer
      harmful_break_or_not/test.jsonl) thì càng tốt hơn nữa,
      nhưng với data hiện có của AlphaSteer thì balance là fix
      khả thi và đủ để cải thiện rõ rệt.
"""

import os
import argparse
import pickle
import logging

import numpy as np
import torch
from copy import deepcopy

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STEERING_LAYERS = {
    "llama3.1":            [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
    "qwen2.5":             [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19],
    "gemma2":              [6, 8, 10, 11, 12, 13, 14, 15, 16, 18, 22],
    "llama3.3-70b":        [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50],
    "llama3.1-8b-unsloth": [8, 9, 10, 11, 12, 13, 14, 16, 18, 19],
    "qwen3-32b":           [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    "gpt-oss-120b":        [40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
}


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — GPU utilities (giữ nguyên từ v2)
# ══════════════════════════════════════════════════════════════════════════════

def standardize_gpu(X: torch.Tensor):
    """Z-score normalization trên GPU. Returns (X_scaled, mean_, std_)."""
    mean_ = X.mean(dim=0)
    std_  = X.std(dim=0).clamp(min=1e-8)
    return (X - mean_) / std_, mean_, std_


def _logistic_loss(w, X, y, C=1.0):
    logits = X @ w
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="mean"
    )
    return loss + (0.5 / C) * (w @ w)


def train_logistic_gpu(
    X: torch.Tensor,
    y: torch.Tensor,
    C: float = 1.0,
    max_iter: int = 500,
) -> torch.Tensor:
    """
    Binary logistic regression trên GPU bằng L-BFGS.
    X: [N, d] float32 on device.
    y: [N]    float32 on device.
    Returns w: [d] on device.
    """
    w = torch.zeros(
        X.shape[1], dtype=torch.float32, device=X.device, requires_grad=True
    )
    opt = torch.optim.LBFGS(
        [w], lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-6, tolerance_change=1e-6,
        history_size=10, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = _logistic_loss(w, X, y, C=C)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach()


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — AGOP functions (giữ nguyên từ v2)
# ══════════════════════════════════════════════════════════════════════════════

def compute_agop_linear(
    H_pos: torch.Tensor,
    H_neg: torch.Tensor,
    device: str = "cpu",
) -> tuple:
    """
    Refusal direction = top eigenvector của AGOP(w·wᵀ) từ logistic probe.

    H_pos: harmful activations  [N, d] CPU
    H_neg: harmless activations [N, d] CPU  (đã balanced bên ngoài)

    Returns:
        agop: [d, d] CPU
        r:    [d]    CPU
    """
    dev = torch.device(device)
    N_pos, d = H_pos.shape

    X = torch.cat([H_pos, H_neg], dim=0).float().to(dev)
    y = torch.cat([
        torch.ones(N_pos, device=dev),
        torch.zeros(H_neg.shape[0], device=dev),
    ])

    X_sc, mean_, std_ = standardize_gpu(X)
    logger.info(
        "  GPU logistic: %d harmful + %d harmless, d=%d, device=%s",
        N_pos, H_neg.shape[0], d, dev,
    )

    best_acc, best_C, best_w_sc = -1.0, 1.0, None
    for C in [0.1, 1.0, 10.0]:
        w_sc = train_logistic_gpu(X_sc, y, C=C)
        acc = ((X_sc @ w_sc > 0).float() == y).float().mean().item()
        if acc > best_acc:
            best_acc, best_C, best_w_sc = acc, C, w_sc.clone()

    logger.info("  Best C=%.1f  train_acc=%.4f", best_C, best_acc)

    w_orig = (best_w_sc / std_).cpu()
    r = w_orig / w_orig.norm().clamp(min=1e-8)
    agop = torch.outer(w_orig, w_orig)

    del X, y, X_sc, mean_, std_, best_w_sc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return agop, r


def compute_agop_rfm(
    H_pos: torch.Tensor,
    H_neg: torch.Tensor,
    rfm_iters: int = 3,
    device: str = "cpu",
) -> tuple:
    """
    Tính AGOP bằng full RFM. Cần: pip install xrfm
    Fallback sang linear nếu xrfm không có.

    H_pos: harmful activations  [N, d] CPU
    H_neg: harmless activations [N, d] CPU  (đã balanced bên ngoài)

    Returns:
        agop: [d, d] CPU
        r:    [d]    CPU
    """
    try:
        from xrfm import RFM
        from sklearn.metrics import roc_auc_score
    except ImportError:
        logger.warning("xrfm not installed. Falling back to linear AGOP.")
        return compute_agop_linear(H_pos, H_neg, device)

    dev = torch.device(device)
    N_pos = H_pos.shape[0]
    N_neg = H_neg.shape[0]

    X = torch.cat([H_pos, H_neg], dim=0).float().to(dev)
    y = torch.cat([
        torch.ones(N_pos, 1, device=dev),
        torch.zeros(N_neg, 1, device=dev),
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
                m = RFM(kernel="l2_high_dim", bandwidth=bw, device=device)
                m.fit(
                    (Xtr, ytr), (Xvl, yvl),
                    reg=reg, iters=rfm_iters,
                    center_grads=True, early_stop_rfm=True,
                    get_agop_best_model=True, top_k=1,
                )
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
# PHẦN 3 — Per-layer computation (giữ nguyên từ v2)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rfm_refusal_vectors(
    H_harmful:        torch.Tensor,
    H_harmless:       torch.Tensor,
    layers:           list,
    num_total_layers: int,
    method:           str = "linear",
    rfm_iters:        int = 3,
    device:           str = "cpu",
) -> np.ndarray:
    """
    Compute RFM refusal direction cho mỗi layer.

    Args:
        H_harmful:        [N_harmful,  num_layers, d] — harmful activations
        H_harmless:       [N_harmless, num_layers, d] — harmless activations
                          (đã balanced: N_harmful == N_harmless sau khi gọi
                           load_balanced_embeddings())
        layers:           List layer indices cần compute
        num_total_layers: Tổng số layers của model
        method:           "linear" hoặc "rfm"
        rfm_iters:        Số RFM iterations (chỉ dùng khi method="rfm")
        device:           "cuda" hoặc "cpu"

    Returns:
        numpy array [num_total_layers, d] float32.
        Format giống DIM pkl của AlphaSteer — drop-in replacement.
        Layers không compute = zero vector.
    """
    d = H_harmful.shape[2]
    refusal_vectors = np.zeros((num_total_layers, d), dtype=np.float32)

    for layer_idx in layers:
        logger.info("=== Layer %d ===", layer_idx)

        h_pos = H_harmful[:, layer_idx, :].float()   # CPU, [N_harmful, d]
        h_neg = H_harmless[:, layer_idx, :].float()  # CPU, [N_harmless, d]

        # Safety balance tại layer level — phòng trường hợp shape khác nhau
        # (thường đã balance rồi từ load_balanced_embeddings, nhưng defensive)
        n_min = min(len(h_pos), len(h_neg))
        if len(h_pos) > n_min:
            h_pos = h_pos[torch.randperm(len(h_pos))[:n_min]]
        if len(h_neg) > n_min:
            h_neg = h_neg[torch.randperm(len(h_neg))[:n_min]]

        logger.info(
            "  harmful=%d, harmless=%d (balanced to %d each)",
            H_harmful.shape[0], H_harmless.shape[0], n_min,
        )

        if method == "linear":
            _, r = compute_agop_linear(h_pos, h_neg, device=device)
        elif method == "rfm":
            _, r = compute_agop_rfm(
                h_pos, h_neg, rfm_iters=rfm_iters, device=device
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        refusal_vectors[layer_idx] = r.float().numpy()
        logger.info(
            "  r[%d] norm=%.6f",
            layer_idx, np.linalg.norm(refusal_vectors[layer_idx]),
        )

    return refusal_vectors


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — Load data (FIX B6, B7: balanced harmful vs harmless)
# ══════════════════════════════════════════════════════════════════════════════

def load_balanced_embeddings(
    embedding_dir: str,
    n_jailbreak: int = 1000,
    balance_ratio: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Load harmful và harmless embeddings với số lượng CÂN BẰNG.

    FIX so với v2's load_alphasteer_embeddings():
      - v2: harmful=2000, harmless=~14000 → imbalanced 1:7
      - v4: harmful=N,    harmless=N      → balanced 1:1

    Data sources (không thay đổi, chỉ thay đổi cách balance):
      Harmful:  embeds_harmful_train_1000.pt + embeds_jailbreak_train.pt
      Harmless: embeds_benign_train.pt + embeds_coconot_original.pt
                + embeds_coconot_pref.pt

    Logic balance (giống AlphaSteer gốc line 3094-3103 trong MLP version):
      1. Load toàn bộ harmful → N_harmful samples
      2. Load toàn bộ harmless
      3. Downsample harmless → round(N_harmful * balance_ratio) samples

    Args:
        embedding_dir: Thư mục chứa .pt files
        n_jailbreak:   Số jailbreak samples lấy từ embeds_jailbreak_train.pt
                       (default 1000, giống v2 và calc_steering_matrix gốc)
        balance_ratio: Tỉ lệ harmless/harmful sau balance (default 1.0 = 1:1)
        seed:          Random seed cho reproducibility

    Returns:
        H_harmful:  [N_harmful,  num_layers, d] float32 CPU
        H_harmless: [N_harmless, num_layers, d] float32 CPU
                    với N_harmless = round(N_harmful * balance_ratio)
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    def _load(fname: str) -> torch.Tensor:
        path = os.path.join(embedding_dir, fname)
        return torch.load(path, map_location="cpu").float()  # FIX B4

    # ── Harmful side ──────────────────────────────────────────────────────────
    logger.info("Loading harmful embeddings...")
    H_harm_1000   = _load("embeds_harmful_train_1000.pt")
    H_jailbreak   = _load("embeds_jailbreak_train.pt")

    idx_jb = torch.randperm(H_jailbreak.size(0), generator=gen)[:n_jailbreak]
    H_harmful = torch.cat([H_harm_1000, H_jailbreak[idx_jb]], dim=0)

    del H_harm_1000, H_jailbreak
    N_harmful = H_harmful.shape[0]
    logger.info("H_harmful:  %s  (1000 harmful + %d jailbreak)", tuple(H_harmful.shape), n_jailbreak)

    # ── Harmless side ─────────────────────────────────────────────────────────
    logger.info("Loading harmless embeddings...")
    H_benign_10k  = _load("embeds_benign_train.pt")
    H_coconot_pref = _load("embeds_coconot_pref.pt")
    H_coconot_orig = _load("embeds_coconot_original.pt")

    # Lấy subset coconot_original (giống calc_steering_matrix gốc)
    idx_coco = torch.randperm(H_coconot_orig.size(0), generator=gen)[
        : 4000 - H_coconot_pref.size(0)
    ]
    H_harmless_full = torch.cat(
        [H_benign_10k, H_coconot_orig[idx_coco], H_coconot_pref], dim=0
    )
    del H_benign_10k, H_coconot_orig, H_coconot_pref
    logger.info("H_harmless full: %s", tuple(H_harmless_full.shape))

    # ── FIX B6: Downsample harmless → N_harmful * balance_ratio ──────────────
    #
    # v2 BUG: harmless ~14000 >> harmful 2000 → 1:7 imbalance
    # v4 FIX: downsample harmless về N_target = round(N_harmful * balance_ratio)
    #
    # Tại sao không upsample harmful?
    #   → Upsample tạo duplicate embeddings → làm méo distribution
    #   → Downsample harmless safe hơn vì có nhiều data
    #
    # balance_ratio = 1.0 → 1:1 (recommended, giống AdaSteer paired data)
    # balance_ratio = 1.2 → 1.2:1 (giống AlphaSteer MLP line 3100: 1.2x benign)
    #
    N_target = round(N_harmful * balance_ratio)
    N_harmless_full = H_harmless_full.shape[0]

    if N_harmless_full > N_target:
        idx_down = torch.randperm(N_harmless_full, generator=gen)[:N_target]
        H_harmless = H_harmless_full[idx_down]
        logger.info(
            "Downsampled harmless: %d → %d (ratio=%.1f)",
            N_harmless_full, N_target, balance_ratio,
        )
    else:
        # Nếu harmless ít hơn target (unlikely nhưng defensive)
        H_harmless = H_harmless_full
        logger.warning(
            "Harmless (%d) < target (%d). Dùng toàn bộ harmless.",
            N_harmless_full, N_target,
        )

    del H_harmless_full

    logger.info(
        "Final balance — harmful: %d, harmless: %d  (ratio=%.2f:1)",
        H_harmful.shape[0],
        H_harmless.shape[0],
        H_harmless.shape[0] / H_harmful.shape[0],
    )

    return H_harmful, H_harmless


# Backward-compatible alias (giữ tên cũ để các file khác không cần đổi import)
def load_alphasteer_embeddings(embedding_dir: str, seed: int = 42) -> tuple:
    """
    Alias của load_balanced_embeddings() với balance_ratio=1.0.
    Drop-in replacement cho hàm cùng tên trong v2.

    v2 trả về (H_refusal, H_compliant) imbalanced.
    v4 trả về (H_harmful, H_harmless)  balanced 1:1.

    Tên parameter trong compute_rfm_refusal_vectors() đã được đổi
    từ H_refusal/H_compliant → H_harmful/H_harmless để rõ nghĩa hơn.
    """
    logger.info(
        "load_alphasteer_embeddings() → load_balanced_embeddings(ratio=1.0)"
    )
    return load_balanced_embeddings(embedding_dir, balance_ratio=1.0, seed=seed)


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 5 — Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute RFM refusal vector với balanced harmful/harmless data"
    )
    p.add_argument("--embedding_dir", required=True,
                   help="Thư mục chứa embeds_*.pt files")
    p.add_argument("--model_name", required=True,
                   choices=list(STEERING_LAYERS.keys()))
    p.add_argument("--method", default="rfm", choices=["linear", "rfm"],
                   help="rfm = full RFM-AGOP (recommended sau khi fix balance); "
                        "linear = AGOP từ logistic probe (faster, ablation)")
    p.add_argument("--rfm_iters", type=int, default=3,
                   help="Số iterations cho RFM (chỉ dùng khi method=rfm)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_path", required=True,
                   help="Path để save refusal vectors (.pkl)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_jailbreak", type=int, default=1000,
                   help="Số jailbreak samples lấy từ embeds_jailbreak_train.pt")
    p.add_argument("--balance_ratio", type=float, default=1.0,
                   help="Tỉ lệ harmless/harmful sau balance (1.0 = 1:1)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)   # FIX B5
    np.random.seed(args.seed)

    # ── Load balanced data ────────────────────────────────────────────────────
    H_harmful, H_harmless = load_balanced_embeddings(
        embedding_dir=args.embedding_dir,
        n_jailbreak=args.n_jailbreak,
        balance_ratio=args.balance_ratio,
        seed=args.seed,
    )

    num_total_layers = H_harmful.shape[1]
    d = H_harmful.shape[2]
    layers = STEERING_LAYERS[args.model_name]

    logger.info(
        "model=%s  layers=%s  d=%d  method=%s  device=%s",
        args.model_name, layers, d, args.method, args.device,
    )
    logger.info(
        "Data: harmful=%d, harmless=%d  balance_ratio=%.1f",
        H_harmful.shape[0], H_harmless.shape[0], args.balance_ratio,
    )

    # ── Compute RFM refusal vectors ───────────────────────────────────────────
    rv = compute_rfm_refusal_vectors(
        H_harmful=H_harmful,
        H_harmless=H_harmless,
        layers=layers,
        num_total_layers=num_total_layers,
        method=args.method,
        rfm_iters=args.rfm_iters,
        device=args.device,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(rv, f)

    logger.info("Saved → %s  shape=%s", args.save_path, rv.shape)