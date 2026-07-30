"""
calc_steering_matrix_dim_rc.py
===============================
DIM (DiffMean) ablation of calc_steering_matrix_rfm_rc.py: concept vector `r`
is computed as the normalized mean-difference between refusal and compliance
activations on the SAME per-model SORRY-Bench refuse-compliance dataset used
by the RFM pipeline (see build_refusal_compliance_sorrybench.py), instead of
the RFM/AGOP top-eigenvector estimator. Everything else -- gate `u` and
null-space `Q` fit on AlphaSteer's own D_m/D_b, the sigmoid/clip gate at
inference time, the rank1_gate_v1 output format -- is unchanged, so this is a
clean estimator-only ablation (DIM-RC vs RFM-RC) on identical data.

r[l] = normalize( mean(H_refusal[:,l,:]) - mean(H_compliance[:,l,:]) )
     -- points toward refusal, same sign convention as compute_refusal_vectors
        (y=1=refusal, strength>0 == defend).

Usage:
    python src/calc_steering_matrix_dim_rc.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --rc_subdir refusal_compliance_full_hard_refusal \
        --save_path data/steering_matrix/steering_matrix_llama3.1_dim_rc_full_hard_refusal.pt \
        --lambda_reg 10.0 --seed 2706 --device cuda
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
from rfm_refusal_vector import load_alphasteer_embeddings       # noqa: E402
from calc_steering_matrix_rfm_rc import load_refusal_compliance_embeddings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_dim_vectors_rc(H_refusal: torch.Tensor,
                            H_compliance: torch.Tensor,
                            layers,
                            num_total_layers: int) -> tuple[np.ndarray, dict]:
    """
    r[l] = normalize(mean(H_refusal[:,l,:]) - mean(H_compliance[:,l,:])).
    No balancing needed -- DIM is a difference of class means, unaffected by
    class-size imbalance (unlike the RFM probe, which is sensitive to it).
    Uses all available rows per class to minimize mean-estimate variance.
    Returns (num_total_layers, d) float32; layers not in `layers` are left 0,
    matching compute_refusal_vectors' output convention.
    """
    d = H_refusal.shape[2]
    rv = np.zeros((num_total_layers, d), dtype=np.float32)
    meta = {}
    for l in layers:
        mr = H_refusal[:, l, :].mean(0)
        mc = H_compliance[:, l, :].mean(0)
        r = mr - mc
        raw_norm = float(r.norm())
        r = r / r.norm().clamp(min=1e-8)
        rv[l] = r.numpy()
        pooled_std = float(0.5 * (H_refusal[:, l, :].std(0).mean()
                                  + H_compliance[:, l, :].std(0).mean()))
        meta[l] = {
            "probe": "dim_rc",
            "raw_diff_norm": raw_norm,
            "separation_ratio": raw_norm / max(pooled_std * (d ** 0.5), 1e-8),
            "n_refusal": int(H_refusal.shape[0]),
            "n_compliance": int(H_compliance.shape[0]),
        }
        logger.info("  layer %d: ||Δmean||=%.4f  separation=%.3f",
                    l, raw_norm, meta[l]["separation_ratio"])
    return rv, meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--embedding_dir", required=True,
                   help="Base dir with AlphaSteer embeddings (gate/null-space) AND "
                        "a refusal_compliance/ (or --rc_subdir) subfolder (concept vector r).")
    p.add_argument("--rc_subdir", default="refusal_compliance",
                   help="Which refuse-compliance embedding subfolder to compute r on: "
                        "'refusal_compliance' (440 base prompts), 'refusal_compliance_full' "
                        "(9,236 rows) or 'refusal_compliance_full_hard_refusal' (strict judge).")
    p.add_argument("--save_path", required=True)
    p.add_argument("--device", default="cuda")

    p.add_argument("--max_per_class", type=int, default=None,
                   help="Optional cap per class before computing means (uses the first N "
                        "rows after loading; DIM itself does not need balancing).")

    p.add_argument("--lambda_reg", type=float, default=10.0)
    p.add_argument("--nullspace_dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--holdout_benign", type=int, default=1000)
    p.add_argument("--include_math", action="store_true",
                   help="Adds 900 MATH samples to AlphaSteer's D_b (gate/null-space only).")

    p.add_argument("--gate_type", default="sigmoid", choices=["sigmoid", "clip"])
    p.add_argument("--gate_slope", type=float, default=10.0,
                   help="Slope `a` for sigmoid(a*(u^Th - 0.5)); ignored for --gate_type clip.")

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

    # -- 1. AlphaSteer embeddings: gate u + null-space P only (unchanged) --
    H_malicious, H_benign = load_alphasteer_embeddings(
        args.embedding_dir, seed=args.seed, include_math=args.include_math)
    num_total_layers = H_benign.shape[1]
    d_model = H_benign.shape[2]

    g = torch.Generator().manual_seed(args.seed + 1)
    perm = torch.randperm(H_benign.shape[0], generator=g)
    n_ho = min(args.holdout_benign, H_benign.shape[0] // 10)
    ho_idx, fit_idx = perm[:n_ho], perm[n_ho:]
    H_benign_ho = H_benign[ho_idx]
    H_benign_fit = H_benign[fit_idx]
    logger.info("D_b: %d fit / %d held-out | D_m: %d | L=%d d=%d",
                len(fit_idx), len(ho_idx), H_malicious.shape[0], num_total_layers, d_model)

    # -- 2. Concept vector r: DIM on the SAME refuse-compliance set as RFM-RC --
    H_refusal, H_compliance = load_refusal_compliance_embeddings(args.embedding_dir, args.rc_subdir)
    if H_refusal.shape[1] != num_total_layers or H_refusal.shape[2] != d_model:
        raise ValueError(
            f"refuse-compliance activations have shape {tuple(H_refusal.shape)[1:]}, "
            f"expected (num_layers={num_total_layers}, d_model={d_model}) to match "
            f"AlphaSteer embeddings -- check build_refusal_compliance_sorrybench.py output.")

    if args.max_per_class is not None:
        H_refusal = H_refusal[:args.max_per_class]
        H_compliance = H_compliance[:args.max_per_class]

    logger.info("Computing DIM concept vectors on refuse-compliance data "
                "(n_refusal=%d, n_compliance=%d)...",
                H_refusal.shape[0], H_compliance.shape[0])

    refusal_vectors_np, probe_meta = compute_dim_vectors_rc(
        H_refusal, H_compliance, layers, num_total_layers)
    refusal_vectors = torch.from_numpy(refusal_vectors_np).float()
    logger.info("refusal_vectors %s dtype=%s (||r||=1 per layer)",
                tuple(refusal_vectors.shape), refusal_vectors.dtype)

    # -- 3. Null-space + gate (unchanged math, new storage: rank-1, not dense) --
    factors: dict[int, dict] = {}
    diagnostics: dict[str, dict] = {}

    for layer in layers:
        ratio = ratios[layer]
        logger.info("=== layer %d (rho=%.2f) ===", layer, ratio)

        h_b = H_benign_fit[:, layer, :].to(device).float()
        ns_dtype = torch.float64 if args.nullspace_dtype == "float64" else None
        Q_layer = null_space_basis_l(h_b, abs_nullspace_ratio=ratio, dtype=ns_dtype).float()
        del h_b

        leak = benign_leakage(H_benign_ho[:, layer, :], Q_layer)
        logger.info("  benign held-out leakage ||Q^Th||/||h||: mean=%.4f p95=%.4f max=%.4f",
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
        logger.info("  gate u^Th: malicious=%.4f+-%.4f benign_holdout=%.4f+-%.4f "
                    "-> selectivity=%.1fx", st_m["gate_mean"], st_m["gate_std"],
                    st_b["gate_mean"], st_b["gate_std"], sel)

        diagnostics[str(layer)] = {
            "rho": ratio, "k_nullspace": int(round(ratio * d_model)),
            "u_norm": float(u.norm()), "benign_holdout_leakage": leak,
            "gate_malicious": st_m, "gate_benign_holdout": st_b,
            "selectivity_ratio": sel,
            "probe": probe_meta[layer],
        }
        del u, r
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- 4. Save: rank1_gate_v1 dict (nonlinear-gate inference) --
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    payload = {
        "format": "rank1_gate_v1",
        "num_layers": num_total_layers, "d_model": d_model, "layers": layers,
        "gate_type": args.gate_type, "gate_slope": args.gate_slope,
        "factors": factors,
    }
    torch.save(payload, args.save_path)

    # Companion plain [L, d] r-only tensor, for a future no-nullspace ablation
    # (keeps r IDENTICAL between the two files, no refit).
    r_only_path = os.path.splitext(args.save_path)[0] + "_r.pt"
    torch.save(refusal_vectors, r_only_path)

    meta = {
        "model_name": args.model_name, "probe": "dim",
        "concept_source": f"sorry-bench/sorry-bench-202503 ({args.rc_subdir})",
        "n_refusal": int(H_refusal.shape[0]), "n_compliance": int(H_compliance.shape[0]),
        "gate_data": "alphasteer_malicious", "nullspace_data": "alphasteer_benign",
        "lambda_reg": args.lambda_reg, "seed": args.seed,
        "gate_type": args.gate_type, "gate_slope": args.gate_slope,
        "N_malicious": int(H_malicious.shape[0]), "N_benign_total": int(H_benign.shape[0]),
        "N_benign_fit_P": int(len(fit_idx)), "N_benign_holdout": int(len(ho_idx)),
        "rho_per_layer": {str(l): ratios[l] for l in layers},
        "layers": layers, "per_layer": diagnostics,
    }
    meta_path = os.path.splitext(args.save_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved -> %s (format=rank1_gate_v1)", args.save_path)
    logger.info("Saved r-only -> %s", r_only_path)
    logger.info("Meta  -> %s", meta_path)
    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
