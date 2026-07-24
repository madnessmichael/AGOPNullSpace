"""
calc_steering_matrix_rfm_rc.py
===============================
Drop-in replacement for calc_steering_matrix_rfm_hh.py: concept vector `r` is
trained on a per-model SORRY-Bench refuse-compliance dataset (see
build_refusal_compliance_sorrybench.py) instead of
justinphan3110/harmful_harmless_instructions -- HH is not used anywhere in
this file.

Label convention: y=1=refusal, y=0=compliance, so `r` points toward refusal
(strength>0 == defend, consistent with the rest of the repo). Gate `u` and
the null-space projector `Q` are still fit on AlphaSteer's own D_m/D_b
(14k benign + 2k malicious, via load_alphasteer_embeddings) -- unchanged
from the HH version. Only the source of `r` changed.

New: the linear gate `u^T h` (unbounded) is replaced at *inference* time by
a bounded squash g(u^T h) (sigmoid or clip). Because that gate is nonlinear,
it can no longer be folded into a single dense [d,d] matrix -- this script
saves rank-1 factors {u, r} per layer plus the gate spec instead:

    {
        "format": "rank1_gate_v1",
        "num_layers": L, "d_model": d, "layers": [...],
        "gate_type": "sigmoid" | "clip", "gate_slope": float,
        "factors": {layer: {"u": Tensor[d], "r": Tensor[d]}},
    }

AlphaLlama.py / AlphaQwen.py / AlphaGemma.py and generate_response.py know
how to load this format (see their rank1_gate_v1 branch). A companion plain
[L, d] tensor of `r` alone is also saved (same basename + "_r.pt") so the
no-nullspace script can reuse the identical concept vector via
--refusal_vectors_path without recomputing it.

Usage:
    python src/calc_steering_matrix_rfm_rc.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --device cuda \
        --save_path data/steering_matrix/steering_matrix_llama3.1_rfm_rc.pt \
        --probe rfm --rfm_iters 8 --n_components 1 --seed 2706
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


def load_refusal_compliance_embeddings(embedding_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load per-model SORRY-Bench refuse-compliance activations produced by
    build_refusal_compliance_sorrybench.py: [N, num_layers, d_model] each.
    """
    rc_dir = os.path.join(embedding_dir, "refusal_compliance")
    refusal_path = os.path.join(rc_dir, "embeds_refusal.pt")
    compliance_path = os.path.join(rc_dir, "embeds_compliance.pt")
    if not (os.path.exists(refusal_path) and os.path.exists(compliance_path)):
        raise FileNotFoundError(
            f"Refuse-compliance embeddings not found under {rc_dir}. "
            "Run build_refusal_compliance_sorrybench.py first."
        )
    H_refusal = torch.load(refusal_path, map_location="cpu").float()
    H_compliance = torch.load(compliance_path, map_location="cpu").float()
    logger.info("Loaded refuse-compliance activations: refusal=%s compliance=%s",
                tuple(H_refusal.shape), tuple(H_compliance.shape))
    return H_refusal, H_compliance


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--embedding_dir", required=True,
                   help="Base dir with AlphaSteer embeddings (gate/null-space) AND "
                        "a refusal_compliance/ subfolder (concept vector r).")
    p.add_argument("--save_path", required=True)
    p.add_argument("--device", default="cuda")

    p.add_argument("--probe", default="rfm", choices=["rfm", "linear"])
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--tuning_metric", default="auc", choices=["auc", "accuracy", "mse"])
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--include_math", action="store_true",
                   help="Adds 900 MATH samples to AlphaSteer's D_b (gate/null-space only).")

    p.add_argument("--lambda_reg", type=float, default=10.0)
    p.add_argument("--nullspace_dtype", default="float32", choices=["float32", "float64"])
    p.add_argument("--holdout_benign", type=int, default=1000)

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

    # -- 2. Concept vector r from the SORRY-Bench refuse-compliance set --
    H_refusal, H_compliance = load_refusal_compliance_embeddings(args.embedding_dir)
    if H_refusal.shape[1] != num_total_layers or H_refusal.shape[2] != d_model:
        raise ValueError(
            f"refuse-compliance activations have shape {tuple(H_refusal.shape)[1:]}, "
            f"expected (num_layers={num_total_layers}, d_model={d_model}) to match "
            f"AlphaSteer embeddings -- check build_refusal_compliance_sorrybench.py output.")

    logger.info("Computing %s concept vectors on refuse-compliance data "
                "(metric=%s, iters=%d, n_compliance=%d)...",
                args.probe.upper(), args.tuning_metric, args.rfm_iters, H_compliance.shape[0])

    refusal_vectors_np, probe_meta = compute_refusal_vectors(
        H_malicious=H_refusal,      # y=1=refusal
        H_benign=H_compliance,      # y=0=compliance
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
            "probe": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                      for k, v in probe_meta[layer].items()},
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

    # Companion plain [L, d] r-only tensor, for calc_steering_matrix_rfm_rc_no_nullspace.py
    # --refusal_vectors_path (keeps r IDENTICAL between the two files, no refit).
    r_only_path = os.path.splitext(args.save_path)[0] + "_r.pt"
    torch.save(refusal_vectors, r_only_path)

    meta = {
        "model_name": args.model_name, "probe": args.probe,
        "concept_source": "sorry-bench/sorry-bench-202503 (refusal_compliance)",
        "n_refusal": int(H_refusal.shape[0]), "n_compliance": int(H_compliance.shape[0]),
        "gate_data": "alphasteer_malicious", "nullspace_data": "alphasteer_benign",
        "rfm_iters": args.rfm_iters, "tuning_metric": args.tuning_metric,
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
