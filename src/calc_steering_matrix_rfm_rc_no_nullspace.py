"""
calc_steering_matrix_rfm_rc_no_nullspace.py
=============================================
ABLATION: pure additive steering, no gate, no null-space, no regression --
h' = h + strength * r_l applied at EVERY token position. This matches the
paper's original anti-refusal formula A~_{l,i}(X) := A_{l,i}(X) + eps*v_l
(Phan A of the refuse-compliance design discussion).

v2: previously saved a plain [L, d] tensor consumed by NaiveSteerModel /
Steer_MODELS_DICT via --steering_vector_path. That path had two latent bugs
in NaiveSteerModel (set_steering_parameters wiped the vector to None on every
strength-only update, so all 8 strengths in a sweep silently collapsed to
the same unsteered output; and an uninitialized/meta `torch.empty` sentinel
crashed once that wipe was fixed). Rather than keep patching a second model
class, this now saves the same rank1_gate_v1 dict format as the null-space
script, with `u=None` per layer -- AlphaLlama/AlphaQwen/AlphaGemma's forward()
detects u=None and takes a no-gate branch: h + strength*r on every token
position, with no last-token/gate computation at all. Same model classes,
same state-preserving set_steering_parameters as the gated path, one fewer
model family to keep correct. Load via --steering_matrix_path (AlphaSteer_
MODELS_DICT), not --steering_vector_path.

`r` should be the IDENTICAL vector trained by calc_steering_matrix_rfm_rc.py
on the SORRY-Bench refuse-compliance set -- pass its companion
"<save_path>_r.pt" via --refusal_vectors_path to reuse it verbatim (avoids
any stochastic refit divergence). Standalone recompute (same refuse-
compliance loader, same hyperparams) is supported as a fallback if you want
an independent run instead.

HH (justinphan3110/harmful_harmless_instructions) is not used anywhere in
this file. AlphaSteer's D_m/D_b are not used either -- this script only
ever touches the refuse-compliance data.

Usage:
    # Preferred: reuse r from the null-space run
    python src/calc_steering_matrix_rfm_rc_no_nullspace.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --save_path data/steering_matrix/steering_matrix_llama3.1_rfm_rc_no_nullspace.pt \
        --refusal_vectors_path data/steering_matrix/steering_matrix_llama3.1_rfm_rc_r.pt

    # Standalone recompute
    python src/calc_steering_matrix_rfm_rc_no_nullspace.py \
        --model_name llama3.1 \
        --embedding_dir data/embeddings/llama3.1 \
        --save_path data/steering_matrix/steering_matrix_llama3.1_rfm_rc_no_nullspace.pt \
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

from utils.const import AlphaSteer_CALCULATION_CONFIG  # noqa: E402
from rfm_refusal_vector import compute_refusal_vectors  # noqa: E402
from calc_steering_matrix_rfm_rc import load_refusal_compliance_embeddings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--embedding_dir", required=True,
                   help="Must contain a refusal_compliance/ subfolder "
                        "(build_refusal_compliance_sorrybench.py output).")
    p.add_argument("--save_path", required=True)
    p.add_argument("--device", default="cuda")

    p.add_argument("--refusal_vectors_path", default=None,
                   help="Reuse r from the null-space run's companion *_r.pt file "
                        "(STRONGLY recommended -- keeps r identical between the two "
                        "files). Omit to recompute r from the refuse-compliance set here.")

    p.add_argument("--probe", default="rfm", choices=["rfm", "linear"])
    p.add_argument("--rfm_iters", type=int, default=8)
    p.add_argument("--tuning_metric", default="auc", choices=["auc", "accuracy", "mse"])
    p.add_argument("--n_components", type=int, default=1)
    p.add_argument("--max_per_class", type=int, default=None)

    p.add_argument("--seed", type=int, default=2706)
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layers_ratio_list = AlphaSteer_CALCULATION_CONFIG[args.model_name]
    layers = [l for l, _ in layers_ratio_list]

    concept_data = None
    if args.refusal_vectors_path:
        refusal_vectors = torch.load(args.refusal_vectors_path, map_location="cpu").float()
        concept_source = args.refusal_vectors_path
        num_total_layers, d_model = refusal_vectors.shape
        logger.info("Reusing r from %s -- shape=%s", args.refusal_vectors_path,
                    tuple(refusal_vectors.shape))
    else:
        logger.warning("No --refusal_vectors_path -> recomputing r from refuse-compliance "
                        "data. Use identical --seed/--probe/--rfm_iters as the null-space "
                        "run if you need the two ablations to differ by ONLY the null-space.")
        H_refusal, H_compliance = load_refusal_compliance_embeddings(args.embedding_dir)
        num_total_layers, d_model = H_refusal.shape[1], H_refusal.shape[2]
        concept_data = {"n_refusal": int(H_refusal.shape[0]), "n_compliance": int(H_compliance.shape[0])}

        refusal_vectors_np, _probe_meta = compute_refusal_vectors(
            H_malicious=H_refusal, H_benign=H_compliance,
            layers=layers, num_total_layers=num_total_layers,
            probe=args.probe, rfm_iters=args.rfm_iters,
            n_components=args.n_components, tuning_metric=args.tuning_metric,
            max_per_class=args.max_per_class, seed=args.seed, device=args.device,
        )
        refusal_vectors = torch.from_numpy(refusal_vectors_np).float()
        concept_source = "sorry-bench/sorry-bench-202503 (refusal_compliance, recomputed)"

    # -- pure additive: h' = h + strength * r, every token position --
    # No regression, no gate, no null-space. Saved as rank1_gate_v1 with
    # u=None per steered layer so AlphaLlama/AlphaQwen/AlphaGemma take the
    # no-gate branch (see module docstring) -- load via --steering_matrix_path.
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    factors = {l: {"u": None, "r": refusal_vectors[l].clone()} for l in layers}
    payload = {
        "format": "rank1_gate_v1",
        "num_layers": num_total_layers, "d_model": d_model, "layers": layers,
        "gate_type": None, "gate_slope": None,
        "factors": factors,
    }
    torch.save(payload, args.save_path)

    meta = {
        "ablation": "vector", "model_name": args.model_name,
        "concept_source": concept_source, "concept_data": concept_data,
        "refusal_vectors_path": args.refusal_vectors_path,
        "seed": args.seed, "num_layers": num_total_layers, "d_model": d_model,
        "layers": layers, "nullspace": False, "gate": None,
    }
    meta_path = os.path.splitext(args.save_path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved -> %s (format=rank1_gate_v1, no-gate)  shape=%s",
                args.save_path, tuple(refusal_vectors.shape))
    logger.info("Meta  -> %s", meta_path)
    logger.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
