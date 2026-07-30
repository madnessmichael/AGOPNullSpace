"""
compare_dim_vs_rfm.py
======================
Geometric comparison of the two concept-vector estimators trained on the
SAME refuse-compliance data (rc_hr: refusal_compliance_full_hard_refusal):
r_RFM (top AGOP eigenvector) vs r_DIM (mean(refusal) - mean(compliance)).

Both are loaded from the companion "_r.pt" plain [L, d] tensors saved by
calc_steering_matrix_rfm_rc.py / calc_steering_matrix_dim_rc.py -- these are
the raw per-layer unit vectors BEFORE the null-space/gate regression, so
this comparison isolates exactly what changed between the two pipelines.

Usage:
    python evaluation/compare_dim_vs_rfm.py --model_name llama3.1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils.const import AlphaSteer_CALCULATION_CONFIG  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--steering_dir", default="data/steering_matrix")
    p.add_argument("--rfm_r_path", default=None,
                   help="Defaults to steering_matrix_<model>_rfm_rc_full_hard_refusal_r.pt")
    p.add_argument("--dim_r_path", default=None,
                   help="Defaults to steering_matrix_<model>_dim_rc_full_hard_refusal_r.pt")
    p.add_argument("--out_fig", default=None,
                   help="Defaults to figures/dim_vs_rfm_cosine_<model>.png")
    return p.parse_args()


def main():
    args = parse_args()
    m = args.model_name

    rfm_path = args.rfm_r_path or os.path.join(
        args.steering_dir, f"steering_matrix_{m}_rfm_rc_full_hard_refusal_r.pt")
    dim_path = args.dim_r_path or os.path.join(
        args.steering_dir, f"steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt")
    out_fig = args.out_fig or f"figures/dim_vs_rfm_cosine_{m}.png"

    r_rfm = torch.load(rfm_path, map_location="cpu").float()
    r_dim = torch.load(dim_path, map_location="cpu").float()
    assert r_rfm.shape == r_dim.shape, (r_rfm.shape, r_dim.shape)

    layers = [l for l, _ in AlphaSteer_CALCULATION_CONFIG[m]]
    cos = []
    for l in layers:
        a, b = r_rfm[l], r_dim[l]
        na, nb = a.norm().item(), b.norm().item()
        c = float((a @ b) / (na * nb)) if na > 0 and nb > 0 else float("nan")
        cos.append(c)
    cos = np.array(cos)

    print(f"{'layer':>6} {'cos(r_RFM, r_DIM)':>20}")
    for l, c in zip(layers, cos):
        print(f"{l:>6} {c:>20.4f}")
    print()
    print(f"mean={cos.mean():.4f}  min={cos.min():.4f} (layer {layers[cos.argmin()]})  "
          f"max={cos.max():.4f} (layer {layers[cos.argmax()]})  std={cos.std():.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, cos, marker="o", color="#3b6fa0", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(r_RFM, r_DIM)")
    ax.set_title(f"{m}: RFM vs DIM concept-vector agreement per layer\n"
                 f"(same refusal_compliance_full_hard_refusal data)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"\nSaved figure -> {out_fig}")


if __name__ == "__main__":
    main()
