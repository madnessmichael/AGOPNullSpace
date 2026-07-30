"""
rc_hr_subspace_viz.py
=======================
PCA visualization of the ACTUAL training data used to fit the concept
vectors -- refuse-compliance activations from
refusal_compliance_full_hard_refusal (rc_hr) -- colored by refusal/
compliance label, with r_RFM and r_DIM (both trained on this exact data)
overlaid as arrows. Unlike attack_subspace_viz.py (which projects onto the
held-out attack-family activation cloud), this shows the vectors on the
space they were actually fit on -- the direct sanity check for "does each
estimator's direction align with the label split it was trained to find".

Usage:
    python evaluation/rc_hr_subspace_viz.py --model_name llama3.1 --layer 15
    python evaluation/rc_hr_subspace_viz.py --model_name llama3.1 --layer 27
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--rc_subdir", default="refusal_compliance_full_hard_refusal")
    p.add_argument("--embedding_dir", default=None,
                   help="Defaults to data/embeddings/<model>")
    p.add_argument("--steering_dir", default="data/steering_matrix")
    p.add_argument("--out_fig", default=None,
                   help="Defaults to figures/rc_hr_subspace_<model>_layer<layer>.png")
    p.add_argument("--max_per_class", type=int, default=1500,
                   help="Subsample per class for a less crowded/slow plot.")
    p.add_argument("--seed", type=int, default=2706)
    return p.parse_args()


def main():
    args = parse_args()
    m = args.model_name
    torch.manual_seed(args.seed)
    embedding_dir = args.embedding_dir or os.path.join("data/embeddings", m)
    rc_dir = os.path.join(embedding_dir, args.rc_subdir)
    out_fig = args.out_fig or f"figures/rc_hr_subspace_{m}_layer{args.layer}.png"

    rfm_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_rfm_rc_full_hard_refusal_r.pt")
    dim_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt")
    r_rfm = torch.load(rfm_path, map_location="cpu").float()[args.layer]
    r_dim = torch.load(dim_path, map_location="cpu").float()[args.layer]

    H_refusal = torch.load(os.path.join(rc_dir, "embeds_refusal.pt"), map_location="cpu").float()[:, args.layer, :]
    H_compliance = torch.load(os.path.join(rc_dir, "embeds_compliance.pt"), map_location="cpu").float()[:, args.layer, :]

    def subsample(H, n):
        if H.shape[0] <= n:
            return H
        idx = torch.randperm(H.shape[0])[:n]
        return H[idx]

    H_refusal_s = subsample(H_refusal, args.max_per_class)
    H_compliance_s = subsample(H_compliance, args.max_per_class)
    labels = np.array(["refusal"] * H_refusal_s.shape[0] + ["compliance"] * H_compliance_s.shape[0])
    H_all = torch.cat([H_refusal_s, H_compliance_s], dim=0)
    print(f"Loaded refusal={H_refusal.shape[0]} (plotting {H_refusal_s.shape[0]}), "
          f"compliance={H_compliance.shape[0]} (plotting {H_compliance_s.shape[0]}) "
          f"at layer {args.layer}, d={H_all.shape[1]}")

    # PCA fit on the FULL data (not the subsample) for a stable basis, mean-centered.
    H_full = torch.cat([H_refusal, H_compliance], dim=0)
    mean = H_full.mean(0, keepdim=True)
    Hc_full = H_full - mean
    U, S, Vt = torch.linalg.svd(Hc_full, full_matrices=False)
    basis = Vt[:2]
    explained = (S[:2] ** 2 / (S ** 2).sum()).numpy()

    Hc_plot = H_all - mean
    proj = (Hc_plot @ basis.T).numpy()

    scale = float(np.linalg.norm(proj, axis=1).mean()) * 1.5
    r_rfm_2d = (r_rfm @ basis.T).numpy()
    r_rfm_2d = r_rfm_2d / np.linalg.norm(r_rfm_2d) * scale
    r_dim_2d = (r_dim @ basis.T).numpy()
    r_dim_2d = r_dim_2d / np.linalg.norm(r_dim_2d) * scale

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    mask_r = labels == "refusal"
    mask_c = labels == "compliance"
    ax.scatter(proj[mask_c, 0], proj[mask_c, 1], s=10, alpha=0.35,
               color="#3b6fa0", label=f"compliance (n={mask_c.sum()})")
    ax.scatter(proj[mask_r, 0], proj[mask_r, 1], s=10, alpha=0.35,
               color="#c0392b", label=f"refusal (n={mask_r.sum()})")

    ax.annotate("", xy=r_rfm_2d, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5))
    ax.text(*(r_rfm_2d * 1.08), "r_RFM", fontsize=11, fontweight="bold", color="black")
    ax.annotate("", xy=r_dim_2d, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2.5, linestyle="--"))
    ax.text(*(r_dim_2d * 1.08), "r_DIM", fontsize=11, fontweight="bold", color="darkorange")
    ax.scatter([0], [0], color="black", s=30, zorder=5)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title(f"{m}, layer {args.layer}: refusal vs compliance activations (rc_hr training data)\n"
                 f"with r_RFM / r_DIM directions overlaid")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"Saved figure -> {out_fig}")


if __name__ == "__main__":
    main()
