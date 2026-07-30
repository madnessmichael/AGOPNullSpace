"""
fig1_teaser_v2.py
===================
Redesign of fig1_teaser_encoding.py after feedback that the 3D scatter,
while visually clean, did not explain WHY RFM/AGOP beats DIM or show how the
top-1 AGOP eigenvector arises, and did not make RFM's advantage visually
obvious.

Two panels:
  LEFT  -- a schematic "recipe" panel (real formulas, not fabricated toy
           data -- an earlier attempt to reproduce the effect on synthetic
           2D Gaussian-cluster data did NOT reproduce the real phenomenon,
           see conversation; presenting a forced toy demo would misrepresent
           the mechanism, so this panel is explicitly conceptual/diagrammatic).
           Shows DIM's one global mean-difference vs RFM's per-point local
           gradient -> averaged outer product -> top eigenvector, with one
           sentence explaining why the local-gradient route can retain signal
           on a rare subgroup that a single global average washes out.
  RIGHT -- REAL data: a joint scatter + marginal histograms of the encoding-
           family (caesar/morse/atbash/ascii) activations at the single layer
           with the largest AUC gap (layer 25 for llama3.1: AUC_RFM=0.633 vs
           AUC_DIM=0.273), projected onto (r_DIM, r_RFM). The marginal
           histograms make the separation-or-lack-thereof immediately
           legible in 1D, which a 2D/3D scatter cloud alone does not.

Usage:
    python evaluation/fig1_teaser_v2.py --model_name llama3.1 --layer 25
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ENCODING_STYLES = ["caesar", "morse", "atbash", "ascii"]

COL_COMPLIANCE = "#2a78d6"
COL_REFUSAL = "#eb6834"
COL_RFM = "#111111"
COL_DIM = "#4a3aa7"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--layer", type=int, default=25)
    p.add_argument("--sample_path", default=None)
    p.add_argument("--steering_dir", default="data/steering_matrix")
    p.add_argument("--out_fig", default=None)
    return p.parse_args()


def draw_schematic(ax, m):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def box(xy, w, h, text, fc="#f5f5f4", ec="#333333", fontsize=9.5, weight="normal"):
        from matplotlib.patches import FancyBboxPatch
        x, y = xy
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.12",
                            facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=2)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, fontweight=weight, zorder=3, linespacing=1.4)

    def arrow(xy1, xy2, color="#555555", lw=1.6, style="-|>"):
        ax.annotate("", xy=xy2, xytext=xy1,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=1)

    box((2.6, 8.6), 4.8, 1.0, "activations {(h_i, y_i)}\ny=1 refusal, y=0 compliance",
        fc="#eef0ef", fontsize=9.5)

    arrow((3.6, 8.6), (1.8, 7.3))
    arrow((6.6, 8.6), (8.2, 7.3))

    box((0.2, 6.1), 3.2, 1.2,
        "DIM\nr = mean(h|y=1) − mean(h|y=0)\n(ONE global subtraction)",
        fc="#efeaf7", fontsize=8.7)

    box((6.6, 6.1), 3.2, 1.4,
        "RFM / AGOP\nfit nonlinear f(h) (kernel + iterated metric M)\n"
        "then EVERY point gets its own ∇f(h_i)",
        fc="#eef2fa", fontsize=8.3)

    arrow((2.2, 6.1), (2.2, 5.0))
    box((0.3, 4.0), 3.0, 1.0, "one arrow,\nblind to sub-population\nstructure",
        fc="#f7f0f0", fontsize=8.3)

    arrow((8.2, 6.1), (8.2, 5.2))
    box((6.4, 3.7), 3.6, 1.5,
        "G = (1/n) Σᵢ ∇f(hᵢ) ∇f(hᵢ)ᵀ\nr_RFM = top eigenvector(G)\n"
        "(local pushes from EVERY subgroup\naveraged into the metric, not just\nthe majority's global shift)",
        fc="#eef2fa", fontsize=8.0)

    arrow((1.8, 4.0), (1.8, 2.1))
    arrow((8.2, 3.7), (8.2, 2.1))
    box((0.2, 0.9), 3.2, 1.2,
        "dominated by whichever\nsubgroup has the most rows +\ncleanest separation",
        fc="#f7f0f0", fontsize=8.3)
    box((6.6, 0.9), 3.2, 1.2,
        "retains a usable signal even for\nrare / hard subgroups the\nmajority direction washes out",
        fc="#eef2fa", fontsize=8.3)

    ax.text(5, 10.8, "How the two directions are built", ha="center", va="top",
            fontsize=11.5, fontweight="bold")


def main():
    args = parse_args()
    m = args.model_name
    l = args.layer
    sample_path = args.sample_path or f"data/embeddings/{m}/rc_style_full_dir.pt"
    out_fig = args.out_fig or f"figures/fig1_teaser_v2_{m}.png"

    data = torch.load(sample_path, map_location="cpu")
    H = data["H"].float()
    styles = np.array(data["prompt_style"])
    y = np.array(data["label"])
    mask = np.isin(styles, ENCODING_STYLES)
    H_l = H[mask][:, l, :]
    is_ref = y[mask].astype(bool)

    r_rfm = torch.load(os.path.join(args.steering_dir, f"steering_matrix_{m}_rfm_rc_full_hard_refusal_r.pt"),
                        map_location="cpu").float()[l]
    r_dim = torch.load(os.path.join(args.steering_dir, f"steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt"),
                        map_location="cpu").float()[l]

    x_raw = (H_l @ r_dim).numpy()
    y_raw = (H_l @ r_rfm).numpy()
    auc_dim = roc_auc_score(is_ref, x_raw)
    auc_rfm = roc_auc_score(is_ref, y_raw)
    x = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-8)
    yv = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-8)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(14, 6.6))
    fig.patch.set_facecolor("white")
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[0.85, 1.15], wspace=0.18,
                         left=0.03, right=0.98, top=0.86, bottom=0.09)

    ax_schema = fig.add_subplot(gs_outer[0, 0])
    draw_schematic(ax_schema, m)

    gs_right = GridSpec(4, 4, figure=fig,
                         left=gs_outer.get_grid_positions(fig)[2][1] + 0.06,
                         right=0.985, top=0.83, bottom=0.11, wspace=0.05, hspace=0.05)
    ax_scatter = fig.add_subplot(gs_right[1:4, 0:3])
    ax_top = fig.add_subplot(gs_right[0, 0:3], sharex=ax_scatter)
    ax_rightm = fig.add_subplot(gs_right[1:4, 3], sharey=ax_scatter)

    ax_scatter.scatter(x[~is_ref], yv[~is_ref], s=10, color=COL_COMPLIANCE, alpha=0.45,
                        linewidths=0, label=f"compliance (n={(~is_ref).sum()})")
    ax_scatter.scatter(x[is_ref], yv[is_ref], s=42, color=COL_REFUSAL, alpha=0.95,
                        edgecolors="white", linewidths=0.6, label=f"hard refusal (n={is_ref.sum()})")
    ax_scatter.set_xlabel("projection onto r_DIM  (z-scored)", fontsize=10)
    ax_scatter.set_ylabel("projection onto r_RFM  (z-scored)", fontsize=10)
    ax_scatter.grid(alpha=0.25)
    ax_scatter.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    bins = np.linspace(-3.5, 3.5, 36)
    ax_top.hist(x[~is_ref], bins=bins, color=COL_COMPLIANCE, alpha=0.55, density=True)
    ax_top.hist(x[is_ref], bins=bins, color=COL_REFUSAL, alpha=0.75, density=True)
    ax_top.axis("off")
    ax_top.text(0.02, 0.85, f"AUC(r_DIM) = {auc_dim:.2f}", transform=ax_top.transAxes,
                fontsize=11, fontweight="bold", color=COL_DIM, va="top")

    ax_rightm.hist(yv[~is_ref], bins=bins, color=COL_COMPLIANCE, alpha=0.55,
                    density=True, orientation="horizontal")
    ax_rightm.hist(yv[is_ref], bins=bins, color=COL_REFUSAL, alpha=0.75,
                    density=True, orientation="horizontal")
    ax_rightm.axis("off")
    ax_rightm.text(0.5, 0.985, f"AUC(r_RFM)\n= {auc_rfm:.2f}", transform=ax_rightm.transAxes,
                    fontsize=11, fontweight="bold", color=COL_RFM, ha="center", va="top",
                    rotation=0)

    fig.suptitle("Why AGOP beats a global mean-difference on heterogeneous jailbreaks",
                 fontsize=15, fontweight="bold", y=0.99)
    fig.text(gs_outer.get_grid_positions(fig)[2][1] + 0.06, 0.895,
             f"{m}, layer {l}: encoded-jailbreak activations (caesar / morse / atbash / ascii)",
             fontsize=10.5, ha="left", va="bottom")

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=300, facecolor="white")
    print(f"AUC_DIM={auc_dim:.3f}  AUC_RFM={auc_rfm:.3f}")
    print(f"Saved -> {out_fig}")


if __name__ == "__main__":
    main()
