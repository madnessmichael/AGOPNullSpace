"""
fig1_teaser_encoding.py
=========================
Figure-1 "teaser" for the DIM-vs-RFM ablation: a compact 3D visualization
showing that on the character-level ENCODING family of SORRY-Bench
prompt_style mutations (caesar, morse, atbash, ascii) -- the hardest,
most heterogeneous jailbreak surface in the rc_hr training data -- the
RFM/AGOP-derived concept direction separates refusal from compliance, while
the DIM (mean-difference) direction does not (dim_vs_rfm_by_style_family.py
found DIM AUC often falls BELOW 0.5 here, i.e. anti-informative).

Axes are literal, not a generic PCA: X = projection onto r_DIM, Y =
projection onto r_RFM, both z-scored per layer for visual comparability
across layers whose activation norms grow with depth. Layer is the
(discretized) 3rd axis, so the same 2D relationship is shown "growing
apart" with depth in one glance -- five slices at layers 15/18/21/24/27,
the range where the AUC gap is largest and most consistent (see
dim_vs_rfm_by_style_family.py's full-data run).

Requires: data/embeddings/<model>/rc_style_full_dir.pt (extract_rc_style_sample.py
+ scripts_claude/run_extract_rc_style_full.sh) and the two _r.pt concept vectors.

Usage:
    python evaluation/fig1_teaser_encoding.py --model_name llama3.1
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

# Validated categorical pair (dataviz skill palette, slots 1 & 2 -- the
# "first three slots validate all-pairs in both modes" guarantee):
COL_COMPLIANCE = "#2a78d6"   # blue
COL_REFUSAL = "#eb6834"      # orange
COL_RFM = "#111111"          # near-black -- the method being argued for
COL_DIM = "#4a3aa7"          # violet (slot 7) -- kept far from blue/orange in hue


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--sample_path", default=None)
    p.add_argument("--steering_dir", default="data/steering_matrix")
    p.add_argument("--layers", default="15,18,21,24,27")
    p.add_argument("--out_fig", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    m = args.model_name
    layers = [int(x) for x in args.layers.split(",")]
    sample_path = args.sample_path or f"data/embeddings/{m}/rc_style_full_dir.pt"
    out_fig = args.out_fig or f"figures/fig1_teaser_encoding_{m}.png"

    data = torch.load(sample_path, map_location="cpu")
    H = data["H"].float()
    styles = np.array(data["prompt_style"])
    y = np.array(data["label"])
    mask = np.isin(styles, ENCODING_STYLES)
    H = H[mask]
    y = y[mask]
    print(f"encoding rows: {mask.sum()} (refusal={y.sum()})")

    r_rfm_all = torch.load(
        os.path.join(args.steering_dir, f"steering_matrix_{m}_rfm_rc_full_hard_refusal_r.pt"),
        map_location="cpu").float()
    r_dim_all = torch.load(
        os.path.join(args.steering_dir, f"steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt"),
        map_location="cpu").float()

    slices = []
    for l in layers:
        h_l = H[:, l, :]
        x_raw = (h_l @ r_dim_all[l]).numpy()
        y_raw = (h_l @ r_rfm_all[l]).numpy()
        x = (x_raw - x_raw.mean()) / (x_raw.std() + 1e-8)
        yv = (y_raw - y_raw.mean()) / (y_raw.std() + 1e-8)
        auc_rfm = roc_auc_score(y, y_raw)
        auc_dim = roc_auc_score(y, x_raw)
        slices.append(dict(layer=l, x=x, y=yv, auc_rfm=auc_rfm, auc_dim=auc_dim))
        print(f"layer {l}: AUC_RFM={auc_rfm:.3f} AUC_DIM={auc_dim:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(10, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([-0.02, 0.03, 1.04, 0.90])
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    z_positions = list(range(len(slices)))
    pad = 3.2

    for zi, s in zip(z_positions, slices):
        # Faint "shelf" plane behind each slice for depth cues.
        plane = [[-pad, -pad, zi], [pad, -pad, zi], [pad, pad, zi], [-pad, pad, zi]]
        poly = Poly3DCollection([plane], facecolor="#f2f2f0", edgecolor="#dddddb",
                                 linewidth=0.6, alpha=0.55, zorder=1)
        ax.add_collection3d(poly)

    # y (refusal label) is identical across slices (same underlying rows).
    is_ref = y.astype(bool)

    for zi, s in zip(z_positions, slices):
        zc = np.full(mask.sum(), zi)
        ax.scatter(s["x"][~is_ref], s["y"][~is_ref], zc[~is_ref],
                   s=10, color=COL_COMPLIANCE, alpha=0.55, linewidths=0, zorder=2)
        ax.scatter(s["x"][is_ref], s["y"][is_ref], zc[is_ref],
                   s=48, color=COL_REFUSAL, alpha=0.95, edgecolors="white",
                   linewidths=0.6, zorder=5)
        ax.text(pad * 0.98, pad * 0.98, zi,
                f"L{s['layer']}\nRFM {s['auc_rfm']:.2f} | DIM {s['auc_dim']:.2f}",
                fontsize=7.5, color="#333333", ha="right", va="top", zorder=6)

    ax.set_xlabel("projection onto r_DIM  (z-scored)", fontsize=10, labelpad=10)
    ax.set_ylabel("projection onto r_RFM  (z-scored)", fontsize=10, labelpad=10)
    ax.set_zlabel("layer (depth)", fontsize=10, labelpad=6)
    ax.set_zticks(z_positions)
    ax.set_zticklabels([f"L{s['layer']}" for s in slices], fontsize=8)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    ax.view_init(elev=18, azim=-58)
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor((0.85, 0.85, 0.85, 0.6))

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_REFUSAL,
               markersize=8, label=f"hard refusal (n={is_ref.sum()})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COL_COMPLIANCE,
               markersize=8, alpha=0.5, label=f"compliance (n={(~is_ref).sum()})"),
    ]
    fig.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.03, 0.93),
               fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Encoded jailbreaks (caesar / morse / atbash / ascii): the RFM axis separates\n"
        "refusal from compliance at depth -- the DIM axis does not",
        fontsize=13, fontweight="bold", y=1.0)

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=300, facecolor="white")
    print(f"Saved -> {out_fig}")


if __name__ == "__main__":
    main()
