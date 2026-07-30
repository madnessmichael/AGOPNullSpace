"""
fig1_teaser_diffmean_vs_agop_topk.py
=====================================
Figure-1 teaser: DiffMean (r_DIM) vs AGOP top-K ridge-combo (r_topk10),
plus the null-space gate `u`, all from REAL computed quantities -- no
synthetic/toy data.

IMPORTANT precedent (see evaluation/fig1_teaser_v2.py's own docstring): an
earlier attempt to reproduce "DIM fails, AGOP succeeds" on hand-picked
synthetic 2D Gaussian clusters did NOT reproduce the real phenomenon, and
shipping a forced toy demo was judged to misrepresent the mechanism. This
script therefore uses only real activations and real saved steering-matrix
factors (r_DIM, r_topk10, u), llama3.1 only -- it's the only model with the
encoding-style-labeled sample (`data/embeddings/llama3.1/rc_style_full_dir.pt`)
and a DIM `_r.pt` companion file to compare against.

Two panels:
  LEFT  (3D) -- the null-space / gate story. Real benign_val (n=1000) +
          harmful_val (n=1000) activations at the chosen layer, projected
          onto (r_DIM, r_topk10, u). `u` is the real per-layer null-space gate
          vector saved in the rank1_gate_v1 dict -- benign activations form a
          near-flat disk at u^Th ~ 0 (real numbers below) while malicious
          activations spread upward, which is exactly the mechanism that lets
          the gate leave benign prompts ~untouched.
  RIGHT (2D + marginal histograms) -- the "DIM fails on obfuscated prompts"
          story. Real activations restricted to the 4 hardest encoding
          styles (caesar/morse/atbash/ascii), projected onto (r_DIM, r_topk10).
          This is where DIM's AUC actually drops below 0.5 (worse than
          random) while the top-K ridge-combo direction still separates well
          -- the layer is picked by an actual AUC-gap search over all 26
          steering layers, not cherry-picked by eye.

Usage:
    python experimental/fig1_teaser_diffmean_vs_agop_topk.py
    python experimental/fig1_teaser_diffmean_vs_agop_topk.py --layer 12
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

# palette (dataviz skill's validated categorical order; blue/orange = the
# same compliance/refusal roles fig1_teaser_v2.py already established)
COL_COMPLIANCE = "#2a78d6"
COL_MALICIOUS = "#eb6834"
COL_DIM = "#4a3aa7"
COL_TOPK = "#111111"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="llama3.1",
                    help="only llama3.1 has the encoding-style sample + DIM _r.pt companion file")
    p.add_argument("--layer", type=int, default=None,
                    help="default: auto-pick the layer with the largest (AUC_topk - AUC_dim) gap on the encoding subset")
    p.add_argument("--out_fig", default=None)
    return p.parse_args()


def find_best_layer(H, is_ref_enc, r_dim_all, r_topk_all, layers):
    best = None
    for l in layers:
        H_l = H[:, l, :]
        auc_dim = roc_auc_score(is_ref_enc, (H_l @ r_dim_all[l]).numpy())
        auc_topk = roc_auc_score(is_ref_enc, (H_l @ r_topk_all[l]).numpy())
        gap = auc_topk - auc_dim
        if best is None or gap > best[3]:
            best = (l, auc_dim, auc_topk, gap)
    return best


def main():
    args = parse_args()
    m = args.model_name
    out_fig = args.out_fig or f"figures/fig1_teaser_diffmean_vs_agop_{m}.png"

    steer_dir = "data/steering_matrix"
    main_dict = torch.load(f"{steer_dir}/steering_matrix_{m}_rfm_rc_full_hard_refusal_topk10.pt", map_location="cpu")
    r_topk_all = torch.load(f"{steer_dir}/steering_matrix_{m}_rfm_rc_full_hard_refusal_topk10_r.pt", map_location="cpu").float()
    r_dim_all = torch.load(f"{steer_dir}/steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt", map_location="cpu").float()
    layers = main_dict["layers"]
    slope = main_dict["gate_slope"]

    # --- encoding-family (hard) subset, for the right panel + layer search ---
    enc_data = torch.load(f"data/embeddings/{m}/rc_style_full_dir.pt", map_location="cpu")
    H_enc = enc_data["H"].float()
    styles = np.array(enc_data["prompt_style"])
    labels_enc = np.array(enc_data["label"])
    mask = np.isin(styles, ENCODING_STYLES)
    H_enc = H_enc[mask]
    is_ref_enc = labels_enc[mask].astype(bool)

    if args.layer is not None:
        l = args.layer
        H_l = H_enc[:, l, :]
        auc_dim = roc_auc_score(is_ref_enc, (H_l @ r_dim_all[l]).numpy())
        auc_topk = roc_auc_score(is_ref_enc, (H_l @ r_topk_all[l]).numpy())
    else:
        l, auc_dim, auc_topk, gap = find_best_layer(H_enc, is_ref_enc, r_dim_all, r_topk_all, layers)
    print(f"layer {l}: AUC_dim={auc_dim:.3f}  AUC_topk10={auc_topk:.3f}  gap={auc_topk-auc_dim:+.3f}")

    r_dim, r_topk = r_dim_all[l], r_topk_all[l]
    u = main_dict["factors"][l]["u"].float()

    x_enc = (H_enc[:, l, :] @ r_dim).numpy()
    y_enc = (H_enc[:, l, :] @ r_topk).numpy()
    x_enc_z = (x_enc - x_enc.mean()) / (x_enc.std() + 1e-8)
    y_enc_z = (y_enc - y_enc.mean()) / (y_enc.std() + 1e-8)

    # --- benign_val / harmful_val (the actual gate/null-space fitting distribution) ---
    b = torch.load(f"data/embeddings/{m}/embeds_benign_val.pt", map_location="cpu").float()[:, l, :]
    h = torch.load(f"data/embeddings/{m}/embeds_harmful_val.pt", map_location="cpu").float()[:, l, :]

    def proj(X, v):
        return (X @ v).numpy()

    bx, by, bz = proj(b, r_dim), proj(b, r_topk), proj(b, u)
    hx, hy, hz = proj(h, r_dim), proj(h, r_topk), proj(h, u)
    print(f"benign_val  u^Th: mean={bz.mean():+.4f} std={bz.std():.4f}")
    print(f"harmful_val u^Th: mean={hz.mean():+.4f} std={hz.std():.4f}")

    # ─────────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(14.5, 6.8))
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.1], wspace=0.28,
                  left=0.03, right=0.98, top=0.85, bottom=0.10)

    # ---- LEFT: 3D null-space panel ----
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.scatter(bx, by, bz, s=9, color=COL_COMPLIANCE, alpha=0.35, linewidths=0,
                 label=f"benign (n={len(b)}): uᵀh={bz.mean():+.3f}±{bz.std():.3f}")
    ax3d.scatter(hx, hy, hz, s=9, color=COL_MALICIOUS, alpha=0.35, linewidths=0,
                 label=f"malicious (n={len(h)}): uᵀh={hz.mean():+.3f}±{hz.std():.3f}")

    # translucent disk at u^Th ~ 0 marking the "safe" (near-zero-gate) plane
    xx, yy = np.meshgrid(
        np.linspace(min(bx.min(), hx.min()), max(bx.max(), hx.max()), 2),
        np.linspace(min(by.min(), hy.min()), max(by.max(), hy.max()), 2),
    )
    ax3d.plot_surface(xx, yy, np.zeros_like(xx), color=COL_COMPLIANCE, alpha=0.08, linewidth=0)

    ax3d.set_xlabel("proj. onto r_DIM", fontsize=8.5, color=INK_SECONDARY, labelpad=2)
    ax3d.set_ylabel("proj. onto r_topk10", fontsize=8.5, color=INK_SECONDARY, labelpad=2)
    ax3d.set_zlabel("proj. onto u  (null-space gate)", fontsize=8.5, color=INK_SECONDARY, labelpad=2)
    ax3d.tick_params(labelsize=7, colors=INK_SECONDARY)
    ax3d.view_init(elev=18, azim=-58)
    ax3d.xaxis.pane.set_alpha(0.03)
    ax3d.yaxis.pane.set_alpha(0.03)
    ax3d.zaxis.pane.set_alpha(0.03)
    ax3d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=8, framealpha=0.9)

    # ---- RIGHT: 2D hard-case scatter + marginal histograms ----
    gs_right = GridSpec(4, 4, figure=fig,
                         left=gs.get_grid_positions(fig)[3][0] + 0.05,
                         right=0.985, top=0.80, bottom=0.11, wspace=0.05, hspace=0.05)
    ax_scatter = fig.add_subplot(gs_right[1:4, 0:3])
    ax_top = fig.add_subplot(gs_right[0, 0:3], sharex=ax_scatter)
    ax_rightm = fig.add_subplot(gs_right[1:4, 3], sharey=ax_scatter)

    is_ref = is_ref_enc
    ax_scatter.scatter(x_enc_z[~is_ref], y_enc_z[~is_ref], s=10, color=COL_COMPLIANCE, alpha=0.45,
                        linewidths=0, label=f"compliance (n={(~is_ref).sum()})")
    ax_scatter.scatter(x_enc_z[is_ref], y_enc_z[is_ref], s=42, color=COL_MALICIOUS, alpha=0.95,
                        edgecolors="white", linewidths=0.6, label=f"hard refusal (n={is_ref.sum()})")
    ax_scatter.set_xlabel("projection onto r_DIM  (z-scored)", fontsize=10)
    ax_scatter.set_ylabel("projection onto r_topk10  (z-scored)", fontsize=10)
    ax_scatter.grid(alpha=0.25, color=GRIDLINE)
    ax_scatter.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

    bins = np.linspace(-3.5, 3.5, 36)
    ax_top.hist(x_enc_z[~is_ref], bins=bins, color=COL_COMPLIANCE, alpha=0.55, density=True)
    ax_top.hist(x_enc_z[is_ref], bins=bins, color=COL_MALICIOUS, alpha=0.75, density=True)
    ax_top.axis("off")
    ax_top.text(0.02, 0.85, f"AUC(r_DIM) = {auc_dim:.2f}", transform=ax_top.transAxes,
                fontsize=11, fontweight="bold", color=COL_DIM, va="top")

    ax_rightm.hist(y_enc_z[~is_ref], bins=bins, color=COL_COMPLIANCE, alpha=0.55,
                    density=True, orientation="horizontal")
    ax_rightm.hist(y_enc_z[is_ref], bins=bins, color=COL_MALICIOUS, alpha=0.75,
                    density=True, orientation="horizontal")
    ax_rightm.axis("off")
    ax_rightm.text(0.5, 0.985, f"AUC(r_topk10)\n= {auc_topk:.2f}", transform=ax_rightm.transAxes,
                    fontsize=11, fontweight="bold", color=COL_TOPK, ha="center", va="top")

    fig.suptitle("DiffMean vs. AGOP top-K ridge-combo, and the null-space gate — real activations, no synthetic data",
                 fontsize=14.5, fontweight="bold", y=0.975)
    fig.text(0.03, 0.895, "Null-space gate u (real steering-matrix factor)", fontsize=11, fontweight="bold")
    fig.text(gs_right.get_grid_positions(fig)[2][0], 0.895,
             f"Hardest case: {m}, layer {l}, encoded jailbreaks (caesar/morse/atbash/ascii)",
             fontsize=11, fontweight="bold")

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=220, facecolor="white", bbox_inches="tight")
    print(f"Saved -> {out_fig}")


if __name__ == "__main__":
    main()
