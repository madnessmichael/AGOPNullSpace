"""
fig1_teaser_diffmean_vs_agop_topk.py
=====================================
Figure-1 teaser: DiffMean (r_DIM) vs AGOP top-K ridge-combo (r_topk10), each
as its OWN 3D figure -- from REAL computed quantities, no synthetic/toy data.

IMPORTANT precedent (see evaluation/fig1_teaser_v2.py's own docstring): an
earlier attempt to reproduce "DIM fails, AGOP succeeds" on hand-picked
synthetic 2D Gaussian clusters did NOT reproduce the real phenomenon, and
shipping a forced toy demo was judged to misrepresent the mechanism. This
script uses only real activations and real saved steering-matrix factors
(r_DIM, r_topk10, u), llama3.1 only -- it's the only model with the
encoding-style-labeled sample (`data/embeddings/llama3.1/rc_style_full_dir.pt`)
and a DIM `_r.pt` companion file to compare against.

Design (revised after feedback that overlaying both methods' axes on one
scatter -- e.g. x=r_DIM, y=r_topk10 on the SAME plot -- made it hard to see
which method's *own* space actually separates the classes): each method now
gets its own dedicated 3D figure, sharing the same interpretation across
both so they're a fair side-by-side comparison:
  x = projection onto that method's OWN direction (r_DIM or r_topk10) --
      this is the axis that actually does the classifying; if the two
      colors don't separate along x, that method's direction doesn't work
      for this data.
  y = projection onto u, the real per-layer null-space gate (same u in both
      figures -- it's shared infrastructure, not method-specific -- so this
      axis also carries the null-space story: benign flat near u^Th~0).
  z = PC1 of the residual after projecting out x's own direction -- a
      generic "everything else" axis so the scatter isn't degenerate, not
      cherry-picked to flatter either method.

Data: real activations restricted to the 4 hardest encoding styles
(caesar/morse/atbash/ascii). The layer is picked by an actual AUC-gap search
over all 26 steering layers (largest AUC_topk10 - AUC_dim), not eyeballed.

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
INK_SECONDARY = "#52514e"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="llama3.1",
                    help="only llama3.1 has the encoding-style sample + DIM _r.pt companion file")
    p.add_argument("--layer", type=int, default=None,
                    help="default: auto-pick the layer with the largest (AUC_topk - AUC_dim) gap")
    p.add_argument("--out_dir", default="figures")
    return p.parse_args()


def find_best_layer(H, is_ref, r_dim_all, r_topk_all, layers):
    best = None
    for l in layers:
        H_l = H[:, l, :]
        auc_dim = roc_auc_score(is_ref, (H_l @ r_dim_all[l]).numpy())
        auc_topk = roc_auc_score(is_ref, (H_l @ r_topk_all[l]).numpy())
        gap = auc_topk - auc_dim
        if best is None or gap > best[3]:
            best = (l, auc_dim, auc_topk, gap)
    return best


def residual_pc1(H, r_vec, styles=None):
    """PC1 of H after projecting out the r_vec direction -- a 'everything
    else' axis that's guaranteed orthogonal to r_vec, not cherry-picked.

    If `styles` is given, each encoding style's own mean is subtracted first.
    Without this, PC1 of the raw residual is dominated by which encoding
    style a prompt uses (real, but a confound for this figure's actual
    point -- checked numerically: per-style residual means differ by ~1-3
    units before de-styling, ~1e-7 after), producing a distracting bimodal
    band in the plot that has nothing to do with DIM vs AGOP. De-styling
    isolates genuine within-style residual variance instead.
    """
    r_hat = r_vec / r_vec.norm()
    proj = (H @ r_hat).unsqueeze(1) * r_hat.unsqueeze(0)
    residual = (H - proj).numpy()
    if styles is not None:
        for st in set(styles.tolist()):
            idx = styles == st
            residual[idx] -= residual[idx].mean(axis=0, keepdims=True)
    else:
        residual = residual - residual.mean(axis=0, keepdims=True)
    # top-1 right singular vector of the residual = PC1
    _, _, Vt = np.linalg.svd(residual, full_matrices=False)
    pc1 = Vt[0]
    return residual @ pc1


def zscore(v):
    return (v - v.mean()) / (v.std() + 1e-8)


def youden_threshold(score, is_ref):
    """ROC-optimal cut point (max TPR-FPR, Youden's J) -- a principled,
    computed decision threshold, not an eyeballed one."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(is_ref, score)
    j = tpr - fpr
    return thr[np.argmax(j)]


def plot_method_figure(method_label, r_vec, u, H, is_ref, styles, layer, auc, out_path):
    x_raw = (H @ r_vec).numpy()
    x = zscore(x_raw)
    y = zscore((H @ u).numpy())
    z = zscore(residual_pc1(H, r_vec, styles=styles))

    # decision threshold on the RAW projection, then mapped into the same
    # z-scored coordinate the plot actually uses
    thr_raw = youden_threshold(x_raw, is_ref)
    thr_z = (thr_raw - x_raw.mean()) / (x_raw.std() + 1e-8)
    # AUC<0.5 means the score runs backwards (refusal scores LOWER) -- the
    # "refusal side" of the cut is then the left side, not the right
    refusal_side = "right (higher)" if auc >= 0.5 else "left (lower)"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(7.6, 8.8))
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[3.4, 0.55, 0.85], hspace=0.05)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax_hist = fig.add_subplot(gs[2, 0])

    ax.scatter(x[~is_ref], y[~is_ref], z[~is_ref], s=11, color=COL_COMPLIANCE, alpha=0.45,
               linewidths=0, label=f"compliance (n={(~is_ref).sum()})")
    ax.scatter(x[is_ref], y[is_ref], z[is_ref], s=55, color=COL_MALICIOUS, alpha=0.95,
               edgecolors="white", linewidths=0.7, label=f"hard refusal (n={is_ref.sum()})")

    # decision plane: perpendicular to x, at the ROC-optimal threshold
    yy, zz = np.meshgrid(
        np.linspace(y.min(), y.max(), 2), np.linspace(z.min(), z.max(), 2))
    ax.plot_surface(np.full_like(yy, thr_z), yy, zz, color="#111111", alpha=0.12, linewidth=0)

    ax.set_xlabel(f"projection onto {method_label}'s own direction", fontsize=9, color=INK_SECONDARY, labelpad=8)
    ax.set_ylabel("projection onto u (null-space gate)", fontsize=9, color=INK_SECONDARY, labelpad=8)
    ax.set_zlabel("residual PC1 (within-style variance)", fontsize=9, color=INK_SECONDARY, labelpad=8)
    ax.tick_params(labelsize=7.5, colors=INK_SECONDARY)
    # elev/azim chosen empirically over a 6-angle sweep (see CLAUDE.md) for the
    # clearest visual separation between the two classes, not the default view
    ax.view_init(elev=25, azim=-45)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # short, single-line title -- full context (dataset, layer, verdict) goes in
    # the README caption instead, so it doesn't eat vertical space here and
    # shrink the actual 3D plot
    ax.set_title(f"{method_label}:  AUC = {auc:.2f}",
                 fontsize=13, fontweight="bold", color="#0b0b0b", pad=6)

    # ---- marginal 1D histogram along x, with the SAME decision plane as a vertical line ----
    bins = np.linspace(min(x.min(), thr_z) - 0.2, max(x.max(), thr_z) + 0.2, 40)
    ax_hist.hist(x[~is_ref], bins=bins, color=COL_COMPLIANCE, alpha=0.6, density=True,
                 label="compliance")
    ax_hist.hist(x[is_ref], bins=bins, color=COL_MALICIOUS, alpha=0.75, density=True,
                 label="hard refusal")
    ax_hist.axvline(thr_z, color="#111111", linewidth=1.6, linestyle="--",
                     label=f"decision cut (Youden's J)\nrefusal side: {refusal_side}")
    ax_hist.set_xlabel(f"same x-axis, 1D: projection onto {method_label}'s own direction",
                        fontsize=8.5, color=INK_SECONDARY)
    ax_hist.set_yticks([])
    for spine in ["top", "right", "left"]:
        ax_hist.spines[spine].set_visible(False)
    ax_hist.tick_params(labelsize=7.5, colors=INK_SECONDARY)
    # legend outside/above the axes so it never overlaps a tall bar regardless
    # of where the decision cut happens to fall for this method
    ax_hist.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                    fontsize=7.5, frameon=False, handlelength=1.4, columnspacing=1.0)

    fig.savefig(out_path, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)
    print(f"Saved -> {out_path}  (AUC={auc:.3f}, Youden thr(raw)={thr_raw:.4f})")


def main():
    args = parse_args()
    m = args.model_name

    steer_dir = "data/steering_matrix"
    main_dict = torch.load(f"{steer_dir}/steering_matrix_{m}_rfm_rc_full_hard_refusal_topk10.pt", map_location="cpu")
    r_topk_all = torch.load(f"{steer_dir}/steering_matrix_{m}_rfm_rc_full_hard_refusal_topk10_r.pt", map_location="cpu").float()
    r_dim_all = torch.load(f"{steer_dir}/steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt", map_location="cpu").float()
    layers = main_dict["layers"]

    enc_data = torch.load(f"data/embeddings/{m}/rc_style_full_dir.pt", map_location="cpu")
    H_all = enc_data["H"].float()
    styles_all = np.array(enc_data["prompt_style"])
    labels_all = np.array(enc_data["label"])

    # The 4 encoding styles are llama3.1's hardest real case for DIM (n=28
    # hard-refusal rows) -- but checked numerically, qwen2.5 has only 5 such
    # rows and gemma2 only 1 across ALL FOUR encoding styles combined in the
    # full 9,236-row dataset (not an extraction artifact -- verified against
    # sorrybench_rc_dataset.json directly: these two models essentially never
    # produce a strict "hard refusal" to an encoding-obfuscated prompt at
    # all, a real finding in itself). AUC on n=1 or n=5 positives is not a
    # meaningful comparison, so for those models this falls back to the full
    # 21-style dataset instead, which still isolates a genuine per-model
    # DIM-vs-AGOP gap on real data -- just not framed as "encoding" specifically.
    mask = np.isin(styles_all, ENCODING_STYLES)
    n_pos_encoding = int(labels_all[mask].astype(bool).sum())
    MIN_POSITIVES = 20
    if n_pos_encoding >= MIN_POSITIVES:
        subset_desc = "encoded jailbreaks (caesar/morse/atbash/ascii)"
        H_enc_full, is_ref, styles_masked = H_all[mask], labels_all[mask].astype(bool), styles_all[mask]
    else:
        print(f"NOTE: only {n_pos_encoding} hard-refusal rows in the 4 encoding styles for {m} "
              f"(need >={MIN_POSITIVES}) -- falling back to all 21 SORRY-Bench prompt styles")
        subset_desc = "all 21 SORRY-Bench prompt styles (encoding styles alone too sparse for this model)"
        H_enc_full, is_ref, styles_masked = H_all, labels_all.astype(bool), styles_all

    if args.layer is not None:
        l = args.layer
        H_l = H_enc_full[:, l, :]
        auc_dim = roc_auc_score(is_ref, (H_l @ r_dim_all[l]).numpy())
        auc_topk = roc_auc_score(is_ref, (H_l @ r_topk_all[l]).numpy())
    else:
        l, auc_dim, auc_topk, gap = find_best_layer(H_enc_full, is_ref, r_dim_all, r_topk_all, layers)
    print(f"subset: {subset_desc}  (n_pos={int(is_ref.sum())}, n_neg={int((~is_ref).sum())})")
    print(f"layer {l}: AUC_dim={auc_dim:.3f}  AUC_topk10={auc_topk:.3f}  gap={auc_topk-auc_dim:+.3f}")

    H = H_enc_full[:, l, :]
    r_dim, r_topk = r_dim_all[l], r_topk_all[l]
    u = main_dict["factors"][l]["u"].float()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_method_figure("DiffMean (r_DIM)", r_dim, u, H, is_ref, styles_masked, l, auc_dim,
                        f"{args.out_dir}/fig1_teaser_dim_{m}.png")
    plot_method_figure("AGOP top-K ridge-combo (r_topk10)", r_topk, u, H, is_ref, styles_masked, l, auc_topk,
                        f"{args.out_dir}/fig1_teaser_agop_topk_{m}.png")


if __name__ == "__main__":
    main()
