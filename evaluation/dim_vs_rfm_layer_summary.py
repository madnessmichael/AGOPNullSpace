"""
dim_vs_rfm_layer_summary.py
=============================
Comprehensive ALL-LAYERS comparison of r_RFM vs r_DIM (both trained on the
same refusal_compliance_full_hard_refusal / rc_hr data). Two complementary
per-layer signals, in one figure:

  (A) cos(r_RFM[l], r_DIM[l])           -- how different are the two directions
  (B) AUC(y, H[:,l,:] @ r) for each of  -- how well does EACH direction separate
      r_RFM and r_DIM                      refusal (y=1) from compliance (y=0)
                                            at that layer, on the actual training data

(A) alone (see compare_dim_vs_rfm.py) only says "they disagree"; (B) says
"and here's which one actually separates the classes better, at every layer" --
directly addresses "is AGOP's benefit more than just picking a different but
equally-good direction". Both are cheap: no GPU/model needed, only the
existing embeddings + saved _r.pt concept vectors.

Usage:
    python evaluation/dim_vs_rfm_layer_summary.py --model_name llama3.1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils.const import AlphaSteer_CALCULATION_CONFIG  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--rc_subdir", default="refusal_compliance_full_hard_refusal")
    p.add_argument("--embedding_dir", default=None)
    p.add_argument("--steering_dir", default="data/steering_matrix")
    p.add_argument("--out_fig", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    m = args.model_name
    device = torch.device(args.device)
    embedding_dir = args.embedding_dir or os.path.join("data/embeddings", m)
    rc_dir = os.path.join(embedding_dir, args.rc_subdir)
    out_fig = args.out_fig or f"figures/dim_vs_rfm_layer_summary_{m}.png"

    rfm_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_rfm_rc_full_hard_refusal_r.pt")
    dim_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt")
    r_rfm_all = torch.load(rfm_path, map_location="cpu").float()
    r_dim_all = torch.load(dim_path, map_location="cpu").float()

    print("Loading refuse-compliance activations (may take a moment)...")
    H_refusal = torch.load(os.path.join(rc_dir, "embeds_refusal.pt"), map_location="cpu").float()
    H_compliance = torch.load(os.path.join(rc_dir, "embeds_compliance.pt"), map_location="cpu").float()
    y = np.concatenate([np.ones(H_refusal.shape[0]), np.zeros(H_compliance.shape[0])])
    print(f"refusal={H_refusal.shape[0]} compliance={H_compliance.shape[0]}")

    layers = [l for l, _ in AlphaSteer_CALCULATION_CONFIG[m]]
    cos_vals, auc_rfm, auc_dim = [], [], []

    for l in layers:
        h_r = H_refusal[:, l, :].to(device)
        h_c = H_compliance[:, l, :].to(device)
        h_all = torch.cat([h_r, h_c], dim=0)

        r_rfm = r_rfm_all[l].to(device)
        r_dim = r_dim_all[l].to(device)

        cos = float((r_rfm @ r_dim) / (r_rfm.norm() * r_dim.norm()))
        cos_vals.append(cos)

        score_rfm = (h_all @ r_rfm).cpu().numpy()
        score_dim = (h_all @ r_dim).cpu().numpy()
        auc_rfm.append(roc_auc_score(y, score_rfm))
        auc_dim.append(roc_auc_score(y, score_dim))

        print(f"layer {l:>2}: cos={cos:+.3f}  AUC_RFM={auc_rfm[-1]:.4f}  AUC_DIM={auc_dim[-1]:.4f}")

        del h_r, h_c, h_all, r_rfm, r_dim
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cos_vals, auc_rfm, auc_dim = map(np.array, (cos_vals, auc_rfm, auc_dim))

    print()
    print(f"cos:      mean={cos_vals.mean():.4f}  min={cos_vals.min():.4f}  max={cos_vals.max():.4f}")
    print(f"AUC_RFM:  mean={auc_rfm.mean():.4f}  min={auc_rfm.min():.4f}  max={auc_rfm.max():.4f}")
    print(f"AUC_DIM:  mean={auc_dim.mean():.4f}  min={auc_dim.min():.4f}  max={auc_dim.max():.4f}")
    print(f"layers where RFM beats DIM: {int((auc_rfm > auc_dim).sum())}/{len(layers)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(layers, cos_vals, marker="o", color="#3b6fa0", linewidth=2)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("cos(r_RFM, r_DIM)")
    ax1.set_ylim(-0.1, 1.0)
    ax1.set_title(f"{m}: RFM vs DIM concept-vector comparison across ALL layers\n"
                  f"(both trained on {args.rc_subdir})")
    ax1.grid(alpha=0.3)

    ax2.plot(layers, auc_rfm, marker="o", color="black", linewidth=2, label="r_RFM")
    ax2.plot(layers, auc_dim, marker="s", color="darkorange", linewidth=2,
             linestyle="--", label="r_DIM")
    ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("AUC(refusal vs compliance)\nprojection onto r")
    ax2.set_ylim(0.45, 1.02)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"\nSaved figure -> {out_fig}")


if __name__ == "__main__":
    main()
