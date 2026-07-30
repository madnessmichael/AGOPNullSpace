"""
attack_subspace_viz.py
=======================
Per-attack subspace visualization for the DIM-vs-RFM ablation (reviewer
question #5: how does attack geometry differ from what the concept vectors
capture?).

For a chosen layer, PCA-projects last-token activations from the 7 jailbreak
attack datasets (aim, autodan, cipher, gcg, jailbroken, pair, renellm) into
2D, colors points by attack family, and overlays r_RFM / r_DIM (from the
SAME refusal_compliance_full_hard_refusal training data) as arrows in the
same PCA basis -- showing which attack clusters each direction actually
points toward.

Requires activations extracted via:
    python src/extract_embeddings.py --model_name meta-llama/Llama-3.1-8B-Instruct \
        --input_file data/instructions/test/llama3.1/<name>_llama3.1.json \
        --prompt_column query --batch_size 16 --device cuda:X \
        --output_file data/embeddings/llama3.1/attack_prompts/<name>.pt
for each of the 7 datasets.

Usage:
    python evaluation/attack_subspace_viz.py --model_name llama3.1 --layer 15
    python evaluation/attack_subspace_viz.py --model_name llama3.1 --layer 27
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ATTACK_NAMES = ["aim", "autodan", "cipher", "gcg", "jailbroken", "pair", "renellm"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--embed_dir", default=None,
                   help="Defaults to data/embeddings/<model>/attack_prompts")
    p.add_argument("--steering_dir", default="data/steering_matrix")
    p.add_argument("--out_fig", default=None,
                   help="Defaults to figures/attack_subspace_<model>_layer<layer>.png")
    p.add_argument("--max_per_attack", type=int, default=None,
                   help="Optional subsample per attack for a less crowded plot.")
    return p.parse_args()


def main():
    args = parse_args()
    m = args.model_name
    embed_dir = args.embed_dir or os.path.join("data/embeddings", m, "attack_prompts")
    out_fig = args.out_fig or f"figures/attack_subspace_{m}_layer{args.layer}.png"

    rfm_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_rfm_rc_full_hard_refusal_r.pt")
    dim_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt")
    r_rfm = torch.load(rfm_path, map_location="cpu").float()[args.layer]
    r_dim = torch.load(dim_path, map_location="cpu").float()[args.layer]

    Hs, labels = [], []
    for name in ATTACK_NAMES:
        path = os.path.join(embed_dir, f"{name}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path} -- run extract_embeddings.py for '{name}' first.")
        H = torch.load(path, map_location="cpu").float()[:, args.layer, :]
        if args.max_per_attack is not None and H.shape[0] > args.max_per_attack:
            idx = torch.randperm(H.shape[0])[:args.max_per_attack]
            H = H[idx]
        Hs.append(H)
        labels.extend([name] * H.shape[0])
    H_all = torch.cat(Hs, dim=0)
    labels = np.array(labels)
    print(f"Loaded {H_all.shape[0]} activations across {len(ATTACK_NAMES)} attacks "
          f"at layer {args.layer}, d={H_all.shape[1]}")

    # PCA (mean-centered, via SVD) fit on the attack-activation cloud itself.
    mean = H_all.mean(0, keepdim=True)
    Hc = H_all - mean
    U, S, Vt = torch.linalg.svd(Hc, full_matrices=False)
    basis = Vt[:2]  # [2, d]
    proj = (Hc @ basis.T).numpy()  # [N, 2]
    explained = (S[:2] ** 2 / (S ** 2).sum()).numpy()

    # Project concept vectors (unit direction, scaled to a visible arrow length)
    scale = float(np.linalg.norm(proj, axis=1).mean()) * 1.5
    r_rfm_2d = (r_rfm @ basis.T).numpy()
    r_rfm_2d = r_rfm_2d / np.linalg.norm(r_rfm_2d) * scale
    r_dim_2d = (r_dim @ basis.T).numpy()
    r_dim_2d = r_dim_2d / np.linalg.norm(r_dim_2d) * scale

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    palette = plt.get_cmap("tab10")
    for i, name in enumerate(ATTACK_NAMES):
        mask = labels == name
        ax.scatter(proj[mask, 0], proj[mask, 1], s=14, alpha=0.6,
                   color=palette(i), label=name)

    ax.annotate("", xy=r_rfm_2d, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.5))
    ax.text(*(r_rfm_2d * 1.08), "r_RFM", fontsize=11, fontweight="bold", color="black")
    ax.annotate("", xy=r_dim_2d, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="crimson", lw=2.5, linestyle="--"))
    ax.text(*(r_dim_2d * 1.08), "r_DIM", fontsize=11, fontweight="bold", color="crimson")
    ax.scatter([0], [0], color="black", s=30, zorder=5)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
    ax.set_title(f"{m}, layer {args.layer}: attack-family activation clusters\n"
                 f"vs r_RFM / r_DIM directions (both trained on rc_hr data)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150)
    print(f"Saved figure -> {out_fig}")


if __name__ == "__main__":
    main()
