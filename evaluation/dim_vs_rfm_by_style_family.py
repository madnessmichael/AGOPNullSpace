"""
dim_vs_rfm_by_style_family.py
================================
The key test for "does RFM's advantage show up specifically on the harder,
heterogeneous jailbreak styles, even though DIM wins on AGGREGATE AUC
(dim_vs_rfm_layer_summary.py)?"

SORRY-Bench's 21 prompt_style mutations split naturally into families with
very different refusal-rate base rates (see extract_rc_style_sample.py
docstring): "natural"-language styles are 64-78% refusal, "persuasion"
techniques are 6-17%, character-level "encoding" styles are 0-5%. A global
mean-difference (DIM) is a sample-size-weighted average over ALL of this --
dominated by whichever family has the most rows and the cleanest
separation. If RFM's benefit is about handling heterogeneous/nonlinear
structure, it should show up as relatively BETTER (or less-worse) AUC than
DIM specifically within the minority/harder families, even if DIM wins in
aggregate.

Requires the stratified sample from extract_rc_style_sample.py (which,
unlike the production embeds_refusal.pt/embeds_compliance.pt, keeps
prompt_style/label aligned per row).

Usage:
    python evaluation/dim_vs_rfm_by_style_family.py --model_name llama3.1 --layer 27
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

FAMILIES = {
    "natural": ["base", "role_play", "slang", "misspellings", "uncommon_dialects",
                "technical_terms", "question"],
    "persuasion": ["logical_appeal", "misrepresentation", "evidence-based_persuasion",
                   "authority_endorsement", "expert_endorsement"],
    "encoding": ["caesar", "morse", "atbash", "ascii"],
    "translation": ["translate-fr", "translate-mr", "translate-ta", "translate-ml",
                    "translate-zh-cn"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--sample_path", default=None,
                   help="Defaults to data/embeddings/<model>/rc_style_sample.pt")
    p.add_argument("--steering_dir", default="data/steering_matrix")
    p.add_argument("--layer", type=int, default=None,
                   help="If omitted, sweeps all layers and plots a per-family line chart.")
    p.add_argument("--out_fig", default=None)
    return p.parse_args()


def family_auc(H_all, y, r, family_styles, styles):
    mask = np.isin(styles, family_styles)
    if mask.sum() == 0 or len(set(y[mask])) < 2:
        return None
    score = (H_all[mask] @ r).numpy()
    return roc_auc_score(y[mask], score)


def main():
    args = parse_args()
    m = args.model_name
    sample_path = args.sample_path or f"data/embeddings/{m}/rc_style_sample.pt"

    data = torch.load(sample_path, map_location="cpu")
    H = data["H"].float()  # [N, L, d]
    styles = np.array(data["prompt_style"])
    y = np.array(data["label"])
    print(f"Loaded sample: {H.shape}, {len(set(styles))} styles, "
          f"refusal={int(y.sum())}/{len(y)}")

    rfm_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_rfm_rc_full_hard_refusal_r.pt")
    dim_path = os.path.join(args.steering_dir, f"steering_matrix_{m}_dim_rc_full_hard_refusal_r.pt")
    r_rfm_all = torch.load(rfm_path, map_location="cpu").float()
    r_dim_all = torch.load(dim_path, map_location="cpu").float()

    layers = [args.layer] if args.layer is not None else \
        [l for l, _ in AlphaSteer_CALCULATION_CONFIG[m]]

    results = {fam: {"rfm": [], "dim": []} for fam in FAMILIES}
    results["overall"] = {"rfm": [], "dim": []}

    for l in layers:
        H_l = H[:, l, :]
        r_rfm, r_dim = r_rfm_all[l], r_dim_all[l]

        overall_rfm = family_auc(H_l, y, r_rfm, list(set(styles)), styles)
        overall_dim = family_auc(H_l, y, r_dim, list(set(styles)), styles)
        results["overall"]["rfm"].append(overall_rfm)
        results["overall"]["dim"].append(overall_dim)

        line = [f"layer {l:>2}: overall RFM={overall_rfm:.3f} DIM={overall_dim:.3f}"]
        for fam, fam_styles in FAMILIES.items():
            auc_rfm = family_auc(H_l, y, r_rfm, fam_styles, styles)
            auc_dim = family_auc(H_l, y, r_dim, fam_styles, styles)
            results[fam]["rfm"].append(auc_rfm)
            results[fam]["dim"].append(auc_dim)
            if auc_rfm is not None and auc_dim is not None:
                diff = auc_rfm - auc_dim
                flag = " <-- RFM better" if diff > 0 else ""
                line.append(f"{fam}: RFM={auc_rfm:.3f} DIM={auc_dim:.3f} (Δ={diff:+.3f}){flag}")
        print("  ".join(line))

    if args.layer is not None:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams_to_plot = ["overall"] + list(FAMILIES.keys())
    fig, axes = plt.subplots(len(fams_to_plot), 1, figsize=(9, 3 * len(fams_to_plot)), sharex=True)
    for ax, fam in zip(axes, fams_to_plot):
        rfm_vals = [v if v is not None else np.nan for v in results[fam]["rfm"]]
        dim_vals = [v if v is not None else np.nan for v in results[fam]["dim"]]
        ax.plot(layers, rfm_vals, marker="o", color="black", label="r_RFM")
        ax.plot(layers, dim_vals, marker="s", color="darkorange", linestyle="--", label="r_DIM")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
        ax.set_ylabel("AUC")
        ax.set_title(f"{fam} ({'+'.join(FAMILIES[fam]) if fam != 'overall' else 'all styles'})",
                     fontsize=9)
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[-1].set_xlabel("Layer")
    fig.suptitle(f"{m}: RFM vs DIM AUC by prompt_style family (refusal vs compliance)", y=1.0)
    fig.tight_layout()

    out_fig = args.out_fig or f"figures/dim_vs_rfm_by_family_{m}.png"
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure -> {out_fig}")


if __name__ == "__main__":
    main()
