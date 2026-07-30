# ## 11. Unified per-style-labeled RC loader (all 3 models)
#
# Confirmed via check_rc_json_alignment.py: filtering sorrybench_rc_dataset.json
# by label (preserving JSON row order) exactly matches the row order of
# embeds_refusal.pt (label==1) and embeds_compliance.pt (label==0), for all
# three models. So we can recover a per-row `prompt_style` array aligned to
# each activation tensor directly from the JSON, with no separate per-style
# files needed.

STYLE_COL = "prompt_style"
LABEL_COL = "label"
ENCODING_STYLES = {"caesar", "atbash", "morse", "ascii"}


def load_style_labeled_rc(model: str, subdir: str = RC_SUBDIR):
    rc_dir = os.path.join(EMBEDDING_DIR_TMPL.format(model=model), subdir)
    with open(os.path.join(rc_dir, "sorrybench_rc_dataset.json")) as f:
        meta = json.load(f)
    meta_df = pd.DataFrame(meta)

    H_refusal = torch.load(os.path.join(rc_dir, "embeds_refusal.pt"), map_location="cpu").float()
    H_compliance = torch.load(os.path.join(rc_dir, "embeds_compliance.pt"), map_location="cpu").float()

    style_refusal = meta_df.loc[meta_df[LABEL_COL] == 1, STYLE_COL].to_numpy()
    style_compliance = meta_df.loc[meta_df[LABEL_COL] == 0, STYLE_COL].to_numpy()

    assert len(style_refusal) == H_refusal.shape[0], f"{model}: refusal alignment mismatch"
    assert len(style_compliance) == H_compliance.shape[0], f"{model}: compliance alignment mismatch"

    return H_refusal, H_compliance, style_refusal, style_compliance
# ## 12. Refit DIM/RFM EXCLUDING encoding styles; evaluate on the FULL (100%) encoding subset
#
# Why refit instead of reusing section 5's `results[model]["dim_vectors"/"rfm_vectors"]`:
# those were fit on a random 80/20 split of the FULL RC set, which already
# contains the encoding rows -- so ~80% of the (already scarce) encoding
# positives were literally inside the fit data. That's leakage, not a fair
# diagnostic. Here we exclude all four encoding styles from the fit
# distribution entirely, so the resulting DIM/RFM directions have genuinely
# never seen an encoding-style example, and we can evaluate on ALL 1760
# encoding rows (not just a 20% slice) -- important given how few positives
# exist to begin with.

def build_fit_excluding_styles(H_refusal, H_compliance, style_refusal, style_compliance,
                                 exclude_styles=ENCODING_STYLES):
    keep_r = ~pd.Series(style_refusal).astype(str).str.lower().isin(exclude_styles).to_numpy()
    keep_c = ~pd.Series(style_compliance).astype(str).str.lower().isin(exclude_styles).to_numpy()
    return H_refusal[keep_r], H_compliance[keep_c]


def build_encoding_eval_set(H_refusal, H_compliance, style_refusal, style_compliance,
                             styles=ENCODING_STYLES):
    mask_r = pd.Series(style_refusal).astype(str).str.lower().isin(styles).to_numpy()
    mask_c = pd.Series(style_compliance).astype(str).str.lower().isin(styles).to_numpy()
    H_enc = torch.cat([H_refusal[mask_r], H_compliance[mask_c]], dim=0)
    y_enc = np.concatenate([np.ones(int(mask_r.sum()), dtype=int),
                             np.zeros(int(mask_c.sum()), dtype=int)])
    style_enc = np.concatenate([np.asarray(style_refusal)[mask_r],
                                  np.asarray(style_compliance)[mask_c]])
    return H_enc, y_enc, style_enc


def natural_sign(H_refusal_ref, H_compliance_ref, layer, r):
    """Fix direction sign using the (encoding-excluded) fit reference
    distribution -- NOT the encoding eval data -- so checking for a sign flip
    later isn't circular."""
    z_r = project_onto_direction(H_refusal_ref[:, layer, :], r)
    z_c = project_onto_direction(H_compliance_ref[:, layer, :], r)
    return 1 if z_r.mean().item() >= z_c.mean().item() else -1


def auc_inv(H_subset, y_subset, layer, r):
    if len(np.unique(y_subset)) < 2:
        return np.nan
    z = project_onto_direction(H_subset[:, layer, :], r).numpy()
    auc = roc_auc_score(y_subset, z)
    return max(auc, 1 - auc)


def auc_signed(H_subset, y_subset, layer, r, sign):
    if len(np.unique(y_subset)) < 2:
        return np.nan
    z = project_onto_direction(H_subset[:, layer, :], r).numpy()
    return roc_auc_score(y_subset, sign * z)


MIN_POS_FOR_TRUST = 10  # below this, flag the model's numbers as unreliable

enc_fit_results = {}
for model in MODELS:
    print(f"=== {model}: refit excluding encoding styles ===")
    H_refusal, H_compliance, style_refusal, style_compliance = load_style_labeled_rc(model)
    num_total_layers = H_refusal.shape[1]
    layers = [l for l, _ in AlphaSteer_CALCULATION_CONFIG[model]]

    H_refusal_fit, H_compliance_fit = build_fit_excluding_styles(
        H_refusal, H_compliance, style_refusal, style_compliance
    )
    H_enc, y_enc, style_enc = build_encoding_eval_set(
        H_refusal, H_compliance, style_refusal, style_compliance
    )
    n_pos, n_total = int(y_enc.sum()), len(y_enc)

    per_style_counts = (
        pd.DataFrame({"style": style_enc, "label": y_enc})
        .groupby("style")["label"].agg(n_total="count", n_refusal="sum")
    )
    print(f"  fit set (encoding excluded): {H_refusal_fit.shape[0]} refusal / "
          f"{H_compliance_fit.shape[0]} compliance rows")
    print(f"  encoding eval set (100% held out): {n_pos} refusals / {n_total} rows")
    print(per_style_counts)

    if n_pos < MIN_POS_FOR_TRUST:
        print(f"  *** CAUTION: only {n_pos} true refusals for {model} -- per-layer AUC "
              f"has very high variance and should NOT be reported as evidence on its own "
              f"(n_pos={n_pos} is close to or at the degenerate limit). ***")

    dim_vectors = compute_dim_vectors(H_refusal_fit, H_compliance_fit, layers)
    rfm_vectors_np, _ = compute_refusal_vectors(
        H_malicious=H_refusal_fit, H_benign=H_compliance_fit,
        layers=layers, num_total_layers=num_total_layers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **RFM_KW,
    )
    rfm_vectors = torch.from_numpy(rfm_vectors_np).float()

    row = {l: {} for l in layers}
    for l in layers:
        d_l, r_l = dim_vectors[l], rfm_vectors[l]
        row[l]["auc_dim_inv"] = auc_inv(H_enc, y_enc, l, d_l)
        row[l]["auc_rfm_inv"] = auc_inv(H_enc, y_enc, l, r_l)
        sign_dim = natural_sign(H_refusal_fit, H_compliance_fit, l, d_l)
        sign_rfm = natural_sign(H_refusal_fit, H_compliance_fit, l, r_l)
        row[l]["auc_dim_signed"] = auc_signed(H_enc, y_enc, l, d_l, sign_dim)
        row[l]["auc_rfm_signed"] = auc_signed(H_enc, y_enc, l, r_l, sign_rfm)

    enc_fit_results[model] = dict(
        layers=layers, per_layer=row, n_pos=n_pos, n_total=n_total,
        per_style_counts=per_style_counts,
    )
# ## 13. Summary table, all 3 models
rows_enc = []
for model, res in enc_fit_results.items():
    for l in res["layers"]:
        r = res["per_layer"][l]
        rows_enc.append(dict(
            model=model, layer=l,
            auc_dim_inv=r["auc_dim_inv"], auc_rfm_inv=r["auc_rfm_inv"],
            delta_inv=r["auc_rfm_inv"] - r["auc_dim_inv"],
            auc_dim_signed=r["auc_dim_signed"], auc_rfm_signed=r["auc_rfm_signed"],
            delta_signed=r["auc_rfm_signed"] - r["auc_dim_signed"],
            n_pos=res["n_pos"], n_total=res["n_total"],
        ))
df_enc = pd.DataFrame(rows_enc)
display(df_enc)

print("\nn_pos by model (READ THIS before trusting any AUC number above):")
for model, res in enc_fit_results.items():
    flag = " <-- too small to trust" if res["n_pos"] < MIN_POS_FOR_TRUST else ""
    print(f"  {model}: n_pos={res['n_pos']} / n_total={res['n_total']}{flag}")
# ## 14. Figure -- llama3.1 only (the one model with enough positives to plot meaningfully)
#
# qwen2.5 (n_pos=5) and gemma2 (n_pos=1) are reported in the table above for
# completeness, but are NOT plotted as per-layer curves: with n_pos this
# small (gemma2 especially -- a single positive example), a per-layer AUC
# line is not a meaningful signal, just noise dressed up as a trend.
fig, (ax_auc, ax_n) = plt.subplots(
    2, 1, figsize=(8, 7), gridspec_kw=dict(height_ratios=[3, 1.2])
)

TEASER_MODEL = "llama3.1"
res = enc_fit_results[TEASER_MODEL]
layers = res["layers"]
auc_dim_arr = np.array([res["per_layer"][l]["auc_dim_signed"] for l in layers])
auc_rfm_arr = np.array([res["per_layer"][l]["auc_rfm_signed"] for l in layers])
delta_arr = auc_rfm_arr - auc_dim_arr

ax_auc.axhspan(0.0, 0.5, color="#C44E52", alpha=0.08, zorder=0)
ax_auc.axhline(0.5, color="k", linewidth=0.8, linestyle="--")
ax_auc.plot(layers, auc_dim_arr, marker="o", color="black", label="AUC signed (DIM)")
ax_auc.plot(layers, auc_rfm_arr, marker="o", color="#55A868", label="AUC signed (RFM top-1)")

valid = ~np.isnan(delta_arr)
top_idx = np.argsort(np.where(valid, np.abs(delta_arr), -np.inf))[::-1][:4]
for i in top_idx:
    if not valid[i]:
        continue
    l = layers[i]
    ax_auc.annotate(
        f"L{l}: Δ={delta_arr[i]:+.3f}",
        xy=(l, auc_rfm_arr[i]),
        xytext=(l, auc_rfm_arr[i] + 0.08 * np.sign(delta_arr[i] + 1e-9)),
        ha="center", fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8),
    )

ax_auc.set_ylim(-0.05, 1.05)
ax_auc.set_xlabel("layer")
ax_auc.set_ylabel("AUC (sign fixed by encoding-excluded fit reference)")
ax_auc.set_title(
    f"{TEASER_MODEL}: DIM vs RFM(top-1) refit WITHOUT encoding styles,\n"
    f"evaluated on 100% held-out caesar+atbash+morse+ascii "
    f"(n={res['n_total']}, {res['n_pos']} true refusals)"
)
ax_auc.legend(loc="lower left", fontsize=9)

sc = res["per_style_counts"]
ax_n.bar(sc.index.astype(str), sc["n_total"], color="#4C72B0", alpha=0.5, label="n_total")
ax_n.bar(sc.index.astype(str), sc["n_refusal"], color="#C44E52", alpha=0.8, label="n_refusal")
ax_n.set_ylabel("count")
ax_n.set_title("per-style sample sizes (natural data ceiling)")
ax_n.legend(fontsize=8)

plt.tight_layout()
plt.show()
# ### Notes for the rebuttal text
#
# - This IS now a genuine held-out generalization test: DIM/RFM never see any
#   caesar/atbash/morse/ascii example during fitting, and are evaluated on
#   100% of those rows (not a 20% slice) -- state the methodology this way.
# - llama3.1 (n_pos=28) is the only model with enough positives to report a
#   per-layer curve. qwen2.5 (n_pos=5) and gemma2 (n_pos=1) are in the summary
#   table for completeness/transparency, but should NOT be cited as evidence
#   of a pattern -- especially gemma2, where n_pos=1 means the "AUC" is fully
#   determined by the rank of a single example.
# - If you want a qwen2.5/gemma2 number for the rebuttal at all, report it
#   alongside n_pos explicitly (e.g. "gemma2: AUC not meaningful, n_pos=1") --
#   do not present a bare AUC value without that context.
# - Compare auc_dim_signed at llama3.1 layer ~24-26 to your originally quoted
#   numbers -- refitting without encoding styles will shift these values
#   somewhat from the earlier (leaky) computation, since the fit distribution
#   itself has changed; that's expected and is the point of this correction.
