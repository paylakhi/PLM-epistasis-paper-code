#!/usr/bin/env python3
"""Validate VESM-35M single-mutation scores against ClinVar annotations."""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLINVAR_ORDER_MAIN = ["Benign", "Pathogenic"]
CLINVAR_ORDER_ALL = ["Benign/Likely benign", "VUS/Conflicting", "Pathogenic/Likely pathogenic"]
COLORS = {
    "Benign": "#4C72B0",
    "Pathogenic": "#C44E52",
    "Benign/Likely benign": "#4C72B0",
    "VUS/Conflicting": "#999999",
    "Pathogenic/Likely pathogenic": "#C44E52",
}


def find_file_case_insensitive(gene_dir, gene, suffixes):
    if not os.path.isdir(gene_dir):
        return None
    for suffix in suffixes:
        target = f"{gene}{suffix}".lower()
        for fn in os.listdir(gene_dir):
            if fn.lower() == target:
                return os.path.join(gene_dir, fn)
    return None


def discover_genes(base_dir):
    return sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_one_gene(base_dir, gene):
    gene_dir = os.path.join(base_dir, gene)
    epi_path = find_file_case_insensitive(gene_dir, gene, ["_epistasis_VESM_35M.tsv"])
    meta_path = find_file_case_insensitive(gene_dir, gene, ["_double_muts_for_esm_VALID_with_metadata.tsv"])
    if epi_path is None:
        return None
    epi = pd.read_csv(epi_path, sep="\t", dtype=str)
    if epi.empty:
        return None
    epi["gene"] = epi.get("gene", gene)
    if meta_path is not None:
        meta = pd.read_csv(meta_path, sep="\t", dtype=str)
        cols_to_add = [c for c in meta.columns if c not in epi.columns]
        if "double_mut" in meta.columns and cols_to_add:
            epi = epi.merge(meta[["double_mut"] + cols_to_add], on="double_mut", how="left")
    return epi


def parse_mutation(mut):
    mut = str(mut).strip()
    m = re.match(r"^([A-Za-z\*])(\d+)([A-Za-z\*])$", mut)
    if not m:
        return pd.Series([np.nan, np.nan, np.nan])
    return pd.Series([m.group(1), int(m.group(2)), m.group(3)])


def find_first_existing(row, candidates):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return np.nan


def build_single_table(pair_df):
    rows = []
    for _, r in pair_df.iterrows():
        dm = str(r["double_mut"])
        if ":" not in dm:
            continue
        parts = dm.split(":")
        if len(parts) != 2:
            continue
        mut1, mut2 = parts[0].strip(), parts[1].strip()
        rows.append({
            "gene": r["gene"], "mutation": mut1, "single_score": pd.to_numeric(r["d1"], errors="coerce"),
            "chrom": find_first_existing(r, ["chrom1", "chrom_1", "locus1_contig"]),
            "pos": find_first_existing(r, ["pos1", "pos_1", "locus1_position"]),
            "ref": find_first_existing(r, ["ref1", "ref_1", "alleles1_0"]),
            "alt": find_first_existing(r, ["alt1", "alt_1", "alleles1_1"]),
        })
        rows.append({
            "gene": r["gene"], "mutation": mut2, "single_score": pd.to_numeric(r["d2"], errors="coerce"),
            "chrom": find_first_existing(r, ["chrom2", "chrom_2", "locus2_contig"]),
            "pos": find_first_existing(r, ["pos2", "pos_2", "locus2_position"]),
            "ref": find_first_existing(r, ["ref2", "ref_2", "alleles2_0"]),
            "alt": find_first_existing(r, ["alt2", "alt_2", "alleles2_1"]),
        })
    single_df = pd.DataFrame(rows)
    parsed = single_df["mutation"].apply(parse_mutation)
    parsed.columns = ["wt_aa", "aa_pos", "mut_aa"]
    single_df = pd.concat([single_df, parsed], axis=1)
    single_df = safe_numeric(single_df, ["single_score", "pos", "aa_pos"])
    for c in ["chrom", "ref", "alt"]:
        if c in single_df.columns:
            single_df[c] = single_df[c].astype(str)
    group_cols = ["gene", "mutation", "wt_aa", "aa_pos", "mut_aa", "chrom", "pos", "ref", "alt"]
    single_df = single_df.groupby(group_cols, dropna=False, as_index=False).agg(single_score=("single_score", "mean"), n_obs=("single_score", "count"))
    return single_df.dropna(subset=["wt_aa", "aa_pos", "mut_aa"]).copy()


def load_clinvar_table(path):
    clin = pd.read_csv(path, sep="\t", dtype=str, names=["chrom", "pos", "ref", "alt", "clinvar_id", "clnsig", "clnrevstat", "clndn"], header=None)
    clin = safe_numeric(clin, ["pos"])
    for c in ["chrom", "ref", "alt", "clinvar_id", "clnsig", "clnrevstat", "clndn"]:
        clin[c] = clin[c].astype(str)
    clin["chrom"] = clin["chrom"].str.replace("^chr", "", regex=True)
    return clin


def normalize_single_keys(df):
    out = df.copy()
    out["chrom"] = out["chrom"].astype(str).str.replace("^chr", "", regex=True)
    out["ref"] = out["ref"].astype(str)
    out["alt"] = out["alt"].astype(str)
    return out


def classify_clinvar(clnsig):
    if pd.isna(clnsig):
        return np.nan
    s = str(clnsig).strip().lower().replace("_", " ").replace("|", "/").replace(",", "/")
    if "conflict" in s or "uncertain" in s:
        return "VUS/Conflicting"
    if "pathogenic" in s and "benign" not in s:
        return "Pathogenic/Likely pathogenic"
    if "benign" in s and "pathogenic" not in s:
        return "Benign/Likely benign"
    return np.nan


def simplify_review_status(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower()
    if "practice guideline" in s:
        return "practice_guideline"
    if "expert panel" in s:
        return "expert_panel"
    if "multiple submitters" in s and "no conflicts" in s:
        return "multiple_submitters_no_conflicts"
    if "single submitter" in s:
        return "single_submitter"
    if "conflicting" in s:
        return "conflicting"
    return "other"


def unique_join(values):
    vals = sorted({str(v) for v in values if pd.notna(v) and str(v) != "nan"})
    return "|".join(vals) if vals else np.nan


def collapse_mutation_level_group(values):
    vals = {str(v) for v in values if pd.notna(v) and str(v) != "nan"}
    has_path = "Pathogenic/Likely pathogenic" in vals
    has_benign = "Benign/Likely benign" in vals
    has_vus = "VUS/Conflicting" in vals
    if has_path and has_benign:
        return "VUS/Conflicting"
    if has_path:
        return "Pathogenic/Likely pathogenic"
    if has_benign:
        return "Benign/Likely benign"
    if has_vus:
        return "VUS/Conflicting"
    return np.nan


def collapse_review_status(values):
    vals = {str(v) for v in values if pd.notna(v) and str(v) != "nan"}
    for p in ["practice_guideline", "expert_panel", "multiple_submitters_no_conflicts", "single_submitter", "conflicting", "other"]:
        if p in vals:
            return p
    return np.nan


def aggregate_to_mutation_level(merged):
    group_cols = ["gene", "mutation", "wt_aa", "aa_pos", "mut_aa"]
    agg = merged.groupby(group_cols, as_index=False).agg(
        single_score=("single_score", "mean"),
        n_genomic_mappings=("chrom", "size"),
        n_clinvar_rows=("clnsig", lambda x: pd.Series(x).notna().sum()),
        genomic_loci=("pos", unique_join),
        genomic_refs=("ref", unique_join),
        genomic_alts=("alt", unique_join),
        clinvar_ids=("clinvar_id", unique_join),
        clnsig_all=("clnsig", unique_join),
        clnrevstat_all=("clnrevstat", unique_join),
        clinvar_group=("clinvar_group", collapse_mutation_level_group),
        review_simple=("review_simple", collapse_review_status),
    )

    def classify_label_purity(row):
        vals = str(row["clnsig_all"]) if pd.notna(row["clnsig_all"]) else ""
        vals_set = set(vals.split("|")) if vals else set()
        has_path = "Pathogenic/Likely pathogenic" in vals_set
        has_benign = "Benign/Likely benign" in vals_set
        has_vus = "VUS/Conflicting" in vals_set
        if has_path and not has_benign and not has_vus:
            return "pathogenic_only"
        if has_benign and not has_path and not has_vus:
            return "benign_only"
        if has_vus and not has_path and not has_benign:
            return "vus_only"
        if has_path or has_benign or has_vus:
            return "mixed"
        return np.nan

    agg["label_purity"] = agg.apply(classify_label_purity, axis=1)
    agg["is_high_confidence_review"] = agg["review_simple"].isin({"practice_guideline", "expert_panel", "multiple_submitters_no_conflicts"})
    agg["clinvar_group_strict"] = pd.Series(
        np.select(
            [agg["label_purity"].eq("pathogenic_only"), agg["label_purity"].eq("benign_only")],
            ["Pathogenic", "Benign"],
            default=None,
        ),
        index=agg.index,
        dtype="object",
    )
    return agg


def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum(np.sum(xi > y) for xi in x)
    lt = sum(np.sum(xi < y) for xi in x)
    return (gt - lt) / (len(x) * len(y))


def summarize_groups(df, group_col):
    return df.groupby(group_col, as_index=False).agg(n=("single_score", "size"), mean_score=("single_score", "mean"), median_score=("single_score", "median"), min_score=("single_score", "min"), max_score=("single_score", "max"))


def run_test(df, group_col, a, b, out_path, label):
    xa = df.loc[df[group_col] == a, "single_score"].dropna()
    xb = df.loc[df[group_col] == b, "single_score"].dropna()
    if len(xa) > 0 and len(xb) > 0:
        u, p = mannwhitneyu(xa, xb, alternative="two-sided")
    else:
        u, p = np.nan, np.nan
    pd.DataFrame([{
        "analysis": label, "group_a": a, "group_b": b, "n_a": len(xa), "n_b": len(xb),
        "median_a": np.median(xa) if len(xa) else np.nan,
        "median_b": np.median(xb) if len(xb) else np.nan,
        "median_difference_a_minus_b": (np.median(xa) - np.median(xb)) if len(xa) and len(xb) else np.nan,
        "mannwhitney_u": u, "pvalue": p, "cliffs_delta": cliffs_delta(xa.to_numpy(), xb.to_numpy()) if len(xa) and len(xb) else np.nan,
        "interpretation": "More negative single_score = more deleterious",
    }]).to_csv(out_path, sep="\t", index=False)
    return p


def run_spearman(df, group_col, score_map, out_path, label):
    corr_df = df[[group_col, "single_score"]].copy()
    corr_df["ordinal"] = corr_df[group_col].map(score_map)
    corr_df = corr_df.dropna()
    if len(corr_df) >= 3:
        rho, p = spearmanr(corr_df["ordinal"], corr_df["single_score"])
    else:
        rho, p = np.nan, np.nan
    pd.DataFrame([{"analysis": label, "x": f"{group_col}_ordinal", "y": "single_score", "n": len(corr_df), "rho": rho, "pvalue": p}]).to_csv(out_path, sep="\t", index=False)


def format_p(p):
    if pd.isna(p):
        return "p=NA"
    return f"p={p:.2e}" if p < 1e-4 else f"p={p:.3g}"


def boxplot(df, group_col, order, title, out_png, p_text=None):
    sub = df.dropna(subset=[group_col, "single_score"]).copy()
    sub[group_col] = pd.Categorical(sub[group_col], categories=order, ordered=True)
    sub = sub.sort_values(group_col)
    present = [g for g in order if (sub[group_col] == g).sum() > 0]
    data = [sub.loc[sub[group_col] == g, "single_score"].dropna().values for g in present]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=present, patch_artist=True, showfliers=False, widths=0.65)
    for patch, label in zip(bp["boxes"], present):
        patch.set_facecolor(COLORS.get(label, "#cccccc"))
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    ax.set_title(title)
    ax.set_ylabel("Single-mutation score")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(0.02, 0.02, "Lower / more negative = more deleterious", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    if p_text:
        ax.text(0.02, 0.09, p_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.06 * (y_max - y_min)
    for i, g in enumerate(present, start=1):
        ax.text(i, y_text, f"n={(sub[group_col] == g).sum()}", ha="center", va="top", fontsize=10)
    fig.tight_layout()
    base = os.path.splitext(out_png)[0]
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close(fig)


def barplot(summary_df, group_col, order, title, out_png, p_text=None):
    sub = summary_df.copy()
    sub[group_col] = pd.Categorical(sub[group_col], categories=order, ordered=True)
    sub = sub.sort_values(group_col).dropna(subset=["median_score"])
    if sub.empty:
        return
    labels = sub[group_col].tolist()
    values = sub["median_score"].tolist()
    counts = sub["n"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=[COLORS.get(x, "#cccccc") for x in labels], alpha=0.9, width=0.7)
    ax.set_title(title)
    ax.set_ylabel("Median single-mutation score")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(0.02, 0.02, "Lower / more negative = more deleterious", transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    if p_text:
        ax.text(0.02, 0.09, p_text, transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    for bar, n in zip(bars, counts):
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h - 0.15 if h < 0 else h + 0.08, f"n={n}", ha="center", va="top" if h < 0 else "bottom", fontsize=10)
    fig.tight_layout()
    base = os.path.splitext(out_png)[0]
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Single-mutation ClinVar validation with strict recommended analysis")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clinvar-tsv", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dfs = [d for gene in discover_genes(args.base_dir) if (d := merge_one_gene(args.base_dir, gene)) is not None and len(d) > 0]
    if not dfs:
        raise RuntimeError("No epistasis result files found.")

    pair_df = pd.concat(dfs, ignore_index=True)
    pair_df = safe_numeric(pair_df, ["d1", "d2"])
    single_df = normalize_single_keys(build_single_table(pair_df))
    clin = load_clinvar_table(args.clinvar_tsv)
    merged = single_df.merge(clin, on=["chrom", "pos", "ref", "alt"], how="left")
    merged["clinvar_group"] = merged["clnsig"].apply(classify_clinvar)
    merged["review_simple"] = merged["clnrevstat"].apply(simplify_review_status)
    merged.to_csv(os.path.join(args.out_dir, "single_mutations_with_clinvar_genomic_level.tsv"), sep="\t", index=False)

    mutation_level = aggregate_to_mutation_level(merged)
    mutation_level.to_csv(os.path.join(args.out_dir, "single_mutations_with_clinvar_mutation_level_all.tsv"), sep="\t", index=False)

    # Recommended main analysis: strict pathogenic-only vs benign-only
    strict_df = mutation_level[mutation_level["clinvar_group_strict"].isin(CLINVAR_ORDER_MAIN)].copy()
    strict_summary = summarize_groups(strict_df, "clinvar_group_strict")
    strict_summary.to_csv(os.path.join(args.out_dir, "clinvar_strict_group_summary.tsv"), sep="\t", index=False)
    p_main = run_test(strict_df, "clinvar_group_strict", "Pathogenic", "Benign", os.path.join(args.out_dir, "clinvar_strict_mannwhitney.tsv"), "clinvar_strict")
    run_spearman(strict_df, "clinvar_group_strict", {"Benign": 0, "Pathogenic": 1}, os.path.join(args.out_dir, "clinvar_strict_spearman.tsv"), "clinvar_strict")
    strict_df.to_csv(os.path.join(args.out_dir, "single_mutations_with_clinvar_strict.tsv"), sep="\t", index=False)

    # Sensitivity: all mutation-level classes
    all_df = mutation_level.dropna(subset=["clinvar_group", "single_score"]).copy()
    all_summary = summarize_groups(all_df, "clinvar_group")
    all_summary.to_csv(os.path.join(args.out_dir, "clinvar_all_group_summary.tsv"), sep="\t", index=False)
    run_test(all_df[all_df["clinvar_group"].isin(["Pathogenic/Likely pathogenic", "Benign/Likely benign"])], "clinvar_group", "Pathogenic/Likely pathogenic", "Benign/Likely benign", os.path.join(args.out_dir, "clinvar_all_path_vs_benign_mannwhitney.tsv"), "clinvar_all")
    run_spearman(all_df, "clinvar_group", {"Benign/Likely benign": 0, "VUS/Conflicting": 1, "Pathogenic/Likely pathogenic": 2}, os.path.join(args.out_dir, "clinvar_all_spearman.tsv"), "clinvar_all")

    # High-confidence sensitivity set
    highconf_df = mutation_level[mutation_level["is_high_confidence_review"] & mutation_level["clinvar_group_strict"].isin(CLINVAR_ORDER_MAIN)].copy()
    highconf_df.to_csv(os.path.join(args.out_dir, "single_mutations_with_clinvar_highconf_strict.tsv"), sep="\t", index=False)
    if len(highconf_df) > 0:
        summarize_groups(highconf_df, "clinvar_group_strict").to_csv(os.path.join(args.out_dir, "clinvar_highconf_strict_group_summary.tsv"), sep="\t", index=False)
        run_test(highconf_df, "clinvar_group_strict", "Pathogenic", "Benign", os.path.join(args.out_dir, "clinvar_highconf_strict_mannwhitney.tsv"), "clinvar_highconf_strict")

    boxplot(strict_df, "clinvar_group_strict", CLINVAR_ORDER_MAIN, "Single-mutation scores across ClinVar classes (strict)", os.path.join(args.out_dir, "box_single_score_by_clinvar_strict.png"), f"Pathogenic vs benign: {format_p(p_main)}")
    barplot(strict_summary, "clinvar_group_strict", CLINVAR_ORDER_MAIN, "Median single-mutation score across ClinVar classes (strict)", os.path.join(args.out_dir, "bar_median_single_score_by_clinvar_strict.png"), f"Pathogenic vs benign: {format_p(p_main)}")
    boxplot(all_df, "clinvar_group", CLINVAR_ORDER_ALL, "Single-mutation scores across ClinVar classes (all)", os.path.join(args.out_dir, "box_single_score_by_clinvar_all.png"))
    barplot(all_summary, "clinvar_group", CLINVAR_ORDER_ALL, "Median single-mutation score across ClinVar classes (all)", os.path.join(args.out_dir, "bar_median_single_score_by_clinvar_all.png"))

    pd.DataFrame([{
        "n_single_mutations_total": len(single_df),
        "n_with_any_clinvar_match_genomic_level": int(merged["clnsig"].notna().sum()),
        "n_mutation_level_rows": len(mutation_level),
        "n_strict_rows": len(strict_df),
        "n_highconf_strict_rows": len(highconf_df),
        "recommended_main_clinvar_analysis": "Strict pathogenic-only vs benign-only at mutation level",
        "recommended_supporting_analysis": "All ClinVar classes as sensitivity only",
        "interpretation": "More negative single_score = more deleterious",
    }]).to_csv(os.path.join(args.out_dir, "clinvar_run_summary.tsv"), sep="\t", index=False)

    print("Done.")
    print(f"Output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
