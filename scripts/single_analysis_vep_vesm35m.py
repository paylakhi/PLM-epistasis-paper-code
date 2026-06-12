#!/usr/bin/env python3
"""Validate VESM-35M single-mutation scores against VEP SIFT and PolyPhen annotations."""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

BASE_DIR = os.environ.get("BASE_DIR", ".")
OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.join(BASE_DIR, "vep_single_mutation_results_vesm35m")
)

os.makedirs(OUT_DIR, exist_ok=True)

# This matches the VEP files you are creating with the SLURM array
VEP_SUFFIXES = [
    "_annotated_sift_polyphen.vep.vcf",
    "_annotated.vep.vcf",
    "_annotated_vep.vcf",
]

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SIFT_ORDER = ["tolerated", "deleterious"]
POLYPHEN_ORDER = ["benign", "probably_damaging"]

COLORS = {
    "tolerated": "#4C72B0",
    "deleterious": "#C44E52",
    "benign": "#4C72B0",
    "probably_damaging": "#C44E52",
}


def find_file_case_insensitive(gene_dir, gene, suffixes):
    if not os.path.isdir(gene_dir):
        return None

    files = os.listdir(gene_dir)

    for suffix in suffixes:
        target = f"{gene}{suffix}".lower()
        for fn in files:
            if fn.lower() == target:
                return os.path.join(gene_dir, fn)

    return None


def discover_genes(base_dir):
    genes = []

    for d in os.listdir(base_dir):
        gene_dir = os.path.join(base_dir, d)

        if not os.path.isdir(gene_dir):
            continue

        epi_path = find_file_case_insensitive(
            gene_dir,
            d,
            ["_epistasis_VESM_35M.tsv"]
        )

        if epi_path is not None:
            genes.append(d)

    return sorted(genes)


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_one_gene(base_dir, gene):
    gene_dir = os.path.join(base_dir, gene)

    epi_path = find_file_case_insensitive(
        gene_dir,
        gene,
        ["_epistasis_VESM_35M.tsv"]
    )

    meta_path = find_file_case_insensitive(
        gene_dir,
        gene,
        [
            "_double_muts_for_esm_VALID_with_metadata.tsv",
            "_double_muts_for_esm_VALID.tsv",
        ]
    )

    if epi_path is None:
        return None

    epi = pd.read_csv(epi_path, sep="\t", dtype=str)

    if epi.empty:
        return None

    if "gene" not in epi.columns:
        epi["gene"] = gene
    else:
        epi["gene"] = epi["gene"].fillna(gene)

    if meta_path is not None and "double_mut" in epi.columns:
        meta = pd.read_csv(meta_path, sep="\t", dtype=str)

        if "double_mut" in meta.columns:
            cols_to_add = [c for c in meta.columns if c not in epi.columns]

            if cols_to_add:
                epi = epi.merge(
                    meta[["double_mut"] + cols_to_add],
                    on="double_mut",
                    how="left"
                )

    return epi


def parse_mutation(mut):
    mut = str(mut).strip()
    m = re.match(r"^([A-Za-z\*])(\d+)([A-Za-z\*])$", mut)

    if not m:
        return pd.Series([np.nan, np.nan, np.nan])

    return pd.Series([m.group(1), int(m.group(2)), m.group(3)])


def find_first_existing(row, candidates):
    for c in candidates:
        if c in row.index and pd.notna(row[c]) and str(row[c]) != "":
            return row[c]
    return np.nan


def build_single_table(pair_df):
    required = ["double_mut", "d1", "d2", "gene"]
    missing = [c for c in required if c not in pair_df.columns]

    if missing:
        raise ValueError(f"Missing required columns in epistasis files: {missing}")

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
            "gene": r["gene"],
            "mutation": mut1,
            "single_score": pd.to_numeric(r["d1"], errors="coerce"),
            "chrom": find_first_existing(r, ["chrom1", "chrom_1", "locus1_contig"]),
            "pos": find_first_existing(r, ["pos1", "pos_1", "locus1_position"]),
            "ref": find_first_existing(r, ["ref1", "ref_1", "alleles1_0"]),
            "alt": find_first_existing(r, ["alt1", "alt_1", "alleles1_1"]),
        })

        rows.append({
            "gene": r["gene"],
            "mutation": mut2,
            "single_score": pd.to_numeric(r["d2"], errors="coerce"),
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
            single_df[c] = single_df[c].astype(str).str.replace("^chr", "", regex=True)

    group_cols = [
        "gene", "mutation", "wt_aa", "aa_pos", "mut_aa",
        "chrom", "pos", "ref", "alt"
    ]

    single_df = (
        single_df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            single_score_mean=("single_score", "mean"),
            single_score_median=("single_score", "median"),
            single_score_n=("single_score", "count"),
        )
    )

    single_df["single_score"] = single_df["single_score_mean"]

    # Strict single amino-acid substitutions only
    single_df = single_df.dropna(subset=["wt_aa", "aa_pos", "mut_aa"]).copy()

    return single_df


def parse_sift_label(x):
    if pd.isna(x) or x == "":
        return np.nan

    s = str(x).lower()

    if "low_confidence" in s:
        return np.nan

    if s.startswith("deleterious"):
        return "deleterious"

    if s.startswith("tolerated"):
        return "tolerated"

    return np.nan


def parse_polyphen_label(x):
    if pd.isna(x) or x == "":
        return np.nan

    s = str(x).lower()

    if s.startswith("probably_damaging"):
        return "probably_damaging"

    if s.startswith("benign"):
        return "benign"

    return np.nan


def parse_prediction_score(x):
    if pd.isna(x) or x == "":
        return np.nan

    m = re.search(r"\(([-+]?[0-9]*\.?[0-9]+)\)", str(x))

    if m:
        return float(m.group(1))

    return np.nan


def hgvsp_to_mutation(hgvsp):
    if pd.isna(hgvsp) or hgvsp == "":
        return np.nan

    s = str(hgvsp)

    m = re.search(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|\=|\?)$", s)

    if not m:
        return np.nan

    aa3_to_1 = {
        "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
        "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
        "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
        "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
        "Ter": "*"
    }

    wt3, pos, mut3 = m.group(1), m.group(2), m.group(3)

    if mut3 in ["=", "?"]:
        return np.nan

    if wt3 not in aa3_to_1 or mut3 not in aa3_to_1:
        return np.nan

    return f"{aa3_to_1[wt3]}{pos}{aa3_to_1[mut3]}"


def parse_vep_vcf(vcf_path, gene_name):
    csq_fields = None
    rows = []

    with open(vcf_path) as f:
        for line in f:
            if line.startswith("##INFO=<ID=CSQ"):
                m = re.search(r'Format: (.+?)">', line)
                if m:
                    csq_fields = m.group(1).split("|")
                continue

            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")

            if len(parts) < 8:
                continue

            chrom, pos, ref, alt, info = parts[0], parts[1], parts[3], parts[4], parts[7]

            if csq_fields is None:
                continue

            csq_match = re.search(r"CSQ=([^;]+)", info)

            if not csq_match:
                continue

            annotations = csq_match.group(1).split(",")

            for ann in annotations:
                vals = ann.split("|")

                if len(vals) < len(csq_fields):
                    vals += [""] * (len(csq_fields) - len(vals))

                d = dict(zip(csq_fields, vals))

                consequence = d.get("Consequence", "")
                canonical = d.get("CANONICAL", "")
                hgvsp = d.get("HGVSp", "")
                sift = d.get("SIFT", "")
                polyphen = d.get("PolyPhen", "")

                # Strict filters
                if "missense_variant" not in consequence:
                    continue

                if canonical != "YES":
                    continue

                if not hgvsp:
                    continue

                mutation = hgvsp_to_mutation(hgvsp)

                if pd.isna(mutation):
                    continue

                rows.append({
                    "gene": gene_name,
                    "mutation": mutation,
                    "chrom": str(chrom).replace("chr", ""),
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "hgvsp": hgvsp,
                    "consequence": consequence,
                    "canonical": canonical,
                    "sift_raw": sift,
                    "polyphen_raw": polyphen,
                    "sift_group": parse_sift_label(sift),
                    "polyphen_group": parse_polyphen_label(polyphen),
                    "sift_score": parse_prediction_score(sift),
                    "polyphen_score": parse_prediction_score(polyphen),
                })

    if not rows:
        return pd.DataFrame(columns=[
            "gene", "mutation", "chrom", "pos", "ref", "alt",
            "hgvsp", "consequence", "canonical",
            "sift_raw", "polyphen_raw",
            "sift_group", "polyphen_group",
            "sift_score", "polyphen_score"
        ])

    df = pd.DataFrame(rows)

    def first_nonempty(series):
        for v in series:
            if pd.notna(v) and str(v) != "":
                return v
        return np.nan

    df = (
        df.groupby(["gene", "mutation"], as_index=False)
        .agg(
            chrom=("chrom", first_nonempty),
            pos=("pos", "first"),
            ref=("ref", first_nonempty),
            alt=("alt", first_nonempty),
            hgvsp=("hgvsp", first_nonempty),
            consequence=("consequence", first_nonempty),
            canonical=("canonical", first_nonempty),
            sift_raw=("sift_raw", first_nonempty),
            polyphen_raw=("polyphen_raw", first_nonempty),
            sift_group=("sift_group", first_nonempty),
            polyphen_group=("polyphen_group", first_nonempty),
            sift_score=("sift_score", "first"),
            polyphen_score=("polyphen_score", "first"),
        )
    )

    return df


def load_all_vep_annotations(base_dir):
    genes = discover_genes(base_dir)

    dfs = []
    missing = []
    empty = []

    for gene in genes:
        gene_dir = os.path.join(base_dir, gene)

        vcf_path = find_file_case_insensitive(
            gene_dir,
            gene,
            VEP_SUFFIXES
        )

        if vcf_path is None:
            missing.append(gene)
            continue

        d = parse_vep_vcf(vcf_path, gene)

        if len(d) > 0:
            dfs.append(d)
        else:
            empty.append(gene)

    pd.DataFrame({"missing_vep_gene": missing}).to_csv(
        os.path.join(OUT_DIR, "missing_vep_files.tsv"),
        sep="\t",
        index=False
    )

    pd.DataFrame({"empty_vep_after_filters_gene": empty}).to_csv(
        os.path.join(OUT_DIR, "empty_vep_after_filters.tsv"),
        sep="\t",
        index=False
    )

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)

    out["chrom"] = out["chrom"].astype(str).str.replace("^chr", "", regex=True)
    out["ref"] = out["ref"].astype(str)
    out["alt"] = out["alt"].astype(str)

    return out


def summarize_group(df, group_col):
    sub = df.dropna(subset=[group_col, "single_score"]).copy()

    if sub.empty:
        return pd.DataFrame()

    return (
        sub.groupby(group_col, as_index=False)
        .agg(
            n=("single_score", "size"),
            mean_score=("single_score", "mean"),
            median_score=("single_score", "median"),
            min_score=("single_score", "min"),
            max_score=("single_score", "max"),
        )
    )


def compute_spearman(df, xcol, ycol):
    sub = df[[xcol, ycol]].copy()

    sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
    sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")

    sub = sub.dropna()

    if len(sub) < 3:
        return {
            "x": xcol,
            "y": ycol,
            "n": len(sub),
            "rho": np.nan,
            "pvalue": np.nan
        }

    rho, pval = spearmanr(sub[xcol], sub[ycol])

    return {
        "x": xcol,
        "y": ycol,
        "n": len(sub),
        "rho": rho,
        "pvalue": pval
    }


def boxplot_two_group(df, group_col, order, title, ylabel, out_png):
    sub = df.dropna(subset=[group_col, "single_score"]).copy()

    sub[group_col] = pd.Categorical(
        sub[group_col],
        categories=order,
        ordered=True
    )

    sub = sub.sort_values(group_col)

    present = [g for g in order if (sub[group_col] == g).sum() > 0]
    data = [
        sub.loc[sub[group_col] == g, "single_score"].dropna().values
        for g in present
    ]

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    bp = ax.boxplot(
        data,
        labels=present,
        patch_artist=True,
        showfliers=False,
        widths=0.65
    )

    for patch, label in zip(bp["boxes"], present):
        patch.set_facecolor(COLORS.get(label, "#cccccc"))
        patch.set_alpha(0.85)

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    y_min, y_max = ax.get_ylim()
    y_text = y_max - 0.06 * (y_max - y_min)

    for i, g in enumerate(present, start=1):
        n = (sub[group_col] == g).sum()
        ax.text(i, y_text, f"n={n}", ha="center", va="top", fontsize=10)

    fig.tight_layout()
    base = os.path.splitext(out_png)[0]
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close(fig)


def barplot_group(summary_df, group_col, order, title, ylabel, out_png):
    if summary_df.empty:
        return

    sub = summary_df.copy()

    sub[group_col] = pd.Categorical(
        sub[group_col],
        categories=order,
        ordered=True
    )

    sub = sub.sort_values(group_col).dropna(subset=["median_score"])

    if sub.empty:
        return

    labels = sub[group_col].tolist()
    values = sub["median_score"].tolist()
    counts = sub["n"].tolist()
    colors = [COLORS.get(x, "#cccccc") for x in labels]

    fig, ax = plt.subplots(figsize=(10, 5.0))

    bars = ax.bar(
        labels,
        values,
        color=colors,
        alpha=0.9,
        width=0.7
    )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    for bar, n in zip(bars, counts):
        h = bar.get_height()

        if h >= 0:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h + 0.08,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=10
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                h - 0.15,
                f"n={n}",
                ha="center",
                va="top",
                fontsize=10
            )

    fig.tight_layout()
    base = os.path.splitext(out_png)[0]
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    print("Base directory:", BASE_DIR)
    print("Output directory:", OUT_DIR)

    genes = discover_genes(BASE_DIR)

    print(f"Found {len(genes)} gene folders with epistasis files.")

    dfs = []
    genes_with_epi = []

    for gene in genes:
        d = merge_one_gene(BASE_DIR, gene)

        if d is not None and len(d) > 0:
            dfs.append(d)
            genes_with_epi.append(gene)

    if not dfs:
        raise RuntimeError("No epistasis result files found.")

    pair_df = pd.concat(dfs, ignore_index=True)

    pair_df = safe_numeric(pair_df, ["d1", "d2"])

    print("Pair rows:", pair_df.shape[0])

    single_df = build_single_table(pair_df)

    single_df["chrom"] = single_df["chrom"].astype(str).str.replace("^chr", "", regex=True)
    single_df["ref"] = single_df["ref"].astype(str)
    single_df["alt"] = single_df["alt"].astype(str)

    single_df.to_csv(
        os.path.join(OUT_DIR, "single_mutations_from_pairs.tsv"),
        sep="\t",
        index=False
    )

    print("Single mutations from pair table:", single_df.shape[0])

    vep_df = load_all_vep_annotations(BASE_DIR)

    if vep_df.empty:
        raise RuntimeError(
            "No VEP files with SIFT/PolyPhen annotations found. "
            "Check that *_annotated_sift_polyphen.vep.vcf files exist."
        )

    print("VEP mutation-level rows after filters:", vep_df.shape[0])

    merged = single_df.merge(
        vep_df,
        on=["gene", "mutation"],
        how="left",
        suffixes=("", "_vep")
    )

    merged.to_csv(
        os.path.join(OUT_DIR, "single_mutations_with_vep.tsv"),
        sep="\t",
        index=False
    )

    # ========================================================
    # SIFT
    # ========================================================

    sift_df = merged.dropna(subset=["sift_group", "single_score"]).copy()
    sift_summary = summarize_group(sift_df, "sift_group")

    sift_summary.to_csv(
        os.path.join(OUT_DIR, "sift_group_summary.tsv"),
        sep="\t",
        index=False
    )

    sift_del = sift_df.loc[
        sift_df["sift_group"] == "deleterious",
        "single_score"
    ].dropna()

    sift_tol = sift_df.loc[
        sift_df["sift_group"] == "tolerated",
        "single_score"
    ].dropna()

    if len(sift_del) > 0 and len(sift_tol) > 0:
        u_sift, p_sift = mannwhitneyu(
            sift_del,
            sift_tol,
            alternative="two-sided"
        )
    else:
        u_sift, p_sift = np.nan, np.nan

    pd.DataFrame([{
        "comparison": "SIFT_deleterious_vs_tolerated",
        "n_deleterious": len(sift_del),
        "n_tolerated": len(sift_tol),
        "median_deleterious": np.median(sift_del) if len(sift_del) else np.nan,
        "median_tolerated": np.median(sift_tol) if len(sift_tol) else np.nan,
        "mannwhitney_u": u_sift,
        "pvalue": p_sift,
    }]).to_csv(
        os.path.join(OUT_DIR, "sift_mannwhitney.tsv"),
        sep="\t",
        index=False
    )

    # ========================================================
    # PolyPhen
    # ========================================================

    poly_df = merged.dropna(subset=["polyphen_group", "single_score"]).copy()
    poly_summary = summarize_group(poly_df, "polyphen_group")

    poly_summary.to_csv(
        os.path.join(OUT_DIR, "polyphen_group_summary.tsv"),
        sep="\t",
        index=False
    )

    poly_dmg = poly_df.loc[
        poly_df["polyphen_group"] == "probably_damaging",
        "single_score"
    ].dropna()

    poly_ben = poly_df.loc[
        poly_df["polyphen_group"] == "benign",
        "single_score"
    ].dropna()

    if len(poly_dmg) > 0 and len(poly_ben) > 0:
        u_poly, p_poly = mannwhitneyu(
            poly_dmg,
            poly_ben,
            alternative="two-sided"
        )
    else:
        u_poly, p_poly = np.nan, np.nan

    pd.DataFrame([{
        "comparison": "PolyPhen_probably_damaging_vs_benign",
        "n_probably_damaging": len(poly_dmg),
        "n_benign": len(poly_ben),
        "median_probably_damaging": np.median(poly_dmg) if len(poly_dmg) else np.nan,
        "median_benign": np.median(poly_ben) if len(poly_ben) else np.nan,
        "mannwhitney_u": u_poly,
        "pvalue": p_poly,
    }]).to_csv(
        os.path.join(OUT_DIR, "polyphen_mannwhitney.tsv"),
        sep="\t",
        index=False
    )

    # ========================================================
    # Numeric correlations
    # ========================================================

    corr_rows = []

    if "sift_score" in merged.columns:
        corr_rows.append(
            compute_spearman(merged, "single_score", "sift_score")
        )

    if "polyphen_score" in merged.columns:
        corr_rows.append(
            compute_spearman(merged, "single_score", "polyphen_score")
        )

    pd.DataFrame(corr_rows).to_csv(
        os.path.join(OUT_DIR, "vep_score_spearman.tsv"),
        sep="\t",
        index=False
    )

    # ========================================================
    # Plots
    # ========================================================

    ylab = "Single-mutation score\n(more negative = more deleterious)"

    boxplot_two_group(
        sift_df,
        "sift_group",
        SIFT_ORDER,
        "Single-mutation scores by SIFT class",
        ylab,
        os.path.join(OUT_DIR, "box_single_score_by_sift.png")
    )

    barplot_group(
        sift_summary,
        "sift_group",
        SIFT_ORDER,
        "Median single-mutation score by SIFT class",
        "Median single-mutation score\n(more negative = more deleterious)",
        os.path.join(OUT_DIR, "bar_median_single_score_by_sift.png")
    )

    boxplot_two_group(
        poly_df,
        "polyphen_group",
        POLYPHEN_ORDER,
        "Single-mutation scores by PolyPhen class",
        ylab,
        os.path.join(OUT_DIR, "box_single_score_by_polyphen.png")
    )

    barplot_group(
        poly_summary,
        "polyphen_group",
        POLYPHEN_ORDER,
        "Median single-mutation score by PolyPhen class",
        "Median single-mutation score\n(more negative = more deleterious)",
        os.path.join(OUT_DIR, "bar_median_single_score_by_polyphen.png")
    )

    pd.DataFrame([{
        "n_genes_with_epistasis": len(genes_with_epi),
        "n_pair_rows": len(pair_df),
        "n_single_mutations_total": len(single_df),
        "n_vep_mutation_level_rows_after_filters": len(vep_df),
        "n_rows_with_sift_group": int(sift_df.shape[0]),
        "n_rows_with_polyphen_group": int(poly_df.shape[0]),
        "merge_unit": "gene+mutation",
        "score_direction": "more negative single_score = more deleterious",
        "vep_suffixes_checked": ";".join(VEP_SUFFIXES),
        "filters": "strict missense only, canonical YES, non-empty HGVSp, SIFT low_confidence removed, PolyPhen only probably_damaging vs benign",
    }]).to_csv(
        os.path.join(OUT_DIR, "vep_run_summary.tsv"),
        sep="\t",
        index=False
    )

    print("Done.")
    print("Output directory:", OUT_DIR)
    print("Main outputs:")
    print("  - single_mutations_with_vep.tsv")
    print("  - sift_group_summary.tsv")
    print("  - sift_mannwhitney.tsv")
    print("  - polyphen_group_summary.tsv")
    print("  - polyphen_mannwhitney.tsv")
    print("  - vep_score_spearman.tsv")
    print("  - vep_run_summary.tsv")


if __name__ == "__main__":
    main()
