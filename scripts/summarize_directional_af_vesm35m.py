#!/usr/bin/env python3
"""Summarize gnomAD haplotype-derived allele-frequency metrics for VESM-35M epistasis scores."""

import os
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


DEFAULT_BASE_DIR = os.environ.get("BASE_DIR", ".")

BIN_CUTS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
BIN_LABELS = [
    "Top 1%",
    "Top 2%",
    "Top 3%",
    "Top 4%",
    "Top 5%",
    "Top 10%",
]


def find_file_case_insensitive(gene_dir, gene, suffixes):
    if not os.path.isdir(gene_dir):
        return None
    files_lower = {fn.lower(): fn for fn in os.listdir(gene_dir)}
    for suffix in suffixes:
        target = f"{gene}{suffix}".lower()
        if target in files_lower:
            return os.path.join(gene_dir, files_lower[target])
    return None


def discover_genes(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    return sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ])


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def merge_one_gene(base_dir, gene):
    gene_dir = os.path.join(base_dir, gene)

    epi_path = find_file_case_insensitive(
        gene_dir, gene, ["_epistasis_VESM_35M.tsv", "_epistasis_VESM_35M.tsv.gz"]
    )
    meta_path = find_file_case_insensitive(
        gene_dir, gene, [
            "_double_muts_for_esm_VALID_with_metadata.tsv",
            "_double_muts_for_esm_VALID_with_metadata.tsv.gz",
        ]
    )

    if epi_path is None:
        return None, {
            "gene": gene,
            "status": "missing_epistasis_file",
            "epi_path": "",
            "meta_path": meta_path or "",
            "n_rows": 0,
        }

    try:
        epi = pd.read_csv(epi_path, sep="\t", dtype=str, low_memory=False)
    except Exception as e:
        return None, {
            "gene": gene,
            "status": f"failed_read_epistasis: {e}",
            "epi_path": epi_path,
            "meta_path": meta_path or "",
            "n_rows": 0,
        }

    if epi.empty:
        return None, {
            "gene": gene,
            "status": "empty_epistasis_file",
            "epi_path": epi_path,
            "meta_path": meta_path or "",
            "n_rows": 0,
        }

    if "gene" not in epi.columns:
        epi["gene"] = gene
    else:
        epi["gene"] = epi["gene"].fillna(gene)

    if meta_path is not None:
        try:
            meta = pd.read_csv(meta_path, sep="\t", dtype=str, low_memory=False)
            cols_to_add = [c for c in meta.columns if c not in epi.columns]
            if "double_mut" in meta.columns and "double_mut" in epi.columns and cols_to_add:
                epi = epi.merge(meta[["double_mut"] + cols_to_add], on="double_mut", how="left")
            elif "double_mut" not in epi.columns:
                warnings.warn(f"{gene}: epistasis file has no double_mut column; metadata not merged.")
        except Exception as e:
            warnings.warn(f"{gene}: could not read/merge metadata file {meta_path}: {e}")

    return epi, {
        "gene": gene,
        "status": "ok",
        "epi_path": epi_path,
        "meta_path": meta_path or "",
        "n_rows": len(epi),
    }


def add_enrichment_columns(df, pseudocount=1e-12):
    out = df.copy()
    needed = [
        "all_hap_counts_0",
        "all_hap_counts_1",
        "all_hap_counts_2",
        "all_hap_counts_3",
    ]

    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError("Missing required haplotype count columns: " + ", ".join(missing))

    h0 = pd.to_numeric(out["all_hap_counts_0"], errors="coerce")
    h1 = pd.to_numeric(out["all_hap_counts_1"], errors="coerce")
    h2 = pd.to_numeric(out["all_hap_counts_2"], errors="coerce")
    h3 = pd.to_numeric(out["all_hap_counts_3"], errors="coerce")

    total_haps = h0 + h1 + h2 + h3

    valid = total_haps > 0
    out.loc[~valid, needed] = np.nan

    out["total_haplotypes"] = total_haps
    out["f_AB_obs"] = (h3 + pseudocount) / (total_haps + 4 * pseudocount)
    out["f_A"] = (h1 + h3 + pseudocount) / (total_haps + 4 * pseudocount)
    out["f_B"] = (h2 + h3 + pseudocount) / (total_haps + 4 * pseudocount)
    out["f_AB_expected_indep"] = out["f_A"] * out["f_B"]
    out["enrichment_ratio"] = (out["f_AB_obs"] + pseudocount) / (
        out["f_AB_expected_indep"] + pseudocount
    )
    out["log10_enrichment"] = np.log10(out["enrichment_ratio"])
    out["depletion_score"] = -out["log10_enrichment"]
    out["never_cooccur"] = h3.fillna(0).eq(0)

    return out


def add_pair_af_columns_from_haps(df, pseudocount=1e-12):
    out = df.copy()

    out["pair_mean_af"] = out[["f_A", "f_B"]].mean(axis=1, skipna=True)
    out["pair_median_af"] = out[["f_A", "f_B"]].median(axis=1, skipna=True)
    out["pair_min_af"] = out[["f_A", "f_B"]].min(axis=1, skipna=True)
    out["pair_max_af"] = out[["f_A", "f_B"]].max(axis=1, skipna=True)

    for c in [
        "f_A",
        "f_B",
        "pair_mean_af",
        "pair_median_af",
        "pair_min_af",
        "pair_max_af",
        "f_AB_obs",
        "f_AB_expected_indep",
    ]:
        out[f"log10_{c}"] = np.log10(pd.to_numeric(out[c], errors="coerce") + pseudocount)

    return out


def add_directional_score_ranks(df, score_col="E_cond", more_deleterious="more_negative"):
    out = df.copy()

    if score_col not in out.columns:
        raise ValueError(f"Score column not found: {score_col}")

    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.dropna(subset=[score_col]).copy()

    ascending = more_deleterious == "more_negative"

    out = out.sort_values(score_col, ascending=ascending).reset_index(drop=True)
    out["score_rank"] = np.arange(1, len(out) + 1)
    out["score_rank_pct"] = out["score_rank"] / len(out)
    out["deleterious_percentile"] = 100 * out["score_rank_pct"]

    for cut, label in zip(BIN_CUTS, BIN_LABELS):
        key = label.lower().replace(" ", "_").replace("%", "pct").replace(".", "p")
        out[f"is_{key}"] = out["score_rank_pct"] <= cut

    def assign_exclusive_bin(p):
        prev = 0.0
        for cut, label in zip(BIN_CUTS, BIN_LABELS):
            if prev < p <= cut:
                return label
            prev = cut
        return "Rest"

    out["score_bin_exclusive"] = out["score_rank_pct"].apply(assign_exclusive_bin)

    return out


def bootstrap_ci(x, stat="median", n_boot=1000, seed=42):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna().values

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = []

    for _ in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        if stat == "mean":
            boots.append(np.mean(sample))
        else:
            boots.append(np.median(sample))

    point = np.mean(x) if stat == "mean" else np.median(x)
    low = np.percentile(boots, 2.5)
    high = np.percentile(boots, 97.5)

    return point, low, high


def summarize_metric(df, metric_col):
    rows = []

    groups = []

    for cut, label in zip(BIN_CUTS, BIN_LABELS):
        groups.append((label, df["score_rank_pct"] <= cut))

    groups.append(("Rest", df["score_rank_pct"] > 0.10))

    for group_name, mask in groups:
        sub = pd.to_numeric(
            df.loc[mask, metric_col],
            errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()

        mean_val, mean_low, mean_high = bootstrap_ci(sub, stat="mean")
        median_val, median_low, median_high = bootstrap_ci(sub, stat="median")

        rows.append({
            "group": group_name,
            "metric": metric_col,
            "n": int(len(sub)),

            "mean": mean_val,
            "mean_low95": mean_low,
            "mean_high95": mean_high,

            "median": median_val,
            "median_low95": median_low,
            "median_high95": median_high,

            "min": sub.min() if len(sub) else np.nan,
            "max": sub.max() if len(sub) else np.nan,
        })

    return pd.DataFrame(rows)


def compute_spearman(df, xcol, ycol):
    sub = df[[xcol, ycol]].copy()

    sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
    sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    if len(sub) < 3 or spearmanr is None:
        return {
            "x": xcol,
            "y": ycol,
            "n": len(sub),
            "rho": np.nan,
            "pvalue": np.nan,
        }

    rho, pval = spearmanr(sub[xcol], sub[ycol])

    return {
        "x": xcol,
        "y": ycol,
        "n": len(sub),
        "rho": rho,
        "pvalue": pval,
    }


def make_barplot(summary_df, metric_col, out_png, stat="median", title=None, ylabel=None, yscale=None):
    sub = summary_df[summary_df["metric"] == metric_col].copy()

    order = BIN_LABELS + ["Rest"]

    sub["group"] = pd.Categorical(
        sub["group"],
        categories=order,
        ordered=True
    )

    sub = sub.sort_values("group")
    sub = sub[sub["group"].notna()].copy()

    if sub.empty:
        return

    y = pd.to_numeric(sub[stat], errors="coerce")

    low_col = f"{stat}_low95"
    high_col = f"{stat}_high95"

    y_low = pd.to_numeric(sub[low_col], errors="coerce")
    y_high = pd.to_numeric(sub[high_col], errors="coerce")

    yerr = np.vstack([
        y.values - y_low.values,
        y_high.values - y.values
    ])

    x = np.arange(len(sub))

    plt.figure(figsize=(10, 5.5))

    plt.bar(x, y.values)
    plt.errorbar(
        x,
        y.values,
        yerr=yerr,
        fmt="none",
        capsize=4,
        linewidth=1.5,
        color="black"
    )

    if yscale == "log":
        positive = y[y > 0]
        if len(positive) > 0:
            plt.yscale("log")

    plt.xticks(
        x,
        sub["group"].astype(str),
        rotation=30,
        ha="right"
    )

    plt.ylabel(ylabel if ylabel else f"{stat} {metric_col}", fontsize=12)
    plt.xlabel(
        "Epistatic score percentile bin\n(more negative E_cond = more deleterious)",
        fontsize=12
    )
    plt.title(
        title if title else f"{metric_col} across epistatic score bins",
        fontsize=14
    )

    plt.tight_layout()

    base = os.path.splitext(out_png)[0]
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close()


def make_scatter(df, xcol, ycol, out_png, alpha=0.08, s=5, max_points=500000, seed=1, yscale=None):
    sub = df[[xcol, ycol]].copy()

    sub[xcol] = pd.to_numeric(sub[xcol], errors="coerce")
    sub[ycol] = pd.to_numeric(sub[ycol], errors="coerce")

    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()

    if yscale == "log":
        sub = sub[sub[ycol] > 0].copy()

    if sub.empty:
        return

    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=seed)

    plt.figure(figsize=(10, 5))
    plt.scatter(sub[xcol], sub[ycol], alpha=alpha, s=s)

    if yscale == "log":
        plt.yscale("log")

    plt.xlabel(xcol, fontsize=12)
    plt.ylabel(ycol, fontsize=12)

    title_suffix = " (raw AF, log y-axis)" if yscale == "log" else ""
    plt.title(f"{ycol} vs {xcol}{title_suffix}", fontsize=14)

    plt.tight_layout()

    base = os.path.splitext(out_png)[0]
    plt.savefig(f"{base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.savefig(f"{base}.svg", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Directional epistasis vs allele frequency summary for all genes."
    )

    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help="Base directory containing all-gene per-gene folders"
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory; default: BASE_DIR/michel_summary_directional_af_all_genes_CI"
    )
    parser.add_argument(
        "--score-col",
        default="E_cond",
        help="Signed score column used for ranking"
    )
    parser.add_argument(
        "--more-deleterious",
        choices=["more_negative", "more_positive"],
        default="more_negative"
    )
    parser.add_argument(
        "--gene-list",
        default=None,
        help="Optional text file with one gene per line. If omitted, all folders are used."
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        default=1e-12
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir or os.path.join(
        base_dir,
        "michel_summary_directional_af_all_genes_CI_VESM35M"
    )

    os.makedirs(out_dir, exist_ok=True)

    if args.gene_list:
        with open(args.gene_list) as f:
            genes = sorted([
                line.strip()
                for line in f
                if line.strip() and not line.startswith("#")
            ])
    else:
        genes = discover_genes(base_dir)

    dfs = []
    logs = []

    for i, gene in enumerate(genes, start=1):
        d, log_row = merge_one_gene(base_dir, gene)
        logs.append(log_row)

        if d is not None and len(d) > 0:
            dfs.append(d)

        if i % 50 == 0:
            print(f"Processed {i}/{len(genes)} gene folders", flush=True)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(
        os.path.join(out_dir, "gene_file_discovery_log.tsv"),
        sep="\t",
        index=False
    )

    if not dfs:
        raise RuntimeError("No epistasis result files found. Check --base-dir and file names.")

    df = pd.concat(dfs, ignore_index=True)

    numeric_cols = [
        args.score_col,
        "E_cond",
        "d1",
        "d2",
        "d1_given_m2",
        "d2_given_m1",
        "additive_score",
        "additive_sum",
        "contextual_double_score",
        "pair_model_score",
        "all_p_chet",
        "all_same_haplotype",
        "all_different_haplotype",
        "all_hap_counts_0",
        "all_hap_counts_1",
        "all_hap_counts_2",
        "all_hap_counts_3",
        "all_gt_counts_0",
        "all_gt_counts_1",
        "all_gt_counts_2",
        "all_gt_counts_3",
        "all_gt_counts_4",
        "all_gt_counts_5",
        "all_gt_counts_6",
        "all_gt_counts_7",
        "all_gt_counts_8",
    ]

    df = safe_numeric(df, numeric_cols)

    df = add_enrichment_columns(df, pseudocount=args.pseudocount)
    df = add_pair_af_columns_from_haps(df, pseudocount=args.pseudocount)
    df = add_directional_score_ranks(
        df,
        score_col=args.score_col,
        more_deleterious=args.more_deleterious
    )

    combined_path = os.path.join(
        out_dir,
        "combined_directional_with_pair_af_all_genes_VESM35M.tsv.gz"
    )

    df.to_csv(
        combined_path,
        sep="\t",
        index=False,
        compression="gzip"
    )

    metrics = [
        args.score_col,
        "f_A",
        "f_B",
        "pair_mean_af",
        "pair_median_af",
        "pair_min_af",
        "pair_max_af",
        "log10_f_A",
        "log10_f_B",
        "log10_pair_mean_af",
        "log10_pair_median_af",
        "log10_pair_min_af",
        "log10_pair_max_af",
        "log10_enrichment",
        "depletion_score",
        "f_AB_obs",
        "f_AB_expected_indep",
        "log10_f_AB_obs",
        "log10_f_AB_expected_indep",
        "all_hap_counts_3",
        "total_haplotypes",
    ]

    summary_df = pd.concat(
        [summarize_metric(df, m) for m in metrics if m in df.columns],
        ignore_index=True
    )

    summary_path = os.path.join(
        out_dir,
        "directional_af_bin_summaries_all_genes_VESM35M.tsv"
    )

    summary_df.to_csv(summary_path, sep="\t", index=False)

    for cut, label in zip(BIN_CUTS, BIN_LABELS):
        safe_label = label.lower().replace(" ", "_").replace("%", "pct").replace(".", "p")
        df[df["score_rank_pct"] <= cut].to_csv(
            os.path.join(out_dir, f"{safe_label}_pairs_directional.tsv.gz"),
            sep="\t",
            index=False,
            compression="gzip"
        )

    df[df["score_rank_pct"] > 0.10].to_csv(
        os.path.join(out_dir, "rest_pairs_directional_VESM35M.tsv.gz"),
        sep="\t",
        index=False,
        compression="gzip"
    )

    corr_rows = []

    for af_metric in [
        "f_A",
        "f_B",
        "pair_mean_af",
        "pair_median_af",
        "pair_min_af",
        "pair_max_af",
        "log10_f_A",
        "log10_f_B",
        "log10_pair_mean_af",
        "log10_pair_median_af",
        "log10_pair_min_af",
        "log10_pair_max_af",
        "log10_enrichment",
        "depletion_score",
    ]:
        if af_metric in df.columns:
            corr_rows.append(compute_spearman(df, args.score_col, af_metric))

    corr_df = pd.DataFrame(corr_rows)

    corr_path = os.path.join(
        out_dir,
        "spearman_score_vs_pair_af_all_genes_VESM35M.tsv"
    )

    corr_df.to_csv(corr_path, sep="\t", index=False)

    make_scatter(
        df,
        args.score_col,
        "pair_mean_af",
        os.path.join(out_dir, f"scatter_pair_mean_af_vs_{args.score_col}_VESM35M.png")
    )

    make_scatter(
        df,
        args.score_col,
        "pair_mean_af",
        os.path.join(out_dir, f"scatter_pair_mean_af_vs_{args.score_col}_raw_AF_log_y_axis_VESM35M.png"),
        yscale="log"
    )

    make_barplot(
        summary_df,
        "pair_mean_af",
        os.path.join(out_dir, "MAIN_bar_pair_mean_af_by_score_bin_median_raw_AF_log_y_axis_VESM35M.png"),
        stat="median",
        title="Median raw mean pair AF across epistatic score bins",
        ylabel="Median raw mean pair AF",
        yscale="log"
    )

    make_barplot(
        summary_df,
        "pair_mean_af",
        os.path.join(out_dir, "MAIN_bar_pair_mean_af_by_score_bin_mean_raw_AF_log_y_axis_VESM35M.png"),
        stat="mean",
        title="Mean raw mean pair AF across epistatic score bins",
        ylabel="Mean raw mean pair AF",
        yscale="log"
    )

    make_barplot(
        summary_df,
        "pair_mean_af",
        os.path.join(out_dir, "bar_pair_mean_af_by_score_bin_median_raw_AF_linear_y_axis_VESM35M.png"),
        stat="median",
        title="Median raw mean pair AF across epistatic score bins",
        ylabel="Median raw mean pair AF"
    )

    make_barplot(
        summary_df,
        "pair_mean_af",
        os.path.join(out_dir, "bar_pair_mean_af_by_score_bin_mean_raw_AF_linear_y_axis_VESM35M.png"),
        stat="mean",
        title="Mean raw mean pair AF across epistatic score bins",
        ylabel="Mean raw mean pair AF"
    )

    make_scatter(
        df,
        args.score_col,
        "log10_pair_mean_af",
        os.path.join(out_dir, f"QC_scatter_log10_pair_mean_af_vs_{args.score_col}_VESM35M.png")
    )

    make_barplot(
        summary_df,
        "log10_pair_mean_af",
        os.path.join(out_dir, "QC_bar_log10_pair_mean_af_by_score_bin_median_VESM35M.png"),
        stat="median",
        title="QC: Median log10(mean pair AF) across epistatic score bins",
        ylabel="Median log10(mean pair AF)"
    )

    make_barplot(
        summary_df,
        "depletion_score",
        os.path.join(out_dir, "bar_depletion_score_by_score_bin_median_VESM35M.png"),
        stat="median",
        title="Median depletion score across epistatic score bins",
        ylabel="Median depletion score = -log10(enrichment)"
    )

    run_summary = pd.DataFrame([{
        "base_dir": base_dir,
        "out_dir": out_dir,
        "n_gene_folders_requested": len(genes),
        "n_genes_with_epistasis_files": int((log_df["status"] == "ok").sum()),
        "n_total_pairs_after_score_filter": len(df),
        "score_col": args.score_col,
        "more_deleterious": args.more_deleterious,
        "af_source": "derived_from_gnomad_haplotype_counts",
        "ranking_mode": "signed_directional_no_score_flipping",
        "recommended_plot": "MAIN_bar_pair_mean_af_by_score_bin_median_raw_AF_log_y_axis_VESM35M.png",
        "recommended_plot_note": "Uses raw AF values, not -log10(AF); only the y-axis is logarithmic. Error bars are bootstrap 95% CIs.",
        "combined_table": combined_path,
        "summary_table": summary_path,
        "spearman_table": corr_path,
    }])

    run_summary.to_csv(
        os.path.join(out_dir, "run_summary_all_genes_VESM35M.tsv"),
        sep="\t",
        index=False
    )

    print("Done.", flush=True)
    print(f"Output directory: {out_dir}", flush=True)
    print(f"Combined table:   {combined_path}", flush=True)
    print(f"Summary table:    {summary_path}", flush=True)
    print(f"Correlations:     {corr_path}", flush=True)
    print("Most useful plot: MAIN_bar_pair_mean_af_by_score_bin_median_raw_AF_log_y_axis_VESM35M.png", flush=True)
    print("Note: main plot uses raw AF values, not -log10(AF); only the y-axis is logarithmic.", flush=True)
    print("Error bars are bootstrap 95% confidence intervals.", flush=True)


if __name__ == "__main__":
    main()
