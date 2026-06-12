#!/usr/bin/env python3
"""Summarize gnomAD haplotype counts across epistatic deleteriousness thresholds.

This script merges structural-contact results with per-gene epistasis outputs
containing gnomAD haplotype counts. It then ranks pairs by epistatic score and
compares variant haplotype counts across cumulative top deleteriousness bins
relative to the remaining background pairs.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


DEFAULT_CUTOFFS = [10, 5, 4, 3, 2, 1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare gnomAD variant counts across epistatic score thresholds."
    )
    parser.add_argument("--base-dir", required=True, help="Base directory containing per-gene folders")
    parser.add_argument(
        "--contact-file",
        required=True,
        help="TSV file with gene, double_mut, E_cond, and is_contact columns",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--cutoffs",
        default=",".join(str(x) for x in DEFAULT_CUTOFFS),
        help="Comma-separated cumulative top-percent thresholds; default: 10,5,4,3,2,1",
    )
    parser.add_argument(
        "--model-label",
        default="model",
        help="Label used in output filenames, for example 650M or 35M",
    )
    return parser.parse_args()


def parse_cutoffs(cutoff_string):
    cutoffs = [float(x.strip()) for x in cutoff_string.split(",") if x.strip()]
    if not cutoffs:
        raise ValueError("At least one cutoff must be provided.")
    return cutoffs


def safe_label(value):
    text = str(value).replace(".", "p")
    return text.rstrip("0").rstrip("p") if "p" in text else text


def load_contact_table(contact_file):
    contact = pd.read_csv(
        contact_file,
        sep="\t",
        usecols=["gene", "double_mut", "E_cond", "is_contact"],
    )
    contact["E_cond"] = pd.to_numeric(contact["E_cond"], errors="coerce")
    contact["is_contact"] = pd.to_numeric(contact["is_contact"], errors="coerce").astype("Int64")
    contact = contact.dropna(subset=["gene", "double_mut", "E_cond", "is_contact"]).copy()
    contact["is_contact"] = contact["is_contact"].astype(int)
    return contact


def load_epistasis_counts(base_dir):
    files = sorted(glob.glob(os.path.join(base_dir, "*", "*_epistasis_esmcnd.tsv")))
    cols = [
        "gene",
        "double_mut",
        "all_hap_counts_0",
        "all_hap_counts_1",
        "all_hap_counts_2",
        "all_hap_counts_3",
    ]
    dfs = []
    for i, path in enumerate(files, start=1):
        try:
            tmp = pd.read_csv(path, sep="\t", usecols=cols)
            dfs.append(tmp)
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}", flush=True)
        if i % 1000 == 0:
            print(f"Loaded {i}/{len(files)} epistasis files", flush=True)

    if not dfs:
        raise RuntimeError("No epistasis files with haplotype-count columns were found.")

    epi = pd.concat(dfs, ignore_index=True)
    for col in cols[2:]:
        epi[col] = pd.to_numeric(epi[col], errors="coerce")
    epi = epi.dropna(subset=cols[2:]).copy()

    epi["variant1_hap_count"] = epi["all_hap_counts_1"] + epi["all_hap_counts_3"]
    epi["variant2_hap_count"] = epi["all_hap_counts_2"] + epi["all_hap_counts_3"]
    epi["mean_variant_hap_count"] = (
        epi["variant1_hap_count"] + epi["variant2_hap_count"]
    ) / 2
    epi["max_variant_hap_count"] = epi[["variant1_hap_count", "variant2_hap_count"]].max(axis=1)
    epi["min_variant_hap_count"] = epi[["variant1_hap_count", "variant2_hap_count"]].min(axis=1)
    epi["log10_mean_variant_hap_count"] = np.log10(epi["mean_variant_hap_count"] + 1)
    return epi


def build_summary(df, cutoffs):
    rows = []
    n = len(df)
    for cutoff in cutoffs:
        k = int(np.ceil(n * cutoff / 100))
        sub = df.iloc[:k].copy()
        rows.append(summarize_group(sub, f"Top {safe_label(cutoff)}%", cutoff))

    k10 = int(np.ceil(n * 10 / 100))
    rest = df.iloc[k10:].copy()
    rows.append(summarize_group(rest, "Rest >10%", np.nan))
    return pd.DataFrame(rows), rest


def summarize_group(sub, group_name, cutoff_percent):
    return {
        "group": group_name,
        "cutoff_percent": cutoff_percent,
        "n_pairs": len(sub),
        "mean_mean_variant_hap_count": sub["mean_variant_hap_count"].mean(),
        "median_mean_variant_hap_count": sub["mean_variant_hap_count"].median(),
        "mean_log10_mean_variant_hap_count": sub["log10_mean_variant_hap_count"].mean(),
        "median_log10_mean_variant_hap_count": sub["log10_mean_variant_hap_count"].median(),
        "mean_max_variant_hap_count": sub["max_variant_hap_count"].mean(),
        "median_max_variant_hap_count": sub["max_variant_hap_count"].median(),
        "mean_min_variant_hap_count": sub["min_variant_hap_count"].mean(),
        "median_min_variant_hap_count": sub["min_variant_hap_count"].median(),
        "contact_rate": sub["is_contact"].mean(),
        "E_cond_max_in_group": sub["E_cond"].max() if group_name != "Rest >10%" else np.nan,
    }


def add_threshold_labels(df, cutoffs):
    out = df.copy()
    out["threshold_group"] = "Rest >10%"
    n = len(out)
    for cutoff in cutoffs:
        k = int(np.ceil(n * cutoff / 100))
        out.loc[out.index[:k], "threshold_group"] = f"Top {safe_label(cutoff)}%"
    return out


def run_tests(df, rest, cutoffs):
    test_rows = []
    rest_vals = rest["log10_mean_variant_hap_count"].dropna()
    n = len(df)
    for cutoff in cutoffs:
        k = int(np.ceil(n * cutoff / 100))
        sub = df.iloc[:k].copy()
        vals = sub["log10_mean_variant_hap_count"].dropna()
        if len(vals) > 0 and len(rest_vals) > 0:
            u, p = mannwhitneyu(vals, rest_vals, alternative="two-sided")
        else:
            u, p = np.nan, np.nan
        test_rows.append({
            "comparison": f"Top {safe_label(cutoff)}% vs Rest >10%",
            "n_top": len(vals),
            "n_rest": len(rest_vals),
            "median_top_log10_count": vals.median(),
            "median_rest_log10_count": rest_vals.median(),
            "median_top_raw_count": sub["mean_variant_hap_count"].median(),
            "median_rest_raw_count": rest["mean_variant_hap_count"].median(),
            "U_statistic": u,
            "p_value": p,
        })
    return pd.DataFrame(test_rows)


def make_line_plot(plot_df, y_col, ylabel, title, out_base):
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, plot_df[y_col], marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["group"], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epistatic deleteriousness group")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".svg", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    cutoffs = parse_cutoffs(args.cutoffs)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading contact file...", flush=True)
    contact = load_contact_table(args.contact_file)
    print(f"Contact rows: {len(contact)}", flush=True)

    print("Loading epistasis files with gnomAD haplotype counts...", flush=True)
    epi = load_epistasis_counts(args.base_dir)
    print(f"Epistasis rows with counts: {len(epi)}", flush=True)

    df = contact.merge(epi, on=["gene", "double_mut"], how="inner")
    df = df.dropna(subset=["E_cond", "log10_mean_variant_hap_count"]).copy()
    df = df.sort_values("E_cond", ascending=True).reset_index(drop=True)
    print(f"Merged rows: {len(df)}", flush=True)

    if df.empty:
        raise RuntimeError("Merged table is empty. Check identifiers and input files.")

    summary, rest = build_summary(df, cutoffs)
    label = args.model_label

    summary_out = os.path.join(args.out_dir, f"gnomad_variant_counts_by_epistasis_threshold_{label}.tsv")
    summary.to_csv(summary_out, sep="\t", index=False)

    labeled = add_threshold_labels(df, cutoffs)
    merged_out = os.path.join(args.out_dir, f"merged_contact_epistasis_gnomad_counts_{label}.tsv")
    labeled.to_csv(merged_out, sep="\t", index=False)

    tests = run_tests(df, rest, cutoffs)
    tests_out = os.path.join(args.out_dir, f"mannwhitney_tests_top_thresholds_vs_rest_{label}.tsv")
    tests.to_csv(tests_out, sep="\t", index=False)

    plot_order = [f"Top {safe_label(c)}%" for c in sorted(cutoffs)] + ["Rest >10%"]
    plot_df = summary.set_index("group").reindex(plot_order).dropna(how="all").reset_index()

    make_line_plot(
        plot_df,
        "median_mean_variant_hap_count",
        "Median mean variant haplotype count",
        "Human variant counts across epistatic deleteriousness thresholds",
        os.path.join(args.out_dir, f"median_gnomad_variant_count_by_epistasis_threshold_{label}"),
    )
    make_line_plot(
        plot_df,
        "median_log10_mean_variant_hap_count",
        "Median log10(mean variant haplotype count + 1)",
        "Human variant counts across epistatic deleteriousness thresholds",
        os.path.join(args.out_dir, f"median_log10_gnomad_variant_count_by_epistasis_threshold_{label}"),
    )

    print("Done.", flush=True)
    print(f"Summary: {summary_out}", flush=True)
    print(f"Merged table: {merged_out}", flush=True)
    print(f"Tests: {tests_out}", flush=True)


if __name__ == "__main__":
    main()
