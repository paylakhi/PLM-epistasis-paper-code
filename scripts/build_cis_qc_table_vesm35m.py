#!/usr/bin/env python3
"""
Build a VESM-35M cis/co-occurrence QC table.

This script maps pair keys in a pre-existing cis/co-occurrence table to protein
mutation pairs and then merges those rows with VESM-35M epistasis scores.
"""

import argparse
import glob
import os

import pandas as pd


METADATA_COLUMNS = [
    "gene", "double_mut",
    "chrom1", "pos1", "ref1", "alt1",
    "chrom2", "pos2", "ref2", "alt2",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a cis/co-occurrence QC table with VESM-35M epistasis scores."
    )
    parser.add_argument("--base-dir", required=True, help="Base directory containing per-gene folders.")
    parser.add_argument("--cis-file", required=True, help="Input cis/co-occurrence QC TSV with gene and pair_key.")
    parser.add_argument("--vesm-score-file", required=True, help="VESM-35M score TSV with gene, double_mut, E_cond, d1, d2.")
    parser.add_argument("--out-file", required=True, help="Output merged TSV path.")
    return parser.parse_args()


def build_pair_key(chrom1, pos1, ref1, alt1, chrom2, pos2, ref2, alt2):
    return (
        f"{chrom1}:{pos1}:{ref1}:{alt1}__"
        f"{chrom2}:{pos2}:{ref2}:{alt2}"
    )


def load_metadata_pair_map(base_dir):
    meta_files = glob.glob(
        os.path.join(base_dir, "*", "*double_muts_for_esm_VALID_with_metadata.tsv")
    )

    rows = []
    for path in meta_files:
        try:
            meta = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        except Exception as exc:
            print(f"[WARN] Could not read metadata file {path}: {exc}", flush=True)
            continue

        if not all(c in meta.columns for c in METADATA_COLUMNS):
            continue

        meta = meta[METADATA_COLUMNS].copy()
        meta["pair_key"] = meta.apply(
            lambda r: build_pair_key(
                r["chrom1"], r["pos1"], r["ref1"], r["alt1"],
                r["chrom2"], r["pos2"], r["ref2"], r["alt2"],
            ),
            axis=1,
        )
        rows.append(meta[["gene", "double_mut", "pair_key"]])

        meta["pair_key"] = meta.apply(
            lambda r: build_pair_key(
                r["chrom2"], r["pos2"], r["ref2"], r["alt2"],
                r["chrom1"], r["pos1"], r["ref1"], r["alt1"],
            ),
            axis=1,
        )
        rows.append(meta[["gene", "double_mut", "pair_key"]])

    if not rows:
        raise RuntimeError("No usable metadata files were found.")

    return pd.concat(rows, ignore_index=True).drop_duplicates()


def main():
    args = parse_args()

    print(f"Loading cis/co-occurrence file: {args.cis_file}", flush=True)
    cis_df = pd.read_csv(args.cis_file, sep="\t", low_memory=False)
    if not {"gene", "pair_key"}.issubset(cis_df.columns):
        raise ValueError("--cis-file must contain columns: gene and pair_key")
    print(f"Cis rows: {cis_df.shape[0]}", flush=True)

    print(f"Loading VESM-35M scores: {args.vesm_score_file}", flush=True)
    scores = pd.read_csv(args.vesm_score_file, sep="\t", low_memory=False)
    required_scores = ["gene", "double_mut", "E_cond", "d1", "d2"]
    missing_scores = [c for c in required_scores if c not in scores.columns]
    if missing_scores:
        raise ValueError(f"VESM-35M score file is missing columns: {missing_scores}")
    scores = scores[required_scores].copy()
    print(f"VESM-35M score rows: {scores.shape[0]}", flush=True)

    print("Building pair-key metadata map...", flush=True)
    meta = load_metadata_pair_map(args.base_dir)
    print(f"Metadata map rows: {meta.shape[0]}", flush=True)

    cis_meta = cis_df.merge(meta, on=["gene", "pair_key"], how="inner")
    print(f"Rows after cis + metadata merge: {cis_meta.shape[0]}", flush=True)

    merged = cis_meta.merge(scores, on=["gene", "double_mut"], how="inner")
    print(f"Rows after VESM-35M merge: {merged.shape[0]}", flush=True)

    if "epistasis_score" in merged.columns:
        merged = merged.drop(columns=["epistasis_score"])
    merged = merged.rename(columns={"E_cond": "epistasis_score"})

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.out_file, sep="\t", index=False)
    print(f"Saved: {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
