#!/usr/bin/env python3
"""
Build a VESM-35M structural-contact analysis table.

This script combines an existing structural-contact table with VESM-35M
pairwise epistasis scores. It keeps the structural metadata from the contact
analysis and replaces the score columns with VESM-35M-derived values.
"""

import argparse
import os

import pandas as pd


STRUCTURAL_COLUMNS = [
    "gene", "double_mut", "mut1", "mut2", "pos1", "pos2",
    "fragment", "linear_dist", "dist_3d", "is_contact",
]

SCORE_COLUMNS = ["gene", "double_mut", "d1", "d2", "E_cond"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge structural-contact metadata with VESM-35M epistasis scores."
    )
    parser.add_argument(
        "--contact-file",
        required=True,
        help="TSV containing structural-contact metadata, including gene and double_mut.",
    )
    parser.add_argument(
        "--vesm-score-file",
        required=True,
        help="TSV containing VESM-35M scores with gene, double_mut, d1, d2, and E_cond.",
    )
    parser.add_argument(
        "--out-file",
        required=True,
        help="Output TSV path for the merged VESM-35M contact table.",
    )
    return parser.parse_args()


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def main():
    args = parse_args()

    print(f"Loading structural-contact file: {args.contact_file}", flush=True)
    contact_df = pd.read_csv(args.contact_file, sep="\t", low_memory=False)
    require_columns(contact_df, STRUCTURAL_COLUMNS, "contact file")
    contact_df = contact_df[STRUCTURAL_COLUMNS].copy()
    print(f"Structural-contact rows: {contact_df.shape[0]}", flush=True)

    print(f"Loading VESM-35M score file: {args.vesm_score_file}", flush=True)
    score_df = pd.read_csv(args.vesm_score_file, sep="\t", low_memory=False)
    require_columns(score_df, SCORE_COLUMNS, "VESM-35M score file")
    score_df = score_df[SCORE_COLUMNS].copy()
    print(f"VESM-35M score rows: {score_df.shape[0]}", flush=True)

    merged = contact_df.merge(score_df, on=["gene", "double_mut"], how="inner")
    print(f"Merged rows: {merged.shape[0]}", flush=True)

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(args.out_file, sep="\t", index=False)
    print(f"Saved: {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
