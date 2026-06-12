#!/usr/bin/env python3
"""
Merge PLM epistasis scores with per-gene variant-pair metadata.

For each gene directory under --out-base, this script looks for:
  {GENE}_epistasis_esmcnd.tsv
  {GENE}_double_muts_for_esm_VALID_with_metadata.tsv

It backs up the original epistasis file, merges available metadata columns, and
writes the merged table back to {GENE}_epistasis_esmcnd.tsv.

Designed for chunked/array execution on an HPC cluster.
"""

import os
import glob
import argparse
import shutil
import traceback
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-base", required=True)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str)


def find_merge_columns(epi, meta):
    possible = [
        "pair_key",
        "double_mut",
        "mut_pair",
        "pair",
        "mutation_pair",
        "esm_mutation_pair",
    ]

    for c in possible:
        if c in epi.columns and c in meta.columns:
            return [c]

    # fallback: use shared mutation columns if present
    shared = [c for c in ["mut1", "mut2"] if c in epi.columns and c in meta.columns]
    if len(shared) == 2:
        return shared

    shared = [c for c in ["mutation1", "mutation2"] if c in epi.columns and c in meta.columns]
    if len(shared) == 2:
        return shared

    raise ValueError(
        "Could not find merge columns.\n"
        f"Epistasis columns: {list(epi.columns)}\n"
        f"Metadata columns: {list(meta.columns)}"
    )


def process_gene(gene_dir, overwrite=False):
    gene = os.path.basename(gene_dir)

    epi_file = os.path.join(gene_dir, f"{gene}_epistasis_esmcnd.tsv")
    meta_file = os.path.join(gene_dir, f"{gene}_double_muts_for_esm_VALID_with_metadata.tsv")
    backup_file = os.path.join(gene_dir, f"{gene}_epistasis_esmcnd.original.tsv")
    tmp_file = os.path.join(gene_dir, f"{gene}_epistasis_esmcnd.tmp.tsv")

    if not os.path.exists(epi_file):
        return gene, "MISSING", "missing epistasis file"

    if not os.path.exists(meta_file):
        return gene, "MISSING", "missing metadata file"

    if os.path.exists(backup_file) and not overwrite:
        return gene, "SKIPPED_EXISTS", "backup already exists; use --overwrite if rerunning"

    epi = read_tsv(epi_file)
    meta = read_tsv(meta_file)

    merge_cols = find_merge_columns(epi, meta)

    # Avoid duplicating columns already in epistasis file
    keep_meta_cols = merge_cols + [c for c in meta.columns if c not in epi.columns]

    merged = epi.merge(
        meta[keep_meta_cols],
        on=merge_cols,
        how="left"
    )

    if not os.path.exists(backup_file):
        shutil.copy2(epi_file, backup_file)

    merged.to_csv(tmp_file, sep="\t", index=False)
    os.replace(tmp_file, epi_file)

    return gene, "OK", f"rows={len(merged)}; merge_cols={','.join(merge_cols)}"


def main():
    args = parse_args()
    out_base = args.out_base

    summary_dir = os.path.join(out_base, "epistasis_metadata_merge_summaries")
    os.makedirs(summary_dir, exist_ok=True)

    gene_dirs = sorted([
        d for d in glob.glob(os.path.join(out_base, "*"))
        if os.path.isdir(d)
        and os.path.basename(d) not in {
            "logs", "qc_reports", "results", "env",
            "chunk_summaries", "cooccurrence_chunk_summaries",
            "metadata_merge_summaries",
            "epistasis_metadata_merge_summaries"
        }
    ])

    start = args.task_id * args.chunk_size
    end = min(start + args.chunk_size, len(gene_dirs))
    chunk_dirs = gene_dirs[start:end]

    print(f"Total gene folders: {len(gene_dirs)}", flush=True)
    print(f"Task ID: {args.task_id}", flush=True)
    print(f"Chunk size: {args.chunk_size}", flush=True)
    print(f"Genes in this chunk: {len(chunk_dirs)}", flush=True)

    rows = []

    for gene_dir in chunk_dirs:
        gene = os.path.basename(gene_dir)
        print(f"Processing {gene}", flush=True)

        try:
            gene, status, message = process_gene(gene_dir, overwrite=args.overwrite)
        except Exception as e:
            status = "FAILED"
            message = str(e).replace("\n", " ")
            print(traceback.format_exc(), flush=True)

        rows.append({
            "gene": gene,
            "status": status,
            "message": message
        })

        print(f"{gene}: {status} - {message}", flush=True)

    summary = pd.DataFrame(rows)
    summary_file = os.path.join(
        summary_dir,
        f"epistasis_metadata_merge_chunk_{args.task_id}.tsv"
    )
    summary.to_csv(summary_file, sep="\t", index=False)

    print("Done.", flush=True)
    print("Summary:", summary_file, flush=True)


if __name__ == "__main__":
    main()
