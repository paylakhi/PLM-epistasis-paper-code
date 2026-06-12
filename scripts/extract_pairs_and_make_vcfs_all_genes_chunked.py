"""
Extract phased co-occurrence pair metadata and create per-gene VCF files.

Input:
  --gene-intervals  TSV file with columns: gene, interval
  --coocc-ht        Hail Table containing phased co-occurrence information
  --out-base        Base directory containing per-gene folders

For each gene, the script writes:
  OUT_BASE/GENE/GENE_pairs_from_cooccurrence.tsv.bgz
  OUT_BASE/GENE/GENE_unique_variants.txt
  OUT_BASE/GENE/GENE_unique_variants.vcf

The script is intended for SLURM array jobs. Each task processes one chunk
of genes based on --task-id and --chunk-size.
"""

import os
import argparse
import traceback
from pathlib import Path

import pandas as pd
import hail as hl


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract co-occurrence pairs and make per-gene VCFs from Hail table for all genes, chunked."
    )
    p.add_argument(
        "--gene-intervals",
        required=True,
        help="TSV with columns: gene, interval"
    )
    p.add_argument(
        "--coocc-ht",
        required=True,
        help="Path to Hail co-occurrence table (.ht)"
    )
    p.add_argument(
        "--out-base",
        required=True,
        help="Base output directory"
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("CHUNK_SIZE", "500")),
        help="Number of genes per array task/chunk"
    )
    p.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("TASK_ID", "0"))),
        help="Array task id"
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pair/VCF files"
    )
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def interval_contains_both_variant(ht, interval):
    return ht.filter(interval.contains(ht.locus1) & interval.contains(ht.locus2))


def make_export_table(ht):
    """
    Export explicit pair identity fields plus raw phase/co-occurrence fields
    from phase_info['all'] for later frequency/count analysis.
    """
    all_phase = ht.phase_info["all"]

    return ht.select(
        chrom1=hl.str(ht.locus1.contig),
        pos1=hl.int32(ht.locus1.position),
        ref1=hl.str(ht.alleles1[0]),
        alt1=hl.str(ht.alleles1[1]),

        chrom2=hl.str(ht.locus2.contig),
        pos2=hl.int32(ht.locus2.position),
        ref2=hl.str(ht.alleles2[0]),
        alt2=hl.str(ht.alleles2[1]),

        all_gt_counts_0=hl.int32(all_phase.gt_counts[0]),
        all_gt_counts_1=hl.int32(all_phase.gt_counts[1]),
        all_gt_counts_2=hl.int32(all_phase.gt_counts[2]),
        all_gt_counts_3=hl.int32(all_phase.gt_counts[3]),
        all_gt_counts_4=hl.int32(all_phase.gt_counts[4]),
        all_gt_counts_5=hl.int32(all_phase.gt_counts[5]),
        all_gt_counts_6=hl.int32(all_phase.gt_counts[6]),
        all_gt_counts_7=hl.int32(all_phase.gt_counts[7]),
        all_gt_counts_8=hl.int32(all_phase.gt_counts[8]),

        all_hap_counts_0=hl.float64(all_phase.em.hap_counts[0]),
        all_hap_counts_1=hl.float64(all_phase.em.hap_counts[1]),
        all_hap_counts_2=hl.float64(all_phase.em.hap_counts[2]),
        all_hap_counts_3=hl.float64(all_phase.em.hap_counts[3]),

        all_p_chet=hl.float64(all_phase.em.p_chet),
        all_same_haplotype=hl.bool(all_phase.em.same_haplotype),
        all_different_haplotype=hl.bool(all_phase.em.different_haplotype),
    )


def read_pairs_tsv(tsv_path):
    compression = "gzip" if str(tsv_path).endswith((".gz", ".bgz")) else None
    return pd.read_csv(tsv_path, sep="\t", dtype=str, compression=compression)


def make_unique_variants_txt(tsv_path, out_txt):
    df = read_pairs_tsv(tsv_path)

    needed = ["chrom1", "pos1", "ref1", "alt1", "chrom2", "pos2", "ref2", "alt2"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{tsv_path} is missing required column: {c}")

    left = df[["chrom1", "pos1", "ref1", "alt1"]].rename(columns={
        "chrom1": "chrom", "pos1": "pos", "ref1": "ref", "alt1": "alt"
    })
    right = df[["chrom2", "pos2", "ref2", "alt2"]].rename(columns={
        "chrom2": "chrom", "pos2": "pos", "ref2": "ref", "alt2": "alt"
    })

    uni = pd.concat([left, right], axis=0).drop_duplicates()

    for col in ["chrom", "pos", "ref", "alt"]:
        uni[col] = uni[col].astype(str).str.strip()

    uni = uni[(uni["ref"] != "") & (uni["alt"] != "")]
    uni = uni.sort_values(["chrom", "pos", "ref", "alt"])
    uni.to_csv(out_txt, sep="\t", index=False)


def chrom_sort_key(chrom):
    chrom = str(chrom).replace("chr", "")
    chrom_map = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    if chrom in chrom_map:
        return (0, chrom_map[chrom])
    try:
        return (0, int(chrom))
    except ValueError:
        return (1, chrom)


def make_vcf_from_unique_txt(in_txt, out_vcf):
    df = pd.read_csv(in_txt, sep="\t", dtype=str)

    needed = ["chrom", "pos", "ref", "alt"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"{in_txt} is missing required column: {c}")

    records = []
    for _, row in df.iterrows():
        chrom = str(row["chrom"]).strip()
        pos = int(str(row["pos"]).strip())
        ref = str(row["ref"]).strip()
        alt = str(row["alt"]).strip()

        if ref == "" or alt == "":
            raise ValueError(f"Empty REF/ALT at {chrom}:{pos}")

        records.append((chrom, pos, ref, alt))

    records = sorted(records, key=lambda x: (chrom_sort_key(x[0]), x[1], x[2], x[3]))

    with open(out_vcf, "w") as w:
        w.write("##fileformat=VCFv4.2\n")
        w.write("##source=extract_pairs_and_make_vcfs_all_genes_chunked.py\n")
        w.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, ref, alt in records:
            w.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\n")


def select_chunk(df, task_id, chunk_size):
    start = task_id * chunk_size
    end = min(start + chunk_size, len(df))
    if start >= len(df):
        return df.iloc[0:0].copy()
    return df.iloc[start:end].copy()


def main():
    args = parse_args()

    out_base = Path(args.out_base)
    ensure_dir(out_base)
    ensure_dir(out_base / "logs")
    ensure_dir(out_base / "cooccurrence_chunk_summaries")

    print("===== CONFIG =====", flush=True)
    print(f"OUT_BASE: {out_base}", flush=True)
    print(f"COOCC_HT: {args.coocc_ht}", flush=True)
    print(f"GENE_INTERVALS: {args.gene_intervals}", flush=True)
    print(f"TASK_ID: {args.task_id}", flush=True)
    print(f"CHUNK_SIZE: {args.chunk_size}", flush=True)
    print(f"OVERWRITE: {args.overwrite}", flush=True)

    genes = pd.read_csv(args.gene_intervals, sep="\t", dtype=str)
    if not {"gene", "interval"}.issubset(genes.columns):
        raise ValueError("gene_intervals TSV must contain columns: gene and interval")

    genes = genes.dropna(subset=["gene", "interval"]).copy()
    genes["gene"] = genes["gene"].astype(str)
    genes["interval"] = genes["interval"].astype(str)

    # Keep only genes that have a folder in OUT_BASE.
    genes = genes[genes["gene"].apply(lambda g: (out_base / g).is_dir())].copy()
    genes = genes.sort_values("gene").reset_index(drop=True)

    chunk = select_chunk(genes, args.task_id, args.chunk_size)

    print(f"[INFO] Total genes with folders and intervals: {len(genes)}", flush=True)
    print(f"[INFO] Genes in this chunk: {len(chunk)}", flush=True)

    if len(chunk) == 0:
        print("[INFO] Empty chunk. Exiting safely.", flush=True)
        return

    print("Initializing Hail with GRCh37...", flush=True)
    hl.init(default_reference="GRCh37")

    print(f"Reading Hail table: {args.coocc_ht}", flush=True)
    ht = hl.read_table(args.coocc_ht)

    success_rows = []
    fail_rows = []

    for idx, row in chunk.iterrows():
        gene = row["gene"]
        interval_str = row["interval"]

        print("\n" + "=" * 70, flush=True)
        print(f"Processing gene: {gene}", flush=True)
        print(f"Interval: {interval_str}", flush=True)

        gene_dir = out_base / gene
        ensure_dir(gene_dir)

        pairs_out = gene_dir / f"{gene}_pairs_from_cooccurrence.tsv.bgz"
        uniq_out = gene_dir / f"{gene}_unique_variants.txt"
        vcf_out = gene_dir / f"{gene}_unique_variants.vcf"

        if pairs_out.exists() and uniq_out.exists() and vcf_out.exists() and not args.overwrite:
            print(f"[INFO] Existing outputs found for {gene}; skipping.", flush=True)
            success_rows.append({
                "gene": gene,
                "interval": interval_str,
                "n_rows_in_hail_subset": None,
                "n_unique_variants": None,
                "pairs_tsv_bgz": str(pairs_out),
                "unique_variants_txt": str(uniq_out),
                "unique_variants_vcf": str(vcf_out),
                "status": "SKIPPED_EXISTS"
            })
            continue

        try:
            interval = hl.parse_locus_interval(interval_str, reference_genome="GRCh37")

            ht_gene = interval_contains_both_variant(ht, interval)
            n_rows = ht_gene.count()
            print(f"Rows matching interval for {gene}: {n_rows}", flush=True)

            if n_rows == 0:
                raise RuntimeError(f"No rows found in Hail table for interval {interval_str}")

            ht_export = make_export_table(ht_gene)

            print(f"Exporting pairs to: {pairs_out}", flush=True)
            ht_export.export(str(pairs_out))

            print(f"Creating unique variant list: {uniq_out}", flush=True)
            make_unique_variants_txt(str(pairs_out), str(uniq_out))

            n_unique = sum(1 for _ in open(uniq_out)) - 1
            print(f"Unique variants for {gene}: {n_unique}", flush=True)

            if n_unique <= 0:
                raise RuntimeError(f"No unique variants found after export for {gene}")

            print(f"Creating VCF: {vcf_out}", flush=True)
            make_vcf_from_unique_txt(str(uniq_out), str(vcf_out))

            success_rows.append({
                "gene": gene,
                "interval": interval_str,
                "n_rows_in_hail_subset": n_rows,
                "n_unique_variants": n_unique,
                "pairs_tsv_bgz": str(pairs_out),
                "unique_variants_txt": str(uniq_out),
                "unique_variants_vcf": str(vcf_out),
                "status": "OK"
            })

            print(f"Finished successfully: {gene}", flush=True)

        except Exception as e:
            tb = traceback.format_exc()
            fail_rows.append({
                "gene": gene,
                "interval": interval_str,
                "status": "FAILED",
                "error": str(e),
                "traceback": tb
            })
            print(f"FAILED for {gene}", flush=True)
            print(tb, flush=True)

    success_df = pd.DataFrame(success_rows)
    fail_df = pd.DataFrame(fail_rows)

    success_path = out_base / "cooccurrence_chunk_summaries" / f"chunk_{args.task_id}_success.tsv"
    fail_path = out_base / "cooccurrence_chunk_summaries" / f"chunk_{args.task_id}_failed.tsv"

    success_df.to_csv(success_path, sep="\t", index=False)
    fail_df.to_csv(fail_path, sep="\t", index=False)

    print("\nDone.", flush=True)
    print(f"Success summary: {success_path}", flush=True)
    print(f"Failure summary: {fail_path}", flush=True)


if __name__ == "__main__":
    main()
