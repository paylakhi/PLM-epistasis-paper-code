#!/usr/bin/env python3
"""
Attach gnomAD co-occurrence metadata to valid double-mutant PLM inputs.

For each gene directory under --out-base, this script joins:
  {GENE}_double_muts_for_esm_VALID.tsv
with:
  {GENE}_pairs_from_cooccurrence.tsv.bgz
and VEP-derived missense annotations.

The output file per gene is:
  {GENE}_double_muts_for_esm_VALID_with_metadata.tsv
"""

import os
import glob
import argparse
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
    compression = "gzip" if path.endswith((".gz", ".bgz")) else None
    return pd.read_csv(path, sep="\t", dtype=str, compression=compression)


def variant_key(chrom, pos, ref, alt):
    chrom = str(chrom).replace("chr", "").strip()
    return f"{chrom}:{str(pos).strip()}:{str(ref).strip()}:{str(alt).strip()}"


def norm_mut(x):
    return (
        str(x)
        .replace("p.", "")
        .replace("(", "")
        .replace(")", "")
        .replace("%3D", "=")
        .strip()
    )


def protein_pair_key(m1, m2):
    return "||".join(sorted([norm_mut(m1), norm_mut(m2)]))


def parse_vep_mutation_map(vep_vcf, gene):
    """
    Parse existing VEP VCF and keep only:
      - SYMBOL == gene
      - missense_variant
      - preferably CANONICAL == YES

    Returns:
      genomic variant key -> protein mutation like M1L
    """

    csq_fields = None
    mapping = {}

    with open(vep_vcf) as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("##INFO=<ID=CSQ"):
                fmt = line.split("Format: ")[1].split('">')[0]
                csq_fields = fmt.split("|")
                continue

            if line.startswith("#"):
                continue

            if csq_fields is None:
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            chrom, pos, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            info = parts[7]

            gkey = variant_key(chrom, pos, ref, alt)

            csq_block = None
            for item in info.split(";"):
                if item.startswith("CSQ="):
                    csq_block = item.replace("CSQ=", "")
                    break

            if csq_block is None:
                continue

            best_mut = None
            fallback_mut = None

            for csq in csq_block.split(","):
                vals = csq.split("|")
                rec = dict(zip(csq_fields, vals))

                symbol = rec.get("SYMBOL", "")
                consequence = rec.get("Consequence", "")
                canonical = rec.get("CANONICAL", "")
                aa = rec.get("Amino_acids", "")
                protein_pos = rec.get("Protein_position", "")

                if symbol != gene:
                    continue

                if "missense_variant" not in consequence:
                    continue

                if "/" not in aa:
                    continue

                if protein_pos == "":
                    continue

                aa_ref, aa_alt = aa.split("/")[:2]

                if aa_ref == "" or aa_alt == "":
                    continue

                mut = norm_mut(f"{aa_ref}{protein_pos}{aa_alt}")

                if canonical == "YES":
                    best_mut = mut
                    break

                if fallback_mut is None:
                    fallback_mut = mut

            if best_mut is not None:
                mapping[gkey] = best_mut
            elif fallback_mut is not None:
                mapping[gkey] = fallback_mut

    return mapping


def process_gene(gene_dir, overwrite=False):
    gene = os.path.basename(gene_dir)

    valid_file = os.path.join(gene_dir, f"{gene}_double_muts_for_esm_VALID.tsv")
    pair_file = os.path.join(gene_dir, f"{gene}_pairs_from_cooccurrence.tsv.bgz")
    vep_file = os.path.join(gene_dir, f"{gene}_annotated.vep.vcf")
    out_file = os.path.join(gene_dir, f"{gene}_double_muts_for_esm_VALID_with_metadata.tsv")

    if os.path.exists(out_file) and os.path.getsize(out_file) > 0 and not overwrite:
        return gene, "SKIPPED_EXISTS", "already exists"

    if not os.path.exists(valid_file):
        return gene, "MISSING", "missing VALID file"

    if not os.path.exists(pair_file):
        return gene, "MISSING", "missing pairs_from_cooccurrence file"

    if not os.path.exists(vep_file):
        return gene, "MISSING", "missing annotated VEP file"

    valid = read_tsv(valid_file)
    pairs = read_tsv(pair_file)

    if not {"mut1", "mut2"}.issubset(valid.columns):
        raise ValueError(
            f"VALID file does not have mut1/mut2 columns. Columns: {list(valid.columns)}"
        )

    required_pair_cols = {
        "chrom1", "pos1", "ref1", "alt1",
        "chrom2", "pos2", "ref2", "alt2"
    }

    if not required_pair_cols.issubset(pairs.columns):
        raise ValueError(
            f"Pair file missing required columns. Needed: {required_pair_cols}. "
            f"Found: {list(pairs.columns)}"
        )

    mut_map = parse_vep_mutation_map(vep_file, gene)

    if len(mut_map) == 0:
        if os.path.exists(out_file):
            os.remove(out_file)
        return gene, "NO_MISSENSE_MAP", "no gene-specific missense variants found in VEP"

    pairs["variant_key_1_meta"] = pairs.apply(
        lambda r: variant_key(r["chrom1"], r["pos1"], r["ref1"], r["alt1"]),
        axis=1
    )

    pairs["variant_key_2_meta"] = pairs.apply(
        lambda r: variant_key(r["chrom2"], r["pos2"], r["ref2"], r["alt2"]),
        axis=1
    )

    pairs["mut1_meta"] = pairs["variant_key_1_meta"].map(mut_map)
    pairs["mut2_meta"] = pairs["variant_key_2_meta"].map(mut_map)

    pairs = pairs.dropna(subset=["mut1_meta", "mut2_meta"]).copy()

    if len(pairs) == 0:
        if os.path.exists(out_file):
            os.remove(out_file)
        return gene, "NO_MISSENSE_PAIRS", "no pairs where both variants are gene-specific missense"

    pairs["protein_pair_key"] = pairs.apply(
        lambda r: protein_pair_key(r["mut1_meta"], r["mut2_meta"]),
        axis=1
    )

    valid["protein_pair_key"] = valid.apply(
        lambda r: protein_pair_key(r["mut1"], r["mut2"]),
        axis=1
    )

    meta_cols = [
        "protein_pair_key",
        "variant_key_1_meta", "variant_key_2_meta",
        "mut1_meta", "mut2_meta",
        "chrom1", "pos1", "ref1", "alt1",
        "chrom2", "pos2", "ref2", "alt2",
        "all_gt_counts_0", "all_gt_counts_1", "all_gt_counts_2",
        "all_gt_counts_3", "all_gt_counts_4", "all_gt_counts_5",
        "all_gt_counts_6", "all_gt_counts_7", "all_gt_counts_8",
        "all_hap_counts_0", "all_hap_counts_1",
        "all_hap_counts_2", "all_hap_counts_3",
        "all_p_chet",
        "all_same_haplotype",
        "all_different_haplotype",
    ]

    meta_cols = [c for c in meta_cols if c in pairs.columns]

    pairs_meta = pairs[meta_cols].drop_duplicates("protein_pair_key")

    merged = valid.merge(
        pairs_meta,
        on="protein_pair_key",
        how="left"
    )

    n_total = len(merged)
    n_matched = merged["chrom1"].notna().sum() if "chrom1" in merged.columns else 0

    if n_matched == 0:
        if os.path.exists(out_file):
            os.remove(out_file)
        return (
            gene,
            "NO_VALID_MATCH",
            f"rows={n_total}; matched_metadata=0; VEP missense mutations did not match VALID protein mutations"
        )

    merged.to_csv(out_file, sep="\t", index=False)

    return gene, "OK", f"rows={n_total}; matched_metadata={n_matched}"


def main():
    args = parse_args()
    out_base = args.out_base

    summary_dir = os.path.join(out_base, "metadata_merge_summaries")
    os.makedirs(summary_dir, exist_ok=True)

    gene_dirs = sorted([
        d for d in glob.glob(os.path.join(out_base, "*"))
        if os.path.isdir(d)
        and os.path.basename(d) not in {
            "logs",
            "qc_reports",
            "results",
            "env",
            "chunk_summaries",
            "cooccurrence_chunk_summaries",
            "metadata_merge_summaries",
            "epistasis_metadata_merge_summaries",
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
    summary_file = os.path.join(summary_dir, f"metadata_merge_chunk_{args.task_id}.tsv")
    summary.to_csv(summary_file, sep="\t", index=False)

    print("Done.", flush=True)
    print("Summary:", summary_file, flush=True)


if __name__ == "__main__":
    main()
