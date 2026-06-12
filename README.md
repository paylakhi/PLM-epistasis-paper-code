# PLM Epistasis Paper Code

This repository contains analysis scripts for the proof-of-principle PLM epistasis analyses.

The main manuscript analyses use Synthyra/ESM2-650M. The 35M model analyses are included as supplementary/scalability analyses.

## Pipeline

1. Extract phased variant pairs from gnomAD.
2. Generate per-gene VCF files.
3. Annotate variants using VEP.
4. Construct double-mutant protein inputs.
5. Compute PLM single and pairwise epistasis scores.
6. Merge epistasis scores with metadata.
7. Validate single-mutation scores using ClinVar, SIFT, and PolyPhen.
8. Analyze structural contacts using AlphaFold-derived structures.
9. Analyze gnomAD cis co-occurrence and allele-frequency patterns.

## Data

Large datasets are not included. Users must obtain gnomAD, VEP cache files, and AlphaFold structure files separately.

## Example

```bash
source configs/config_650M.env
python scripts/04_run_all_genes_epistasis_esmcnd_chunked.py
