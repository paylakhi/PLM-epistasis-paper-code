#!/usr/bin/env python3
"""
Run conditional epistasis / non-additivity scoring for ALL genes in chunks.

Expected per gene:
  {BASE_DIR}/{GENE}/{GENE}_canonical_protein.fasta
  {BASE_DIR}/{GENE}/{GENE}_double_muts_for_esm_VALID_with_metadata.tsv
    OR
  {BASE_DIR}/{GENE}/{GENE}_double_muts_for_esm_VALID.tsv

Writes per gene:
  {BASE_DIR}/{GENE}/{GENE}_epistasis_esmcnd.tsv
  {BASE_DIR}/{GENE}/{GENE}_epistasis_skipped_rows.tsv

Writes per chunk:
  {BASE_DIR}/chunk_summaries/chunk_{TASK_ID}_epistasis_summary.tsv
  {BASE_DIR}/chunk_summaries/chunk_{TASK_ID}_epistasis_failures.tsv

This script is designed for GPU-based execution and can be run as a
SLURM array job. Each array task processes one chunk of genes.
"""

import os
import re
import math
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM


# -----------------------------
# Configuration from environment
# -----------------------------
BASE_DIR = os.environ.get(
    "BASE_DIR",
    "."
)

MODEL_NAME = os.environ.get("MODEL_NAME", "Synthyra/ESM2-650M")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("TASK_ID", "0")))

START_BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
MIN_BATCH_SIZE = int(os.environ.get("MIN_BATCH_SIZE", "4"))

NROWS_ENV = os.environ.get("NROWS", "").strip()
NROWS = int(NROWS_ENV) if NROWS_ENV else None

GENE_LIST_PATH = os.environ.get("GENE_LIST", "").strip()
OVERWRITE = os.environ.get("OVERWRITE", "0").strip() == "1"

TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "0").strip() == "1"
ATTN_BACKEND = os.environ.get("ATTN_BACKEND", "sdpa")

DEVICE = "cuda"

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


# -----------------------------
# Basic helpers
# -----------------------------
def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. This job must run on a GPU node.")


def read_single_fasta(path: str) -> str:
    seq_lines = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                seq_lines.append(line.strip())
    seq = "".join(seq_lines).strip().upper()
    if not seq:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return seq


def parse_mut(m: str) -> Tuple[str, int, str]:
    m = str(m).strip()
    mm = MUT_RE.match(m)
    if not mm:
        raise ValueError(f"Bad mutation format: {m}")
    return mm.group(1), int(mm.group(2)), mm.group(3)


def parse_double_mut(s: str) -> Tuple[str, str]:
    s = str(s).strip()
    for sep in [":", ",", ";"]:
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
    raise ValueError(f"Bad double_mut format: {s}")


def apply_mut(seq: str, pos: int, alt: str) -> str:
    i = pos - 1
    if i < 0 or i >= len(seq):
        raise ValueError(f"Position {pos} out of range for sequence length {len(seq)}")
    return seq[:i] + alt + seq[i + 1:]


def verify_mut_against_seq(seq: str, mut: str, seq_name: str = "sequence") -> None:
    wt, pos, _alt = parse_mut(mut)
    i = pos - 1
    if i < 0 or i >= len(seq):
        raise ValueError(f"{mut}: position {pos} out of range for {seq_name} length {len(seq)}")
    seq_wt = seq[i]
    if seq_wt != wt:
        raise ValueError(
            f"{mut}: WT amino acid mismatch in {seq_name}. Expected {wt} at position {pos}, found {seq_wt}"
        )


def build_context_seq(wt_seq: str, bg_mut: Optional[str]) -> str:
    if bg_mut is None:
        return wt_seq
    verify_mut_against_seq(wt_seq, bg_mut, seq_name="WT sequence")
    _, pos, alt = parse_mut(bg_mut)
    return apply_mut(wt_seq, pos, alt)


def build_masked_sequence(context_seq: str, target_mut: str, mask_token: str) -> Tuple[str, str, str, int]:
    wt, pos, alt = parse_mut(target_mut)
    i = pos - 1
    if i < 0 or i >= len(context_seq):
        raise ValueError(f"{target_mut}: position {pos} out of range for context length {len(context_seq)}")
    if context_seq[i] != wt:
        raise ValueError(
            f"{target_mut}: WT amino acid mismatch in context. Expected {wt} at position {pos}, found {context_seq[i]}"
        )
    masked_seq = context_seq[:i] + mask_token + context_seq[i + 1:]
    return masked_seq, wt, alt, pos


def chunked(lst: List, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or isinstance(exc, torch.cuda.OutOfMemoryError)


def find_existing_file(gene_dir: str, gene: str, suffixes: List[str]) -> Optional[str]:
    if not os.path.isdir(gene_dir):
        return None
    candidates = os.listdir(gene_dir)
    for suffix in suffixes:
        target = f"{gene}{suffix}".lower()
        for fn in candidates:
            if fn.lower() == target:
                return os.path.join(gene_dir, fn)
    return None


def discover_genes(base_dir: str, gene_list_path: str) -> List[str]:
    if gene_list_path:
        with open(gene_list_path) as f:
            genes = [x.strip() for x in f if x.strip()]
    else:
        gene_list_default = os.path.join(base_dir, "all_genes_gene_list.txt")
        if os.path.exists(gene_list_default):
            with open(gene_list_default) as f:
                genes = [x.strip() for x in f if x.strip()]
        else:
            genes = sorted(
                d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d))
                and d not in {"logs", "chunk_summaries"}
            )

    # Keep only genes with folders, avoid accidental non-gene folders.
    genes = [g for g in genes if os.path.isdir(os.path.join(base_dir, g))]
    return sorted(set(genes))


def select_chunk(genes: List[str], task_id: int, chunk_size: int) -> List[str]:
    start = task_id * chunk_size
    end = min(start + chunk_size, len(genes))
    if start >= len(genes):
        return []
    return genes[start:end]


# -----------------------------
# Scoring functions
# -----------------------------
def filter_valid_rows(df: pd.DataFrame, wt_seq: str, gene: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows = []
    skipped = []

    for idx, row in df.iterrows():
        dm = row["double_mut"]
        try:
            m1, m2 = parse_double_mut(dm)
            _, pos1, _ = parse_mut(m1)
            _, pos2, _ = parse_mut(m2)

            if pos1 == pos2:
                skipped.append({
                    "gene": gene,
                    "row_index": idx,
                    "double_mut": dm,
                    "reason": "same_position",
                    "detail": f"{m1} and {m2}"
                })
                continue

            verify_mut_against_seq(wt_seq, m1, seq_name="WT sequence")
            verify_mut_against_seq(wt_seq, m2, seq_name="WT sequence")

            tmp = row.to_dict()
            tmp["mut1"] = m1
            tmp["mut2"] = m2
            valid_rows.append(tmp)

        except Exception as e:
            skipped.append({
                "gene": gene,
                "row_index": idx,
                "double_mut": dm,
                "reason": "validation_error",
                "detail": str(e)
            })

    return pd.DataFrame(valid_rows), pd.DataFrame(skipped)


def score_requests_in_batches_once(
    requests: List[Dict],
    tokenizer,
    model,
    batch_size: int,
    aa_token_ids: Dict[str, int],
    mask_token_id: int
) -> Dict[Tuple[Optional[str], str], float]:
    score_map: Dict[Tuple[Optional[str], str], float] = {}

    total_batches = math.ceil(len(requests) / batch_size)
    for batch_idx, batch in enumerate(chunked(requests, batch_size), start=1):
        seqs = [x["masked_seq"] for x in batch]
        enc = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=False
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**enc)

        logits = outputs.logits
        input_ids = enc["input_ids"]

        for i, req in enumerate(batch):
            mask_positions = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) != 1:
                raise ValueError(f"Expected exactly 1 mask token for request {req['key']}, found {len(mask_positions)}")

            mask_pos = mask_positions.item()
            wt_id = aa_token_ids[req["wt_aa"]]
            mut_id = aa_token_ids[req["mut_aa"]]

            log_probs = torch.log_softmax(logits[i, mask_pos], dim=-1)
            delta = (log_probs[mut_id] - log_probs[wt_id]).item()
            score_map[req["key"]] = float(delta)

        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            print(f"[INFO] Finished model batch {batch_idx}/{total_batches} at batch_size={batch_size}", flush=True)

    return score_map


def score_requests_with_fallback(
    requests: List[Dict],
    tokenizer,
    model,
    start_batch_size: int,
    min_batch_size: int,
    aa_token_ids: Dict[str, int],
    mask_token_id: int
) -> Tuple[Dict[Tuple[Optional[str], str], float], int]:
    batch_size = start_batch_size

    while batch_size >= min_batch_size:
        try:
            print(f"[INFO] Trying batch_size={batch_size}", flush=True)
            score_map = score_requests_in_batches_once(
                requests=requests,
                tokenizer=tokenizer,
                model=model,
                batch_size=batch_size,
                aa_token_ids=aa_token_ids,
                mask_token_id=mask_token_id
            )
            return score_map, batch_size

        except RuntimeError as e:
            if is_cuda_oom(e):
                print(f"[WARN] CUDA OOM at batch_size={batch_size}; retrying with half", flush=True)
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
                continue
            raise

    raise RuntimeError(f"All batch sizes failed with CUDA OOM down to MIN_BATCH_SIZE={min_batch_size}")


def process_gene(gene: str, tokenizer, model) -> Dict:
    gene_dir = os.path.join(BASE_DIR, gene)

    fasta_path = find_existing_file(
        gene_dir, gene,
        ["_canonical_protein.fasta", "_canonical_protein.fa", ".fasta", ".fa"]
    )
    pairs_path = find_existing_file(
        gene_dir, gene,
        ["_double_muts_for_esm_VALID_with_metadata.tsv", "_double_muts_for_esm_VALID.tsv"]
    )

    out_path = os.path.join(gene_dir, f"{gene}_epistasis_esmcnd.tsv")
    skipped_path = os.path.join(gene_dir, f"{gene}_epistasis_skipped_rows.tsv")

    if fasta_path is None:
        return {"gene": gene, "status": "SKIPPED_NO_FASTA", "error": ""}
    if pairs_path is None:
        return {"gene": gene, "status": "SKIPPED_NO_PAIRS", "error": ""}
    if (not OVERWRITE) and os.path.exists(out_path):
        return {"gene": gene, "status": "SKIPPED_ALREADY_EXISTS", "error": "", "out_path": out_path}

    print("=" * 100, flush=True)
    print(f"[INFO] Processing gene: {gene}", flush=True)
    print(f"[INFO] FASTA: {fasta_path}", flush=True)
    print(f"[INFO] PAIRS: {pairs_path}", flush=True)

    wt_seq = read_single_fasta(fasta_path)
    df = pd.read_csv(pairs_path, sep="\t", nrows=NROWS, dtype=str)

    if "double_mut" not in df.columns:
        raise ValueError(f"{pairs_path} must contain column 'double_mut'")

    df = df.dropna(subset=["double_mut"]).copy()
    n_input_rows = len(df)

    if n_input_rows == 0:
        return {"gene": gene, "status": "SKIPPED_ZERO_PAIRS", "error": ""}

    valid_df, skipped_df = filter_valid_rows(df, wt_seq, gene)

    if len(skipped_df) > 0:
        skipped_df.to_csv(skipped_path, sep="\t", index=False)

    if len(valid_df) == 0:
        return {
            "gene": gene,
            "status": "SKIPPED_NO_VALID_ROWS",
            "error": "",
            "n_input_rows": n_input_rows,
            "n_valid_rows": 0,
            "n_skipped_rows": len(skipped_df)
        }

    mask_token = tokenizer.mask_token
    mask_token_id = tokenizer.mask_token_id
    if mask_token is None or mask_token_id is None:
        raise ValueError("Tokenizer does not have a mask token.")

    aa_token_ids = {}
    for aa in AA_ALPHABET:
        tok_id = tokenizer.convert_tokens_to_ids(aa)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer could not map amino acid token: {aa}")
        aa_token_ids[aa] = tok_id

    parsed_rows = []
    request_keys = set()

    for _, row in valid_df.iterrows():
        m1 = row["mut1"]
        m2 = row["mut2"]

        parsed_rows.append((m1, m2))
        request_keys.add((None, m1))
        request_keys.add((None, m2))
        request_keys.add((m2, m1))
        request_keys.add((m1, m2))

    request_keys = sorted(request_keys, key=lambda x: ("" if x[0] is None else x[0], x[1]))

    requests = []
    for bg_mut, target_mut in request_keys:
        context_seq = build_context_seq(wt_seq, bg_mut)
        masked_seq, wt_aa, mut_aa, pos = build_masked_sequence(context_seq, target_mut, mask_token)
        requests.append({
            "key": (bg_mut, target_mut),
            "masked_seq": masked_seq,
            "wt_aa": wt_aa,
            "mut_aa": mut_aa,
            "pos": pos
        })

    score_map, used_batch_size = score_requests_with_fallback(
        requests=requests,
        tokenizer=tokenizer,
        model=model,
        start_batch_size=START_BATCH_SIZE,
        min_batch_size=MIN_BATCH_SIZE,
        aa_token_ids=aa_token_ids,
        mask_token_id=mask_token_id
    )

    d1_list = []
    d2_list = []
    d1_given_m2_list = []
    d2_given_m1_list = []
    additive_score_list = []
    contextual_double_score_list = []
    econd_list = []

    for m1, m2 in parsed_rows:
        d1 = score_map[(None, m1)]
        d2 = score_map[(None, m2)]
        d1_given_m2 = score_map[(m2, m1)]
        d2_given_m1 = score_map[(m1, m2)]

        additive_score = d1 + d2
        contextual_double_score = d1_given_m2 + d2_given_m1
        e_cond = (d1 - d1_given_m2) + (d2 - d2_given_m1)

        d1_list.append(d1)
        d2_list.append(d2)
        d1_given_m2_list.append(d1_given_m2)
        d2_given_m1_list.append(d2_given_m1)
        additive_score_list.append(additive_score)
        contextual_double_score_list.append(contextual_double_score)
        econd_list.append(e_cond)

    out_df = valid_df.copy()
    out_df["gene"] = gene
    out_df["d1"] = d1_list
    out_df["d2"] = d2_list
    out_df["d1_given_m2"] = d1_given_m2_list
    out_df["d2_given_m1"] = d2_given_m1_list
    out_df["additive_score"] = additive_score_list
    out_df["contextual_double_score"] = contextual_double_score_list
    out_df["pair_model_score"] = contextual_double_score_list
    out_df["E_cond"] = econd_list

    out_df.to_csv(out_path, sep="\t", index=False)

    reason_counts = {}
    if len(skipped_df) > 0:
        reason_counts = skipped_df["reason"].value_counts().to_dict()

    return {
        "gene": gene,
        "status": "OK",
        "error": "",
        "fasta_path": fasta_path,
        "pairs_path": pairs_path,
        "out_path": out_path,
        "skipped_path": skipped_path if len(skipped_df) > 0 else "",
        "seq_len": len(wt_seq),
        "n_input_rows": n_input_rows,
        "n_valid_rows": len(valid_df),
        "n_skipped_rows": len(skipped_df),
        "n_skipped_same_position": int(reason_counts.get("same_position", 0)),
        "n_skipped_validation_error": int(reason_counts.get("validation_error", 0)),
        "n_unique_requests": len(requests),
        "used_batch_size": used_batch_size,
        "mean_E_cond": float(out_df["E_cond"].mean()),
        "min_E_cond": float(out_df["E_cond"].min()),
        "max_E_cond": float(out_df["E_cond"].max()),
    }


# -----------------------------
# Main
# -----------------------------
def load_model_and_tokenizer():
    print(f"[INFO] Loading model once per array task: {MODEL_NAME}", flush=True)

    if MODEL_NAME.startswith("Synthyra/"):
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        config.attn_backend = ATTN_BACKEND

        model = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME,
            config=config,
            dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = model.tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16
        )

    model.to(DEVICE)
    model.eval()

    if TORCH_COMPILE:
        print("[INFO] Compiling model with torch.compile()", flush=True)
        model = torch.compile(model)

    return tokenizer, model


def main():
    require_gpu()

    print("===== CONFIG =====", flush=True)
    print(f"BASE_DIR: {BASE_DIR}", flush=True)
    print(f"MODEL_NAME: {MODEL_NAME}", flush=True)
    print(f"DEVICE: {DEVICE}", flush=True)
    print(f"TASK_ID: {TASK_ID}", flush=True)
    print(f"CHUNK_SIZE: {CHUNK_SIZE}", flush=True)
    print(f"START_BATCH_SIZE: {START_BATCH_SIZE}", flush=True)
    print(f"MIN_BATCH_SIZE: {MIN_BATCH_SIZE}", flush=True)
    print(f"NROWS: {NROWS}", flush=True)
    print(f"GENE_LIST: {GENE_LIST_PATH if GENE_LIST_PATH else '[BASE_DIR/all_genes_gene_list.txt or all subdirs]'}", flush=True)
    print(f"OVERWRITE: {OVERWRITE}", flush=True)

    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "chunk_summaries"), exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    all_genes = discover_genes(BASE_DIR, GENE_LIST_PATH)
    genes = select_chunk(all_genes, TASK_ID, CHUNK_SIZE)

    print(f"[INFO] Total discovered genes: {len(all_genes)}", flush=True)
    print(f"[INFO] This array task processes chunk {TASK_ID}: {len(genes)} genes", flush=True)

    if len(genes) == 0:
        print("[INFO] Empty chunk. Exiting safely.", flush=True)
        return

    print(f"[INFO] First gene in chunk: {genes[0]}", flush=True)
    print(f"[INFO] Last gene in chunk:  {genes[-1]}", flush=True)

    tokenizer, model = load_model_and_tokenizer()

    summary_rows = []
    failure_rows = []

    for gene in genes:
        try:
            result = process_gene(gene, tokenizer, model)
            if result["status"] == "OK":
                summary_rows.append(result)
            else:
                failure_rows.append(result)
                print(f"[WARN] {gene}: {result['status']}", flush=True)
        except Exception as e:
            failure_rows.append({
                "gene": gene,
                "status": "FAILED",
                "error": str(e)
            })
            print(f"[ERROR] {gene}: {e}", flush=True)
        finally:
            torch.cuda.empty_cache()

    summary_path = os.path.join(BASE_DIR, "chunk_summaries", f"chunk_{TASK_ID}_epistasis_summary.tsv")
    failures_path = os.path.join(BASE_DIR, "chunk_summaries", f"chunk_{TASK_ID}_epistasis_failures.tsv")

    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame(failure_rows).to_csv(failures_path, sep="\t", index=False)

    print(f"[INFO] Chunk summary saved:  {summary_path}", flush=True)
    print(f"[INFO] Chunk failures saved: {failures_path}", flush=True)
    print("[INFO] Done.", flush=True)


if __name__ == "__main__":
    main()
