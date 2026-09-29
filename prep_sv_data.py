#!/usr/bin/env python3
"""
Prepare a large structural-variant table for the SV-set model.

What this script does
---------------------
1. Validates/cleans the raw SV table.
2. Normalizes genomic coordinates by chromosome length.
3. Normalizes the supplied gender and inheritance annotations.
4. Applies deterministic log transforms to skewed numeric annotations.
5. Encodes chromosome, SV type, nearest gene, and overlapping gene names.
6. Saves:
   - normalized SV-level CSV
   - gene vocabulary CSV
   - JSON manifest describing the preprocessing
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


CHR_LENGTHS = {
    "hg38": {
        "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
        "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
        "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
        "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
        "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
        "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
        "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
        "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
        "chrM": 16569,
    },
    "hg19": {
        "chr1": 249250621, "chr2": 243199373, "chr3": 198022430,
        "chr4": 191154276, "chr5": 180915260, "chr6": 171115067,
        "chr7": 159138663, "chr8": 146364022, "chr9": 141213431,
        "chr10": 135534747, "chr11": 135006516, "chr12": 133851895,
        "chr13": 115169878, "chr14": 107349540, "chr15": 102531392,
        "chr16": 90354753, "chr17": 81195210, "chr18": 78077248,
        "chr19": 59128983, "chr20": 63025520, "chr21": 48129895,
        "chr22": 51304566, "chrX": 155270560, "chrY": 59373566,
        "chrM": 16571,
    },
}

CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
CHR_TO_ID = {chrom: i for i, chrom in enumerate(CHR_ORDER)}
SV_TYPE_TO_ID = {"DEL": 0, "INS": 1, "DUP": 2, "INV": 3}
GENDER_TO_ID = {"unknown": 0, "M": 1, "F": 2}
INHERITANCE_TO_ID = {"unknown": 0, "c": 1, "m": 2, "f": 3, "mf": 4}


def canonical_chrom(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if not s:
        return np.nan
    if not s.lower().startswith("chr"):
        s = "chr" + s
    # normalize X/Y/M casing and numeric prefixes
    body = s[3:]
    if body.upper() in {"X", "Y", "M", "MT"}:
        return "chrM" if body.upper() in {"M", "MT"} else "chr" + body.upper()
    try:
        return f"chr{int(body)}"
    except ValueError:
        return s


def split_genes(value):
    """Split strings like 'TSNAX;TSNAX-DISC1' into unique gene names."""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "."}:
        return []
    parts = re.split(r"[;,|]", s)
    genes = []
    seen = set()
    for p in parts:
        g = p.strip()
        if g and g.lower() not in {"nan", "none", "."} and g not in seen:
            genes.append(g)
            seen.add(g)
    return genes


def to_nonnegative_numeric(series, name):
    out = pd.to_numeric(series, errors="coerce").fillna(0.0)
    bad = out < 0
    if bad.any():
        print(f"[warning] {name}: {(bad).sum()} negative values were clipped to 0.")
        out = out.clip(lower=0)
    return out.astype(float)


def build_gene_vocab(df):
    genes = set()
    if "nearest_gene" in df.columns:
        for x in df["nearest_gene"]:
            genes.update(split_genes(x))
    if "gene_overlap" in df.columns:
        for x in df["gene_overlap"]:
            genes.update(split_genes(x))
    genes = sorted(genes)
    # 0 reserved for unknown/no gene
    return {gene: i + 1 for i, gene in enumerate(genes)}


def canonical_gender(value):
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().upper()
    return value if value in {"M", "F"} else "unknown"


def canonical_inheritance(value):
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().lower()
    if value in {"", ".", "na", "nan", "none", "unknown"}:
        return "unknown"
    return value if value in {"c", "m", "f", "mf"} else "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--genome_build", choices=["hg19", "hg38"], default="hg38")
    args = p.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)
    n_initial = len(df)

    required = [
        "sample_id", "source", "chrom", "start", "end",
        "sv_type", "sv_length", "distance_to_nearest_tss", "gender",
        "inheritance",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Optional metadata columns
    for c in ["family_id", "sv_id", "strand", "nearest_gene", "gene_overlap"]:
        if c not in df.columns:
            df[c] = np.nan

    # Optional overlap counts
    count_cols = [
        "exon_overlap_count",
        "promoter_overlap_count",
        "enhancer_overlap_count",
        "insulator_overlap_count",
    ]
    for c in count_cols:
        if c not in df.columns:
            df[c] = 0

    # Canonicalize source
    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df = df[df["source"].isin([0, 1, 2])].copy()
    df["source"] = df["source"].astype(int)

    # Chromosome cleaning
    df["chrom"] = df["chrom"].map(canonical_chrom)
    lengths = CHR_LENGTHS[args.genome_build]
    valid_chrom = df["chrom"].isin(lengths.keys())
    if (~valid_chrom).any():
        bad = sorted(df.loc[~valid_chrom, "chrom"].dropna().astype(str).unique().tolist())
        print(f"[warning] Dropping {int((~valid_chrom).sum())} rows with unsupported chromosome values: {bad[:10]}")
        df = df[valid_chrom].copy()

    # Coordinates
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df[df["start"].notna() & df["end"].notna()].copy()

    # Swap reversed intervals
    swap = df["start"] > df["end"]
    if swap.any():
        old_start = df.loc[swap, "start"].copy()
        df.loc[swap, "start"] = df.loc[swap, "end"]
        df.loc[swap, "end"] = old_start

    df["start"] = df["start"].clip(lower=0).astype(np.int64)
    df["end"] = df["end"].clip(lower=0).astype(np.int64)

    # SV type
    df["sv_type"] = df["sv_type"].astype(str).str.upper().str.strip()
    valid_sv = df["sv_type"].isin(SV_TYPE_TO_ID)
    if (~valid_sv).any():
        bad = sorted(df.loc[~valid_sv, "sv_type"].unique().tolist())
        print(f"[warning] Dropping {int((~valid_sv).sum())} unsupported SV types: {bad}")
        df = df[valid_sv].copy()

    # Trust the supplied annotations; do not infer them from sample IDs or overlaps.
    raw_gender = df["gender"].copy()
    raw_inheritance = df["inheritance"].copy()
    df["gender"] = raw_gender.map(canonical_gender)
    df["inheritance"] = raw_inheritance.map(canonical_inheritance)
    n_bad_gender = int((raw_gender.notna() & df["gender"].eq("unknown")).sum())
    expected_missing = raw_inheritance.astype(str).str.strip().str.lower().isin(
        ["", ".", "na", "nan", "none", "unknown"]
    )
    n_bad_inheritance = int(
        (raw_inheritance.notna() & df["inheritance"].eq("unknown") &
         ~expected_missing).sum()
    )
    if n_bad_gender:
        print(f"[warning] Mapped {n_bad_gender:,} unrecognized gender values to unknown.")
    if n_bad_inheritance:
        print(f"[warning] Mapped {n_bad_inheritance:,} unrecognized inheritance values to unknown.")
    df["gender_id"] = df["gender"].map(GENDER_TO_ID).astype(np.int8)
    df["inheritance_id"] = df["inheritance"].map(
        INHERITANCE_TO_ID
    ).astype(np.int8)
    inheritance_flags = {
        "inheritance_child_only": "c",
        "inheritance_maternal": "m",
        "inheritance_paternal": "f",
        "inheritance_both": "mf",
        "inheritance_unknown": "unknown",
    }
    for column, value in inheritance_flags.items():
        df[column] = df["inheritance"].eq(value).astype(np.int8)

    # Frequency values are retained as model inputs; rows are already frequency-filtered.
    if "freq" in df.columns:
        df["freq"] = pd.to_numeric(df["freq"], errors="coerce")
    else:
        df["freq"] = np.nan
        print("[warning] No `freq` column found. Frequency-derived features will be missing.")

    # Numeric annotations
    for c in count_cols:
        df[c] = to_nonnegative_numeric(df[c], c)

    # Gene-overlap count derived from actual names
    gene_lists = df["gene_overlap"].map(split_genes)
    df["gene_overlap_count"] = gene_lists.map(len).astype(int)

    df["distance_to_nearest_tss"] = to_nonnegative_numeric(
        df["distance_to_nearest_tss"], "distance_to_nearest_tss"
    )
    df["sv_length"] = to_nonnegative_numeric(df["sv_length"], "sv_length")

    # Use coordinate-derived span as fallback if sv_length is missing/0.
    coord_span = (df["end"] - df["start"]).abs().astype(float)
    replace_len = df["sv_length"].le(0)
    df.loc[replace_len, "sv_length"] = coord_span[replace_len]

    # Fixed-reference coordinate normalization (safe before train/test split)
    df["chrom_length"] = df["chrom"].map(lengths).astype(float)
    df["start_norm"] = (df["start"] / df["chrom_length"]).clip(0, 1)
    df["end_norm"] = (df["end"] / df["chrom_length"]).clip(0, 1)
    df["center_norm"] = ((df["start"] + df["end"]) / 2.0 / df["chrom_length"]).clip(0, 1)
    df["span_norm"] = ((df["end"] - df["start"]).abs() / df["chrom_length"]).clip(0, 1)

    # Deterministic transforms for strongly right-skewed variables
    df["sv_length_log1p"] = np.log1p(df["sv_length"])
    df["distance_to_nearest_tss_log1p"] = np.log1p(df["distance_to_nearest_tss"])
    for c in count_cols + ["gene_overlap_count"]:
        df[f"{c}_log1p"] = np.log1p(df[c].astype(float))

    # Frequency rarity score. Missing frequency stays 0 and gets a missingness indicator.
    df["freq_missing"] = df["freq"].isna().astype(int)
    freq_safe = df["freq"].fillna(1.0).clip(lower=0, upper=1)
    df["freq_log10_rarity"] = -np.log10(freq_safe + 1e-6)

    # Categorical IDs
    df["chrom_id"] = df["chrom"].map(CHR_TO_ID).astype(int)
    df["sv_type_id"] = df["sv_type"].map(SV_TYPE_TO_ID).astype(int)

    gene_vocab = build_gene_vocab(df)
    df["nearest_gene_id"] = df["nearest_gene"].map(
        lambda x: gene_vocab.get(split_genes(x)[0], 0) if split_genes(x) else 0
    ).astype(int)

    df["gene_overlap_ids"] = gene_lists.map(
        lambda gs: "|".join(str(gene_vocab[g]) for g in gs if g in gene_vocab)
    )

    # Strand is kept as metadata. If constant, it should not be a model feature.
    if "strand" in df.columns:
        unique_strand = df["strand"].dropna().astype(str).unique()
        if len(unique_strand) <= 1:
            print("[info] `strand` is constant/near-constant and will be metadata only.")

    # Stable sort improves reproducibility
    sort_cols = [c for c in ["sample_id", "chrom_id", "start", "end", "sv_id"] if c in df.columns]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    # Save
    df.to_csv(out_path, index=False)

    vocab_path = out_path.with_name(out_path.stem + "_gene_vocab.csv")
    vocab_df = pd.DataFrame(
        [{"gene": "<UNK>", "gene_id": 0}] +
        [{"gene": gene, "gene_id": idx} for gene, idx in gene_vocab.items()]
    )
    vocab_df.to_csv(vocab_path, index=False)

    manifest = {
        "input_csv": str(in_path),
        "output_csv": str(out_path),
        "genome_build": args.genome_build,
        "gender_mapping": GENDER_TO_ID,
        "inheritance_mapping": INHERITANCE_TO_ID,
        "gender_counts_rows": {
            str(k): int(v) for k, v in df["gender"].value_counts().items()
        },
        "inheritance_counts_rows": {
            str(k): int(v) for k, v in df["inheritance"].value_counts().items()
        },
        "inheritance_columns": list(inheritance_flags),
        "inheritance_feature_warning": (
            "Inheritance is highly associated with family role. Report an ablation "
            "without inheritance features to quantify possible label shortcutting."
        ),
        "rows_initial": int(n_initial),
        "rows_final": int(len(df)),
        "n_samples": int(df["sample_id"].nunique()),
        "n_families": int(df["family_id"].nunique(dropna=True)),
        "n_genes_in_vocab": int(len(gene_vocab)),
        "source_counts_rows": {str(k): int(v) for k, v in df["source"].value_counts().sort_index().items()},
        "sv_type_counts_rows": {str(k): int(v) for k, v in df["sv_type"].value_counts().items()},
        "model_numeric_columns_recommended": [
            "start_norm",
            "end_norm",
            "center_norm",
            "span_norm",
            "sv_length_log1p",
            "distance_to_nearest_tss_log1p",
            "exon_overlap_count_log1p",
            "gene_overlap_count_log1p",
            "promoter_overlap_count_log1p",
            "enhancer_overlap_count_log1p",
            "insulator_overlap_count_log1p",
            "freq_log10_rarity",
            "freq_missing",
        ],
        "model_categorical_columns_recommended": [
            "chrom_id",
            "sv_type_id",
            "nearest_gene_id",
            "gender_id",
            "inheritance_id",
        ],
        "metadata_not_model_inputs": [
            "family_id",
            "sample_id",
            "sv_id",
            "source",
            "chrom",
            "start",
            "end",
            "strand",
            "nearest_gene",
            "gene_overlap",
            "gene_overlap_ids",
        ],
    }
    manifest_path = out_path.with_name(out_path.stem + "_prep_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\nDone")
    print(f"Prepared CSV : {out_path}")
    print(f"Gene vocab   : {vocab_path}")
    print(f"Manifest     : {manifest_path}")
    print(f"Rows         : {n_initial:,} -> {len(df):,}")
    print(f"Samples      : {df['sample_id'].nunique():,}")


if __name__ == "__main__":
    main()
