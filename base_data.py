#!/usr/bin/env python3
"""Data utilities for phenotype-blind structural-variant set learning."""

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


NUMERIC_COLUMNS = [
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
]
COORDINATE_COLUMNS = ["start_norm", "end_norm", "center_norm", "span_norm"]
AUTOSOME_IDS = set(range(22))
METADATA_COLUMNS = [
    "family_id",
    "sample_id",
    "sv_id",
    "source",
    "phenotype",
    "CL_CLP",
    "cleft_type",
    "gender",
    "sex",
    "ancestry",
    "inheritance",
    "chrom",
    "start",
    "end",
    "sv_type",
    "sv_length",
    "freq",
    "nearest_gene",
    "gene_overlap",
    "exon_overlap_gene",
]


def parse_gene_ids(value):
    if pd.isna(value) or str(value).strip() == "":
        return np.empty(0, dtype=np.int64)
    return np.asarray(
        [int(item) for item in str(value).split("|") if item], dtype=np.int64
    )


def validate_prepared_frame(frame):
    required = [
        "family_id",
        "sample_id",
        "source",
        "chrom_id",
        "sv_type_id",
        "inheritance_id",
        "nearest_gene_id",
    ] + NUMERIC_COLUMNS
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Prepared table is missing {missing}. Run prep_sv_data.py first."
        )
    if frame["sample_id"].isna().any():
        raise ValueError("sample_id contains missing values")


def select_training_rows(frame, population="trio", autosomes_only=False):
    """Use source only to select a population; it is never returned as a tensor."""
    if population == "trio":
        selected = frame[frame["source"].isin([0, 1])]
    elif population == "all":
        selected = frame[frame["source"].isin([0, 1, 2])]
    else:
        raise ValueError(f"Unsupported training population: {population}")
    if autosomes_only:
        selected = selected[selected["chrom_id"].isin(AUTOSOME_IDS)]
    return selected


def family_validation_split(frame, validation_fraction=0.1, seed=42):
    """Phenotype-blind family-grouped validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    samples = frame[["sample_id", "family_id"]].drop_duplicates("sample_id").copy()
    samples["family_id"] = samples["family_id"].fillna(samples["sample_id"]).astype(str)
    families = samples["family_id"].drop_duplicates().tolist()
    if len(families) < 2:
        raise ValueError("At least two independent families/samples are required")
    rng = random.Random(seed)
    rng.shuffle(families)
    n_validation = max(
        1, min(len(families) - 1, round(len(families) * validation_fraction))
    )
    validation_families = set(families[:n_validation])
    validation_samples = samples.loc[
        samples["family_id"].isin(validation_families), "sample_id"
    ].tolist()
    training_samples = samples.loc[
        ~samples["family_id"].isin(validation_families), "sample_id"
    ].tolist()
    return training_samples, validation_samples


class SVPersonDataset(Dataset):
    """One item is one variable-length SV set; phenotype metadata is excluded.

    Returned token tensors have shape [N, feature_dim], where N varies by person.
    Global row indices are returned only for downstream metadata joins.
    """

    def __init__(
        self,
        frame,
        sample_ids,
        numeric_columns,
        scaler,
        max_svs_per_sample=0,
        training=False,
        seed=42,
    ):
        wanted = set(sample_ids)
        frame = frame[frame["sample_id"].isin(wanted)]
        self.frame = frame
        self.numeric_columns = list(numeric_columns)
        self.max_svs = int(max_svs_per_sample)
        self.training = training
        self.seed = seed

        # Store compact arrays once. No source/phenotype array is exposed to the model.
        self.row_index = frame.index.to_numpy(np.int64)
        self.numeric = scaler.transform(
            frame[self.numeric_columns].to_numpy(np.float32)
        ).astype(np.float32)
        self.coordinates = frame[COORDINATE_COLUMNS].to_numpy(np.float32)
        self.chrom = frame["chrom_id"].to_numpy(np.int64)
        self.svtype = frame["sv_type_id"].to_numpy(np.int64)
        self.inheritance = frame["inheritance_id"].fillna(0).to_numpy(np.int64)
        self.nearest_gene = frame["nearest_gene_id"].fillna(0).to_numpy(np.int64)
        if "gene_overlap_ids" in frame:
            self.overlap_gene = [
                parse_gene_ids(value) for value in frame["gene_overlap_ids"]
            ]
        else:
            self.overlap_gene = [np.empty(0, dtype=np.int64) for _ in range(len(frame))]

        positions = defaultdict(list)
        for position, sample_id in enumerate(frame["sample_id"].to_numpy()):
            if sample_id in wanted:
                positions[sample_id].append(position)
        self.groups = [
            (sample_id, np.asarray(index, dtype=np.int64))
            for sample_id, index in positions.items()
        ]
        missing = wanted.difference(positions)
        if missing:
            raise ValueError(f"No SV rows found for {len(missing)} requested samples")
        self.lengths = [
            min(len(index), self.max_svs) if self.max_svs > 0 else len(index)
            for _, index in self.groups
        ]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, item):
        sample_id, positions = self.groups[item]
        if self.max_svs > 0 and len(positions) > self.max_svs:
            if self.training:
                rng = np.random.default_rng(
                    self.seed + item + random.randint(0, 2**16)
                )
                positions = np.sort(rng.choice(positions, self.max_svs, replace=False))
            else:
                positions = positions[: self.max_svs]
        return {
            "sample_id": sample_id,
            "row_index": self.row_index[positions],
            "numeric": self.numeric[positions],
            "coordinates": self.coordinates[positions],
            "chrom": self.chrom[positions],
            "svtype": self.svtype[positions],
            "inheritance": self.inheritance[positions],
            "nearest_gene": self.nearest_gene[positions],
            "overlap_gene": [self.overlap_gene[position] for position in positions],
        }


class LengthBucketBatchSampler(Sampler):
    """Shuffle within similarly sized sets to reduce padding waste."""

    def __init__(
        self, lengths, batch_size, shuffle=True, seed=42, bucket_multiplier=20
    ):
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_size = max(self.batch_size, self.batch_size * bucket_multiplier)
        self.epoch = 0

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            batches.extend(
                bucket[offset : offset + self.batch_size]
                for offset in range(0, len(bucket), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def collate_sv_sets(items):
    """Pad token fields to [B, N_max, ...] and return a Boolean [B, N_max] mask."""
    batch_size = len(items)
    max_tokens = max(len(item["numeric"]) for item in items)
    numeric_dim = items[0]["numeric"].shape[1]
    max_overlap = max(
        1, max((len(ids) for item in items for ids in item["overlap_gene"]), default=0)
    )

    numeric = torch.zeros(batch_size, max_tokens, numeric_dim)
    coordinates = torch.zeros(batch_size, max_tokens, len(COORDINATE_COLUMNS))
    chrom = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    svtype = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    inheritance = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    nearest_gene = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    overlap_gene = torch.zeros(batch_size, max_tokens, max_overlap, dtype=torch.long)
    overlap_gene_mask = torch.zeros(
        batch_size, max_tokens, max_overlap, dtype=torch.bool
    )
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    sample_ids, row_indices = [], []

    for batch_index, item in enumerate(items):
        n_tokens = len(item["numeric"])
        numeric[batch_index, :n_tokens] = torch.from_numpy(item["numeric"])
        coordinates[batch_index, :n_tokens] = torch.from_numpy(item["coordinates"])
        chrom[batch_index, :n_tokens] = torch.from_numpy(item["chrom"])
        svtype[batch_index, :n_tokens] = torch.from_numpy(item["svtype"])
        inheritance[batch_index, :n_tokens] = torch.from_numpy(item["inheritance"])
        nearest_gene[batch_index, :n_tokens] = torch.from_numpy(item["nearest_gene"])
        for token_index, gene_ids in enumerate(item["overlap_gene"]):
            n_genes = len(gene_ids)
            if n_genes:
                overlap_gene[batch_index, token_index, :n_genes] = torch.from_numpy(
                    gene_ids
                )
                overlap_gene_mask[batch_index, token_index, :n_genes] = True
        token_mask[batch_index, :n_tokens] = True
        sample_ids.append(item["sample_id"])
        row_indices.append(item["row_index"])

    return {
        "numeric": numeric,
        "coordinates": coordinates,
        "chrom": chrom,
        "svtype": svtype,
        "inheritance": inheritance,
        "nearest_gene": nearest_gene,
        "overlap_gene": overlap_gene,
        "overlap_gene_mask": overlap_gene_mask,
        "token_mask": token_mask,
        "sample_ids": sample_ids,
        "row_indices": row_indices,
    }


def move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
