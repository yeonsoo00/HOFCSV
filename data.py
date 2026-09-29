"""Person-level data, family splits, and balanced phenotype batches."""

import math
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from base_data import (  # deliberate reuse of Model A
    AUTOSOME_IDS,
    COORDINATE_COLUMNS,
    NUMERIC_COLUMNS,
    SVPersonDataset as _SSL2PersonDataset,
    collate_sv_sets as _ssl2_collate,
    move_batch,
    select_training_rows,
    validate_prepared_frame,
)


SOURCE_NAMES = {0: "proband", 1: "parent", 2: "1KG"}


def sample_metadata(frame):
    columns = [c for c in ["sample_id", "family_id", "source", "sex", "gender", "ancestry"] if c in frame]
    metadata = frame[columns].drop_duplicates("sample_id").copy()
    metadata["family_id"] = metadata.get("family_id", metadata["sample_id"]).fillna(metadata["sample_id"]).astype(str)
    metadata["source"] = metadata["source"].astype(int)
    return metadata


def family_train_validation_test_split(frame, validation_fraction=0.1, test_fraction=0.1, seed=42):
    """Deterministic family-grouped three-way split, stratified by family source profile."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction and test_fraction must be nonnegative and sum to < 1")
    people = sample_metadata(frame)
    family_profiles = (
        people.assign(n=1).pivot_table(index="family_id", columns="source", values="n", aggfunc="sum", fill_value=0)
    )
    rng = random.Random(seed)
    strata = defaultdict(list)
    for family, row in family_profiles.iterrows():
        signature = tuple(int(row.get(source, 0) > 0) for source in (0, 1, 2))
        strata[signature].append(str(family))
    allocation = {"train": [], "validation": [], "test": []}
    for families in strata.values():
        rng.shuffle(families)
        n = len(families)
        n_test = min(n, int(round(n * test_fraction)))
        n_validation = min(n - n_test, int(round(n * validation_fraction)))
        if n >= 3 and test_fraction > 0 and n_test == 0:
            n_test = 1
        if n - n_test >= 2 and validation_fraction > 0 and n_validation == 0:
            n_validation = 1
        allocation["test"].extend(families[:n_test])
        allocation["validation"].extend(families[n_test:n_test + n_validation])
        allocation["train"].extend(families[n_test + n_validation:])
    # Tiny datasets may have singleton strata; rebalance whole families.
    all_families = family_profiles.index.astype(str).tolist()
    for split, fraction in [("test", test_fraction), ("validation", validation_fraction)]:
        if fraction > 0 and not allocation[split] and len(allocation["train"]) > 1:
            allocation[split].append(allocation["train"].pop())
    family_sets = {name: set(values) for name, values in allocation.items()}
    if any(family_sets[a] & family_sets[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise RuntimeError("Family leakage detected during splitting")
    if set().union(*family_sets.values()) != set(all_families):
        raise RuntimeError("Family split did not cover every family")
    result = {}
    for name, families in family_sets.items():
        result[name] = people.loc[people["family_id"].isin(families), "sample_id"].tolist()
    return result


def fit_gene_idf(frame, training_samples):
    """Fit gene inverse-person-frequency weights using training people only."""
    work = frame[frame["sample_id"].isin(training_samples)]
    pairs = []
    for row in work.itertuples():
        ids = {int(getattr(row, "nearest_gene_id", 0) or 0)}
        value = getattr(row, "gene_overlap_ids", "")
        if not pd.isna(value):
            ids.update(int(x) for x in str(value).split("|") if x)
        pairs.extend((row.sample_id, gene) for gene in ids if gene > 0)
    counts = pd.DataFrame(pairs, columns=["sample_id", "gene_id"]).drop_duplicates().groupby("gene_id").size() if pairs else pd.Series(dtype=int)
    n_people = max(1, len(set(training_samples)))
    result = pd.DataFrame({"gene_id": counts.index.astype(int), "n_training_people_with_gene": counts.values})
    result["idf"] = np.log((n_people + 1) / (result["n_training_people_with_gene"] + 1))
    result["n_training_people"] = n_people
    return result


def functional_weights(frame, config):
    """Modest, claim-free multiplicative token weights from raw annotations."""
    score = np.ones(len(frame), dtype=np.float32)
    freq = pd.to_numeric(frame.get("freq", np.nan), errors="coerce")
    score += config["rare"] * np.asarray(freq.fillna(np.inf) <= config["rare_threshold"])
    for column, key in [
        ("gene_overlap_count", "gene_overlap"), ("exon_overlap_count", "exon"),
        ("promoter_overlap_count", "promoter"), ("enhancer_overlap_count", "enhancer"),
    ]:
        values = pd.to_numeric(frame.get(column, 0), errors="coerce")
        if not isinstance(values, pd.Series):
            values = pd.Series(values, index=frame.index)
        score += config[key] * np.asarray(values.fillna(0) > 0)
    return score


class OFCPersonDataset(_SSL2PersonDataset):
    """Model A tensors plus labels kept outside the representation batch."""

    def __init__(self, frame, sample_ids, numeric_columns, scaler, max_svs_per_sample=0,
                 training=False, seed=42, functional_weight_config=None):
        super().__init__(frame, sample_ids, numeric_columns, scaler, max_svs_per_sample, training, seed)
        metadata = sample_metadata(frame).set_index("sample_id")
        self.sources = [int(metadata.loc[sample, "source"]) for sample, _ in self.groups]
        self.families = [str(metadata.loc[sample, "family_id"]) for sample, _ in self.groups]
        if functional_weight_config is not None:
            weights = functional_weights(frame, functional_weight_config)
            self.functional_weight = dict(zip(frame.index.astype(int), np.asarray(weights, dtype=np.float32)))
        else:
            self.functional_weight = None

    def __getitem__(self, item):
        result = super().__getitem__(item)
        result["source_target"] = self.sources[item]
        result["family_target"] = self.families[item]
        if self.functional_weight is not None:
            result["functional_weight"] = np.asarray([self.functional_weight[int(i)] for i in result["row_index"]], np.float32)
        return result


def collate_ofc_sets(items):
    result = _ssl2_collate(items)
    result["source_target"] = torch.tensor([item["source_target"] for item in items], dtype=torch.long)
    result["family_target"] = [item["family_target"] for item in items]
    if "functional_weight" in items[0]:
        weights = torch.ones_like(result["token_mask"], dtype=torch.float32)
        for i, item in enumerate(items):
            weights[i, :len(item["functional_weight"])] = torch.from_numpy(item["functional_weight"])
        result["functional_weight"] = weights
    return result


def representation_batch(batch):
    """Whitelist genomic tensors; phenotype/family labels cannot enter the encoder."""
    allowed = {"numeric", "coordinates", "chrom", "svtype", "inheritance", "nearest_gene",
               "overlap_gene", "overlap_gene_mask", "token_mask", "functional_weight"}
    return {key: value for key, value in batch.items() if key in allowed}


class BalancedPhenotypeBatchSampler(Sampler):
    """Balanced source sampling with at most one member of a family per batch."""

    def __init__(self, dataset, probands=8, parents=8, g1k=8, seed=42, ancestry_matched=False):
        self.dataset = dataset
        self.counts = {0: int(probands), 1: int(parents), 2: int(g1k)}
        self.seed = seed
        self.epoch = 0
        self.by_source = {source: [i for i, value in enumerate(dataset.sources) if value == source] for source in (0, 1, 2)}
        active = [s for s, count in self.counts.items() if count > 0]
        missing = [SOURCE_NAMES[s] for s in active if not self.by_source[s]]
        if missing:
            raise ValueError("Balanced guidance sampler has no " + ", ".join(missing))
        if ancestry_matched:
            warnings.warn("Ancestry-matched 1KG requires reliable per-person ancestry strata; falling back to balanced source sampling")
        self.steps = max(math.ceil(len(self.by_source[s]) / self.counts[s]) for s in active)

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {s: rng.sample(v, len(v)) for s, v in self.by_source.items()}
        cursor = {s: 0 for s in pools}
        for _ in range(self.steps):
            batch, used_families = [], set()
            for source in (0, 1, 2):
                wanted, attempts = self.counts[source], 0
                while sum(self.dataset.sources[i] == source for i in batch) < wanted and attempts < max(20, len(pools[source]) * 3):
                    if cursor[source] >= len(pools[source]):
                        rng.shuffle(pools[source]); cursor[source] = 0
                    index = pools[source][cursor[source]]; cursor[source] += 1; attempts += 1
                    family = self.dataset.families[index]
                    if family not in used_families:
                        batch.append(index); used_families.add(family)
            if len(batch) >= 2:
                rng.shuffle(batch)
                yield batch
