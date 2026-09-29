# HOFCSV: Hierarchical OFC-guided SV model

## Project goal

HOFCSV learns one structural-variant representation for each person and uses
weak phenotype guidance to organize that representation around orofacial cleft
(OFC)-related genomic variation.

The goal is not simply to build a binary phenotype classifier. The model is
designed to support a traceable analysis of which SVs and affected genes move a
person's representation in an OFC-oriented direction, while retaining a
general phenotype-blind representation of SV structure.

The intended biological questions are:

- Do OFC probands occupy a different or enriched person-level latent region?
- Are probands more distinct from 1KG population controls than from their
  unaffected parents?
- Which SVs and directly affected genes contribute to the OFC-oriented
  representation?
- Can those candidates subsequently be evaluated for recurrence, inheritance,
  family sharing, and population frequency using complete SV calls?

Model attribution is not evidence of causality or pathogenicity. It reports
how the trained model uses an SV or gene, not whether that feature causes OFC.

## Model D

Model D is the full hierarchical OFC-guided architecture:

```text
SVs for one person
  ↓
SV encoder with genomic, functional, categorical, and Fourier-coordinate inputs
  ↓
SV latent vectors [number of SVs, 128]
  ↓
sparse relation-aware SV-to-gene aggregation
  ↓
gene latent vectors [number of affected genes, 128]
  ↓
four-head gated gene-level MIL pooling
  ↓
general person latent p_i [128]
  ↓
projection head: 128 → 64 → 32, followed by L2 normalization
  ↓
OFC-guided person latent q_i [32]
  ↓
weak auxiliary OFC classifier
```

There is one `p_i` and one `q_i` per person. Phenotype labels, family
IDs, sex, and ancestry do not enter the SV encoder. Inheritance is retained as
decoder context and metadata rather than being used to construct the person
latent directly.

Training combines:

```text
L_total = L_SSL + λ_contrastive L_contrastive + λ_OFC L_OFC
```

- `L_SSL` is a masked SV objective reconstructing numeric features,
  chromosome, SV type, gene identity, and inheritance.
- `L_contrastive` acts on person-level `q_i` vectors. Probands from different
  families form positives; unaffected parents are weaker negatives and 1KG
  individuals are stronger population-background negatives.
- `L_OFC` is a weak weighted binary-classification loss that provides a signed
  OFC direction for later removal-based attribution.

Training uses family-grouped train, validation, and untouched test splits.
Guided batches are balanced across probands, parents, and 1KG controls and
contain at most one person per family. Selected single- or multi-GPU training
is supported with `--gpu_ids`.

## Included files

```text
prep_sv_data.py  Input validation, normalization, and categorical encoding
coordinates.py   Fourier embedding of chromosome-relative SV coordinates
base_data.py     Variable-length person datasets and padded collation
data.py          Family splits and balanced phenotype-guided sampling
backbone.py      SV encoder, SV-to-gene aggregation, gene MIL, SSL decoder
model.py         OFC projection, auxiliary classifier, and combined losses
train.py         Model D training, evaluation, and checkpointing
smoke_test.py    Synthetic checks for dimensions, leakage, gradients, and splits
requirements.txt Minimal Python dependencies
```

## Training example

```bash
python prep_sv_data.py \
  --input_csv /path/to/inheritance.csv \
  --output_csv svs_prepared.csv \
  --genome_build hg38

python train.py \
  --data svs_prepared.csv \
  --output_dir runs/hierarchical_D \
  --training_population all \
  --ofc_guidance \
  --autosomes_only \
  --lambda_contrastive 0.05 \
  --lambda_ofc 0.05 \
  --parent_negative_weight 0.5 \
  --g1k_negative_weight 1.0 \
  --contrastive_probands_per_batch 8 \
  --contrastive_parents_per_batch 8 \
  --contrastive_g1k_per_batch 8 \
  --gpu_ids 0 \
  --seed 42
```

Use `--gpu_ids 0,1` for selected multi-GPU training or `--gpu_ids cpu` for CPU.
Run `python smoke_test.py` to verify the core implementation.
