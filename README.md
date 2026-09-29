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

## Model

The full hierarchical OFC-guided architecture model:

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

## Included files
Note that not all the files we used for training are included here. We will publish the original codes on github for camera-ready version.

```text
prep_sv_data.py  Input validation, normalization, and categorical encoding
coordinates.py   Fourier embedding of chromosome-relative SV coordinates
base_data.py     Variable-length person datasets and padded collation
data.py          Family splits and balanced phenotype-guided sampling
backbone.py      SV encoder, SV-to-gene aggregation, gene MIL, SSL decoder
model.py         OFC projection, auxiliary classifier, and combined losses
train.py         Model training, evaluation, and checkpointing
requirements.txt Minimal Python dependencies
```
