#!/usr/bin/env python3
"""Hierarchical, linear-time SV -> gene -> person masked autoencoder.

The genomic representation path and inheritance-conditioning path are kept in
separate methods.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from coordinates import GenomicCoordinateEmbedding


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class GatedSetPooling(nn.Module):
    """Multi-head gated MIL plus mean/max/count; O(number of valid items)."""

    def __init__(self, dim, heads=4, dropout=0.15, extra_count_features=1):
        super().__init__()
        hidden = max(32, dim // 2)
        self.heads = heads
        self.value_gate = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh())
        self.signal_gate = nn.Sequential(nn.Linear(dim, hidden), nn.Sigmoid())
        self.score = nn.Linear(hidden, heads)
        input_dim = (heads + 2) * dim + extra_count_features
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 2 * dim),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, values, mask, count_features=None):
        # values [B,M,D], mask [B,M], count_features [B,C]
        if values.shape[1] == 0:
            raise ValueError("Every batch must contain at least one pooled item")
        gated = self.value_gate(values) * self.signal_gate(values)
        scores = self.score(gated).masked_fill(~mask.unsqueeze(-1), -1e9)
        attention = torch.softmax(scores, dim=1)
        attention = attention * mask.unsqueeze(-1)
        attention = attention / attention.sum(1, keepdim=True).clamp_min(1e-12)
        attention_pool = torch.einsum("bmh,bmd->bhd", attention, values).flatten(1)
        valid = mask.unsqueeze(-1)
        count = valid.sum(1).clamp_min(1)
        mean_pool = (values * valid).sum(1) / count
        max_pool = values.masked_fill(~valid, -torch.inf).max(1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        if count_features is None:
            count_features = torch.log1p(mask.sum(1, keepdim=True).float()) / math.log(10000.0)
        person = self.projection(
            torch.cat([attention_pool, mean_pool, max_pool, count_features], dim=-1)
        )
        return person, attention.mean(-1)


class SVToGeneAggregator(nn.Module):
    """Sparse relation-aware gated pooling from SV edges to affected genes."""

    RELATION_NEAREST = 0
    RELATION_OVERLAP = 1

    def __init__(self, token_dim, gene_embedding, relation_dim=16, dropout=0.15):
        super().__init__()
        self.gene_embedding = gene_embedding
        self.relation_embedding = nn.Embedding(2, relation_dim)
        self.edge_projection = nn.Sequential(
            nn.Linear(token_dim + relation_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        hidden = max(32, token_dim // 2)
        self.value_gate = nn.Sequential(nn.Linear(token_dim, hidden), nn.Tanh())
        self.signal_gate = nn.Sequential(nn.Linear(token_dim, hidden), nn.Sigmoid())
        self.score = nn.Linear(hidden, 1)
        self.gene_projection = nn.Sequential(
            nn.Linear(2 * token_dim + gene_embedding.embedding_dim + 1, 2 * token_dim),
            nn.LayerNorm(2 * token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def _edges(self, batch, token_z, exclude_tokens=None):
        # Return sparse edges: vectors [E,D], person/gene/relation/token IDs [E].
        token_mask = batch["token_mask"]
        if exclude_tokens is not None:
            token_mask = token_mask & ~exclude_tokens
        edge_vectors, people, genes, relations, tokens = [], [], [], [], []
        batch_size, n_tokens, _ = token_z.shape
        for person in range(batch_size):
            valid_tokens = torch.nonzero(token_mask[person], as_tuple=False).flatten()
            if valid_tokens.numel() == 0:
                continue
            nearest = batch["nearest_gene"][person, valid_tokens]
            nearest_valid = nearest.gt(0)
            if nearest_valid.any():
                selected = valid_tokens[nearest_valid]
                edge_vectors.append(token_z[person, selected])
                people.append(torch.full_like(selected, person))
                genes.append(nearest[nearest_valid])
                relations.append(torch.full_like(selected, self.RELATION_NEAREST))
                tokens.append(selected)
            overlap_mask = batch["overlap_gene_mask"][person, valid_tokens]
            locations = torch.nonzero(overlap_mask, as_tuple=False)
            if locations.numel():
                selected = valid_tokens[locations[:, 0]]
                edge_vectors.append(token_z[person, selected])
                people.append(torch.full_like(selected, person))
                genes.append(batch["overlap_gene"][person, selected, locations[:, 1]])
                relations.append(torch.full_like(selected, self.RELATION_OVERLAP))
                tokens.append(selected)
        if not edge_vectors:
            empty_long = torch.empty(0, dtype=torch.long, device=token_z.device)
            return token_z.new_empty((0, token_z.shape[-1])), *(empty_long for _ in range(4))
        return tuple(torch.cat(parts, dim=0) for parts in (edge_vectors, people, genes, relations, tokens))

    def forward(self, batch, token_z, exclude_tokens=None):
        edge_sv, edge_person, edge_gene, edge_relation, edge_token = self._edges(
            batch, token_z, exclude_tokens
        )
        batch_size, _, dim = token_z.shape
        if edge_sv.shape[0] == 0:
            # A learned-free zero pseudo-gene keeps tensor shapes valid.
            return {
                "gene_z": token_z.new_zeros((batch_size, 1, dim)),
                "gene_mask": torch.ones((batch_size, 1), dtype=torch.bool, device=token_z.device),
                "gene_ids": torch.zeros((batch_size, 1), dtype=torch.long, device=token_z.device),
                "gene_n_svs": torch.zeros((batch_size, 1), dtype=torch.long, device=token_z.device),
                "edge_attention": token_z.new_empty(0),
                "edge_person": edge_person,
                "edge_gene": edge_gene,
                "edge_relation": edge_relation,
                "edge_token": edge_token,
            }

        edge_z = self.edge_projection(
            torch.cat([edge_sv, self.relation_embedding(edge_relation)], dim=-1)
        )  # [E,D]
        # Unique (person,gene) groups. n_gene is a stable key stride.
        stride = self.gene_embedding.num_embeddings
        keys = edge_person * stride + edge_gene
        unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
        n_groups = unique_keys.numel()
        raw_score = self.score(
            self.value_gate(edge_z) * self.signal_gate(edge_z)
        ).squeeze(-1)
        maximum = raw_score.new_full((n_groups,), -torch.inf)
        maximum.scatter_reduce_(0, inverse, raw_score, reduce="amax", include_self=True)
        exp_score = torch.exp(raw_score - maximum[inverse])
        denominator = raw_score.new_zeros(n_groups).index_add_(0, inverse, exp_score)
        edge_attention = exp_score / denominator[inverse].clamp_min(1e-12)
        weighted = edge_z.new_zeros((n_groups, dim)).index_add_(
            0, inverse, edge_z * edge_attention.unsqueeze(-1)
        )
        summed = edge_z.new_zeros((n_groups, dim)).index_add_(0, inverse, edge_z)
        edge_counts = torch.zeros(n_groups, dtype=torch.long, device=edge_z.device).index_add_(
            0, inverse, torch.ones_like(inverse)
        )
        pair_stride = token_z.shape[1] + 1
        unique_group_tokens = torch.unique(inverse * pair_stride + edge_token)
        unique_pair_groups = torch.div(
            unique_group_tokens, pair_stride, rounding_mode="floor"
        )
        sv_counts = torch.zeros(
            n_groups, dtype=torch.long, device=edge_z.device
        ).index_add_(0, unique_pair_groups, torch.ones_like(unique_pair_groups))
        group_gene = unique_keys.remainder(stride)
        group_person = torch.div(unique_keys, stride, rounding_mode="floor")
        gene_z_flat = self.gene_projection(
            torch.cat(
                [
                    weighted,
                    summed / edge_counts.clamp_min(1).unsqueeze(-1),
                    self.gene_embedding(group_gene),
                    torch.log1p(sv_counts.float()).unsqueeze(-1) / math.log(1000.0),
                ],
                dim=-1,
            )
        )  # [K,D]
        per_person = torch.bincount(group_person, minlength=batch_size)
        max_genes = max(1, int(per_person.max().item()))
        gene_z = edge_z.new_zeros((batch_size, max_genes, dim))
        gene_mask = torch.zeros((batch_size, max_genes), dtype=torch.bool, device=edge_z.device)
        gene_ids = torch.zeros((batch_size, max_genes), dtype=torch.long, device=edge_z.device)
        gene_n_svs = torch.zeros_like(gene_ids)
        group_slot = torch.empty(n_groups, dtype=torch.long, device=edge_z.device)
        for person in range(batch_size):
            selected = torch.nonzero(group_person.eq(person), as_tuple=False).flatten()
            n_selected = selected.numel()
            if n_selected:
                gene_z[person, :n_selected] = gene_z_flat[selected]
                gene_mask[person, :n_selected] = True
                gene_ids[person, :n_selected] = group_gene[selected]
                gene_n_svs[person, :n_selected] = sv_counts[selected]
                group_slot[selected] = torch.arange(n_selected, device=edge_z.device)
        return {
            "gene_z": gene_z,
            "gene_mask": gene_mask,
            "gene_ids": gene_ids,
            "gene_n_svs": gene_n_svs,
            "edge_attention": edge_attention,
            "edge_person": edge_person,
            "edge_gene": edge_gene,
            "edge_relation": edge_relation,
            "edge_token": edge_token,
            "edge_gene_slot": group_slot[inverse],
        }


class HierarchicalGeneMIL(nn.Module):
    """Genomic SV encoder, sparse gene MIL, person MIL, conditioned decoder."""

    VALID_INHERITANCE_MODES = {"direct", "excluded", "conditioned"}

    def __init__(
        self,
        n_numeric,
        n_chrom,
        n_svtype,
        n_inheritance,
        n_gene,
        hidden_dim=256,
        token_dim=128,
        gene_emb_dim=64,
        coordinate_emb_dim=96,
        coordinate_fourier_bands=16,
        attention_heads=4,
        dropout=0.15,
        inheritance_mode="conditioned",
        inheritance_auxiliary_loss=True,
        inheritance_adversarial=False,
        adversarial_strength=1.0,
        exclude_sv_count_feature=False,
        nuisance_adversary=False,
        nuisance_burden_dim=9,
    ):
        super().__init__()
        if inheritance_mode not in self.VALID_INHERITANCE_MODES:
            raise ValueError(f"Unknown inheritance mode: {inheritance_mode}")
        if nuisance_adversary and inheritance_mode == "direct":
            raise ValueError(
                "nuisance_adversary requires inheritance_mode conditioned "
                "or excluded so inheritance cannot enter person_z"
            )
        self.n_numeric = n_numeric
        self.n_chrom = n_chrom
        self.n_svtype = n_svtype
        self.n_inheritance = n_inheritance
        self.n_gene = max(2, n_gene)
        self.token_dim = token_dim
        self.inheritance_mode = inheritance_mode
        self.inheritance_auxiliary_loss = bool(inheritance_auxiliary_loss)
        self.inheritance_adversarial = bool(inheritance_adversarial)
        self.adversarial_strength = float(adversarial_strength)
        self.exclude_sv_count_feature = bool(exclude_sv_count_feature)
        self.nuisance_adversary = bool(nuisance_adversary)
        self.nuisance_burden_dim = int(nuisance_burden_dim)
        self.chrom_mask_id = n_chrom
        self.svtype_mask_id = n_svtype
        self.inheritance_mask_id = n_inheritance
        self.gene_mask_id = self.n_gene

        self.chrom_embedding = nn.Embedding(n_chrom + 1, 12)
        self.svtype_embedding = nn.Embedding(n_svtype + 1, 8)
        self.gene_embedding = nn.Embedding(self.n_gene + 1, gene_emb_dim, padding_idx=0)
        self.inheritance_embedding = nn.Embedding(n_inheritance + 1, 12, padding_idx=0)
        self.overlap_gene_mask_vector = nn.Parameter(torch.zeros(gene_emb_dim))
        self.coordinate_embedding = GenomicCoordinateEmbedding(
            embedding_dim=coordinate_emb_dim,
            fourier_bands=coordinate_fourier_bands,
            dropout=dropout,
        )
        genomic_input_dim = 2 * n_numeric + coordinate_emb_dim + 12 + 8 + 2 * gene_emb_dim
        if inheritance_mode == "direct":
            genomic_input_dim += 12
        self.sv_encoder = nn.Sequential(
            nn.Linear(genomic_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.gene_aggregator = SVToGeneAggregator(
            token_dim, self.gene_embedding, relation_dim=16, dropout=dropout
        )
        self.person_aggregator = GatedSetPooling(
            token_dim, attention_heads, dropout,
            extra_count_features=1 if self.exclude_sv_count_feature else 2
        )
        decoder_input = 2 * token_dim + (12 if inheritance_mode != "excluded" else 0)
        self.prediction_trunk = nn.Sequential(
            nn.Linear(decoder_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.GELU(),
        )
        self.numeric_head = nn.Linear(token_dim, n_numeric)
        self.chrom_head = nn.Linear(token_dim, n_chrom)
        self.svtype_head = nn.Linear(token_dim, n_svtype)
        self.inheritance_head = nn.Linear(token_dim, n_inheritance)
        self.gene_query = nn.Linear(token_dim, gene_emb_dim)
        self.adversary = nn.Sequential(
            nn.Linear(token_dim, 64), nn.GELU(), nn.Linear(64, n_inheritance)
        )
        self.nuisance_sex_head = (
            nn.Sequential(
                nn.Linear(token_dim, 64), nn.GELU(), nn.Linear(64, 2)
            )
            if self.nuisance_adversary
            else None
        )
        self.nuisance_burden_head = (
            nn.Sequential(
                nn.Linear(token_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.nuisance_burden_dim),
            )
            if self.nuisance_adversary
            else None
        )

    @property
    def inheritance_enters_person_latent(self):
        return self.inheritance_mode == "direct"

    def _pool_overlap_genes(self, gene_ids, gene_mask, field_mask=None):
        embeddings = self.gene_embedding(gene_ids)
        pooled = (embeddings * gene_mask.unsqueeze(-1)).sum(2) / gene_mask.sum(
            2, keepdim=True
        ).clamp_min(1)
        if field_mask is not None:
            pooled = torch.where(
                field_mask.unsqueeze(-1),
                self.overlap_gene_mask_vector.view(1, 1, -1),
                pooled,
            )
        return pooled

    def corrupt(self, batch, objective="masked", mask_probability=0.15):
        valid = batch["token_mask"]
        if objective == "masked":
            numeric_mask = (torch.rand_like(batch["numeric"]) < mask_probability) & valid.unsqueeze(-1)
            chrom_mask = (torch.rand_like(valid.float()) < mask_probability) & valid
            svtype_mask = (torch.rand_like(valid.float()) < mask_probability) & valid
            gene_mask = (torch.rand_like(valid.float()) < mask_probability) & valid & batch["nearest_gene"].ne(0)
            inheritance_mask = (torch.rand_like(valid.float()) < mask_probability) & valid & batch["inheritance"].ne(0)
        elif objective == "reconstruction":
            numeric_mask = valid.unsqueeze(-1).expand_as(batch["numeric"])
            chrom_mask, svtype_mask = valid, valid
            gene_mask = valid & batch["nearest_gene"].ne(0)
            inheritance_mask = valid & batch["inheritance"].ne(0)
        else:
            raise ValueError(objective)
        if not self.inheritance_auxiliary_loss or self.inheritance_mode == "excluded":
            inheritance_mask = torch.zeros_like(valid)
        values = {
            "numeric": batch["numeric"].clone(),
            "numeric_mask_indicator": numeric_mask.float() if objective == "masked" else torch.zeros_like(batch["numeric"]),
            "coordinates": batch["coordinates"].clone(),
            "chrom": batch["chrom"].clone(),
            "svtype": batch["svtype"].clone(),
            "nearest_gene": batch["nearest_gene"].clone(),
            "inheritance": batch["inheritance"].clone(),
            "gene_field_mask": gene_mask if objective == "masked" else None,
        }
        if objective == "masked":
            values["numeric"][numeric_mask] = 0
            values["coordinates"][numeric_mask[..., :4]] = 0
            values["chrom"][chrom_mask] = self.chrom_mask_id
            values["svtype"][svtype_mask] = self.svtype_mask_id
            values["nearest_gene"][gene_mask] = self.gene_mask_id
            values["inheritance"][inheritance_mask] = self.inheritance_mask_id
        if self.inheritance_mode == "excluded":
            values["inheritance"].zero_()
        values["loss_masks"] = {
            "numeric": numeric_mask,
            "chrom": chrom_mask,
            "svtype": svtype_mask,
            "gene": gene_mask,
            "inheritance": inheritance_mask,
        }
        return values

    def encode_svs(self, batch, values=None):
        """Encode SVs; inheritance is read here only for direct ablation A."""
        if values is None:
            values = {
                "numeric": batch["numeric"],
                "numeric_mask_indicator": torch.zeros_like(batch["numeric"]),
                "coordinates": batch["coordinates"],
                "chrom": batch["chrom"],
                "svtype": batch["svtype"],
                "nearest_gene": batch["nearest_gene"],
                "gene_field_mask": None,
            }
            if self.inheritance_mode == "direct":
                values["inheritance"] = batch["inheritance"]
        pieces = [
            values["numeric"],
            values["numeric_mask_indicator"],
            self.coordinate_embedding(values["coordinates"]),
            self.chrom_embedding(values["chrom"]),
            self.svtype_embedding(values["svtype"]),
            self.gene_embedding(values["nearest_gene"]),
            self._pool_overlap_genes(
                batch["overlap_gene"], batch["overlap_gene_mask"], values.get("gene_field_mask")
            ),
        ]
        if self.inheritance_mode == "direct":
            pieces.append(self.inheritance_embedding(values["inheritance"]))
        return self.sv_encoder(torch.cat(pieces, dim=-1))  # [B,N,128]

    def pool_genes(self, gene_z, gene_mask, sv_count):
        # gene_z [B,G,128] -> person_z [B,128].
        # Gene count remains; explicit SV count is an independent ablation.
        gene_count = (
            torch.log1p(gene_mask.sum(1).float()) / math.log(10000.0)
        ).unsqueeze(-1)
        if self.exclude_sv_count_feature:
            count_features = gene_count  # [B,1]
        else:
            sv_count_feature = (
                torch.log1p(sv_count.float()) / math.log(10000.0)
            ).unsqueeze(-1)
            count_features = torch.cat(
                [gene_count, sv_count_feature], dim=-1
            )  # [B,2]
        return self.person_aggregator(gene_z, gene_mask, count_features)

    def encode_person(self, batch, token_mask_override=None, exclude_gene_tokens=None):
        """Construct p from genomic/functional tensors; conditioned mode ignores H.

        In conditioned/excluded modes callers may remove ``inheritance`` entirely;
        successful execution is a direct executable leakage check.
        """
        representation_batch = batch
        if token_mask_override is not None:
            representation_batch = dict(batch)
            representation_batch["token_mask"] = token_mask_override
        sv_z = self.encode_svs(representation_batch)
        gene = self.gene_aggregator(
            representation_batch, sv_z, exclude_tokens=exclude_gene_tokens
        )
        person_z, gene_attention = self.pool_genes(
            gene["gene_z"],
            gene["gene_mask"],
            representation_batch["token_mask"].sum(1),
        )
        return {
            "token_z": sv_z,
            "person_z": person_z,
            "gene_attention": gene_attention,
            **gene,
        }

    def forward(self, batch, objective="masked", mask_probability=0.15):
        corrupted = self.corrupt(batch, objective, mask_probability)
        sv_z = self.encode_svs(batch, corrupted)
        # A masked gene cannot identify itself through the hierarchical routing.
        gene = self.gene_aggregator(
            batch,
            sv_z,
            exclude_tokens=corrupted["loss_masks"]["gene"] if objective == "masked" else None,
        )
        person_z, gene_attention = self.pool_genes(
            gene["gene_z"], gene["gene_mask"], batch["token_mask"].sum(1)
        )
        person_context = person_z.unsqueeze(1).expand(-1, sv_z.shape[1], -1)
        decoder_parts = [sv_z, person_context]
        if self.inheritance_mode != "excluded":
            # Separate H branch: decoder-only conditioning in proposed mode C.
            decoder_parts.append(self.inheritance_embedding(corrupted["inheritance"]))
        prediction_z = self.prediction_trunk(torch.cat(decoder_parts, dim=-1))
        output = {
            "token_z": sv_z,
            "person_z": person_z,
            "gene_attention": gene_attention,
            **gene,
            "numeric_prediction": self.numeric_head(prediction_z),
            "chrom_logits": self.chrom_head(prediction_z),
            "svtype_logits": self.svtype_head(prediction_z),
            "inheritance_logits": self.inheritance_head(prediction_z),
            "gene_query": self.gene_query(prediction_z),
            "loss_masks": corrupted["loss_masks"],
        }
        if self.inheritance_adversarial:
            reversed_z = GradientReversal.apply(person_z, self.adversarial_strength)
            output["inheritance_adversarial_prediction"] = self.adversary(reversed_z)
        if self.nuisance_adversary:
            # Targets never enter this method. Both heads read only GRL(person_z).
            reversed_z = GradientReversal.apply(
                person_z, self.adversarial_strength
            )  # [B,128]
            output["nuisance_sex_logits"] = self.nuisance_sex_head(
                reversed_z
            )  # [B,2]
            output["nuisance_burden_prediction"] = self.nuisance_burden_head(
                reversed_z
            )  # [B,C]
        return output


def _masked_smooth_l1(prediction, target, mask):
    return F.smooth_l1_loss(prediction[mask], target[mask]) if mask.any() else prediction.sum() * 0


def _masked_cross_entropy(logits, target, mask):
    return F.cross_entropy(logits[mask], target[mask]) if mask.any() else logits.sum() * 0


def sampled_gene_loss(model, query, target, mask, negatives=64):
    valid = mask & target.ne(0)
    if not valid.any():
        return query.sum() * 0
    query = F.normalize(query[valid], dim=-1)
    positive_ids = target[valid]
    positive = F.normalize(model.gene_embedding(positive_ids), dim=-1)
    negative_ids = torch.randint(1, model.n_gene, (len(positive_ids), negatives), device=query.device)
    collision = negative_ids.eq(positive_ids.unsqueeze(1))
    negative_ids[collision] = (negative_ids[collision] % (model.n_gene - 1)) + 1
    negative = F.normalize(model.gene_embedding(negative_ids), dim=-1)
    logits = torch.cat(
        [(query * positive).sum(-1, keepdim=True), torch.einsum("bd,bkd->bk", query, negative)], dim=1
    ) / 0.1
    return F.cross_entropy(logits, torch.zeros(len(query), dtype=torch.long, device=query.device))


def inheritance_burden_target(batch, n_classes):
    # [B,N] inheritance IDs -> [B,C] fractions; used as a target only.
    valid = batch["token_mask"]
    target = torch.zeros((valid.shape[0], n_classes), device=valid.device)
    for value in range(n_classes):
        target[:, value] = (batch["inheritance"].eq(value) & valid).sum(1)
    return target / valid.sum(1, keepdim=True).clamp_min(1)


def self_supervised_loss(
    model, output, batch, weights, gene_negatives=64, nuisance_targets=None
):
    masks = output["loss_masks"]
    parts = {
        "numeric": _masked_smooth_l1(output["numeric_prediction"], batch["numeric"], masks["numeric"]),
        "chrom": _masked_cross_entropy(output["chrom_logits"], batch["chrom"], masks["chrom"]),
        "svtype": _masked_cross_entropy(output["svtype_logits"], batch["svtype"], masks["svtype"]),
        "inheritance": _masked_cross_entropy(output["inheritance_logits"], batch["inheritance"], masks["inheritance"]),
        "gene": sampled_gene_loss(model, output["gene_query"], batch["nearest_gene"], masks["gene"], gene_negatives),
    }
    if model.inheritance_adversarial:
        parts["adversarial"] = F.mse_loss(
            output["inheritance_adversarial_prediction"],
            inheritance_burden_target(batch, model.n_inheritance),
        )
    else:
        parts["adversarial"] = output["person_z"].sum() * 0

    zero = output["person_z"].sum() * 0
    nuisance_parts = []
    parts["nuisance_sex"] = zero
    parts["nuisance_burden"] = zero
    if model.nuisance_adversary:
        if nuisance_targets is None:
            raise ValueError(
                "nuisance_targets are required when nuisance_adversary is enabled"
            )
        sex_target = nuisance_targets["sex"]  # [B], unknown is -100.
        known_sex = sex_target.ne(-100)
        if known_sex.any():
            parts["nuisance_sex"] = F.cross_entropy(
                output["nuisance_sex_logits"][known_sex],
                sex_target[known_sex],
            )
            nuisance_parts.append(parts["nuisance_sex"])
        parts["nuisance_burden"] = F.smooth_l1_loss(
            output["nuisance_burden_prediction"],
            nuisance_targets["burden"],
        )
        nuisance_parts.append(parts["nuisance_burden"])
    parts["nuisance"] = (
        torch.stack(nuisance_parts).mean() if nuisance_parts else zero
    )
    total = sum(float(weights.get(name, 0.0)) * loss for name, loss in parts.items())
    return total, parts
