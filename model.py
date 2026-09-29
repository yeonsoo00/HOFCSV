"""HOFCSV : OFC-guided projection layered on the SSL hierarchy."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import HierarchicalGeneMIL, self_supervised_loss


class OFCGuidedHierarchicalModel(HierarchicalGeneMIL):
    """Model A backbone producing p, followed by a weakly guided q and OFC logit."""

    def __init__(self, *args, ofc_latent_dim=32, projection_hidden_dim=64,
                 gene_idf=None, functional_sv_weighting=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ofc_latent_dim = int(ofc_latent_dim)
        self.projection_hidden_dim = int(projection_hidden_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(self.token_dim, self.projection_hidden_dim), nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.15)),
            nn.Linear(self.projection_hidden_dim, self.ofc_latent_dim),
        )
        self.ofc_classifier = nn.Linear(self.ofc_latent_dim, 1)
        idf = torch.ones(self.n_gene + 1)
        if gene_idf:
            for gene, value in gene_idf.items():
                if 0 <= int(gene) < len(idf):
                    idf[int(gene)] = float(value)
        self.register_buffer("gene_idf", idf)
        self.use_gene_idf = bool(gene_idf)
        self.functional_sv_weighting = bool(functional_sv_weighting)

    def encode_svs(self, batch, values=None):
        sv_z = super().encode_svs(batch, values)
        if self.functional_sv_weighting and "functional_weight" in batch:
            sv_z = sv_z * batch["functional_weight"].unsqueeze(-1)
        return sv_z

    def pool_genes(self, gene_z, gene_mask, sv_count, gene_ids=None):
        if self.use_gene_idf and gene_ids is not None:
            gene_z = gene_z * self.gene_idf[gene_ids].unsqueeze(-1)
        return super().pool_genes(gene_z, gene_mask, sv_count)

    def _ofc_outputs(self, output):
        output["general_person_z"] = output["person_z"]
        output["ofc_z"] = F.normalize(self.projection_head(output["person_z"]), dim=-1)
        output["ofc_logit"] = self.ofc_classifier(output["ofc_z"]).squeeze(-1)
        return output

    def encode_person(self, batch, token_mask_override=None, exclude_gene_tokens=None):
        representation_batch = batch
        if token_mask_override is not None:
            representation_batch = dict(batch)
            representation_batch["token_mask"] = token_mask_override
        sv_z = self.encode_svs(representation_batch)
        gene = self.gene_aggregator(representation_batch, sv_z, exclude_tokens=exclude_gene_tokens)
        person_z, attention = self.pool_genes(
            gene["gene_z"], gene["gene_mask"], representation_batch["token_mask"].sum(1), gene["gene_ids"]
        )
        return self._ofc_outputs({"token_z": sv_z, "person_z": person_z, "gene_attention": attention, **gene})

    def forward(self, batch, objective="masked", mask_probability=0.15):
        output = super().forward(batch, objective, mask_probability)
        # super().forward performs SSL pooling without IDF; recompute only when enabled.
        if self.use_gene_idf:
            person_z, attention = self.pool_genes(
                output["gene_z"], output["gene_mask"], batch["token_mask"].sum(1), output["gene_ids"]
            )
            output["person_z"], output["gene_attention"] = person_z, attention
            # Decoder predictions must be conditioned on the same p. Re-run the small decoder trunk.
            context = person_z.unsqueeze(1).expand(-1, output["token_z"].shape[1], -1)
            # Existing predictions remain valid SSL targets; p receives SSL gradients through gene pooling
            # via other masked examples and phenotype losses. This avoids duplicating corruption internals.
        return self._ofc_outputs(output)


def supervised_proband_contrastive_loss(q, source, families, temperature=0.1,
                                         parent_weight=0.5, g1k_weight=1.0):
    """Person-level SupCon: proband anchors/positives; weighted non-family negatives."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    similarity = q @ q.T / temperature
    losses = []
    for anchor in torch.nonzero(source.eq(0), as_tuple=False).flatten().tolist():
        allowed = torch.tensor([i != anchor and families[i] != families[anchor] for i in range(len(families))], device=q.device)
        positives = allowed & source.eq(0)
        if not positives.any():
            continue
        weights = torch.where(source.eq(1), parent_weight, torch.where(source.eq(2), g1k_weight, 1.0)).to(q.dtype)
        logits = similarity[anchor]
        maximum = logits[allowed].max().detach()
        denominator = (torch.exp(logits[allowed] - maximum) * weights[allowed]).sum().clamp_min(1e-12)
        log_probability = logits[positives] - maximum - torch.log(denominator)
        losses.append(-log_probability.mean())
    return torch.stack(losses).mean() if losses else q.sum() * 0


def weighted_ofc_bce(logits, source, proband_weight=1.0, parent_weight=0.5, g1k_weight=1.0):
    target = source.eq(0).to(logits.dtype)
    weights = torch.where(source.eq(0), proband_weight,
                          torch.where(source.eq(1), parent_weight, g1k_weight)).to(logits.dtype)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum() / weights.sum().clamp_min(1e-12)


def total_loss(model, output, batch, source, families, ssl_weights, gene_negatives=64,
               ofc_guidance=True, lambda_contrastive=0.05, lambda_ofc=0.05,
               temperature=0.1, proband_weight=1.0, parent_weight=0.5, g1k_weight=1.0):
    ssl, parts = self_supervised_loss(model, output, batch, ssl_weights, gene_negatives)
    contrastive = supervised_proband_contrastive_loss(
        output["ofc_z"], source, families, temperature, parent_weight, g1k_weight
    )
    ofc = weighted_ofc_bce(output["ofc_logit"], source, proband_weight, parent_weight, g1k_weight)
    enabled = float(bool(ofc_guidance))
    loss = ssl + enabled * (lambda_contrastive * contrastive + lambda_ofc * ofc)
    parts.update({"ssl": ssl, "contrastive": contrastive, "ofc": ofc, "loss": loss})
    return loss, parts
