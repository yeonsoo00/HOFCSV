#!/usr/bin/env python3
"""Train the OFC-guided hierarchical person-level latent model."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import (
    BalancedPhenotypeBatchSampler, NUMERIC_COLUMNS, OFCPersonDataset,
    collate_ofc_sets, family_train_validation_test_split, fit_gene_idf,
    move_batch, representation_batch, sample_metadata, select_training_rows,
    validate_prepared_frame,
)
from model import OFCGuidedHierarchicalModel, total_loss
def max_gene_id(frame):
    result = int(frame["nearest_gene_id"].max()) if len(frame) else 0
    for value in frame.get("gene_overlap_ids", pd.Series(dtype=str)).dropna():
        result = max([result] + [int(item) for item in str(value).split("|") if item])
    return result



def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def parse_gpu_ids(specification):
    """Resolve auto, cpu, or comma-separated CUDA device indices."""
    value = str(specification).strip().lower()
    if value == "cpu":
        return []
    if not torch.cuda.is_available():
        if value not in {"auto", ""}:
            raise RuntimeError("CUDA is unavailable but --gpu_ids requested CUDA devices")
        return []
    if value in {"auto", ""}:
        return [0]
    try:
        device_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--gpu_ids must be auto, cpu, or comma-separated integers") from error
    if not device_ids or len(device_ids) != len(set(device_ids)):
        raise ValueError("--gpu_ids must contain one or more unique device indices")
    invalid = [index for index in device_ids if index < 0 or index >= torch.cuda.device_count()]
    if invalid:
        raise ValueError(f"Invalid CUDA device IDs {invalid}; visible device count is {torch.cuda.device_count()}")
    return device_ids


class FixedOutputTrainingWrapper(nn.Module):
    """Remove variable-width gene tensors before DataParallel gathers replicas."""
    OUTPUT_KEYS = {"person_z", "general_person_z", "ofc_z", "ofc_logit", "numeric_prediction",
                   "chrom_logits", "svtype_logits", "inheritance_logits", "gene_query", "loss_masks"}

    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, batch, objective, mask_probability):
        output = self.core(batch, objective, mask_probability)
        return {key: value for key, value in output.items() if key in self.OUTPUT_KEYS}


def core_model(model):
    return model.module.core if isinstance(model, nn.DataParallel) else model


def arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--architecture", choices=["hierarchical_ofc_guided"], default="hierarchical_ofc_guided")
    p.add_argument("--ofc_guidance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--training_population", choices=["trio", "all"], default="all")
    p.add_argument("--autosomes_only", action="store_true")
    p.add_argument("--validation_fraction", type=float, default=0.1); p.add_argument("--test_fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=24); p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gpu_ids", "--gpus", default="auto", help="auto, cpu, one CUDA index such as 2, or comma-separated indices such as 0,2")
    p.add_argument("--contrastive_probands_per_batch", type=int, default=8)
    p.add_argument("--contrastive_parents_per_batch", type=int, default=8)
    p.add_argument("--contrastive_g1k_per_batch", type=int, default=8)
    p.add_argument("--contrastive_temperature", type=float, default=0.1)
    p.add_argument("--parent_negative_weight", type=float, default=0.5)
    p.add_argument("--g1k_negative_weight", type=float, default=1.0)
    p.add_argument("--proband_positive_weight", type=float, default=1.0)
    p.add_argument("--lambda_contrastive", type=float, default=0.05); p.add_argument("--lambda_ofc", type=float, default=0.05)
    p.add_argument("--ancestry_matched_g1k", action="store_true")
    p.add_argument("--gene_idf_weighting", action="store_true")
    p.add_argument("--functional_sv_weighting", action="store_true")
    p.add_argument("--functional_rare_weight", type=float, default=0.10)
    p.add_argument("--functional_gene_overlap_weight", type=float, default=0.10)
    p.add_argument("--functional_exon_weight", type=float, default=0.10)
    p.add_argument("--functional_promoter_weight", type=float, default=0.05)
    p.add_argument("--functional_enhancer_weight", type=float, default=0.05)
    p.add_argument("--functional_rare_threshold", type=float, default=0.01)
    p.add_argument("--max_svs_per_sample", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256); p.add_argument("--token_dim", type=int, default=128)
    p.add_argument("--gene_emb_dim", type=int, default=64); p.add_argument("--coordinate_emb_dim", type=int, default=96)
    p.add_argument("--coordinate_fourier_bands", type=int, default=16); p.add_argument("--attention_heads", type=int, default=4)
    p.add_argument("--projection_hidden_dim", type=int, default=64); p.add_argument("--ofc_latent_dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.15); p.add_argument("--mask_probability", type=float, default=0.15)
    p.add_argument("--w_numeric", type=float, default=1.0); p.add_argument("--w_chrom", type=float, default=0.2)
    p.add_argument("--w_svtype", type=float, default=0.2); p.add_argument("--w_gene", type=float, default=0.2)
    p.add_argument("--w_inheritance", type=float, default=0.2); p.add_argument("--gene_negatives", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.token_dim != 128 or args.ofc_latent_dim != 32:
        p.error("Scientific contract requires token_dim=128 and ofc_latent_dim=32")
    if args.training_population == "trio" and args.contrastive_g1k_per_batch > 0 and args.ofc_guidance:
        p.error("Use --training_population all for guided batches containing 1KG, or set --contrastive_g1k_per_batch 0")
    if min(args.lambda_contrastive, args.lambda_ofc, args.parent_negative_weight, args.g1k_negative_weight) < 0:
        p.error("Loss weights must be nonnegative")
    return args


def functional_config(args):
    return {
        "rare": args.functional_rare_weight, "gene_overlap": args.functional_gene_overlap_weight,
        "exon": args.functional_exon_weight, "promoter": args.functional_promoter_weight,
        "enhancer": args.functional_enhancer_weight, "rare_threshold": args.functional_rare_threshold,
    }


def make_model(frame, args, gene_idf):
    return OFCGuidedHierarchicalModel(
        n_numeric=len(NUMERIC_COLUMNS), n_chrom=int(frame["chrom_id"].max()) + 1,
        n_svtype=int(frame["sv_type_id"].max()) + 1,
        n_inheritance=max(5, int(frame["inheritance_id"].max()) + 1), n_gene=max_gene_id(frame) + 1,
        hidden_dim=args.hidden_dim, token_dim=args.token_dim, gene_emb_dim=args.gene_emb_dim,
        coordinate_emb_dim=args.coordinate_emb_dim, coordinate_fourier_bands=args.coordinate_fourier_bands,
        attention_heads=args.attention_heads, dropout=args.dropout, inheritance_mode="conditioned",
        inheritance_auxiliary_loss=True, inheritance_adversarial=False,
        ofc_latent_dim=args.ofc_latent_dim, projection_hidden_dim=args.projection_hidden_dim,
        gene_idf=gene_idf, functional_sv_weighting=args.functional_sv_weighting,
    )


def ordinary_loader(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=workers, collate_fn=collate_ofc_sets, pin_memory=torch.cuda.is_available())


def guided_loader(dataset, args):
    sampler = BalancedPhenotypeBatchSampler(
        dataset, args.contrastive_probands_per_batch, args.contrastive_parents_per_batch,
        args.contrastive_g1k_per_batch, args.seed, args.ancestry_matched_g1k,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers,
                      collate_fn=collate_ofc_sets, pin_memory=torch.cuda.is_available())


def epoch_pass(model, loader, device, optimizer, args, ssl_weights):
    training = optimizer is not None; model.train(training)
    totals = {name: 0.0 for name in ["loss", "ssl", "numeric", "chrom", "svtype", "gene", "inheritance", "contrastive", "ofc"]}
    logits, targets, n = [], [], 0
    with torch.enable_grad() if training else torch.no_grad():
        for raw in loader:
            moved = move_batch(raw, device); batch = representation_batch(moved)
            output = model(batch, "masked", args.mask_probability)
            loss, parts = total_loss(
                core_model(model), output, batch, moved["source_target"], raw["family_target"], ssl_weights,
                args.gene_negatives, args.ofc_guidance, args.lambda_contrastive, args.lambda_ofc,
                args.contrastive_temperature, args.proband_positive_weight,
                args.parent_negative_weight, args.g1k_negative_weight,
            )
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            count = len(raw["sample_ids"]); n += count
            for name in totals: totals[name] += float(parts[name].detach()) * count
            logits.extend(output["ofc_logit"].detach().cpu().tolist()); targets.extend(moved["source_target"].eq(0).cpu().tolist())
    metrics = {name: value / max(1, n) for name, value in totals.items()}
    if len(set(targets)) == 2:
        probability = torch.sigmoid(torch.tensor(logits)).numpy()
        metrics.update(auroc=float(roc_auc_score(targets, probability)), auprc=float(average_precision_score(targets, probability)),
                       balanced_accuracy=float(balanced_accuracy_score(targets, probability >= 0.5)))
    else:
        metrics.update(auroc=float("nan"), auprc=float("nan"), balanced_accuracy=float("nan"))
    return metrics


def split_summary(frame, splits):
    metadata = sample_metadata(frame)
    rows = []
    for name, ids in splits.items():
        group = metadata[metadata["sample_id"].isin(ids)]
        rows.append({"split": name, "n_people": len(group), "n_families": group["family_id"].nunique(),
                     "n_probands": int(group["source"].eq(0).sum()), "n_parents": int(group["source"].eq(1).sum()), "n_1kg": int(group["source"].eq(2).sum())})
    return pd.DataFrame(rows)


def main(argv=None):
    args = arguments(argv); set_seed(args.seed)
    root = Path(args.output_dir)
    for name in ["checkpoints", "preprocessing", "latents", "sv_analysis", "gene_analysis", "diagnostics", "plots"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data, low_memory=False); validate_prepared_frame(frame)
    training_frame = select_training_rows(frame, args.training_population, args.autosomes_only).copy()
    splits = family_train_validation_test_split(training_frame, args.validation_fraction, args.test_fraction, args.seed)
    split_summary(training_frame, splits).to_csv(root / "diagnostics/split_summary.csv", index=False)
    train_rows = training_frame[training_frame["sample_id"].isin(splits["train"])]
    scaler = StandardScaler().fit(train_rows[NUMERIC_COLUMNS].to_numpy(np.float32))
    idf_table = fit_gene_idf(training_frame, splits["train"])
    idf_table.to_csv(root / "preprocessing/gene_idf.csv", index=False)
    gene_idf = dict(zip(idf_table.gene_id, idf_table.idf)) if args.gene_idf_weighting else None
    fconfig = functional_config(args) if args.functional_sv_weighting else None
    sets = {name: OFCPersonDataset(training_frame, ids, NUMERIC_COLUMNS, scaler, args.max_svs_per_sample,
                                   name == "train", args.seed, fconfig) for name, ids in splits.items()}
    train_loader = guided_loader(sets["train"], args) if args.ofc_guidance else ordinary_loader(sets["train"], args.batch_size, args.num_workers, True, args.seed)
    loaders = {"train": train_loader,
               "validation": ordinary_loader(sets["validation"], args.batch_size, args.num_workers, False, args.seed),
               "test": ordinary_loader(sets["test"], args.batch_size, args.num_workers, False, args.seed)}
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    if gpu_ids:
        torch.cuda.set_device(gpu_ids[0])
    base_model = make_model(training_frame, args, gene_idf).to(device)
    model = (nn.DataParallel(FixedOutputTrainingWrapper(base_model), device_ids=gpu_ids,
                             output_device=gpu_ids[0]) if len(gpu_ids) > 1 else base_model)
    ssl_weights = {"numeric": args.w_numeric, "chrom": args.w_chrom, "svtype": args.w_svtype,
                   "gene": args.w_gene, "inheritance": args.w_inheritance, "adversarial": 0.0, "nuisance": 0.0}
    config = {**vars(args), "numeric_columns": NUMERIC_COLUMNS, "loss_weights": ssl_weights,
              "inheritance_mode": "conditioned", "inheritance_enters_person_latent": False,
              "phenotype_enters_encoder": False, "person_latents": {"p": 128, "q": 32},
              "functional_sv_weights": functional_config(args),
              "split_samples": splits, "split_seed": args.seed,
              "resolved_gpu_ids": gpu_ids, "multi_gpu": len(gpu_ids) > 1}
    (root / "model_config.json").write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, patience, history = float("inf"), args.patience, []
    checkpoint_path = root / "checkpoints/best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_pass(model, loaders["train"], device, optimizer, args, ssl_weights)
        torch.manual_seed(args.seed + 100000)
        validation_metrics = epoch_pass(model, loaders["validation"], device, None, args, ssl_weights)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(f"epoch={epoch:03d} train={train_metrics['loss']:.4f} validation={validation_metrics['loss']:.4f}")
        if validation_metrics["loss"] < best - 1e-6:
            best, patience = validation_metrics["loss"], args.patience
            torch.save({"model_state": core_model(model).state_dict(), "model_config": config, "dimensions": {
                "n_chrom": core_model(model).n_chrom, "n_svtype": core_model(model).n_svtype,
                "n_inheritance": core_model(model).n_inheritance, "n_gene": core_model(model).n_gene}, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                "best_epoch": epoch, "best_validation_loss": best, "split_samples": splits}, checkpoint_path)
        else:
            patience -= 1
            if patience == 0: break
    (root / "checkpoints/history.json").write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False); core_model(model).load_state_dict(checkpoint["model_state"])
    test_metrics = epoch_pass(model, loaders["test"], device, None, args, ssl_weights)
    summary = {"status": "training_complete", "device": str(device), "gpu_ids": gpu_ids, "multi_gpu": len(gpu_ids) > 1, "best_validation_loss": best,
               "untouched_test_metrics": test_metrics, "one_p_per_person": True, "one_q_per_person": True,
               "family_grouped_three_way_split": True, "balanced_guidance_batches": bool(args.ofc_guidance)}
    (root / "analysis_summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
