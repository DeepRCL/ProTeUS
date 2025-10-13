"""ProTeUS training module.

This module contains the training loop and experiment class for the ProTeUS model,
including progressive training strategy and comprehensive evaluation.
"""

import logging
import typing as tp
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt

from einops import rearrange, repeat

from medAI.datasets.data_bk import make_temporal_bk_dataloaders
from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)

from src.models.proteus import BKMedSAM
from src.helpers.masked_predictions import MaskedPredictionModule
from src.helpers.loss import ConsistencyLoss, MultiHmapMultiTermCanDetLoss, TLoss




class ProTeUSExperiment:
    """ProTeUS experiment class for training and evaluation.
    
    This class handles the complete training pipeline for the ProTeUS model,
    including progressive training strategy, loss computation, and evaluation.
    
    Attributes:
        config: Configuration object containing training parameters.
    """
    
    def __init__(self, config):
        """Initialize the ProTeUS experiment.
        
        Args:
            config: Configuration object containing all training parameters.
        """
        self.config = config

    def setup(self):
        """Set up the experiment including logging, W&B, data, and model."""
        logging.basicConfig(
            level=logging.INFO if not self.config.debug else logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")

        if getattr(self.config, "debug", False):
            self.config.wandb.name = "debug"

        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("W&B initialized")

        # Optional state snapshot path
        self.exp_state_path = None
        self.state = None

        set_global_seed(self.config.seed)
        self.setup_data()
        self.setup_model()

        if self.config.training.from_ckpt:
            path = f'{self.config.training.from_ckpt}/fold{self.config.data.fold}.pth'
            try:
                checkpoint = torch.load(path, map_location="cpu")
                self.model.load_state_dict(checkpoint)
                logging.info("Loaded checkpoint from %s", path)
            except Exception as e:
                logging.error("Failed to load checkpoint from %s: %s", path, e)
                raise

        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        self.setup_optimizer()

        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])

        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(self.config, "use_amp", False)))
        if self.state is not None:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        self.best_score = 0.0 if self.state is None else float(self.state["best_score"])

        if self.state is not None:
            set_all_rng_states(self.state["rng"])

    def setup_model(self):
        logging.info("Setting up model")
        self.model = BKMedSAM(self.config).to(self.config.device)
        try:
            self.model = torch.compile(self.model)  # no-op on older PyTorch
        except Exception:
            pass

        logging.info(
            "Parameters: total=%d, trainable=%d",
            sum(p.numel() for p in self.model.parameters()),
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )


        # Initialize TLoss for temporal modeling
        
        loss_terms = [
            ConsistencyLoss(
                consistency_mode=self.config.loss.consistency_mode,
                loss_pos_weight=self.config.loss.loss_pos_weight,
                prostate_mask=self.config.loss.prostate_mask,
                needle_mask=self.config.loss.needle_mask,
                weak_factor=self.config.loss.weak_factor,
                strong_factor=self.config.loss.strong_factor
            )
        ]
        loss_weights = [self.config.loss.loss_pos_weight]
        self.loss_fn = MultiHmapMultiTermCanDetLoss(loss_terms, loss_weights)

    def setup_optimizer(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

        encoder_parameters, warmup_parameters = self.model.get_params_groups()
        params = [
            {"params": encoder_parameters, "lr": self.config.optimizer.encoder_lr},
            {"params": warmup_parameters, "lr": self.config.optimizer.main_lr},
        ]
        self.optimizer = AdamW(params, weight_decay=self.config.optimizer.wd)

        class LRCalculator:
            def __init__(self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep):
                self.frozen_epochs = frozen_epochs
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.niter_per_ep = niter_per_ep

            def __call__(self, it):
                if it < self.frozen_epochs * self.niter_per_ep:
                    return 0.0
                if it < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep:
                    return (it - self.frozen_epochs * self.niter_per_ep) / (
                        self.warmup_epochs * self.niter_per_ep
                    )
                cur_it = it - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                total_it = (self.total_epochs - self.warmup_epochs - self.frozen_epochs) * self.niter_per_ep
                return 0.5 * (1 + np.cos(np.pi * cur_it / max(total_it, 1)))

        self.lr_scheduler = LambdaLR(
            self.optimizer,
            [
                LRCalculator(
                    self.config.optimizer.encoder_frozen_epochs,
                    self.config.optimizer.encoder_warmup_epochs,
                    self.config.training.num_epochs,
                    len(self.train_loader),
                ),
                LRCalculator(
                    self.config.optimizer.main_frozen_epochs,
                    self.config.optimizer.main_warmup_epochs,
                    self.config.training.num_epochs,
                    len(self.train_loader),
                ),
                LRCalculator(0, 0, self.config.training.num_epochs, len(self.train_loader)),
            ],
        )

    def remove_outliers(self, ratio: float = 0.93):
        self.run_train_epoch_nobackward(self.train_loader, desc="train_noback")
        loss_values = torch.tensor(list(self.loss_tracker.values()), dtype=torch.float32)
        threshold = torch.quantile(loss_values, ratio)
        excluded = [core for core, loss in self.loss_tracker.items() if loss > threshold]
        self.excluded_cores = list(set(getattr(self, "excluded_cores", [])) | set(excluded))
        logging.warning("Total excluded samples so far: %d", len(self.excluded_cores))

    def setup_data(self):
        self.train_loader, self.val_loader, self.test_loader = make_temporal_bk_dataloaders(
            batch_sz=self.config.data.batch_size,
            im_sz=self.config.data.image_size,
            centers=self.config.data.centers,
            style=self.config.data.frame_to_use,
            splitting="from_file_kfold" if self.config.data.kfold else "from_file",
            fold=self.config.data.fold,
            num_folds=self.config.data.num_folds,
            seed=self.config.seed,
            oversampling=self.config.data.oversampling,
            undersampling=self.config.data.undersampling,
            sampling_ratio=self.config.data.sampling_ratio,
            inv_threshold=0.8,
        )
        logging.info(
            "Batches train=%d val=%d test=%d | Samples train=%d val=%d test=%d",
            len(self.train_loader),
            len(self.val_loader),
            len(self.test_loader),
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            len(self.test_loader.dataset),
        )

    def run(self):
        self.setup()
        self.excluded_cores = []

        for self.epoch in range(self.epoch, self.config.training.num_epochs):
            logging.info("Epoch %d", self.epoch)
            self.save_experiment_state()

            if self.epoch in (4, 8, 12, 16):
                self.remove_outliers(ratio=0.95)
                inv_thr = {4: 0.6, 8: 0.4, 12: 0.2, 16: 0.0}[self.epoch]
                self.train_loader, self.val_loader, self.test_loader = make_temporal_bk_dataloaders(
                    batch_sz=self.config.data.batch_size,
                    im_sz=self.config.data.image_size,
                    centers=self.config.data.centers,
                    style=self.config.data.frame_to_use,
                    splitting="from_file_kfold" if self.config.data.kfold else "from_file",
                    fold=self.config.data.fold,
                    num_folds=self.config.data.num_folds,
                    seed=self.config.seed,
                    oversampling=self.config.data.oversampling,
                    undersampling=self.config.data.undersampling,
                    sampling_ratio=self.config.data.sampling_ratio,
                    inv_threshold=inv_thr,
                    cores_to_exclude=self.excluded_cores,
                )
                logging.info("Number of training batches: %d", len(self.train_loader))

            self.run_train_epoch(self.train_loader, desc="train")
            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            new_record = False
            if val_metrics is not None and "val/core_auc_high_involvement" in val_metrics:
                tracked = val_metrics["val/core_auc_high_involvement"]
                if tracked > self.best_score:
                    self.best_score = tracked
                    new_record = True
                    logging.info("New best score: %.5f", self.best_score)
                    test_metrics = self.run_eval_epoch(self.test_loader, desc="test")
                    _ = test_metrics.get("test/core_auc_high_involvement", None)

        logging.info("Finished training")
        self.teardown()

    def _build_metadata_text(self, psa, prostate_size, core_name) -> str:
        log_meta = bool(getattr(self.config, "privacy", {}).get("log_metadata", False))
        if not log_meta:
            return ""
        parts = [f"Current CoreName: {core_name[0]}"]
        if torch.is_tensor(psa) and psa.numel() > 0 and not torch.isnan(psa).any():
            parts.append(f"PSA: {psa.item():.3f}")
        if (
            torch.is_tensor(psa)
            and torch.is_tensor(prostate_size)
            and psa.numel() > 0
            and prostate_size.numel() > 0
            and not torch.isnan(psa).any()
            and not torch.isnan(prostate_size).any()
            and prostate_size.item() != 0
        ):
            parts.append(f"PSA Density: {psa.item() / prostate_size.item():.3f}")
        if torch.is_tensor(prostate_size) and prostate_size.numel() > 0 and not torch.isnan(prostate_size).any():
            parts.append(f"Prostate Size: {prostate_size.item():.3f}")
        return ", ".join(parts) + ", "

    def run_train_epoch(self, loader, desc="train", enable_backward=True):
        """Run a training epoch with optional backward pass.
        
        Args:
            loader: DataLoader for training data.
            desc: Description for progress bar.
            enable_backward: Whether to perform backward pass and optimization.
        """
        self.model.train()
        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()
        
        # Initialize loss tracker only when backward is disabled
        if not enable_backward:
            self.loss_tracker = {}

        # Robust accumulation factor (only used when backward is enabled)
        accum = self.config.loss.accumulate_grad_steps if enable_backward else 1
        if not isinstance(accum, int) or accum < 1:
            accum = 1

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            if self.config.debug and train_iter > 10:
                break

            (
                bmode1,
                needle_mask,
                prostate_mask,
                ood_mask,
                label,
                signal,
                othercores_list,
                prostate_size,
                psa,
                core_name,
                *metadata,
            ) = batch

            core_id = metadata[1][0]
            othercores = othercores_list[0]

            signal = None if getattr(signal, "size", 0) == 0 else signal.to(self.config.device).float()

            bmode1 = torch.cat([bmode1.unsqueeze(1)] * 3, dim=1) / (bmode1.max() + 1e-8)
            bmode1 = bmode1.to(self.config.device)

            prostate_mask = prostate_mask.unsqueeze(1).to(self.config.device)
            needle_mask = needle_mask.unsqueeze(1).to(self.config.device)
            ood_mask = ood_mask.unsqueeze(1).to(self.config.device)
            label = label.to(self.config.device)

            involvement = metadata[0].to(self.config.device)
            core_ids = metadata[1]
            patient_ids = metadata[2]

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                heatmap_logits = self.model(
                    bmode1,
                    prostate_mask=prostate_mask,
                    needle_mask=needle_mask,
                    timeseries_signal=signal,
                    other_cores=othercores,
                    metadata=self._build_metadata_text(psa, prostate_size, core_name),
                )

                if torch.any(torch.isnan(heatmap_logits)):
                    logging.warning("NaNs in heatmap logits")
                    continue

                # Different loss function calls based on backward pass requirement
                if enable_backward:
                    loss = self.loss_fn(
                        heatmap_logits,
                        None,  # Second view logits (not used in single-view)
                        prostate_mask,
                        needle_mask,
                        ood_mask,
                        label,
                        involvement,
                    )
                else:
                    # For loss tracking without backward pass, use the original signature
                    loss = self.loss_fn(
                        heatmap_logits,
                        prostate_mask,
                        needle_mask,
                        ood_mask,
                        label,
                        involvement,
                    )

                # Compute predictions and metrics only when backward is enabled
                if enable_backward:
                    masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                    preds1, b1 = MaskedPredictionModule()(heatmap_logits, masks)
                    meanpreds1 = []
                    for j in range(len(bmode1)):
                        meanpreds1.append(preds1[b1 == j].sigmoid().mean())
                    meanpreds1 = torch.stack(meanpreds1)

            # Store loss for tracking when backward is disabled
            if not enable_backward:
                self.loss_tracker[core_id] = float(loss.item())
                continue  # Skip optimization steps

            # Optimization steps (only when backward is enabled)
            loss = loss / accum

            if self.config.use_amp:
                self.gradient_scaler.scale(loss).backward()
            else:
                loss.backward()

            if (train_iter + 1) % accum == 0:
                if self.config.use_amp:
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            self.lr_scheduler.step()

            # Accumulate metrics for reporting (only when backward is enabled)
            if enable_backward:
                accumulator(
                    {
                        "average_needle_heatmap_value": meanpreds1,
                        "involvement": involvement,
                        "patient_id": patient_ids,
                        "core_id": core_ids,
                        "label": label,
                    }
                )

                step_metrics = {"train_loss": float(loss.item())}
                step_metrics["encoder_lr"] = float(self.optimizer.param_groups[0]["lr"])
                step_metrics["main_lr"] = float(self.optimizer.param_groups[1]["lr"])
                wandb.log(step_metrics)

        # Return metrics only when backward is enabled
        if enable_backward:
            results_table = accumulator.compute()
            return self.create_and_report_metrics(results_table, desc="train")
        else:
            return None

    def run_train_epoch_nobackward(self, loader, desc="train"):
        """Run a training epoch without backward pass for loss tracking.
        
        Args:
            loader: DataLoader for training data.
            desc: Description for progress bar.
        """
        return self.run_train_epoch(loader, desc, enable_backward=False)

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()
        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            (
                bmode1,
                needle_mask,
                prostate_mask,
                ood_mask,
                label,
                signal,
                othercores,
                prostate_size,
                psa,
                core_name,
                *metadata,
            ) = batch

            signal = None if getattr(signal, "size", 0) == 0 else signal.to(self.config.device).float()

            bmode1 = torch.cat([bmode1.unsqueeze(1)] * 3, dim=1) / (bmode1.max() + 1e-8)

            bmode1 = bmode1.to(self.config.device)

            prostate_mask = prostate_mask.unsqueeze(1).to(self.config.device)
            needle_mask = needle_mask.unsqueeze(1).to(self.config.device)
            ood_mask = ood_mask.unsqueeze(1).to(self.config.device)
            label = label.to(self.config.device)

            involvement = metadata[0].to(self.config.device)
            core_ids = metadata[1]
            patient_ids = metadata[2]

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                heatmap_logits = self.model(
                    bmode1,
                    prostate_mask=prostate_mask,
                    needle_mask=needle_mask,
                    timeseries_signal=signal,
                    metadata=self._build_metadata_text(psa, prostate_size, core_name),
                )

                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)

                mean_predictions_in_needle = []
                for j in range(len(bmode1)):
                    mean_predictions_in_needle.append(predictions[batch_idx == j].sigmoid().mean())
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

                prostate_masks = prostate_mask > 0.5
                predictions_p, batch_idx_p = MaskedPredictionModule()(heatmap_logits, prostate_masks)
                mean_predictions_in_prostate = []
                for j in range(len(bmode1)):
                    mean_predictions_in_prostate.append(predictions_p[batch_idx_p == j].sigmoid().mean())
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)

            accumulator(
                {
                    "average_needle_heatmap_value": mean_predictions_in_needle,
                    "average_prostate_heatmap_value": mean_predictions_in_prostate,
                    "involvement": involvement,
                    "patient_id": patient_ids,
                    "core_id": core_ids,
                    "label": label,
                }
            )

        results_table = accumulator.compute()
        return self.create_and_report_metrics(results_table, desc=desc)

    def create_and_report_metrics(self, results_table, desc="eval"):
        from src.utils import calculate_metrics

        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values

        metrics = {}
        metrics_, g = calculate_metrics(predictions, labels, log_images=self.config.wandb.log_images, involvements=involvement)
        metrics.update(metrics_)
        wandb.log({f"scatterplot/{desc}": wandb.Image(g.get_figure())}, commit=False)

        high_involvement = involvement > 0.4
        benign = labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = predictions[keep]
            core_labels = labels[keep]
            metrics_, g = calculate_metrics(core_probs, core_labels, log_images=self.config.wandb.log_images)
            metrics.update({f"{k}_high_involvement": v for k, v in metrics_.items()})

        metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = self.epoch
        wandb.log(metrics)
        return metrics

    @torch.no_grad()
    def show_example(self, batch):
        if not bool(self.config.wandb.log_images):
            return
        (
            bmode1,
            needle_mask,
            prostate_mask,
            ood_mask,
            label,
            signal,
            othercores,
            prostate_size,
            psa,
            core_name,
            *metadata,
        ) = batch

        bmode1 = torch.cat([bmode1.unsqueeze(1)] * 3, dim=1) / (bmode1.max() + 1e-8)

        bmode1 = bmode1.to(self.config.device)

        prostate_mask = prostate_mask.unsqueeze(1).to(self.config.device)
        needle_mask = needle_mask.unsqueeze(1).to(self.config.device)
        label = label.to(self.config.device)

        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            logits_1 = self.model(bmode1, prostate_mask=prostate_mask, needle_mask=needle_mask, timeseries_signal=signal)

        preds_1 = logits_1.sigmoid().cpu().double()

        image1 = bmode1.cpu().double()
        prostate_mask = prostate_mask.cpu().double()
        needle_mask = needle_mask.cpu().double()
        label = label.cpu().double()

        fig, ax = plt.subplots(1, 4, figsize=(16, 6))
        [a.set_axis_off() for a in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image1[0].permute(1, 2, 0), cmap="gray", **kwargs)

        ax[2].imshow(image1[0].permute(1, 2, 0), cmap="gray", **kwargs)
        ax[2].imshow(preds_1[0, 0], alpha=0.5, cmap="jet", **kwargs)
        ax[2].contour(prostate_mask[0, 0], colors="#ffffff", **kwargs)
        ax[2].contour(needle_mask[0, 0], colors="#00eeff", **kwargs)
        masked_pred = (needle_mask * prostate_mask * preds_1[0, 0])
        ax[2].set_title(f"y={label[0].item()} pred={masked_pred.mean().item():.2f}")


        return wandb.Image(plt)

    def save_experiment_state(self):
        if self.exp_state_path is None:
            return
        logging.info("Saving experiment snapshot to %s", self.exp_state_path)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
                "gradient_scaler": self.gradient_scaler.state_dict(),
                "rng": get_all_rng_states(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            },
            self.exp_state_path,
        )

    def save_model_weights(self, score, is_best_score: bool = False):
        if self.config.checkpoint_dir is None or not is_best_score:
            return
        fname = "best_model.ckpt"
        logging.info("Saving best model to %s", os.path.join(self.config.checkpoint_dir, fname))
        torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, fname))

    def teardown(self):
        if self.exp_state_path is not None and os.path.exists(self.exp_state_path):
            os.remove(self.exp_state_path)
