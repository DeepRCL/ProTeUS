import logging
import os
import typing as tp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import wandb
from einops import rearrange, repeat
from matplotlib import pyplot as plt

from medAI.datasets.data_bk import make_corewise_bk_dataloaders

from src.models.sam import SAMClassifier

from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
from pydantic import BaseModel


from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)

from src.helpers.masked_predictions import MaskedPredictionModule
from src.helpers.loss import CancerDetectionValidRegionLoss

class BKSAMExperiment:
    def __init__(self, config):
        self.config = config

    def setup(self):
        logging.basicConfig(
            level=logging.INFO if not self.config.debug else logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler()],
        )
        logging.info("Setting up experiment")

        if self.config.debug:
            self.config.wandb.name = "debug"
        logging.info("Configuration: %s", self.config)
        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("Wandb initialized")
        logging.info("Wandb url: " + wandb.run.url)

        if self.config.checkpoint_dir is not None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.exp_state_path = os.path.join(
                self.config.checkpoint_dir, "experiment_state.pth"
            )
            if os.path.exists(self.exp_state_path):
                logging.info("Loading experiment state from experiment_state.pth")
                self.state = torch.load(self.exp_state_path)
            else:
                logging.info("No experiment state found - starting from scratch")
                self.state = None
        else:
            self.exp_state_path = None
            self.state = None

        set_global_seed(self.config.seed)

        self.setup_data()

        self.setup_model()
        if self.state is not None:
            self.model.load_state_dict(self.state["model"])

        self.setup_optimizer()
        if self.state is not None:
            self.optimizer.load_state_dict(self.state["optimizer"])
            self.lr_scheduler.load_state_dict(self.state["lr_scheduler"])

        self.gradient_scaler = torch.cuda.amp.GradScaler()
        if self.state is not None:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        logging.info(f"Starting at epoch {self.epoch}")
        self.best_score = 0 if self.state is None else self.state["best_score"]
        logging.info(f"Best score so far: {self.best_score}")
        if self.state is not None:
            rng_state = self.state["rng"]
            set_all_rng_states(rng_state)

    def setup_model(self):
        logging.info("Setting up model")

        self.model = SAMClassifier(self.config)

        self.model.to(self.config.device)
        try:
            self.model = torch.compile(self.model)
            logging.info("Model compiled successfully")
        except Exception as e:
            logging.warning("Failed to compile model: %s", e)

        logging.info("Model setup complete")
        logging.info(
            f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}"
        )
        logging.info(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        # setup criterion
        loss_terms = []
        loss_weights = []
        loss_weights.append(self.config.loss.loss_pos_weight)
        loss_terms.append(
            CancerDetectionValidRegionLoss(
                base_loss=self.config.loss.base_loss,
                loss_pos_weight=self.config.loss.loss_pos_weight,
                prostate_mask=self.config.loss.prostate_mask,
                needle_mask=self.config.loss.needle_mask,
                inv_label_smoothing=self.config.loss.inv_label_smoothing,
                smoothing_factor=self.config.loss.smoothing_factor,
            )
        )

        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=1.0,
            reduction="mean",
        )

    def setup_optimizer(self):
        from torch.optim import AdamW

        (
            encoder_parameters,
            warmup_parameters
        ) = self.model.get_params_groups()

        params = [
            {"params": encoder_parameters, "lr": self.config.optimizer.encoder_lr},
            {"params": warmup_parameters, "lr": self.config.optimizer.main_lr}
        ]

        class LRCalculator:
            def __init__(
                self, frozen_epochs, warmup_epochs, total_epochs, niter_per_ep
            ):
                self.frozen_epochs = frozen_epochs
                self.warmup_epochs = warmup_epochs
                self.total_epochs = total_epochs
                self.niter_per_ep = niter_per_ep

            def __call__(self, iter):
                if iter < self.frozen_epochs * self.niter_per_ep:
                    return 0
                elif (
                    iter < (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                ):
                    return (iter - self.frozen_epochs * self.niter_per_ep) / (
                        self.warmup_epochs * self.niter_per_ep
                    )
                else:
                    cur_iter = (
                        iter
                        - (self.frozen_epochs + self.warmup_epochs) * self.niter_per_ep
                    )
                    total_iter = (
                        self.total_epochs - self.warmup_epochs - self.frozen_epochs
                    ) * self.niter_per_ep
                    return 0.5 * (1 + np.cos(np.pi * cur_iter / total_iter))

        self.optimizer = AdamW(params, weight_decay=self.config.optimizer.wd)
        from torch.optim.lr_scheduler import LambdaLR

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
            ],
        )

    def setup_data(self):
        (
            self.train_loader, 
            self.val_loader, 
            self.test_loader 
        ) = make_corewise_bk_dataloaders(
                batch_sz=self.config.data.batch_size, 
                im_sz=self.config.data.image_size,
                centers=self.config.data.centers,
                style=self.config.data.frame_to_use,
                #splitting='patients_kfold' if self.config.data.kfold else 'patients',
                splitting='from_file_kfold' if self.config.data.kfold else 'from_file',
                fold=self.config.data.fold,
                num_folds=self.config.data.num_folds,
                seed=self.config.seed,
                oversampling=self.config.data.oversampling,
                undersampling=self.config.data.undersampling,
                sampling_ratio=self.config.data.sampling_ratio,
        )
        
        
        logging.info(f"Number of training batches: {len(self.train_loader)}")
        logging.info(f"Number of validation batches: {len(self.val_loader)}")
        logging.info(f"Number of test batches: {len(self.test_loader)}")
        logging.info(f"Number of training samples: {len(self.train_loader.dataset)}")
        logging.info(f"Number of validation samples: {len(self.val_loader.dataset)}")
        logging.info(f"Number of test samples: {len(self.test_loader.dataset)}")

    def run(self):
        self.setup()
        for self.epoch in range(self.epoch, self.config.training.num_epochs):
            logging.info(f"Epoch {self.epoch}")
            self.save_experiment_state()

            self.run_train_epoch(self.train_loader, desc="train")

            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            if val_metrics is not None:
                tracked_metric = val_metrics["val/core_auc_high_involvement"]
                new_record = tracked_metric > self.best_score
            else:
                new_record = None

            if new_record:
                self.best_score = tracked_metric
                logging.info(f"New best score: {self.best_score}")

            metrics = self.run_eval_epoch(self.test_loader, desc="test")
            test_score = metrics["test/core_auc_high_involvement"]

            self.save_model_weights(score=test_score, is_best_score=new_record)

        logging.info("Finished training")
        self.teardown()

    def run_train_epoch(self, loader, desc="train"):
        # setup epoch
        self.model.train()
        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation
            if self.config.debug and train_iter > 10:
                break

            # extracting relevant data from the batch
            bmode, needle_mask, prostate_mask, ood_mask, label, *metadata = batch
            
            bmode = bmode.unsqueeze(1)
            prostate_mask = prostate_mask.unsqueeze(1)
            needle_mask = needle_mask.unsqueeze(1)
            ood_mask = ood_mask.unsqueeze(1)
            
            bmode = torch.cat([bmode, bmode, bmode], dim=1) / bmode.max()
            bmode = bmode.to(self.config.device)
            needle_mask = needle_mask.to(self.config.device)
            prostate_mask = prostate_mask.to(self.config.device)
            label = label.to(self.config.device)
            
            involvement = metadata[0]
            involvement = involvement.to(self.config.device)

            core_ids = metadata[1]
            patient_ids = metadata[2]

            B = len(bmode)

            # run the model
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                logits = self.model(bmode)

                if torch.any(torch.isnan(logits)):
                    logging.warning("NaNs in logits")
                    breakpoint()

                # loss calculation
                loss = self.loss_fn(logits, label)

                # compute predictions
                predictions = logits.sigmoid()
                mean_predictions_in_needle = predictions # in this case, we don't have a needle mask, so we just use the predictions

                mean_predictions_in_prostate = []
                for j in range(B):
                    mean_predictions_in_prostate.append(0)
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate) # no predictions in prostate, so we just use 0

            loss = loss / self.config.loss.accumulate_grad_steps
            # backward pass
            if self.config.use_amp:
                logging.debug("Backward pass")
                self.gradient_scaler.scale(loss).backward()
            else:
                logging.debug("Backward pass")
                loss.backward()

            # gradient accumulation and optimizer step
            if self.config.debug:
                for param in self.optimizer.param_groups[1]["params"]:
                    break
                logging.debug(param.data.view(-1)[0])

            if (train_iter + 1) % self.config.loss.accumulate_grad_steps == 0:
                logging.debug("Optimizer step")
                if self.config.use_amp:
                    self.gradient_scaler.step(self.optimizer)
                    self.gradient_scaler.update()
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.config.debug:
                    for param in self.optimizer.param_groups[1]["params"]:
                        break
                    logging.debug(param.data.view(-1)[0])

            self.lr_scheduler.step()

            # accumulate outputs
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

            # log metrics
            step_metrics = {
                "train_loss": loss.item() / self.config.loss.accumulate_grad_steps
            }
            encoder_lr = self.optimizer.param_groups[0]["lr"]
            main_lr = self.optimizer.param_groups[1]["lr"]
            step_metrics["encoder_lr"] = encoder_lr
            step_metrics["main_lr"] = main_lr

            wandb.log(step_metrics)

            # log images
            #if train_iter % 5 == 0 and self.config.wandb.log_images:
            #    wandb_im = self.show_example(batch_for_image_generation)
            #    wandb.log({f"{desc}_example": wandb_im})
            #    plt.close()

        # compute and log metrics
        results_table = accumulator.compute()

        return self.create_and_report_metrics(results_table, desc="train")

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()

        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            batch_for_image_generation = (
                batch.copy()
            )  # we pop some keys from batch, so we keep a copy for image generation
            
            # extracting relevant data from the batch
            bmode, needle_mask, prostate_mask, ood_mask, label, *metadata = batch

            logging.debug("BMode shape (B,C,H,W): %s", bmode.shape)
            
            bmode = bmode.unsqueeze(1)
            prostate_mask = prostate_mask.unsqueeze(1)
            needle_mask = needle_mask.unsqueeze(1)
            ood_mask = ood_mask.unsqueeze(1)

            bmode = torch.cat([bmode, bmode, bmode], dim=1) / bmode.max()
            bmode = bmode.to(self.config.device)
            needle_mask = needle_mask.to(self.config.device)
            prostate_mask = prostate_mask.to(self.config.device)
            label = label.to(self.config.device)
            
            involvement = metadata[0]
            involvement = involvement.to(self.config.device)

            core_ids = metadata[1]
            patient_ids = metadata[2]
            
            B = len(bmode)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                # forward pass
                logits = self.model(bmode)

                if torch.any(torch.isnan(logits)):
                    logging.warning("NaNs in logits")
                    breakpoint()

                # loss calculation
                loss = self.loss_fn(logits, label)

                # compute predictions
                predictions = logits.sigmoid()
                mean_predictions_in_needle = predictions # in this case, we don't have a needle mask, so we just use the predictions

                mean_predictions_in_prostate = []
                for j in range(B):
                    mean_predictions_in_prostate.append(0)
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate) # no predictions in prostate, so we just use 0

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

        # core predictions
        predictions = results_table.average_needle_heatmap_value.values
        labels = results_table.label.values
        involvement = results_table.involvement.values

        core_probs = predictions
        core_labels = labels

        metrics = {}
        metrics_ = calculate_metrics(
            predictions, labels, log_images=self.config.wandb.log_images
        )
        metrics.update(metrics_)

        # high involvement core predictions
        high_involvement = involvement > 0.4
        benign = core_labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = core_probs[keep]
            core_labels = core_labels[keep]
            metrics_ = calculate_metrics(
                core_probs, core_labels, log_images=self.config.wandb.log_images
            )
            metrics.update(
                {
                    f"{metric}_high_involvement": value
                    for metric, value in metrics_.items()
                }
            )

        metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = self.epoch
        wandb.log(metrics)
        return metrics

    @torch.no_grad()
    def show_example(self, batch):
        # don't log images by default, since they take up a lot of space.
        # should be considered more of a debuagging/demonstration tool
        if self.config.wandb.log_images is False:
            return

        bmode, needle_mask, prostate_mask, ood_mask, label, *metadata = batch
            
        bmode = bmode.unsqueeze(1)
        prostate_mask = prostate_mask.unsqueeze(1)
        needle_mask = needle_mask.unsqueeze(1)
        ood_mask = ood_mask.unsqueeze(1)

        bmode = torch.cat([bmode, bmode, bmode], dim=1) / bmode.max()
        bmode = bmode.to(self.config.device)
        needle_mask = needle_mask.to(self.config.device)
        prostate_mask = prostate_mask.to(self.config.device)
        label = label.to(self.config.device)
        
        involvement = metadata[0]
        involvement = involvement.to(self.config.device)
        
        B = len(bmode)

        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            logits = self.model(
                bmode,
                prostate_mask=prostate_mask,
                needle_mask=needle_mask,
                ood_mask=ood_mask
            )

        pred = logits.sigmoid()

        logits = logits.cpu().double()
        pred = pred.cpu().double()
        label = label.cpu().double()
        image = bmode.cpu().double()
        prostate_mask = prostate_mask.cpu().double()
        needle_mask = needle_mask.cpu().double()

        fig, ax = plt.subplots(1, 4, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image[0].permute(1, 2, 0), cmap='gray')

        ax[1].imshow(image[0].permute(1, 2, 0), cmap='gray', **kwargs)
        ax[1].imshow(
            prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs
        )
        ax[1].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[1].set_title(f"Ground truth label: {label[0].item()}")

        masked_pred = (needle_mask * prostate_mask * pred[0, 0])
        masked_pred_flat_nonzero = masked_pred[masked_pred > 0]

        ax[2].imshow(pred[0, 0], **kwargs)
        ax[2].set_title(f"Predicted label: {masked_pred_flat_nonzero.sigmoid().mean().item():.2f}")

        valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (
            needle_mask[0][0] > 0.5
        ).float()

        mask_size = bmode.shape[-1] // 4

        alpha = torch.nn.functional.interpolate(
            valid_loss_region[None, None],
            size=(mask_size, mask_size),
            mode="nearest",
        )[0, 0]
        ax[3].imshow(pred[0, 0], alpha=alpha, **kwargs)

        return wandb.Image(plt)

    def save_experiment_state(self):
        if self.exp_state_path is None:
            return
        logging.info(f"Saving experiment snapshot to {self.exp_state_path}")
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

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None:
            return

        if not is_best_score:
            fname = f"model_epoch{self.epoch}_auc{score:.2f}.ckpt"
        else:
            fname = "best_model.ckpt"

        logging.info("Saving model to checkpoint directory")
        logging.info(f"Checkpoint directory: {self.config.checkpoint_dir}")
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.checkpoint_dir, fname),
        )

    def teardown(self):
        # remove experiment state file
        if self.exp_state_path is not None:
            os.remove(self.exp_state_path)



