import logging
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import wandb
from matplotlib import pyplot as plt
from monai.networks.nets import UNet

from medAI.datasets.data_bk import make_corewise_bk_dataloaders
from medAI.utils.reproducibility import (
    get_all_rng_states,
    set_all_rng_states,
    set_global_seed,
)

from src.helpers.masked_predictions import MaskedPredictionModule
from src.helpers.loss import (
    CancerDetectionValidRegionLoss,
    MultiTermCanDetLoss,
)


class UNetExperiment:
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

        wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            name=self.config.wandb.name,
            tags=self.config.wandb.tags,
        )
        logging.info("W&B initialized")

        if self.config.checkpoint_dir is not None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            self.exp_state_path = os.path.join(self.config.checkpoint_dir, "experiment_state.pth")
            if os.path.exists(self.exp_state_path):
                logging.info("Loading experiment state from experiment_state.pth")
                self.state = torch.load(self.exp_state_path, map_location="cpu")
            else:
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

        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(self.config, "use_amp", False)))
        if self.state is not None and "gradient_scaler" in self.state:
            self.gradient_scaler.load_state_dict(self.state["gradient_scaler"])

        self.epoch = 0 if self.state is None else self.state["epoch"]
        self.best_score = 0.0 if self.state is None else float(self.state["best_score"])
        if self.state is not None and "rng" in self.state:
            set_all_rng_states(self.state["rng"])

    def setup_model(self):
        logging.info("Setting up model")

        class UNetWrapper(nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, x, *args, **kwargs):
                B, C, H, W = x.shape
                out = self.unet(x)
                out = F.interpolate(out, (H // 4, W // 4))
                return out

        self.model = UNetWrapper(
            UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=1,
                channels=(4, 8, 16, 32, 64),
                strides=(2, 2, 2, 2),
            )
        ).to(self.config.device)

        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass

        logging.info(
            "Parameters: total=%d, trainable=%d",
            sum(p.numel() for p in self.model.parameters()),
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )

        loss_terms = [
            CancerDetectionValidRegionLoss(
                base_loss=self.config.loss.base_loss,
                loss_pos_weight=self.config.loss.loss_pos_weight,
                prostate_mask=self.config.loss.prostate_mask,
                needle_mask=self.config.loss.needle_mask,
                inv_label_smoothing=self.config.loss.inv_label_smoothing,
                smoothing_factor=self.config.loss.smoothing_factor,
            )
        ]
        loss_weights = [self.config.loss.loss_pos_weight]
        self.loss_fn = MultiTermCanDetLoss(loss_terms, loss_weights)

    def setup_optimizer(self):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LambdaLR

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

        self.optimizer = AdamW(self.model.parameters(), weight_decay=self.config.optimizer.wd)
        self.lr_scheduler = LambdaLR(
            self.optimizer,
            [
                LRCalculator(
                    self.config.optimizer.encoder_frozen_epochs,
                    self.config.optimizer.encoder_warmup_epochs,
                    self.config.training.num_epochs,
                    len(self.train_loader),
                )
            ],
        )

    def setup_data(self):
        self.train_loader, self.val_loader, self.test_loader = make_corewise_bk_dataloaders(
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
        for self.epoch in range(self.epoch, self.config.training.num_epochs):
            logging.info("Epoch %d", self.epoch)
            self.save_experiment_state()

            self.run_train_epoch(self.train_loader, desc="train")
            val_metrics = self.run_eval_epoch(self.val_loader, desc="val")

            new_record = False
            if val_metrics is not None and "val/core_auc_high_involvement" in val_metrics:
                tracked = val_metrics["val/core_auc_high_involvement"]
                if tracked > self.best_score:
                    self.best_score = tracked
                    new_record = True
                    logging.info("New best score: %.5f", self.best_score)

            metrics = self.run_eval_epoch(self.test_loader, desc="test")
            test_score = metrics.get("test/core_auc_high_involvement", float("nan"))

            self.save_model_weights(score=test_score, is_best_score=new_record)

        logging.info("Finished training")
        self.teardown()

    def run_train_epoch(self, loader, desc="train"):
        self.model.train()
        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()
        accum = self.config.loss.accumulate_grad_steps
        if not isinstance(accum, int) or accum < 1:
            accum = 1

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            if self.config.debug and train_iter > 10:
                break

            bmode, needle_mask, prostate_mask, ood_mask, label, *metadata = batch

            bmode = torch.cat([bmode.unsqueeze(1)] * 3, dim=1) / (bmode.max() + 1e-8)
            bmode = bmode.to(self.config.device)
            prostate_mask = prostate_mask.unsqueeze(1).to(self.config.device)
            needle_mask = needle_mask.unsqueeze(1).to(self.config.device)
            ood_mask = ood_mask.unsqueeze(1).to(self.config.device)
            label = label.to(self.config.device)

            involvement = metadata[0].to(self.config.device)
            core_ids = metadata[1]
            patient_ids = metadata[2]

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                heatmap_logits = self.model(bmode.float())

                if torch.any(torch.isnan(heatmap_logits)):
                    logging.warning("NaNs in logits")
                    continue

                loss = self.loss_fn(heatmap_logits, prostate_mask, needle_mask, ood_mask, label, involvement)

                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)
                mean_predictions_in_needle = []
                for j in range(len(bmode)):
                    mean_predictions_in_needle.append(predictions[batch_idx == j].sigmoid().mean())
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

                prostate_masks = prostate_mask > 0.5
                predictions_p, batch_idx_p = MaskedPredictionModule()(heatmap_logits, prostate_masks)
                mean_predictions_in_prostate = []
                for j in range(len(bmode)):
                    mean_predictions_in_prostate.append(predictions_p[batch_idx_p == j].sigmoid().mean())
                mean_predictions_in_prostate = torch.stack(mean_predictions_in_prostate)

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

            step_metrics = {"train_loss": float(loss.item())}
            # Keep LR logging consistent with other experiments: read from optimizer
            step_metrics["encoder_lr"] = float(self.optimizer.param_groups[0]["lr"])
            step_metrics["main_lr"] = float(self.optimizer.param_groups[0]["lr"])
            wandb.log(step_metrics)

        results_table = accumulator.compute()
        return self.create_and_report_metrics(results_table, desc="train")

    @torch.no_grad()
    def run_eval_epoch(self, loader, desc="eval"):
        self.model.eval()
        from medAI.utils.accumulators import DataFrameCollector

        accumulator = DataFrameCollector()

        for train_iter, batch in enumerate(tqdm(loader, desc=desc)):
            bmode, needle_mask, prostate_mask, ood_mask, label, *metadata = batch

            bmode = torch.cat([bmode.unsqueeze(1)] * 3, dim=1) / (bmode.max() + 1e-8)
            bmode = bmode.to(self.config.device)
            prostate_mask = prostate_mask.unsqueeze(1).to(self.config.device)
            needle_mask = needle_mask.unsqueeze(1).to(self.config.device)
            ood_mask = ood_mask.unsqueeze(1).to(self.config.device)
            label = label.to(self.config.device)

            involvement = metadata[0].to(self.config.device)
            core_ids = metadata[1]
            patient_ids = metadata[2]

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                heatmap_logits = self.model(bmode.float())

                if torch.any(torch.isnan(heatmap_logits)):
                    logging.warning("NaNs in logits")
                    continue

                loss = self.loss_fn(heatmap_logits, prostate_mask, needle_mask, ood_mask, label, involvement)

                masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
                predictions, batch_idx = MaskedPredictionModule()(heatmap_logits, masks)
                mean_predictions_in_needle = []
                for j in range(len(bmode)):
                    mean_predictions_in_needle.append(predictions[batch_idx == j].sigmoid().mean())
                mean_predictions_in_needle = torch.stack(mean_predictions_in_needle)

                prostate_masks = prostate_mask > 0.5
                predictions_p, batch_idx_p = MaskedPredictionModule()(heatmap_logits, prostate_masks)
                mean_predictions_in_prostate = []
                for j in range(len(bmode)):
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
        metrics_ = calculate_metrics(predictions, labels, log_images=self.config.wandb.log_images)
        metrics.update(metrics_)

        high_involvement = involvement > 0.4
        benign = labels == 0
        keep = np.logical_or(high_involvement, benign)
        if keep.sum() > 0:
            core_probs = predictions[keep]
            core_labels = labels[keep]
            metrics_ = calculate_metrics(core_probs, core_labels, log_images=self.config.wandb.log_images)
            metrics.update({f"{metric}_high_involvement": value for metric, value in metrics_.items()})

        metrics = {f"{desc}/{k}": v for k, v in metrics.items()}
        metrics["epoch"] = self.epoch
        wandb.log(metrics)
        return metrics

    @torch.no_grad()
    def show_example(self, batch):
        if self.config.wandb.log_images is False:
            return

        bmode, needle_mask, prostate_mask, ood_mask, label, *metadata = batch

        bmode = torch.cat([bmode.unsqueeze(1)] * 3, dim=1) / (bmode.max() + 1e-8)
        bmode = bmode.to(self.config.device)
        needle_mask = needle_mask.to(self.config.device).unsqueeze(1)
        prostate_mask = prostate_mask.to(self.config.device).unsqueeze(1)
        label = label.to(self.config.device)

        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            logits = self.model(bmode)

        pred = logits.sigmoid().cpu().double()
        logits = logits.cpu().double()
        label = label.cpu().double()
        image = bmode.cpu().double()
        prostate_mask = prostate_mask.cpu().double()
        needle_mask = needle_mask.cpu().double()

        fig, ax = plt.subplots(1, 4, figsize=(12, 4))
        [a.set_axis_off() for a in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))

        ax[0].imshow(image[0].permute(1, 2, 0), cmap="gray")

        ax[1].imshow(image[0].permute(1, 2, 0), cmap="gray", **kwargs)
        ax[1].imshow(prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs)
        ax[1].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[1].set_title(f"Ground truth label: {label[0].item()}")

        masked_pred = (needle_mask * prostate_mask * pred[0, 0])
        ax[2].imshow(pred[0, 0], **kwargs)
        ax[2].set_title(f"Predicted label: {masked_pred[masked_pred > 0].mean().item():.2f}")

        valid_loss_region = (prostate_mask[0][0] > 0.5).float() * (needle_mask[0][0] > 0.5).float()
        mask_size = bmode.shape[-1] // 4
        alpha = F.interpolate(valid_loss_region[None, None], size=(mask_size, mask_size), mode="nearest")[0, 0]
        ax[3].imshow(pred[0, 0], alpha=alpha, **kwargs)

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

    def save_model_weights(self, score, is_best_score=False):
        if self.config.checkpoint_dir is None:
            return
        fname = "best_model.ckpt" if is_best_score else f"model_epoch{self.epoch}_auc{score:.2f}.ckpt"
        logging.info("Saving model to %s", os.path.join(self.config.checkpoint_dir, fname))
        torch.save(self.model.state_dict(), os.path.join(self.config.checkpoint_dir, fname))

    def teardown(self):
        if self.exp_state_path is not None and os.path.exists(self.exp_state_path):
            os.remove(self.exp_state_path)
