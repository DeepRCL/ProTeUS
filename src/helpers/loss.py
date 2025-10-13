"""Loss functions for ProTeUS prostate cancer detection.

This module contains various loss functions specifically designed for prostate cancer
detection with weak labels and involvement-aware training strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.helpers.masked_predictions import MaskedPredictionModule


class TLoss(nn.Module):
    """Student's t-distribution loss for robust regression.
    
    This loss function is based on the Student's t-distribution and provides
    robustness against outliers in the regression setting.
    
    Args:
        nu: Degrees of freedom parameter for t-distribution.
        epsilon: Small constant for numerical stability.
        reduction: Reduction method ('mean', 'sum', or 'none').
        pos_weight: Weight for positive samples.
    """
    
    def __init__(
        self,
        nu: float = 1.0,
        epsilon: float = 1e-8,
        reduction: str = "mean",
        pos_weight: float = 2.0,
    ):
        super().__init__()
        self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float))
        self.register_buffer("epsilon", torch.tensor(epsilon, dtype=torch.float))
        self.register_buffer("pos_weight_tensor", torch.tensor(pos_weight, dtype=torch.float))
        self.reduction = reduction

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """Compute the t-distribution loss.
        
        Args:
            input_tensor: Predicted values.
            target_tensor: Target values.
            
        Returns:
            Computed loss value.
            
        Raises:
            ValueError: If input and target shapes don't match.
        """
        if input_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Shape mismatch: input_tensor has shape {input_tensor.shape}, "
                f"but target_tensor has shape {target_tensor.shape}."
            )

        N, B = input_tensor.shape
        D = torch.tensor(float(N * B), dtype=torch.float, device=input_tensor.device)

        lambdas = torch.ones((N, B), dtype=torch.float, device=input_tensor.device)
        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon

        first_term = -torch.lgamma((sum_nu_epsilon + D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(lambdas + self.epsilon)
        fourth_term = (D / 2) * torch.log(torch.tensor(np.pi, device=input_tensor.device))
        fifth_term = (D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(lambdas + self.epsilon)
        numerator = torch.sum(delta_squared * lambdas_exp, dim=1)
        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + D) / 2) * torch.log(1 + fraction)

        total_losses = first_term + second_term + third_term + fourth_term + fifth_term + sixth_term

        weights = torch.ones_like(target_tensor, device=input_tensor.device)
        weights = torch.where(target_tensor == 1, self.pos_weight_tensor, weights)
        weighted_losses = total_losses * weights

        if self.reduction == "mean":
            return weighted_losses.mean()
        if self.reduction == "sum":
            return weighted_losses.sum()
        if self.reduction == "none":
            return weighted_losses
        raise ValueError(f"The reduction method '{self.reduction}' is not implemented.")


def involvement_label_smoothing(label, involvement, alpha=0.2):
    """Apply label smoothing to involvement values based on cancer label.
    
    Args:
        label: Binary cancer label (0 or 1).
        involvement: Involvement percentage value.
        alpha: Smoothing factor.
        
    Returns:
        Smoothed involvement value.
    """
    if label == 1:
        if involvement < 0.4:
            inv_ls = (1 - alpha / 2) * involvement + alpha / 20
        elif involvement < 0.65:
            inv_ls = (1 - alpha) * involvement + alpha / 10
        else:
            inv_ls = involvement
        return inv_ls
    else:
        return 0


def involvement_tolerant_loss(patch_logits, patch_labels, core_indices, involvement):
    """Compute involvement-tolerant loss for weak label learning.
    
    This loss function adapts the training objective based on the involvement
    percentage, providing more robust learning with weak labels.
    
    Args:
        patch_logits: Patch-level predictions.
        patch_labels: Patch-level labels.
        core_indices: Core indices for grouping patches.
        involvement: Involvement percentages for each core.
        
    Returns:
        Computed loss value.
    """
    batch_size = len(involvement)
    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)
    for i in range(batch_size):
        patch_logits_for_core = patch_logits[core_indices == i]
        patch_labels_for_core = patch_labels[core_indices == i]
        involvement_for_core = involvement[i]
        if patch_labels_for_core[0].item() == 0:
            loss += F.binary_cross_entropy_with_logits(patch_logits_for_core, patch_labels_for_core)
        elif involvement_for_core.item() > 0.65:
            loss += F.binary_cross_entropy_with_logits(patch_logits_for_core, patch_labels_for_core)
        else:
            pred_index_sorted = torch.argsort(patch_logits_for_core[:, 0], descending=True)
            patch_logits_for_core = patch_logits_for_core[pred_index_sorted]
            patch_labels_for_core = patch_labels_for_core[pred_index_sorted]
            n_predictions = patch_logits_for_core.shape[0]
            k = int(n_predictions * involvement_for_core.item())
            loss += F.binary_cross_entropy_with_logits(
                patch_logits_for_core[:k],
                patch_labels_for_core[:k],
            )


def simple_mil_loss(
    patch_logits,
    patch_labels,
    core_indices,
    top_percentile=0.2,
    pos_weight=torch.tensor(1.0),
):
    """Simple Multiple Instance Learning (MIL) loss.
    
    This loss function implements a simple MIL approach by selecting the top
    percentile of hardest patches from each core.
    
    Args:
        patch_logits: Patch-level predictions.
        patch_labels: Patch-level labels.
        core_indices: Core indices for grouping patches.
        top_percentile: Fraction of hardest patches to select.
        pos_weight: Weight for positive samples.
        
    Returns:
        Computed MIL loss value.
    """
    ce_loss = F.binary_cross_entropy_with_logits(
        patch_logits, patch_labels, pos_weight=pos_weight, reduction="none"
    )

    loss = torch.tensor(0, dtype=torch.float32, device=patch_logits.device)
    for i in torch.unique(core_indices):
        patch_losses_for_core = ce_loss[core_indices == i]
        n_patches = len(patch_losses_for_core)
        n_keep = max(1, int(n_patches * top_percentile))
        patch_losses_sorted = torch.sort(patch_losses_for_core)[0]
        loss += patch_losses_sorted[:n_keep].mean()
    return loss


class CancerDetectionLossBase(nn.Module):
    """Base class for cancer detection loss functions.
    
    This is an abstract base class that defines the interface for
    cancer detection loss functions.
    """
    
    def forward(self, cancer_logits, prostate_mask, needle_mask, label, involvement):
        """Forward pass for cancer detection loss.
        
        Args:
            cancer_logits: Model predictions for cancer detection.
            prostate_mask: Prostate segmentation mask.
            needle_mask: Needle segmentation mask.
            label: Ground truth cancer labels.
            involvement: Cancer involvement percentages.
            
        Returns:
            Computed loss value.
            
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class ConsistencyLoss(CancerDetectionLossBase):
    """Consistency loss for single-view cancer detection.
    
    This loss function supports various consistency modes for single-view
    cancer detection, including involvement-aware objectives.
    
    Supported modes:
      - 'inv_aware_single': MSE loss against label*involvement
      - 'inv_aware_single_mae': MSE + MAE average
      - 'inv_aware_smooth': Log-cosh loss
      - 'tloss_mae_inv_aware': T-Loss against label*involvement
      - 'tloss_mae_single': T-Loss against label
      - 'avg_single': Binary cross-entropy against label
    """
    def __init__(
        self,
        consistency_mode: str = "inv_aware_single",
        loss_pos_weight: float = 1.0,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        weak_factor: float = 0.5,   # kept for BC, unused in single-view
        strong_factor: float = 0.5, # kept for BC, unused in single-view
        tloss: TLoss | None = None,
    ):
        super().__init__()
        self.consistency_mode = consistency_mode
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask
        self.weak_factor = weak_factor
        self.strong_factor = strong_factor
        self.tloss = tloss if tloss is not None else TLoss(nu=1.0, epsilon=1e-8)

    def forward(self, logits_w, _logits_s, prostate_mask, needle_mask, _ood_mask, label, involvement, forget_rate=0):
        # Build valid masks (prostate ∩ needle if enabled)
        masks = []
        for i in range(len(logits_w)):
            mask = torch.ones(prostate_mask[i].shape, device=prostate_mask[i].device).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)

        preds_w, B_w = MaskedPredictionModule()(logits_w, masks)  # shape: (N, 1)
        labels = torch.zeros(len(preds_w), device=preds_w.device)
        for i in range(len(preds_w)):
            labels[i] = label[B_w[i]]
        labels = labels[..., None]  # (N,1)

        # Handle involvement scalar/tensor safely
        if isinstance(involvement, torch.Tensor) and involvement.numel() == 1:
            inv_val = 0.0 if torch.isnan(involvement).any() else float(involvement.item())
        else:
            # Vector per batch item
            inv_val = involvement  # (B,) tensor expected

        mode = self.consistency_mode

        if mode == "inv_aware_single":
            target = (labels * inv_val).float()
            loss_unreduced = F.mse_loss(preds_w, target, reduction="none").float()
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            return loss_unreduced.mean().float()

        if mode == "inv_aware_single_mae":
            target = (labels * inv_val).float()
            mse_part = F.mse_loss(preds_w, target, reduction="none").float()
            mae_part = F.l1_loss(preds_w, target, reduction="none").float()
            mse_part[labels == 1] *= self.loss_pos_weight
            mae_part[labels == 1] *= self.loss_pos_weight
            return ((mse_part.mean() + mae_part.mean()) / 2).float()

        if mode == "inv_aware_smooth":
            target = (labels * inv_val).float()
            diff = preds_w - target
            return torch.mean(torch.log(torch.cosh(diff + 1e-12))).float()

        if mode == "tloss_mae_inv_aware":
            target = (labels * inv_val).float()
            return self.tloss(preds_w, target)

        if mode == "tloss_mae_single":
            target = labels.float()
            return self.tloss(preds_w, target)

        if mode == "avg_single":
            return F.binary_cross_entropy_with_logits(
                preds_w,
                labels,
                pos_weight=torch.tensor(self.loss_pos_weight, device=preds_w.device),
            )

        raise ValueError(f"Unsupported single-view consistency_mode: {mode}")


class CancerDetectionValidRegionLoss(CancerDetectionLossBase):
    def __init__(
        self,
        base_loss: str = "ce",
        loss_pos_weight: float = 1.0,
        prostate_mask: bool = True,
        needle_mask: bool = True,
        inv_label_smoothing: bool = False,
        smoothing_factor: float = 0.2,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.loss_pos_weight = loss_pos_weight
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, cancer_logits, prostate_mask, needle_mask, ood_mask, label, involvement):
        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(prostate_mask[i].shape, device=prostate_mask[i].device).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)

        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        preds_ood, _ = MaskedPredictionModule()(cancer_logits, ood_mask > 0.5)  # used only for *obp modes
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]

        labels_ood = torch.zeros(len(preds_ood), device=preds_ood.device)
        preds_ood = preds_ood.squeeze()

        loss = torch.tensor(0, dtype=torch.float32, device=predictions.device)

        if self.base_loss == "ce":
            loss += F.binary_cross_entropy_with_logits(
                predictions,
                labels,
                pos_weight=torch.tensor(self.loss_pos_weight, device=predictions.device),
            )

        elif self.base_loss == "obp":  # CE + out-of-bounds penalty
            ce = F.binary_cross_entropy_with_logits(
                predictions,
                labels,
                pos_weight=torch.tensor(self.loss_pos_weight, device=predictions.device),
            )
            obp = F.binary_cross_entropy_with_logits(
                preds_ood,
                labels_ood,
                pos_weight=torch.tensor(self.loss_pos_weight, device=predictions.device),
            )
            loss += ce + (labels_ood.shape[0] / labels.shape[0]) * obp

        elif self.base_loss == "inv_mae":
            inv = 0.0 if (isinstance(involvement, torch.Tensor) and torch.isnan(involvement).any()) else involvement
            loss_unreduced = F.l1_loss(predictions, labels * inv, reduction="none")
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
            loss += F.l1_loss(predictions.mean(), inv)

        elif self.base_loss == "inv_mse":
            inv = 0.0 if (isinstance(involvement, torch.Tensor) and torch.isnan(involvement).any()) else involvement
            loss_unreduced = F.mse_loss(predictions, (labels * inv).float(), reduction="none").float()
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()

        elif self.base_loss == "inv_mse_obp":
            inv = 0.0 if (isinstance(involvement, torch.Tensor) and torch.isnan(involvement).any()) else involvement
            loss_unreduced = F.mse_loss(predictions, (labels * inv).float(), reduction="none").float()
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
            obp = F.binary_cross_entropy_with_logits(
                preds_ood,
                labels_ood,
                pos_weight=torch.tensor(self.loss_pos_weight, device=predictions.device),
            )
            loss += (labels_ood.shape[0] / labels.shape[0]) * obp

        elif self.base_loss == "inv_ce_obp":
            inv = 0.0 if (isinstance(involvement, torch.Tensor) and torch.isnan(involvement).any()) else involvement
            inv = involvement_label_smoothing(label, inv)
            ce = F.binary_cross_entropy_with_logits(
                predictions,
                labels * inv,
                pos_weight=torch.tensor(self.loss_pos_weight, device=predictions.device),
            )
            obp = F.binary_cross_entropy_with_logits(
                preds_ood,
                labels_ood,
                pos_weight=torch.tensor(self.loss_pos_weight, device=predictions.device),
            )
            loss += ce + (labels_ood.shape[0] / labels.shape[0]) * obp

        elif self.base_loss == "inv_mae_obp":
            if isinstance(involvement, torch.Tensor):
                if involvement.numel() == 0 or torch.isnan(involvement).any():
                    involvement = torch.tensor(0.0, device=predictions.device)
            loss_unreduced = F.l1_loss(predictions, labels * involvement, reduction="none")
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()
            loss += F.l1_loss(predictions.mean(), involvement)
            obp = F.binary_cross_entropy_with_logits(
                preds_ood,
                labels_ood,
                pos_weight=torch.tensor(self.loss_pos_weight, device=predictions.device),
            )
            loss += (labels_ood.shape[0] / labels.shape[0]) * obp


        elif self.base_loss == "mae":
            loss_unreduced = F.l1_loss(predictions, labels, reduction="none")
            loss_unreduced[labels == 1] *= self.loss_pos_weight
            loss += loss_unreduced.mean()

        else:
            raise ValueError(f"Unknown base loss: {self.base_loss}")

        return lo
