"""Utility functions for ProTeUS evaluation and visualization.

This module contains utility functions for computing evaluation metrics,
creating visualizations, and other helper functions for the ProTeUS framework.
"""

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

# seaborn is optional; plot functions will degrade gracefully if not available
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def scatterplot(inv_c, outputs_c, scores=None, declare_thr=0.35, c_ind=None):
    """Plot predicted involvement versus true involvement.

    Creates a scatter plot showing the relationship between predicted and true
    cancer involvement percentages with performance metrics and decision regions.

    Args:
        inv_c: True involvement values.
        outputs_c: Predicted involvement values.
        scores: Optional dictionary containing correlation and MAE scores.
        declare_thr: Decision threshold for declaring cancer.
        c_ind: Optional center indices for coloring points.

    Returns:
        matplotlib Figure object. Falls back to minimal plot if seaborn unavailable.
    """
    involvement = np.asarray(inv_c)
    predicted = np.asarray(outputs_c, dtype=float)

    # Safe min-max normalization
    min_val = float(np.nanmin(predicted)) if predicted.size else 0.0
    max_val = float(np.nanmax(predicted)) if predicted.size else 1.0
    denom = (max_val - min_val) if (max_val - min_val) > 0 else 1.0
    predicted = (predicted - min_val) / denom

    if c_ind is None:
        c_ind = np.zeros_like(involvement)

    mask_pos = involvement > 0
    inv_pos = involvement[mask_pos]
    pred_pos = predicted[mask_pos]
    c_ind_pos = np.asarray(c_ind)[mask_pos]

    if scores is None:
        if inv_pos.size > 1 and pred_pos.size > 1:
            # np.corrcoef can NaN if constant; guard it
            with np.errstate(invalid="ignore"):
                corr = np.corrcoef(inv_pos, pred_pos)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        mae = float(np.abs(inv_pos - pred_pos).mean()) if inv_pos.size else 0.0
    else:
        corr = float(scores.get("corr", 0.0))
        mae = float(scores.get("mae", 0.0))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    if _HAS_SNS:
        sns.scatterplot(x=inv_pos, y=pred_pos, legend=False, hue=c_ind_pos, ax=ax)
        # Benign points at involvement == 0
        sns.swarmplot(
            x=involvement[~mask_pos],
            y=predicted[~mask_pos],
            size=2,
            legend=False,
            ax=ax,
        )
        diag = np.arange(0, 1.001, 0.05)
        sns.lineplot(x=diag, y=diag, color="r", ax=ax)
    else:
        ax.scatter(inv_pos, pred_pos, s=8)
        diag = np.arange(0, 1.001, 0.05)
        ax.plot(diag, diag, "r-")

    # Quadrants/regions
    ax.axvspan(-0.1, 0.1, ymin=-0.1, ymax=declare_thr + 0.02, alpha=0.2, facecolor="lightgreen")
    ax.axvspan(-0.1, 0.1, ymin=declare_thr + 0.02, ymax=1.0, alpha=0.2, facecolor="red")
    ax.axvspan(0.11, 1.1, ymin=-0.1, ymax=declare_thr + 0.02, alpha=0.2, facecolor="grey")
    ax.axvspan(0.11, 1.1, ymin=declare_thr + 0.02, ymax=1.0, alpha=0.2, facecolor="moccasin")
    ax.axvline(x=0.101, linewidth=0.6, linestyle="--", color="black")
    ax.axhline(y=declare_thr, linewidth=0.6, linestyle="--", color="black")

    ax.axis("square")
    ax.set(ylim=[-0.1, 1.1], xlim=[-0.1, 1.1])
    ax.set(
        title=f"Correlation Coefficient = {corr:.3f} | MAE = {mae:.3f}",
        xlabel="True Involvement",
        ylabel="Predicted Involvement",
    )
    return fig


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Initialize tensor with truncated normal distribution.
    
    Method based on:
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    
    Args:
        tensor: Tensor to initialize.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        a: Lower bound for truncation.
        b: Upper bound for truncation.
    """
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Initialize tensor with truncated normal distribution.
    
    Args:
        tensor: Tensor to initialize.
        mean: Mean of the normal distribution (default: 0.0).
        std: Standard deviation of the normal distribution (default: 1.0).
        a: Lower bound for truncation (default: -2.0).
        b: Upper bound for truncation (default: 2.0).
        
    Returns:
        The initialized tensor.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def calculate_metrics(predictions, labels, log_images=False, involvements=None):
    """Calculate comprehensive metrics for cancer classification.

    Computes various evaluation metrics including AUC, sensitivity at fixed
    specificities, F-scores, and balanced accuracy for prostate cancer detection.

    Args:
        predictions: Model predictions (probabilities or logits).
        labels: Ground truth binary labels.
        log_images: Whether to generate visualization plots.
        involvements: Optional involvement percentages for visualization.

    Returns:
        Tuple of (metrics_dict, figure) where:
        - metrics_dict: Dictionary containing computed metrics
        - figure: matplotlib Figure object or None
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(involvements, torch.Tensor):
        involvements = involvements.detach().cpu().numpy()

    predictions = np.asarray(predictions, dtype=float)
    labels = np.asarray(labels, dtype=int)

    # Drop NaNs from predictions (and align labels/involvements)
    if predictions.ndim == 0:
        predictions = predictions[None]
    if labels.ndim == 0:
        labels = labels[None]

    nan_mask = ~np.isnan(predictions)
    predictions = predictions[nan_mask]
    labels = labels[nan_mask]
    if involvements is not None:
        involvements = np.asarray(involvements)[nan_mask]
    else:
        involvements = None

    metrics = {}

    # Handle degenerate label sets
    has_pos = np.any(labels == 1)
    has_neg = np.any(labels == 0)
    if not (has_pos and has_neg) or predictions.size == 0:
        # Not enough class variety for ROC/F-scores
        metrics["core_auc"] = float("nan")
        for specificity in [0.20, 0.40, 0.60, 0.80]:
            metrics[f"sens_at_{int(specificity*100)}_spe"] = float("nan")
        metrics["f1"] = metrics["f2"] = metrics["f5"] = metrics["f8"] = float("nan")
        metrics["balanced_accuracy"] = float("nan")
        fig = None
        if involvements is not None and predictions.size > 0 and log_images:
            try:
                fig = scatterplot(involvements, predictions, declare_thr=0.5)
            except Exception:
                fig = None
        return metrics, fig

    # ROC / AUC
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        fbeta_score,
        roc_auc_score,
        roc_curve,
    )

    core_probs = predictions
    core_labels = labels

    # AUC
    try:
        metrics["core_auc"] = float(roc_auc_score(core_labels, core_probs))
    except Exception:
        metrics["core_auc"] = float("nan")

    # ROC curve
    fpr, tpr, thresholds = roc_curve(core_labels, core_probs)

    # Sensitivity at fixed specificities (robust indexing)
    for specificity in [0.20, 0.40, 0.60, 0.80]:
        target_fpr = 1 - specificity
        idxs = np.where(fpr <= target_fpr)[0]
        if idxs.size > 0:
            sensitivity = float(tpr[idxs.max()])
        else:
            sensitivity = float(tpr[0]) if tpr.size else float("nan")
        metrics[f"sens_at_{int(specificity*100)}_spe"] = sensitivity

    # Threshold that maximizes Youden's J (tpr - fpr)
    if tpr.size and fpr.size and thresholds.size:
        best_idx = int(np.argmax(tpr - fpr))
        best_threshold = float(thresholds[best_idx])
    else:
        best_threshold = 0.5

    # F-scores / balanced accuracy
    try:
        pred_labels = core_probs > best_threshold
        metrics["f1"] = float(f1_score(core_labels, pred_labels))
        metrics["f2"] = float(fbeta_score(core_labels, pred_labels, beta=2))
        metrics["f5"] = float(fbeta_score(core_labels, pred_labels, beta=5))
        metrics["f8"] = float(fbeta_score(core_labels, pred_labels, beta=8))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(core_labels, pred_labels))
    except Exception:
        metrics["f1"] = metrics["f2"] = metrics["f5"] = metrics["f8"] = float("nan")
        metrics["balanced_accuracy"] = float("nan")

    # Optional scatter plot
    fig = None
    if involvements is not None and log_images:
        try:
            fig = scatterplot(involvements, core_probs, declare_thr=best_threshold)
        except Exception:
            fig = None

    return metrics, fig
