"""Masked prediction module for anatomy-aware pooling.

This module implements region-masked pooling for prostate cancer detection,
ensuring predictions are aggregated only within valid anatomical regions
(prostate ∩ needle).
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat

class MaskedPredictionModule(nn.Module):
    """Module for computing masked predictions within valid anatomical regions.
    
    This module extracts predictions only from regions specified by the mask,
    typically the intersection of prostate and needle regions for anatomy-aware
    pooling in prostate cancer detection.
    """

    def __init__(self):
        """Initialize the masked prediction module."""
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Compute predictions within the valid region specified by the mask.
        
        Args:
            heatmap_logits: Prediction logits of shape (B, C, H, W).
            mask: Boolean mask of shape (B, 1, H, W) specifying valid regions.
            
        Returns:
            Tuple of (patch_logits, core_indices) where:
            - patch_logits: Flattened logits for valid patches
            - core_indices: Core indices for grouping patches
        """
        B, C, H, W = heatmap_logits.shape

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5
        
        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx
