"""ProTeUS: A Spatio-Temporal Enhanced Ultrasound-Based Framework for Prostate Cancer Detection.

This module contains the main ProTeUS model implementation, which integrates:
- MedSAM backbone for spatial feature extraction
- InceptionTime for RF time-series encoding
- PubMedBERT for clinical metadata encoding
- Cross-attention fusion mechanisms
"""

import logging
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

from .inception_1d import InceptionModel


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing image embeddings with RF time-series embeddings.
    
    This module implements a cross-attention fusion mechanism that allows the model
    to attend to temporal RF features when processing spatial image features.
    
    Args:
        dim_img: Dimension of image features.
        dim_time: Dimension of time-series features.
        dim_hidden: Hidden dimension for attention computation.
        num_heads: Number of attention heads.
        
    Input shapes:
        img_feat: (B, C_img, H, W) - Image feature maps
        rf_feat: (B, T, D_time) - RF time-series features
        
    Output shape:
        (B, C_img, H, W) - Fused features projected back to image grid
    """
    def __init__(self, dim_img: int, dim_time: int, dim_hidden: int = 256, num_heads: int = 8):
        super().__init__()
        self.query_proj = nn.Linear(dim_img, dim_hidden)
        self.key_proj = nn.Linear(dim_time, dim_hidden)
        self.value_proj = nn.Linear(dim_time, dim_hidden)
        self.norm = nn.LayerNorm(dim_hidden)
        self.ffn = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.num_heads = num_heads

    def forward(self, img_feat: torch.Tensor, rf_feat: torch.Tensor):
        """Forward pass through cross-attention fusion.
        
        Args:
            img_feat: Image features of shape (B, C_img, H, W).
            rf_feat: RF time-series features of shape (B, T, D_time).
            
        Returns:
            Fused features of shape (B, dim_hidden, H, W).
        """
        B, C, H, W = img_feat.size()
        img_seq = img_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C_img)

        # Expect rf_feat as (B, T, D_time). If different, try to coerce.
        if rf_feat.dim() == 3:
            rf_seq = rf_feat  # (B, T, D_time)
        else:
            # Best-effort fallback
            rf_seq = rf_feat.view(B, rf_feat.shape[-1], -1).transpose(1, 2)

        Q = self.query_proj(img_seq)  # (B, HW, Dh)
        K = self.key_proj(rf_seq)     # (B, T,  Dh)
        V = self.value_proj(rf_seq)   # (B, T,  Dh)

        # scaled dot-product attention (batch-first)
        attn_out = F.scaled_dot_product_attention(
            Q.transpose(0, 1),  # (HW, B, Dh)
            K.transpose(0, 1),  # (T,  B, Dh)
            V.transpose(0, 1),  # (T,  B, Dh)
        ).transpose(0, 1)  # (B, HW, Dh)

        out = self.norm(attn_out + Q)
        out = self.ffn(out) + out
        out = out.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, Dh, H, W)
        return out


class LinearLayers(nn.Module):
    """Simple linear layer stack with GELU activations.
    
    Args:
        in_dim: Input dimension.
        mid: Hidden dimension.
        out: Output dimension.
    """
    
    def __init__(self, in_dim: int = 768, mid: int = 512, out: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.GELU(),
            nn.Linear(mid, out),
            nn.GELU(),
        )

    def forward(self, x):
        """Forward pass through linear layers.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after linear transformations and activations.
        """
        return self.model(x)


class SimpleNN(nn.Module):
    """Simple neural network with two linear layers and GELU activations.
    
    Args:
        in_dim: Input dimension.
        mid: Hidden dimension.
        out: Output dimension.
    """
    
    def __init__(self, in_dim: int = 1024, mid: int = 512, out: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, mid)
        self.fc2 = nn.Linear(mid, out)
        self.gelu = nn.GELU()

    def forward(self, x):
        """Forward pass through the simple neural network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after two linear transformations with GELU activations.
        """
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return x


class BKMedSAM(nn.Module):
    """Main ProTeUS model integrating MedSAM backbone with temporal and metadata encoders.
    
    This is the core ProTeUS model that combines:
    - MedSAM backbone for spatial feature extraction
    - InceptionTime for RF time-series encoding  
    - PubMedBERT for clinical metadata encoding
    - Optional cross-attention fusion mechanisms
    
    Args:
        config: Configuration object containing model architecture parameters.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config

        from medAI.modeling.sam import (
            build_adapter_medsam_224,
            build_adapter_sam,
            build_adapter_sammed_2d,
            build_medsam,
            build_sam,
            build_sammed_2d,
        )

        # Backbone
        if config.architecture.sam_backbone == "medsam":
            self.medsam_model = build_medsam()
            self.image_size_for_features = 1024
        elif config.architecture.sam_backbone == "adapter_medsam":
            self.medsam_model = build_adapter_medsam_224()
            self.image_size_for_features = 1024
        elif config.architecture.sam_backbone == "sam":
            self.medsam_model = build_sam()
            self.image_size_for_features = 1024
        elif config.architecture.sam_backbone == "adapter_sam":
            self.medsam_model = build_adapter_sam()
            self.image_size_for_features = 1024
        elif config.architecture.sam_backbone == "sam_med2d":
            self.medsam_model = build_sammed_2d()
            self.image_size_for_features = 256
        elif config.architecture.sam_backbone == "adapter_sammed_2d":
            self.medsam_model = build_adapter_sammed_2d()
            self.image_size_for_features = 256
        else:
            raise ValueError(f"Unknown sam_backbone: {config.architecture.sam_backbone}")

        if config.architecture.freeze_image_encoder:
            for p in self.medsam_model.image_encoder.parameters():
                p.requires_grad = False
        if config.architecture.freeze_mask_decoder:
            for p in self.medsam_model.mask_decoder.parameters():
                p.requires_grad = False

        # Timeseries encoder (Inception 1D)
        self.timeseries_encoder = InceptionModel(
            in_channels=12,
            num_classes=1,
            out_channels=256,
            stride=1,
            bottleneck_channels=5,
            kernel_sizes=5,
            input_length=200,
            use_residuals="default",
            self_train=True,
            num_positions=0,
        )

        # Text encoders (PubMedBERT) – load by model name (no private filesystem path)
        hf_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        try:
            self.tokenizer = BertTokenizer.from_pretrained(hf_model_name, do_lower_case=True)
            self.text_encoder = BertModel.from_pretrained(
                hf_model_name,
                output_attentions=False,
                output_hidden_states=False,
            )
            for _, param in self.text_encoder.named_parameters():
                param.requires_grad = False
        except Exception as e:
            logging.warning("Failed to load PubMedBERT (%s). Text features will be disabled.", e)
            self.tokenizer = None
            self.text_encoder = None

        self.text_fc = LinearLayers(in_dim=768, mid=512, out=256)
        self.metadata_fc = LinearLayers(in_dim=768, mid=512, out=256)

    def forward(
        self,
        image,
        prostate_mask=None,
        needle_mask=None,
        ood_mask=None,
        return_prompt_embeddings: bool = False,
        timeseries_signal=None,
        other_cores=None,
        metadata=None,
        second_decoder: bool = False,
    ):
        """Forward pass through the ProTeUS model.
        
        Args:
            image: Input ultrasound image tensor.
            prostate_mask: Prostate segmentation mask (optional).
            needle_mask: Needle segmentation mask (optional).
            ood_mask: Out-of-distribution mask (optional).
            return_prompt_embeddings: Whether to return prompt embeddings.
            timeseries_signal: RF time-series signal for temporal encoding.
            other_cores: Text description of other biopsy cores.
            metadata: Clinical metadata text.
            second_decoder: Whether to use second decoder (if available).
            
        Returns:
            If return_prompt_embeddings=True: tuple of (mask_logits, sparse_embedding, dense_embedding)
            Otherwise: mask_logits tensor for cancer prediction
        """
        device = image.device
        B, C, H, W = image.shape

        # Resize if needed to backbone feature size
        if H != self.image_size_for_features or W != self.image_size_for_features:
            img_in = F.interpolate(image, size=(self.image_size_for_features, self.image_size_for_features))
        else:
            img_in = image
        image_feats = self.medsam_model.image_encoder(img_in.float())

        # Timeseries encoding (optional)
        ts_embed = None
        if timeseries_signal is not None:
            try:
                ts_embed = self.timeseries_encoder(timeseries_signal).to(device)
            except Exception as e:
                logging.debug("Timeseries encoder failed: %s", e)
                ts_embed = None

        # Text features: other_cores (optional)
        other_cores_down = None
        if other_cores and self.tokenizer is not None and self.text_encoder is not None:
            try:
                toks = self.tokenizer.encode_plus(
                    other_cores,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=128,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                enc = self.text_encoder(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
                other_cores_down = self.text_fc(enc.pooler_output)
            except Exception as e:
                logging.debug("other_cores text encode failed: %s", e)
                other_cores_down = None

        # Text features: metadata (optional)
        metadata_down = None
        if metadata and self.tokenizer is not None and self.text_encoder is not None:
            try:
                toks = self.tokenizer.encode_plus(
                    metadata,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=16,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                toks = {k: v.to(device) for k, v in toks.items()}
                enc = self.text_encoder(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"])
                metadata_down = self.metadata_fc(enc.pooler_output)
            except Exception as e:
                logging.debug("metadata text encode failed: %s", e)
                metadata_down = None

        # Mask prompt is disabled by default
        mask = None

        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(
            None, None, mask, ts_embed, other_cores_down, metadata_down
        )
        sparse_embedding = sparse_embedding.repeat_interleave(B, 0)

        if second_decoder and hasattr(self, "second_decoder"):
            mask_logits = self.second_decoder.forward(
                image_feats,
                self.medsam_model.prompt_encoder.get_dense_pe(),
                sparse_embedding,
                dense_embedding,
                multimask_output=False,
            )[0]
        else:
            mask_logits = self.medsam_model.mask_decoder.forward(
                image_feats,
                self.medsam_model.prompt_encoder.get_dense_pe(),
                sparse_embedding,
                dense_embedding,
                multimask_output=False,
            )[0]

        if return_prompt_embeddings:
            return mask_logits, sparse_embedding, dense_embedding
        return mask_logits

    def train(self, mode: bool = True):
        super().train(mode)

    def get_params_groups(self):
        """Get parameter groups for different learning rates.
        
        Returns:
            Tuple of (encoder_parameters, warmup_parameters) for optimizer configuration.
        """
        from itertools import chain

        encoder_parameters = [
            p for (k, p) in self.medsam_model.image_encoder.named_parameters() if "neck" not in k
        ]
        warmup_parameters = chain(self.medsam_model.mask_decoder.parameters())
        return encoder_parameters, warmup_parameters
