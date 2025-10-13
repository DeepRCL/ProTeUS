"""SAM-based classifier for prostate cancer detection.

This module contains SAM-based models for prostate cancer detection,
including a convolutional classifier that operates on SAM image features.
"""

import torch
import typing as tp
from dataclasses import dataclass, field
from torch import nn
from enum import StrEnum
import logging
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvNet Classifier for SAM Image Features
class ConvNet(nn.Module):
    """Convolutional neural network classifier for SAM image features.
    
    This network processes SAM image features through convolutional layers
    followed by fully connected layers for binary classification.
    
    Args:
        num_classes: Number of output classes (typically 2 for cancer detection).
    """
    
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        # Input size: [1, 256, 64, 64]
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1) # -> [1, 128, 64, 64]
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)  # -> [1, 64, 64, 64]
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)   # -> [1, 32, 64, 64]
        
        # Pooling layer (downsample)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # -> Reduces size by half [1, C, 32, 32]

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 512)  # Flatten -> FC layer
        self.fc2 = nn.Linear(512, num_classes)   # Output layer

    def forward(self, x):
        """Forward pass through the convolutional network.
        
        Args:
            x: Input tensor of shape (B, 256, 64, 64).
            
        Returns:
            Output tensor of shape (B, num_classes).
        """
        # Convolution + ReLU + Pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # [1, 128, 32, 32]

        x = F.relu(self.conv2(x))
        x = self.pool(x)  # [1, 64, 16, 16]

        x = F.relu(self.conv3(x))
        x = self.pool(x)  # [1, 32, 8, 8]

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 32 * 8 * 8)  # [1, 32*8*8]

        # Fully connected layers
        x = F.relu(self.fc1(x))  # [1, 512]
        x = self.fc2(x)  # [1, num_classes]

        return x

class SAMClassifier(nn.Module):
    """SAM-based classifier for prostate cancer detection.
    
    This model combines SAM backbone with a convolutional classifier for
    binary cancer classification using image features.
    
    Args:
        config: Configuration object containing model architecture parameters.
    """
    
    def __init__(
        self,
        config,
    ):
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

        # BUILD BACKBONE
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
        elif config.architecture.sam_backbone == "adapter_sammed_2d":
            self.medsam_model = build_adapter_sammed_2d()
            self.image_size_for_features = 256

        if config.architecture.freeze_image_encoder:
            logging.debug("Freezing image encoder")
            for param in self.medsam_model.image_encoder.parameters():
                param.requires_grad = False

        if config.architecture.freeze_mask_decoder:
            logging.debug("Freezing mask decoder")
            for param in self.medsam_model.mask_decoder.parameters():
                param.requires_grad = False

        self.classifier = ConvNet(num_classes=2)

    def forward(
        self,
        image,
        prostate_mask=None,
        needle_mask=None,
        ood_mask=None,
        return_prompt_embeddings=False,
    ):
        B, C, H, W = image.shape
        if H != self.image_size_for_features or W != self.image_size_for_features:
            image_resized_for_features = torch.nn.functional.interpolate(
                image, size=(self.image_size_for_features, self.image_size_for_features)
            )
        else:
            image_resized_for_features = image
        image_feats = self.medsam_model.image_encoder(image_resized_for_features.float())

        #if "prostate_mask" in self.config.prompts:
        #    if (
        #        prostate_mask is None
        #        or self.config.prompt_dropout > 0
        #        and self.training
        #        and torch.rand(1) < self.config.prompt_dropout
        #    ):
        #        mask = None
        #    else:
        #        B, C, H, W = prostate_mask.shape
        #        if H != 256 or W != 256:
        #            prostate_mask = torch.nn.functional.interpolate(
        #                prostate_mask, size=(256, 256)
        #            )
        #        mask = prostate_mask
        #else:
        #    mask = None
        mask = None
        
        # use the prompt encoder to get the prompt embeddings for the mask, if provided.
        # otherwise we will use our own custom prompt modules exclusively
        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(
            None, None, mask
        )
        # sparse embeddings will be an empty tensor if the mask is None,
        # and we need to repeat the embeddings for each image in the batch
        sparse_embedding = sparse_embedding.repeat_interleave(len(image), 0)

        logits = self.classifier(image_feats)

        return logits

    def train(self, mode: bool = True):
        super().train(mode)

    def get_params_groups(self):
        """Return the parameters groups for the optimizer,
        which will be used to set different learning rates for different parts of the model.

        Returns:
            Tuple[tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter]]:
                encoder_parameters, warmup_parameters, cnn_parameters
                (warmup_parameters are the parameters for the prompt modules and the prompt encoder and mask decoder)
        """

        from itertools import chain

        encoder_parameters = [
            p
            for (k, p) in self.medsam_model.image_encoder.named_parameters()
            if "neck" not in k
        ]
        warmup_parameters = chain(
            self.medsam_model.mask_decoder.parameters(),
        )

        return encoder_parameters, warmup_parameters
