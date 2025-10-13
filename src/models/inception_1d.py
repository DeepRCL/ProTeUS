"""InceptionTime model for 1D time series classification.

This module implements the InceptionTime architecture for processing
RF time-series signals in prostate cancer detection.
"""

import torch
from torch import nn

from typing import cast, Union, List
import torch.nn.functional as F

ACT = nn.ReLU


class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    
    From https://arxiv.org/abs/1909.04939
    
    This model processes 1D time series data through inception blocks,
    each containing multiple convolutional layers with different kernel sizes.
    
    Attributes:
        num_blocks: The number of inception blocks to use.
        in_channels: The number of input channels.
        out_channels: The number of "hidden channels" to use.
        bottleneck_channels: The number of channels to use for the bottleneck.
        kernel_sizes: The size of the kernels to use for each inception block.
        num_pred_classes: The number of output classes.
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: Union[List[int], int],
                 bottleneck_channels: Union[List[int], int], kernel_sizes: Union[List[int], int],
                 input_length: int, use_residuals: Union[List[bool], bool, str] = 'default',
                 num_pred_classes: int = 1, self_train=False, stride=1,
                 num_positions=0,
                 ) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'num_blocks': num_blocks,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'bottleneck_channels': bottleneck_channels,
            'kernel_sizes': kernel_sizes,
            'use_residuals': use_residuals,
            'num_pred_classes': num_pred_classes,
            'stride': stride,
            'input_length': input_length,
        }
        self.self_train = self_train
        channels = [in_channels] + [out_channels for i in range(num_blocks)]
        bottleneck_channels = [bottleneck_channels] * num_blocks
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        strides = cast(List[int], self._expand_to_blocks(stride, num_blocks))
        if use_residuals == 'default':
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(List[bool], self._expand_to_blocks(
            cast(Union[bool, List[bool]], use_residuals), num_blocks))

        self.blocks = nn.Sequential(*[
            nn.Sequential(InceptionBlock(in_channels=channels[i], out_channels=channels[i + 1],
                                         residual=use_residuals[i], bottleneck_channels=bottleneck_channels[i],
                                         stride=strides[i],
                                         kernel_size=kernel_sizes[i]),
                          ) for i in range(num_blocks)
        ])

        self.feature_size = channels[-1]
        self.num_positions = num_positions

        self.feat_extractor = FeatureExtractor(self.blocks, nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        self.classifier = Classifier(channels[-1], channels[-1], num_pred_classes)

    @staticmethod
    def _expand_to_blocks(value: Union[int, bool, List[int], List[bool]],
                          num_blocks: int) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, \
                f'Length of inputs lists must be the same as num blocks, ' \
                f'expected length {num_blocks}, got {len(value)}'
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.feat_extractor(x, *args)
        if self.num_positions > 0:
            x = self.pos_encoder1(torch.cat((x, args[0].float()), 1))
            x = self.pos_encoder2(torch.cat((x, args[0].float()), 1))
            x = self.out(torch.cat((x, args[0].float()), 1))
        if self.self_train:
            return F.normalize(x, dim=1)

        return x

    def train(self, mode=True, freeze_bn=False):
        """
        Override the default train() to freeze the BN parameters
        """
        super(InceptionModel, self).train(mode)
        self.freeze_bn = freeze_bn
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41, drop: float = None, groups=1) -> None:
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels, kernel_size=1, bias=False, groups=1)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        strides = [1, 1, stride]
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=strides[i], bias=False, groups=groups)
            for i in range(len(kernel_size_s))
        ])

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm1d(out_channels),
                ACT()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.net = nn.Sequential(*args)

    def forward(self, x, *args):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, feat_dim, out_dim, num_pred_classes):
        super().__init__()
        self.linear01 = nn.Sequential(
            nn.Linear(out_dim, num_pred_classes),
        )

    def forward(self, x, *args):
        return self.linear01(x)


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1)) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            # nn.ReLU(),
            ACT(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)
