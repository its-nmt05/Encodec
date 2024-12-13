"""Modified form of LayerNorm for convolutions that moves the channel dimension to the end before applying the norm"""
import torch.nn as nn


class ConvLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, 2, 1).contiguous()
        return x
