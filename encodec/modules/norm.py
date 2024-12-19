import torch.nn as nn
import einops

class ConvLayerNorm(nn.LayerNorm):
    """Modified form of LayerNorm for convolutions that moves the channel dimension to the end before applying the norm"""
    
    def __init__(self, normalized_shape):
        super().__init__(normalized_shape)

    def forward(self, x):
        x = einops.rearrange(x, "b ... t -> b t ...") # (B, ..., T) -> (B, T, ...)
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t") # (B, T, ...) -> (B, ..., T)
        return x
