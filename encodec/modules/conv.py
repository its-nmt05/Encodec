import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from .norm import ConvLayerNorm

import typing as tp
import math

# def apply_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
#     if norm == 'layer_norm':
#         return ConvLayerNorm(module)
#     elif norm == 'weight_norm':
#         return weight_norm(module)
#     else:
#         return module
    
def get_norm_module(module: nn.Module, norm: str = 'none') -> nn.Module:
    """Returns the normalization module based on the given norm type"""
    
    if norm == 'layer_norm':
        return ConvLayerNorm(module.out_channels)
    elif norm == 'weight_norm':
        return weight_norm
    else:
        return nn.Identity()
    
def get_extra_padding(x: torch.Tensor, kernel_size: int, stride: int, padding_total: int):
    """We add some extra padding to the the end of the input tensor to ensure that
    we can utilize all of the input and none of the time steps are lost due to uneven 
    padding """
    length = x.shape[-1]
    n_out = (length + padding_total - kernel_size) / stride + 1 # np_out = (n_in + 2*padding - kernel_size) / stride + 1
    ideal_length = (math.ceil(n_out) - 1) * stride + (kernel_size - padding_total)  
    return ideal_length - length    # extra padding required
    

    
def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'constant', value: float = 0.0) -> torch.Tensor:
    """Applies padding to the input tensor. In case of 'reflect' mode, modifies the input if necessary for padding"""
    length = x.shape[-1]    # number of time steps
    padding_left, padding_right = paddings
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))   # add extra padding to the right
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad  # remove extra padding
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)
    

class NormConv1d(nn.Module):
    def __init__(self, norm: str = 'none', *args):
        super().__init__()
        self.conv = nn.Conv1d(*args)
        self.norm = get_norm_module(self.conv, norm)

    def forward(self, x):
        return self.norm(self.conv(x))


# class NormConv2d(nn.Module):
#     def __init_(self):
#         super().__init__()
#         self.conv = nn.Conv2d()
#         self.norm = nn.LayerNorm()

#     def forward(self, x):
#         return self.norm(self.conv(x))


class NormConvTranspose1d(nn.Module):
    def __init__(self, norm: str = 'none', *args):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(*args)
        self.norm = get_norm_module(self.convtr, norm)

    def forward(self, x):
        return self.norm(self.convtr(x))

# class NormConvTranspose2d(nn.Module):
#     pass


class SConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1, bias: bool = True, causal: bool = False, norm: str = 'none', pad_mode: str = 'reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride, dilation, bias, norm=norm)
        self.causal = causal
        self.pad_mode = pad_mode
        

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilation
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding(x, kernel_size, stride, padding_total)
        if self.causal:
            # only left padding for causal
            pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # apply padding on both sides for non-causal
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)