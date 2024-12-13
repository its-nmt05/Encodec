import torch.nn as nn
from .conv import SConv1d
import typing as tp

class ResnetBlock(nn.Module):
    """Residual block modelled after the SEANet paper.
    compress (int): Authors have implemented a reduced dimensionality
    inside the residual branches (from Demucs v3). 
    Maybe to represent them as bottleneck layers.
    """
    
    def __init__(self, dim: int, kernel_sizes: tp.List[int] = [3, 3], dilations: tp.List[int] = [1, 1], 
                 activation_params: dict = {'alpha': 1.0}, norm: str = 'layer_norm', 
                 causal: bool = False, pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True):
        super().__init__()
        block = []
        act = nn.ELU(**activation_params)
        hidden_dim = dim // compress    
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_channels = dim if i == 0 else hidden_dim
            out_channels = dim if i == len(kernel_sizes) - 1 else hidden_dim
            block += [
                act,
                SConv1d(in_channels, out_channels, kernel_size, dilation, 
                        norm=norm, causal=causal, pad_mode=pad_mode)
            ]
            self.block = nn.Sequential(*block)
            
            if true_skip:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = SConv1d(dim, dim, kernel_size=1, norm=norm, causal=causal, pad_mode=pad_mode)
        
    def forward(self, x):
        return self.block(x) + self.shortcut(x)