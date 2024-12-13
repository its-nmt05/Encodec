import torch.nn as nn
import typing as tp
import numpy as np

from .conv import SConv1d
from .resnet import ResnetBlock
from .lstm import LSTM


class Encoder(nn.Module):
    
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1, 
                ratios: tp.List[int] = [2, 4, 5, 8], activation_params: dict = {'alpha': 1.0}, norm: str = 'layer_norm',
                kernel_size: int = 7, last_kernel_size: int = 7, residual_kernal_size: int = 3, dialation: int = 2, causal: bool = False, 
                pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2, lstm: int = 2):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.n_residual_layers = n_residual_layers
        self.ratios = ratios
        self.hop_length = np.prod(self.ratios)
        
        mult = 1
        act = nn.ELU
        model: tp.List[nn.Module] = [
            SConv1d(channels, n_filters, kernel_size, dialation, norm=norm, causal=causal, pad_mode=pad_mode)
        ]
        
        # Downsampling process
        for i, ratio in enumerate(self.ratios):
            # Residual layers
            # The base model has onr residual unit per conv block
            model += [
                ResnetBlock(mult * n_filters, kernel_sizes=[residual_kernal_size, 1], 
                            dilations=[dialation, 1], 
                            norm=norm, activation_params=activation_params, 
                            causal=causal, pad_mode=pad_mode, 
                            compress=compress, true_skip=true_skip)]
            
            # add downsampling layer
            model += [
                act(**activation_params),
                SConv1d(mult * n_filters, mult * n_filters * 2, kernel_size=ratio * 2, 
                        stride=ratio, norm=norm, causal=causal, pad_mode=pad_mode),
            ]
            mult *= 2
        
            # LSTM layer
            # model += [LSTM()] 
            
            # final conv1D
            model += [
                act(**activation_params),
                SConv1d(mult * n_filters, dimension, kernel_size=last_kernel_size, 
                        norm=norm, causal=causal, pad_mode=pad_mode),    
            ]
            
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
        
        
    