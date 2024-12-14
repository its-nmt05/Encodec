import torch.nn as nn
import typing as tp
import numpy as np

from conv import SConv1d, SConvTranspose1d
from resnet import ResnetBlock
from lstm import LSTM

import torch


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
            SConv1d(channels, n_filters, kernel_size, norm=norm, causal=causal, pad_mode=pad_mode)
        ]
        
        # Downsampling process
        for i, ratio in enumerate(self.ratios):
            # Conv block
            # The base model has one residual unit per conv block
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
        if lstm:
            model += [LSTM(mult * n_filters, num_layers=lstm)] 
            
        # final conv1D
        model += [
            act(**activation_params),
            SConv1d(mult * n_filters, dimension, kernel_size=last_kernel_size, 
                    norm=norm, causal=causal, pad_mode=pad_mode),
        ]
            
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)
        
        
class Decoder(nn.Module):
    
    def __init__(self, channels: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1, 
                 ratios: tp.List[int] = [8, 5, 4, 2], activation_params: dict = {'alpha': 1.0}, 
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None, 
                 norm: str = 'layer_norm', kernel_size: int = 7, last_kernel_size: int = 7, residual_kernal_size: int = 3, 
                 dialation: int = 2, causal: bool = False, pad_mode: str = 'reflect', true_skip: bool = False, 
                 compress: int = 2, lstm: int = 2, trim_right_ratio: float = 1.0):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.n_residual_layers = n_residual_layers
        self.ratios = ratios
        self.hop_length = np.prod(self.ratios)
        
        act = nn.ELU
        mult = int(2 ** len(self.ratios))    # 2 ** 4 = 16  
        model: tp.List[nn.Module] = [
            SConv1d(dimension, mult * n_filters, kernel_size, norm=norm,
                    causal=causal, pad_mode=pad_mode),
        ]
        
        if lstm:
            model += [LSTM(mult * n_filters, num_layers=lstm)]
            
        # Upsampling process
        for i, ratio in enumerate(self.ratios):
            # Conv block
            # add upsampling layer
            model += [
                act(**activation_params),
                SConvTranspose1d(mult * n_filters, mult * n_filters // 2, kernel_size=ratio * 2, 
                                stride=ratio, norm=norm, causal=causal, trim_right_ratio=trim_right_ratio), 
            ]
            
            # The base model has one residual unit per conv block
            model += [
                ResnetBlock(mult * n_filters // 2, kernel_sizes=[residual_kernal_size, 1], 
                            dilations=[dialation, 1], 
                            norm=norm, activation_params=activation_params, 
                            causal=causal, pad_mode=pad_mode, 
                            compress=compress, true_skip=true_skip)]
            mult //= 2
            
        # final conv1D
        model += [
            act(**activation_params),
            SConv1d(n_filters, channels, kernel_size=last_kernel_size, 
                    norm=norm, causal=causal, pad_mode=pad_mode),
        ]
        
        # optional final activation for decoder
        if final_activation is not None:
            fianl_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [
                fianl_act(**final_activation_params)
            ]
        self.model = nn.Sequential(*model)
            
    def forward(self, z):
        return self.model(z)