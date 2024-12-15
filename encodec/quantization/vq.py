import torch
import torch.nn as nn
from core_vq import ResidualVectorQuantization

from dataclasses import dataclass, field
import typing as tp
import math


@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor # bandwidth is stored in kbps
    penalty: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)


class ResiualVectorQuantizer(nn.Module):
    """***Residual Vector Quantizer***"""
    def __init__(self, n_q: int = 8, dim: int = 128, codebook_size: int = 1024, decay: float = 0.99,
                 kmeans_init: bool = True, kmeans_iters: int = 50, threshold_ema_dead_code: int = 2):
        super().__init__()
        self.n_q = n_q
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
                dim=self.dim, codebook_size=self.codebook_size,
                num_quantizers=self.n_q, decay=self.decay,
                kmeans_init=self.kmeans_init, kmeans_iters=self.kmeans_iters,
                threshold_ema_dead_code=self.threshold_ema_dead_code
            )
        
    def forward(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> QuantizedResult:
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        quantized, codes, commit_loss = self.vq(x, n_q)
        bw = torch.tensor(n_q * bw_per_q).to(x.device)
        return QuantizedResult(quantized, codes, bw, penalty=torch.mean(commit_loss))
    
    def get_num_quantizers_for_bandwidth(self, frame_rate: int, bandwidth: tp.Optional[float] = None):
        """get the number of quantizers required for a given bandwidth"""
        bw_per_q = self.get_bandwidth_per_quantizer(frame_rate)
        n_q = self.n_q
        if bandwidth and bandwidth > 0.:
            n_q = int(max(1, math.floor(bandwidth * 1000 / bw_per_q))) # bandwidth = 6.0 for 6kbps
        return n_q
    
    def get_bandwidth_per_quantizer(self, frame_rate: int):
        """each quantizer has a bandwidth of log2(codebook_size) * frame_rate"""
        return math.log2(self.codebook_size) * frame_rate
    
    def encode(self, x: torch.Tensor, frame_rate: int, bandwidth: tp.Optional[float] = None) -> torch.Tensor:
        n_q = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        codes = self.vq.encode(x, n_q)
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.vq.decode(codes)
        return quantized
    