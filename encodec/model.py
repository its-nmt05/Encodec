import torch.nn as nn
import torch

from . import modules as m
from . import quantization as qt

import typing as tp
import numpy as np
import math

from .utils import _linear_overlap_add

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]    # (codes, scale)


class EncodecModel(nn.Module):
    """Encodec model"""
    def __init__(self, encoder: m.Encoder, decoder: m.Decoder, qauntizer: qt.ResiualVectorQuantizer, 
                 target_bandwidths: tp.List[float], sample_rate: int, channels: int, normalize: bool = False, 
                 segment: tp.Optional[float] = None, overlap: float = 0.01, name: str = 'encodec'):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = qauntizer
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.name = name
        self.bits_per_codebook = int(math.log2(self.quantizer.codebook_size))
    
    @property        
    def segment_length(self):
        if not self.segment:
            return None
        return int(self.segment * self.sample_rate)
    
    @property
    def segment_stride(self):
        if not self.segment:
            return None
        return max(1, int((1 - self.overlap) * self.segment_length))

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        if self.normalize:
            mono = x.mean(dim=1, keepdim=True)  # squeeze channel dimension
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt() # calculate volume as RMS
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
            
        emb = self.encoder(x)
        
        if self.training:
            return emb, scale
        
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1) # (Nq, B, T) -> (B, Nq, T)
        return codes, scale
    
    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        codes, scale = encoded_frame
        if self.training:   #training: codes are embeddings 
            emb = codes
        else:
            codes = codes.transpose(0, 1)
            emb = self.quantizer.decode(codes)   
        out = self.decoder(emb) 
        if scale is not None:
            out = out * scale.view(-1, 1, 1) # scale back to original volume
        return out
    
    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        _, channel, length = x.shape
        segment_length = self.segment_length
        if not self.segment_length: # no segegment overlap
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride
            
        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):    # move by stride across the length
            frame = x[:, :, offset : offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames
    
    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        segment_length = self.segment_length
        if segment_length is None:
            return self._decode_frame(encoded_frames[0])
        
        frames = [self._decode_frame(frame) for frame in encoded_frames]
        return _linear_overlap_add(frames, self.segment_stride or 1) # overlap and add frames
    
    def forward(self, x: torch.Tensor):
        encoded_frames = self.encode(x) # input_wav -> encoder
        
        if self.training:
            # training: input_wav -> encoder -> quantizer forward -> decode
            loss_w = torch.tensor([0.0], device=x.device, requires_grad=True)
            codes = []
            index = torch.randint(0, len(self.target_bandwidths), (1,), device=x.device)   
            bw = self.target_bandwidths[index.item()] # variable bandwidth training
            for emb, scale in encoded_frames:
                qv = self.quantizer(emb, self.frame_rate, bw)
                loss_w = loss_w + qv.penalty # RVQ commit loss
                codes.append((qv.quantized, scale))
            return self.decode(codes)[:, :, :x.shape[-1]], loss_w
        else:
            # not training: input_wav -> encoder -> quantizer encode -> decode
            return self.decode(encoded_frames)[:, :, :x.shape[-1]] # trim to original length
    
    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f'Bandwidth {bandwidth} not in target bandwidths {self.target_bandwidths}')
        self.bandwidth = bandwidth
        
    @staticmethod
    def _get_model(target_bandwidths: tp.List[float], sample_rate: int = 24000, 
                  channels: int = 1, causal: bool = False, model_norm: str = 'time_group_norm',
                  audio_normalize: bool = False, segment: tp.Optional[float] = None, name: str = 'encodec'):
        encoder = m.Encoder(channels=channels, norm=model_norm, causal=causal)
        decoder = m.Decoder(channels=channels, norm=model_norm, causal=causal)
        # ex. (6 kbps * 1000) / (24000Hz / 320 * 10 bits) -> 6000 / 750 = 8 quantizers
        n_q = int((target_bandwidths[-1] * 1000) // (math.ceil(sample_rate / encoder.hop_length) * 10)) 
        quantizer = qt.ResiualVectorQuantizer(n_q=n_q, dim=encoder.dimension, codebook_size=1024)
        model = EncodecModel(encoder, decoder, quantizer, target_bandwidths, sample_rate, channels, 
                             normalize=audio_normalize, segment=segment, name=name)
        return model