import torch
import torch.nn as nn
import torch.nn.functional as F

from librosa.filters import mel as librosa_mel_fn


class Audio2Mel(nn.Module):
    """Converts audio to mel spectrogram"""
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, 
                 win_length: int = 1024, sampling_rate: int = 24000, 
                 n_mel_banks: int = 64, mel_fmin: float = 0.0, 
                 mel_fmax: float = None, device: str = 'cuda'):
        super().__init__()
        window = torch.hann_window(win_length, device=device).float()
        # create a mel filter bank to transform fft bins to mel bins
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_banks, fmin=mel_fmin, fmax=mel_fmax) 
        mel_basis = torch.from_numpy(mel_basis).float().to(device) # move to device
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('window', window)
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_banks = n_mel_banks
        
    def forward(self, audioin):
        shape = audioin.shape
        if len(shape) > 2:
            audioin = audioin.reshape(shape[0] * shape[1], -1)
        p = (self.win_length - self.hop_length) // 2 # padding
        audio = F.pad(audioin, (p, p), mode='reflect') # pad audio
        fft = torch.stft(audio, n_fft=self.n_fft, 
                   hop_length=self.hop_length, 
                   win_length=self.win_length, 
                   window=self.window, 
                   center = False, return_complex=True)
        fft = torch.view_as_real(fft) # convert to real
        mel_output = torch.matmul(self.mel_basis, torch.sum(torch.pow(fft, 2), dim=[-1])) # apply mel filter banks
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))   # convert mel to log scale
        # restore original shape [B, H, T]
        if len(shape) > 2:
            log_mel_spec = log_mel_spec.reshape(shape[0], shape[1], -1)
        return log_mel_spec