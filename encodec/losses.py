from encodec.audio_to_mel import Audio2Mel
import torch

def total_loss(input_wav, output_wav, sample_rate=24000, device='cuda'):
    """compute total loss"""
    l1loss = torch.nn.L1Loss(reduction='mean')
    l2loss = torch.nn.MSELoss(reduction='mean')
    # l_t loss - L1 distance b/w target and compressed audio in time domain
    # l_f loss - linear comb. of L1 + L2 losses across frequency domain
    #            over a mel spectrogram using several time scales 
    l_t = torch.tensor([0.0], device=device, requires_grad=True)
    l_f = torch.tensor([0.0], device=device, requires_grad=True)
    
    l_t = l1loss(input_wav, output_wav)
    
    for i in range(5, 12): # time scales, e = 5,..., 11
        window_size = 2**i
        hop_length = window_size // 4
        n_mel_bins = 64
        
        # S_i(64 bins mel-soectrogram) according to the paper
        S_i = Audio2Mel(n_fft=window_size, hop_length=hop_length, 
                  win_length=window_size, sampling_rate=sample_rate, 
                  n_mel_banks=n_mel_bins, device=device)
        l_f = l_f + l1loss(S_i(input_wav), S_i(output_wav)) + l2loss(S_i(input_wav), S_i(output_wav))
        
    return {
        'l_t': l_t, 
        'l_f': l_f
    }