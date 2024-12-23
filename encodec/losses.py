from encodec.audio_to_mel import Audio2Mel
import torch

def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=24000, device='cuda'):
    """compute total loss"""
    relu = torch.nn.ReLU()
    l1loss = torch.nn.L1Loss(reduction='mean')
    l2loss = torch.nn.MSELoss(reduction='mean')
    # l_t loss - L1 distance b/w target and compressed audio in time domain
    # l_f loss - linear comb. of L1 + L2 losses across frequency domain
    #            over a mel spectrogram using several time scales 
    l_t = torch.tensor([0.0], device=device, requires_grad=True)
    l_f = torch.tensor([0.0], device=device, requires_grad=True)
    l_g = torch.tensor([0.0], device=device, requires_grad=True)
    l_feat = torch.tensor([0.0], device=device, requires_grad=True)
    
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
        
    for tt1 in range(len(fmap_real)):
        l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
        for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
            # l_feat = l_feat + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach()))
            l_feat = l_feat + l1loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))
        
    KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
    l_feat /= KL_scale
    K_scale = len(fmap_real) # len(fmap_real[0]) = len(fmap_fake[0]) == L
    l_g /= K_scale
    l_f /= 7
    
    return {
        'l_t': l_t, 
        'l_f': l_f,
        'l_g': l_g,
        'l_feat': l_feat
    }
    
def disc_loss(logits_real, logits_fake):
    """This function is used to compute the loss of the discriminator.
        l_d = \sum max(0, 1 - D_k(x)) + max(0, 1 + D_k(\hat x)) / K, K = disc.num_discriminators = len(logits_real) = len(logits_fake) = 3
    Args:
        logits_real (List[torch.Tensor]): logits_real = disc_model(input_wav)[0]
        logits_fake (List[torch.Tensor]): logits_fake = disc_model(model(input_wav)[0])[0]

    Returns:
        lossd: discriminator loss
    """
    relu = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device='cuda', requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(relu(1-logits_real[tt1])) + torch.mean(relu(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd