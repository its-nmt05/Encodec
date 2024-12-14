import torch.nn as nn


class LSTM(nn.Module):
    """Implementation of LSTM module. 
    LSTM requires input of shape (L, N, Hin) or (T, B, C) 
    if batch_first=False https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html"""
    def __init__(self, dim: int, num_layers: int = 2, skip:bool = True):
        super().__init__()
        self.lstm = nn.LSTM(dim, dim, num_layers)
        self.skip = skip
        
    def forward(self, x):
        x = x.permute(2, 0, 1) # (B, C, T) -> (T, B, C)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0) # (T, B, C) -> (B, C, T)
        return y