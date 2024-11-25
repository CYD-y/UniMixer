import torch
import torch.nn as nn
from einops import rearrange

class PWLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, n_vars, patch_nums, patch_len):
        super(PWLayer, self).__init__()
        self.timeencoder = nn.Sequential(
            nn.Linear(patch_nums * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, patch_nums * d_model),
            nn.Dropout(dropout)
        )
        self.varencoder = nn.Sequential(
            nn.Linear(n_vars * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, n_vars * d_model),
            nn.Dropout(dropout)
        )
        self.norm_layer = nn.LayerNorm(d_model)
    
    def forward(self, x): # bs, nvars, patchnums, d_model.
        b, n, p, d = x.shape
        x_in = rearrange(x, 'b n p d -> b p (n d)')
        x = rearrange(x, 'b n p d -> b p (n d)') # bs, nvars*patchnums, d_model
        x = self.varencoder(x) + x_in
        x_in = rearrange(x_in, 'b p (n d) -> b n (p d)', n=n)
        x = rearrange(x, 'b p (n d) -> b n (p d)', n=n) # bs, patchnums, nvars*d_model
        x = self.timeencoder(x) +  x_in
        x = rearrange(x, 'b n (p d) -> b n p d', p=p) # bs, nvars, patchnums, d_model
        x = self.norm_layer(x) 
        return x

class PWMixer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, n_vars, patch_nums, num_layers=1, patch_len=16):
        super(PWMixer, self).__init__()
        self.layers = nn.ModuleList([PWLayer(d_model, d_ff, dropout, n_vars, patch_nums, patch_len) for _ in range(num_layers)])

    def forward(self, x): # bs, nvars, patchnums, d_model.
        for layer in self.layers:
            x = layer(x)
        return x

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

class FlattenHead(nn.Module):
    def __init__(self, n_vars, d_ff, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.mlp = nn.Sequential(
            nn.Linear(nf, target_window*2), 
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(target_window*2, target_window),
        )

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x) # bs, nvars, d_model × patch_num
        x = self.mlp(x)
        return x

def zero_lag_corr_coef(x, variable_batch_size=32):
    B, C, L = x.shape

    rfft = torch.fft.rfft(x, dim=-1)  # [B, C, F]
    rfft_conj = torch.conj(rfft)

    cross_corr = torch.cat([
        torch.fft.irfft(rfft.unsqueeze(2) * rfft_conj[:, i: i + variable_batch_size].unsqueeze(1),
                        dim=-1, n=L)[..., 0:1]
        for i in range(0, C, variable_batch_size)],
        2)  # [B, C, C, 1]

    return cross_corr.squeeze(-1) / L  # [B, C, C]

