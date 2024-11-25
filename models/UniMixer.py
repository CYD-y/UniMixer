import torch
from torch import nn
from layers.Embed import PatchEmbedding, DataEmbedding_inverted
from layers.RevIN import RevIN
from einops import rearrange
from layers.UniMixerLayers import PWMixer, FlattenHead, zero_lag_corr_coef
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention

def compute_max_corr(matrix, threshold=0.7):
    bs, nvars, _ = matrix.shape
    mask = ~torch.eye(nvars, dtype=bool, device='cuda').unsqueeze(0).repeat(bs, 1, 1)
    masked_matrix = matrix.masked_fill(~mask, float('-inf'))
    max_corr, _ = masked_matrix.max(dim=2)
    result = (max_corr > threshold).int()
    return result

    
class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        padding = self.stride
        self.k = configs.k
        self.patch_num = int((self.seq_len - patch_len) / stride) + 2
    
        self.inverted_embedding = DataEmbedding_inverted(
            configs.seq_len, self.k*configs.d_model, configs.embed, configs.freq, configs.fc_dropout)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.fc_dropout)
        
        self.revin = RevIN(channel=configs.enc_in, output_dim=configs.pred_len)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), (self.k+1)*configs.d_model, configs.n_heads),
                    (self.k+1)*configs.d_model,
                    configs.d_ff,
                    configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm((self.k+1)*configs.d_model)
        )

        self.varencoder = PWMixer(configs.d_model, configs.d_ff, configs.dropout, configs.enc_in, self.patch_num+1, configs.e_layers, patch_len)
        
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        self.patchhead = FlattenHead(configs.enc_in, configs.d_ff, self.head_nf, configs.pred_len,
                                head_dropout=configs.head_dropout)
        self.project = nn.Sequential(
            nn.Linear((self.k+1)*configs.d_model, configs.d_ff),
            nn.GELU(),
            nn.Dropout(configs.fc_dropout),
            nn.Linear(configs.d_ff, self.patch_num*configs.d_model),
        )
        self.linear = nn.Linear(2*configs.d_model, configs.d_model)
        self.corr_linear = nn.Linear(configs.enc_in, configs.d_model)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x_enc = self.revin(x_enc) # bs, seq_len, nvars
        n_vars = x_enc.shape[2]
        
        corr = zero_lag_corr_coef(x_enc.permute(0, 2, 1)) # bs, nvars, nvars
        corr_emb = self.corr_linear(corr) # bs, nvars, d_model

        x_emb = self.inverted_embedding(x_enc, None) # b, n, d
        x_emb = torch.cat([x_emb, corr_emb], dim=-1) # b, n, (k+1)d
        invertedenc_out, attn = self.encoder(x_emb) # bs, nvars, k*d_model
        invertedenc_out = self.project(invertedenc_out) # bs, nvars, (patch_num d_model)
        invertedenc_out = rearrange(invertedenc_out, 'b n (p d) -> b n p d', p=self.patch_num)
        
        x_patch, n_vars = self.patch_embedding(x_enc.permute(0, 2, 1)) # bs * nvars x patch_nums x d_model 
        corr_emb = corr_emb.unsqueeze(-2)
        x_patch = rearrange(x_patch, '(b n) p d -> b n p d', n=n_vars) # bs x nvars x patch_num x d_model
        x_patch = torch.cat([x_patch, corr_emb], dim=-2) # bs * nvars x (patch_nums+1) x d_model
        patchenc_out = self.varencoder(x_patch)[:, :, :-1, :] # bs x nvars x patch_num x d_model

        dec_out = self.linear(torch.cat([patchenc_out, invertedenc_out], dim=-1)) # bs x nvars x patch_num x d_model
        dec_out = self.patchhead(dec_out)  # bs x nvars x pred_len
    
        dec_out = dec_out.permute(0, 2, 1) # bs, pred_len, nvars
        
        dec_out = self.revin.inverse_normalize(dec_out)
    
        return dec_out[:, -self.pred_len:, :] # bs, pred_len, nvars