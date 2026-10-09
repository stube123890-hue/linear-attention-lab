"""V12 multimodal: AR sequence modeling over vision/audio on the frozen V8-A stack.

New code only — models.py / triton_kernels.py / fused_mixer.py are FROZEN
(his V12 discipline). This file adds modality encoders + a continuous
decoder-only AR model reusing the frozen Block.

Task (his call): autoregressive next-element prediction (MSE), the direct
analogue of the text campaign's val-loss comparison.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Block  # frozen


# ---------------------------------------------------------------- encoders

class VisionEncoder(nn.Module):
    """Patchify + linear: (B,3,H,W) -> (B,T,dim); elements: (B,T,E) raw patches."""

    def __init__(self, dim, patch=8, in_ch=3):
        super().__init__()
        self.patch = patch
        self.in_ch = in_ch
        self.elem_dim = in_ch * patch * patch
        self.proj = nn.Linear(self.elem_dim, dim)

    def elements(self, img):
        B, C, H, W = img.shape
        p = self.patch
        assert H % p == 0 and W % p == 0, (H, W, p)
        x = img.view(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        return x.view(B, (H // p) * (W // p), self.elem_dim)

    def forward(self, img):
        return self.proj(self.elements(img))

    @property
    def seq_len(self):
        return None  # set by image size at runtime


class AudioEncoder(nn.Module):
    """Linear on log-mel frames: (B,T,80) -> (B,T,dim); elements = mel itself."""

    def __init__(self, dim, n_mels=80):
        super().__init__()
        self.elem_dim = n_mels
        self.proj = nn.Linear(n_mels, dim)

    def elements(self, mel):
        return mel

    def forward(self, mel):
        return self.proj(mel)


# ------------------------------------------------------- log-mel (no deps)

_mel_fb_cache = {}


def log_mel(wave, sr=16000, n_mels=80, n_fft=400, hop=160, win=400):
    """wave: (B,N) float32 -> (B,T,n_mels) log-mel. Pure torch."""
    key = (sr, n_mels, n_fft)
    if key not in _mel_fb_cache:
        fb = _mel_filterbank(sr, n_fft, n_mels)
        _mel_fb_cache[key] = fb
    else:
        fb = _mel_fb_cache[key]
    fb = fb.to(wave.device, wave.dtype)
    window = torch.hann_window(win, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(wave, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, return_complex=True)  # (B,F,Tt)
    power = spec.abs().pow(2).transpose(1, 2)  # (B,Tt,F)
    mel = power @ fb.T  # (B,Tt,n_mels)
    return torch.log(torch.clamp(mel, min=1e-10))


def _mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    fmax = fmax or sr / 2
    mmin, mmax = _hz_to_mel(fmin), _hz_to_mel(fmax)
    m_pts = torch.linspace(mmin, mmax, n_mels + 2)
    h_pts = _mel_to_hz(m_pts)
    f_pts = torch.floor((n_fft + 1) * h_pts / sr).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        f0, f1, f2 = f_pts[m - 1].item(), f_pts[m].item(), f_pts[m + 1].item()
        if f1 > f0:
            fb[m - 1, f0:f1] = (torch.arange(f0, f1) - f0) / (f1 - f0)
        if f2 > f1:
            fb[m - 1, f1:f2] = (f2 - torch.arange(f1, f2)) / (f2 - f1)
    return fb


def _hz_to_mel(hz):
    return 2595.0 * torch.log10(torch.as_tensor(1.0) + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


# ------------------------------------------------------------------- model

class MultimodalAR(nn.Module):
    """Decoder-only AR model over continuous embeddings.

    encoder: VisionEncoder | AudioEncoder. Causal next-element prediction;
    loss = MSE(pred[:, :-1], elements[:, 1:]).
    """

    def __init__(self, encoder, dim=256, n_layers=8, n_heads=8, seq_len=256,
                 mixer_fn=None, dropout=0.0, ffn_hidden=None):
        super().__init__()
        self.encoder = encoder
        self.elem_dim = encoder.elem_dim
        self.seq_len = seq_len
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mixer_fn(dim, n_heads), dropout, ffn_hidden)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.elem_dim, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, raw):
        B = raw.shape[0]
        T = self.encoder.elements(raw).shape[1]
        x = self.encoder(raw) + self.pos_emb(
            torch.arange(T, device=raw.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))  # (B,T,elem_dim)

    def ar_loss(self, raw):
        pred = self(raw)
        with torch.no_grad():
            tgt = self.encoder.elements(raw)
        return F.mse_loss(pred[:, :-1], tgt[:, 1:])


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def match_ffn_hidden(v8a_ffn, dim, mixer_v8a_params, mixer_attn_params):
    """FFN hidden size for the attention arm so totals match (V4-style).

    Per-unit cost of FFN hidden: 2*dim (weights) + 1 (fc1 bias).
    """
    delta = mixer_v8a_params - mixer_attn_params
    per_unit = 2 * dim + 1
    return v8a_ffn + math.ceil(delta / per_unit)
