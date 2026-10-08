"""
Linear-recurrent replacement for Transformer self-attention.

Experiment: take a standard decoder-only Transformer, rip out the
causal self-attention layer, and replace it with a *moving linear
equation* that compresses the past into a fixed-size running state:

    h_t = d * h_{t-1} + (1 - d) * (B x_t)        (linear recurrence)
    y_t = C h_t                                   (readout)

where d in (0,1)^S is a learned per-channel decay, B/C are learned
projections, and h has FIXED size S independent of sequence length.

This is O(T) time and O(1) memory w.r.t. context length, vs O(T^2)
for attention. The question: how much modeling power do we lose?
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Token mixing layers
# ----------------------------------------------------------------------------

class SelectiveSegmentedState(nn.Module):
    """V5: gated segmented state-space mixer.

    Replaces the static decay with two dynamic mechanisms:

    1. Dynamic Selective Gate (the model's "will"):
         g_t = sigmoid(W_g x_t + b_g)      in (0,1), per token & channel
       acts as a learned per-token filter: low-information tokens get
       g_t -> 0 (forgotten fast), salient tokens get g_t -> 1 (kept).

    2. Hard Memory Reset Mask (segment boundaries):
         r_t = 1 if token t is a structural boundary (e.g. newline)
         h_t = (1 - r_t) * (g_t * h_{t-1} + (1 - g_t) * u_t)
       when r_t = 1 the state is multiplied by zero: the past memory
       buffer is completely destroyed and the new segment starts fresh.
       (The residual stream still carries x_t itself forward.)

    The recurrence stays affine in h (h_t = a_t*h_{t-1} + b_t with
    a_t = (1-r_t)*g_t, b_t = (1-r_t)*(1-g_t)*u_t), so the parallel
    associative scan still applies: O(T) work, log2(T) parallel passes.
    State dim stays frozen at `state_dim` (VRAM edge preserved).
    """
    def __init__(self, dim, state_dim=256, dropout=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, state_dim, bias=False)
        self.gate_proj = nn.Linear(dim, state_dim)  # bias = per-channel prior
        self.out_proj = nn.Linear(state_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, reset=None):
        # x: (B,T,dim); reset: (B,T) binary, 1 = boundary token
        u = self.in_proj(x)                    # (B,T,S)
        g = torch.sigmoid(self.gate_proj(x))   # (B,T,S) dynamic gate
        if reset is not None:
            keep = (1.0 - reset).unsqueeze(-1)  # (B,T,1)
            a = keep * g
            b = keep * (1.0 - g) * u
        else:
            a = g
            b = (1.0 - g) * u
        h_seq = parallel_scan_ab(a, b)          # (B,T,S)
        return self.dropout(self.out_proj(h_seq))


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention (the baseline)."""
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, reset=None):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # (B,T,H,D)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B,H,T,D)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


def parallel_scan_ab(A, B):
    """Parallel associative scan (Hillis-Steele) for h_t = A_t*h_{t-1} + B_t.

    A, B: (B, T, S) per-step coefficients. Returns H: (B, T, S).
    log2(T) fully-parallel passes instead of T sequential steps.
    """
    Bsz, T, S = A.shape
    A = A.contiguous()
    Bh = B.contiguous()
    idx = torch.arange(T, device=A.device).view(1, T, 1)
    o = 1
    while o < T:
        m = idx >= o
        A_sh = torch.roll(A, o, dims=1)
        B_sh = torch.roll(Bh, o, dims=1)
        # compose segment [t-o, t-1] then [t]:
        #   a_new = a_t * a_{t-o},  b_new = a_t * b_{t-o} + b_t
        A_new = A * A_sh
        B_new = A * B_sh + Bh
        A = torch.where(m, A_new, A)
        Bh = torch.where(m, B_new, Bh)
        o *= 2
    return Bh


def parallel_linear_recurrence(u, decay):
    """Parallel associative scan (Hillis-Steele) for h_t = d*h_{t-1} + c_t.

    Reformulates the sequential loop as a tree reduction:
      - each step is an affine map  h -> a*h + b, packed as pair (a, b)
      - pair composition is associative:
            (a1,b1) then (a2,b2)  =  (a2*a1, a2*b1 + b2)
      - a parallel prefix over pairs yields every h_t at once

    log2(T) fully-parallel passes instead of T sequential steps.
    u: (B, T, S) inputs already projected; decay: (S,) in (0, 1).
    Returns h: (B, T, S) — identical math to the for-loop version.
    """
    Bsz, T, S = u.shape
    c = (1.0 - decay) * u
    A = decay.expand(Bsz, T, S).contiguous()   # a-coefficients
    return parallel_scan_ab(A, c)


class LinearRunningState(nn.Module):
    """Attention replacement: fixed-size running state, linear recurrence.

    h_t = decay * h_{t-1} + (1 - decay) * (B x_t)
    y_t = C h_t

    - decay: learned per-state-channel forget factor in (0, 1)
    - B:     input  projection  dim -> state_dim
    - C:     output projection  state_dim -> dim
    State size is FIXED (does not grow with context length).
    mode="loop": native Python for-loop (sequential, slow on GPU)
    mode="scan": parallel associative scan (log2(T) parallel passes)
    """
    def __init__(self, dim, state_dim=None, dropout=0.0, mode="loop"):
        super().__init__()
        self.mode = mode
        self.state_dim = state_dim or dim
        # init decay ~ U(0.9, 0.999) -> long-ish memory at start
        init = torch.empty(self.state_dim).uniform_(math.log(0.9 / 0.1),
                                                   math.log(0.999 / 0.001))
        self.logit_decay = nn.Parameter(init)
        self.in_proj = nn.Linear(dim, self.state_dim, bias=False)
        self.out_proj = nn.Linear(self.state_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, reset=None):
        B, T, _ = x.shape
        decay = torch.sigmoid(self.logit_decay)          # (S,) in (0,1)
        u = self.in_proj(x)                              # (B,T,S)
        if self.mode == "scan":
            h_seq = parallel_linear_recurrence(u, decay)  # (B,T,S)
        else:
            keep = 1.0 - decay
            h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
            outs = []
            for t in range(T):
                h = decay * h + keep * u[:, t]
                outs.append(h)
            h_seq = torch.stack(outs, dim=1)              # (B,T,S)
        return self.dropout(self.out_proj(h_seq))


# ----------------------------------------------------------------------------
# Blocks & models (identical except for the mixing layer)
# ----------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, dim, n_heads, mixer, dropout=0.0, ffn_hidden=None):
        super().__init__()
        ffn_hidden = ffn_hidden or 4 * dim
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mix = mixer
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_hidden), nn.GELU(),
            nn.Linear(ffn_hidden, dim), nn.Dropout(dropout))

    def forward(self, x, reset=None):
        x = x + self.mix(self.ln1(x), reset)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """Decoder-only LM. mixer_fn(dim, n_heads) builds the token-mixing layer."""
    def __init__(self, vocab, dim=128, n_layers=4, n_heads=4,
                 seq_len=128, mixer_fn=None, dropout=0.0, ffn_hidden=None,
                 newline_id=None):
        super().__init__()
        self.seq_len = seq_len
        self.newline_id = newline_id  # boundary token id for reset mask
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mixer_fn(dim, n_heads), dropout, ffn_hidden)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(
            torch.arange(T, device=idx.device))[None]
        reset = None
        if self.newline_id is not None:
            reset = (idx == self.newline_id).float()  # (B,T) boundary mask
        for blk in self.blocks:
            x = blk(x, reset)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temp=1.0, top_k=None):
        for _ in range(max_new):
            ctx = idx[:, -self.seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temp, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def build_models(vocab, dim=128, n_layers=4, n_heads=4, seq_len=128,
                 state_dim=None, dropout=0.0):
    attn = TinyLM(vocab, dim, n_layers, n_heads, seq_len,
                  mixer_fn=lambda d, h: CausalSelfAttention(d, h, dropout),
                  dropout=dropout)
    lin = TinyLM(vocab, dim, n_layers, n_heads, seq_len,
                 mixer_fn=lambda d, h: LinearRunningState(d, state_dim, dropout),
                 dropout=dropout)
    return attn, lin
