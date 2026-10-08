"""V6: Triton-fused selective scan for the gated segmented state space.

Fuses the V5 recurrence (per the V6 execution plan):
  Step A: the 256-wide hidden state lives in SRAM registers for the whole
          sequence -- loaded once per batch element, never spilled to HBM.
  Step B: sigmoid gate, hard reset mask, and the linear update
          h = keep*g*h + keep*(1-g)*u run in registers, every timestep.
  Step C: only h_t (plus a_t, saved for the backward pass) hit HBM.
          The parallel scan's O(log T) intermediate (B,T,S) tensors vanish.

Math is IDENTICAL to V5's SelectiveSegmentedState -- only the execution
path changes. Zero new parameters.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

BLOCK_S = 256  # state dim; the whole state fits in SRAM


@triton.jit
def _sel_scan_fwd_kernel(u_ptr, gp_ptr, keep_ptr, h_ptr, a_ptr,
                         T, BLOCK: tl.constexpr):
    """Forward: h_t = a_t*h_{t-1} + b_t with a_t = keep_t*g_t,
    b_t = keep_t*(1-g_t)*u_t, g_t = sigmoid(gp_t)."""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    bs = T * BLOCK
    u_b = u_ptr + b * bs
    gp_b = gp_ptr + b * bs
    h_b = h_ptr + b * bs
    a_b = a_ptr + b * bs
    k_b = keep_ptr + b * T
    h = tl.zeros([BLOCK], dtype=tl.float32)          # Step A: state in SRAM
    for t in range(T):
        u_t = tl.load(u_b + t * BLOCK + offs).to(tl.float32)
        gp_t = tl.load(gp_b + t * BLOCK + offs).to(tl.float32)
        k = tl.load(k_b + t).to(tl.float32)
        g = 1.0 / (1.0 + tl.exp(-gp_t))              # Step B: gate in registers
        at = k * g
        h = at * h + k * (1.0 - g) * u_t             # Step B: update in registers
        tl.store(h_b + t * BLOCK + offs, h)          # Step C: outputs to HBM
        tl.store(a_b + t * BLOCK + offs, at)


@triton.jit
def _sel_scan_bwd_kernel(dh_ptr, hs_ptr, ap_ptr, u_ptr, gp_ptr, keep_ptr,
                         du_ptr, dgp_ptr, T, BLOCK: tl.constexpr):
    """Backward. hs is (B,T+1,S) with hs[t] = h_{t-1} (hs[0] = 0);
    ap is (B,T+1,S) with ap[t+1] = a_{t+1} (ap[T] = 0).
    D_t = gout_t + a_{t+1}*D_{t+1}; du_t = k(1-g)D_t;
    dgp_t = k*g*(1-g)*(h_{t-1}-u_t)*D_t."""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    bs = T * BLOCK
    bsp = (T + 1) * BLOCK
    dh_b = dh_ptr + b * bs
    hs_b = hs_ptr + b * bsp
    ap_b = ap_ptr + b * bsp
    u_b = u_ptr + b * bs
    gp_b = gp_ptr + b * bs
    k_b = keep_ptr + b * T
    du_b = du_ptr + b * bs
    dgp_b = dgp_ptr + b * bs
    dnext = tl.zeros([BLOCK], dtype=tl.float32)      # D_{t+1}, SRAM-resident
    for ti in range(T):
        t = T - 1 - ti
        gout = tl.load(dh_b + t * BLOCK + offs).to(tl.float32)
        a_next = tl.load(ap_b + (t + 1) * BLOCK + offs).to(tl.float32)
        D = gout + a_next * dnext
        h_prev = tl.load(hs_b + t * BLOCK + offs).to(tl.float32)
        u_t = tl.load(u_b + t * BLOCK + offs).to(tl.float32)
        gp_t = tl.load(gp_b + t * BLOCK + offs).to(tl.float32)
        k = tl.load(k_b + t).to(tl.float32)
        g = 1.0 / (1.0 + tl.exp(-gp_t))
        omg = 1.0 - g
        tl.store(du_b + t * BLOCK + offs, k * omg * D)
        tl.store(dgp_b + t * BLOCK + offs, k * g * omg * (h_prev - u_t) * D)
        dnext = D


class _SelScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, gp, keep):
        assert u.is_cuda and u.dtype == torch.float32, "V6 kernel: CUDA fp32"
        assert u.shape == gp.shape and u.dim() == 3
        assert keep.shape == u.shape[:2]
        assert u.shape[2] == BLOCK_S
        B, T, S = u.shape
        u, gp, keep = u.contiguous(), gp.contiguous(), keep.contiguous()
        h = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        a = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        _sel_scan_fwd_kernel[(B,)](u, gp, keep, h, a, T,
                                   BLOCK=BLOCK_S, num_warps=4)
        ctx.save_for_backward(u, gp, keep, h, a)
        return h

    @staticmethod
    def backward(ctx, dh):
        u, gp, keep, h, a = ctx.saved_tensors
        B, T, S = u.shape
        dh = dh.contiguous()
        hs = torch.zeros(B, T + 1, S, device=u.device, dtype=torch.float32)
        hs[:, 1:] = h    # hs[t] = h_{t-1}
        ap = torch.zeros(B, T + 1, S, device=u.device, dtype=torch.float32)
        ap[:, :T] = a    # ap[t+1] = a_{t+1}
        du = torch.empty_like(u)
        dgp = torch.empty_like(gp)
        _sel_scan_bwd_kernel[(B,)](dh, hs, ap, u, gp, keep, du, dgp, T,
                                   BLOCK=BLOCK_S, num_warps=4)
        return du, dgp, None


def triton_selective_scan(u, gp, keep):
    """Fused selective scan. u, gp: (B,T,256) fp32 CUDA; keep: (B,T) fp32.
    Returns h: (B,T,256). Differentiable."""
    return _SelScanFn.apply(u, gp, keep)


class SelectiveSegmentedStateTriton(nn.Module):
    """V6 mixer: IDENTICAL math and parameters to V5's SelectiveSegmentedState.

    The only change is execution: the recurrence runs in the fused Triton
    kernel (Step A/B/C) instead of the PyTorch Hillis-Steele scan.
    """
    def __init__(self, dim, state_dim=256, dropout=0.0):
        super().__init__()
        assert state_dim == BLOCK_S, "V6 kernel fuses a 256-wide state"
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, state_dim, bias=False)
        self.gate_proj = nn.Linear(dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, reset=None):
        u = self.in_proj(x).float().contiguous()
        gp = self.gate_proj(x).float().contiguous()
        if reset is not None:
            keep = (1.0 - reset).float().contiguous()
        else:
            keep = torch.ones(x.shape[0], x.shape[1],
                              device=x.device, dtype=torch.float32)
        h_seq = triton_selective_scan(u, gp, keep)
        return self.dropout(self.out_proj(h_seq.to(x.dtype)))


class SelectiveSegmentedStateV7(nn.Module):
    """V7 mixer: V6's Triton-fused selective scan + two literature grafts.

    Graft #1 — output gate + state normalization (HGRN2 / Mamba-2 / GDN):
        y_t = C · (o_t ⊙ RMSNorm(h_t)),   o_t = σ(W_o x_t + b_o)
    States accumulate over segments so readout scale drifts; every strong
    post-Mamba block gates the readout. RMSNorm here is parameter-free
    (no scale vector) to keep the parameter budget exact.

    Graft #3 — layer-graded forget floor (HGRN, NeurIPS'23):
        g_t = σ(W_g x_t + b_g + floor_l)
    floor_l is a per-layer learnable scalar, initialized monotonically
    increasing bottom→top, so low layers forget fast (spelling-level) and
    top layers retain (phrase-level). Complements the newline hard reset.

    The recurrence math is UNCHANGED → the same fused Triton kernel as V6.
    With FFN 1023 the budget lands EXACTLY on the attention baseline.
    """
    def __init__(self, dim, state_dim=256, dropout=0.0, forget_floor_init=0.0):
        super().__init__()
        assert state_dim == BLOCK_S, "V7 kernel fuses a 256-wide state"
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, state_dim, bias=False)
        self.gate_proj = nn.Linear(dim, state_dim)
        self.out_gate_proj = nn.Linear(dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim, bias=False)
        self.forget_floor = nn.Parameter(
            torch.tensor(float(forget_floor_init)))
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6

    def forward(self, x, reset=None):
        u = self.in_proj(x).float().contiguous()
        gp = (self.gate_proj(x) + self.forget_floor).float().contiguous()
        og = self.out_gate_proj(x).float()
        if reset is not None:
            keep = (1.0 - reset).float().contiguous()
        else:
            keep = torch.ones(x.shape[0], x.shape[1],
                              device=x.device, dtype=torch.float32)
        h_seq = triton_selective_scan(u, gp, keep)          # (B,T,S)
        h_norm = h_seq / torch.sqrt(
            (h_seq ** 2).mean(dim=-1, keepdim=True) + self.eps)
        y = self.out_proj((torch.sigmoid(og) * h_norm).to(x.dtype))
        return self.dropout(y)
