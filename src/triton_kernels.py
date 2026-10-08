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


# ----------------------------------------------------------------------------
# V8-A: diagonal gated delta-rule scan
# ----------------------------------------------------------------------------

@triton.jit
def _delta_scan_fwd_kernel(u_ptr, ap_ptr, bp_ptr, kp_ptr, keep_ptr,
                           h_ptr, a_ptr, T, BLOCK: tl.constexpr):
    """Forward: h_t = a_t*h_{t-1} + b_t with
        alpha = sigmoid(ap_t), beta = sigmoid(bp_t), kk = sigmoid(kp_t),
        v = silu(u_t),
        a_t = keep_t * alpha * (1 - beta*kk^2),
        b_t = keep_t * beta * kk * v.
    Diagonal restriction of Gated DeltaNet's S_t = a_t S_{t-1}(I-b_t k k^T)
    + b_t v k^T to per-channel 1x1 systems. Strictly contractive:
    sigmoid bounds keep (1-beta*kk^2) in (0,1)."""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    bs = T * BLOCK
    u_b = u_ptr + b * bs
    ap_b = ap_ptr + b * bs
    bp_b = bp_ptr + b * bs
    kp_b = kp_ptr + b * bs
    h_b = h_ptr + b * bs
    a_b = a_ptr + b * bs
    k_b = keep_ptr + b * T
    h = tl.zeros([BLOCK], dtype=tl.float32)          # state in SRAM
    for t in range(T):
        u_t = tl.load(u_b + t * BLOCK + offs).to(tl.float32)
        ap_t = tl.load(ap_b + t * BLOCK + offs).to(tl.float32)
        bp_t = tl.load(bp_b + t * BLOCK + offs).to(tl.float32)
        kp_t = tl.load(kp_b + t * BLOCK + offs).to(tl.float32)
        kr = tl.load(k_b + t).to(tl.float32)
        alpha = 1.0 / (1.0 + tl.exp(-ap_t))
        beta = 1.0 / (1.0 + tl.exp(-bp_t))
        kk = 1.0 / (1.0 + tl.exp(-kp_t))
        su = 1.0 / (1.0 + tl.exp(-u_t))
        v = u_t * su                                 # SiLU in registers
        at = kr * alpha * (1.0 - beta * kk * kk)
        bt = kr * beta * kk * v
        h = at * h + bt
        tl.store(h_b + t * BLOCK + offs, h)
        tl.store(a_b + t * BLOCK + offs, at)


@triton.jit
def _delta_scan_bwd_kernel(dh_ptr, hs_ptr, apn_ptr, u_ptr, ap_ptr, bp_ptr,
                           kp_ptr, keep_ptr, du_ptr, dap_ptr, dbp_ptr,
                           dkp_ptr, T, BLOCK: tl.constexpr):
    """Backward. hs is (B,T+1,S) with hs[t] = h_{t-1} (hs[0] = 0);
    apn is (B,T+1,S) with apn[t+1] = a_{t+1} (apn[T] = 0).
    Adjoint: D_t = gout_t + a_{t+1}*D_{t+1}; then per-step
      du  = D*kr*beta*kk*silu'(u)
      dap = D*h_prev*kr*alpha*(1-alpha)*(1-beta*kk^2)
      dbp = D*kr*beta*(1-beta)*(kk*v - alpha*kk^2*h_prev)
      dkp = D*kr*beta*kk*(1-kk)*(v - 2*alpha*kk*h_prev)
    (hand-derived; verified against autograd on CPU before transcription)."""
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    bs = T * BLOCK
    bsp = (T + 1) * BLOCK
    dh_b = dh_ptr + b * bs
    hs_b = hs_ptr + b * bsp
    apn_b = apn_ptr + b * bsp
    u_b = u_ptr + b * bs
    ap_b = ap_ptr + b * bs
    bp_b = bp_ptr + b * bs
    kp_b = kp_ptr + b * bs
    k_b = keep_ptr + b * T
    du_b = du_ptr + b * bs
    dap_b = dap_ptr + b * bs
    dbp_b = dbp_ptr + b * bs
    dkp_b = dkp_ptr + b * bs
    dnext = tl.zeros([BLOCK], dtype=tl.float32)      # D_{t+1}, SRAM-resident
    for ti in range(T):
        t = T - 1 - ti
        gout = tl.load(dh_b + t * BLOCK + offs).to(tl.float32)
        a_next = tl.load(apn_b + (t + 1) * BLOCK + offs).to(tl.float32)
        D = gout + a_next * dnext
        h_prev = tl.load(hs_b + t * BLOCK + offs).to(tl.float32)
        u_t = tl.load(u_b + t * BLOCK + offs).to(tl.float32)
        ap_t = tl.load(ap_b + t * BLOCK + offs).to(tl.float32)
        bp_t = tl.load(bp_b + t * BLOCK + offs).to(tl.float32)
        kp_t = tl.load(kp_b + t * BLOCK + offs).to(tl.float32)
        kr = tl.load(k_b + t).to(tl.float32)
        alpha = 1.0 / (1.0 + tl.exp(-ap_t))
        beta = 1.0 / (1.0 + tl.exp(-bp_t))
        kk = 1.0 / (1.0 + tl.exp(-kp_t))
        su = 1.0 / (1.0 + tl.exp(-u_t))
        v = u_t * su
        silu_p = su * (1.0 + u_t * (1.0 - su))
        k2 = kk * kk
        omb = 1.0 - beta * k2
        tl.store(du_b + t * BLOCK + offs,
                 D * kr * beta * kk * silu_p)
        tl.store(dap_b + t * BLOCK + offs,
                 D * h_prev * kr * alpha * (1.0 - alpha) * omb)
        tl.store(dbp_b + t * BLOCK + offs,
                 D * kr * beta * (1.0 - beta) * (kk * v - alpha * k2 * h_prev))
        tl.store(dkp_b + t * BLOCK + offs,
                 D * kr * beta * kk * (1.0 - kk) * (v - 2.0 * alpha * kk * h_prev))
        dnext = D


def _delta_scan_cpu(u, ap, bp, kp, keep):
    """CPU reference for the delta scan: differentiable pure-PyTorch loop.

    Used ONLY for local (CPU) verification. On CUDA the fused Triton
    kernel above is used instead (gate-checked against this math on Colab).
    """
    B, T, S = u.shape
    kr = keep.unsqueeze(-1)
    alpha = torch.sigmoid(ap)
    beta = torch.sigmoid(bp)
    kk = torch.sigmoid(kp)
    v = u * torch.sigmoid(u)                          # SiLU
    h = torch.zeros(B, S, dtype=u.dtype, device=u.device)
    hs = []
    for t in range(T):
        a = kr[:, t] * alpha[:, t] * (1.0 - beta[:, t] * kk[:, t] ** 2)
        bb = kr[:, t] * beta[:, t] * kk[:, t] * v[:, t]
        h = a * h + bb
        hs.append(h)
    return torch.stack(hs, dim=1)


class _DeltaScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, ap, bp, kp, keep):
        assert u.is_cuda and u.dtype == torch.float32, \
            "V8-A kernel: CUDA fp32"
        for t in (u, ap, bp, kp):
            assert t.shape == u.shape and t.dim() == 3
        assert keep.shape == u.shape[:2]
        assert u.shape[2] == BLOCK_S
        B, T, S = u.shape
        u, ap, bp, kp = u.contiguous(), ap.contiguous(), bp.contiguous(), kp.contiguous()
        keep = keep.contiguous()
        h = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        a = torch.empty(B, T, S, device=u.device, dtype=torch.float32)
        _delta_scan_fwd_kernel[(B,)](u, ap, bp, kp, keep, h, a, T,
                                    BLOCK=BLOCK_S, num_warps=4)
        ctx.save_for_backward(u, ap, bp, kp, keep, h, a)
        return h

    @staticmethod
    def backward(ctx, dh):
        u, ap, bp, kp, keep, h, a = ctx.saved_tensors
        B, T, S = u.shape
        dh = dh.contiguous()
        hs = torch.zeros(B, T + 1, S, device=u.device, dtype=torch.float32)
        hs[:, 1:] = h      # hs[t] = h_{t-1}
        apn = torch.zeros(B, T + 1, S, device=u.device, dtype=torch.float32)
        apn[:, :T] = a     # apn[t+1] = a_{t+1}
        du = torch.empty_like(u)
        dap = torch.empty_like(ap)
        dbp = torch.empty_like(bp)
        dkp = torch.empty_like(kp)
        _delta_scan_bwd_kernel[(B,)](dh, hs, apn, u, ap, bp, kp, keep,
                                    du, dap, dbp, dkp, T,
                                    BLOCK=BLOCK_S, num_warps=4)
        return du, dap, dbp, dkp, None


def triton_delta_scan(u, ap, bp, kp, keep):
    """Fused diagonal delta-rule scan.

    u: raw Bx (SiLU applied in-kernel); ap/bp/kp: pre-sigmoid
    alpha/beta/key projections; keep: (B,T) reset mask.
    (B,T,256) fp32; CUDA -> Triton kernel, CPU -> differentiable
    reference loop (verification only). Returns h: (B,T,256).
    """
    if u.is_cuda:
        return _DeltaScanFn.apply(u, ap, bp, kp, keep)
    return _delta_scan_cpu(u, ap, bp, kp, keep)


class SelectiveSegmentedStateV8A(nn.Module):
    """V8-A mixer: diagonal gated delta rule on V7's chassis.

    Per channel (delta rule = erase-before-write, Gated DeltaNet Eq.10
    restricted to 1x1 systems):
        alpha_t = sigmoid(W_a x_t + b_a + floor_l)   # reuse gate_proj+floor
        beta_t  = sigmoid(W_b x_t + b_b),  b_b = 0   # NEW write_proj
        k_t     = sigmoid(W_k x_t + b_k),  b_k = +2  # NEW key_proj
        v_t     = SiLU(B x_t)                        # reuse in_proj + SiLU
        a_t = alpha_t * (1 - beta_t * k_t^2)
        b_t = beta_t * k_t * v_t
        h_t = (1 - r_t) * (a_t .* h_{t-1} + b_t)
        y_t = C . (o_t .* RMSNorm(h_t)), o_t = sigmoid(W_o x_t)

    Init puts the model near V7's convex-blend regime (k~=0.88, beta~=0.5)
    and the sigmoid bounds keep every step strictly contractive.
    With FFN 766 the budget lands within the relaxed <=3000 gate.
    """
    def __init__(self, dim, state_dim=256, dropout=0.0, forget_floor_init=0.0):
        super().__init__()
        assert state_dim == BLOCK_S, "V8-A kernel fuses a 256-wide state"
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, state_dim, bias=False)   # B
        self.gate_proj = nn.Linear(dim, state_dim)             # W_alpha
        self.write_proj = nn.Linear(dim, state_dim)            # W_beta (NEW)
        self.key_proj = nn.Linear(dim, state_dim)              # W_k (NEW)
        self.out_gate_proj = nn.Linear(dim, state_dim)         # W_o
        self.out_proj = nn.Linear(state_dim, dim, bias=False)  # C
        self.forget_floor = nn.Parameter(
            torch.tensor(float(forget_floor_init)))
        self.write_proj.bias.data.zero_()       # b_beta = 0  -> beta ~= 0.5
        self.key_proj.bias.data.fill_(2.0)       # b_k = +2.0  -> k ~= 0.88
        self.dropout = nn.Dropout(dropout)
        self.eps = 1e-6
        self.last_state = None  # stashed h_seq for stability logging (no grad)

    def forward(self, x, reset=None):
        u = self.in_proj(x).float().contiguous()
        ap = (self.gate_proj(x) + self.forget_floor).float().contiguous()
        bp = self.write_proj(x).float().contiguous()
        kp = self.key_proj(x).float().contiguous()
        og = self.out_gate_proj(x).float()
        if reset is not None:
            keep = (1.0 - reset).float().contiguous()
        else:
            keep = torch.ones(x.shape[0], x.shape[1],
                              device=x.device, dtype=torch.float32)
        h_seq = triton_delta_scan(u, ap, bp, kp, keep)      # (B,T,S)
        self.last_state = h_seq.detach()  # stability logging only; no numerics change
        h_norm = h_seq / torch.sqrt(
            (h_seq ** 2).mean(dim=-1, keepdim=True) + self.eps)
        y = self.out_proj((torch.sigmoid(og) * h_norm).to(x.dtype))
        return self.dropout(y)
