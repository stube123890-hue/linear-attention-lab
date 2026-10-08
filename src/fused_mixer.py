"""V11-D1 Path B: fused projection+scan via save-x-only autograd Function.

Replaces the SAVED projection tensors (u/ap/bp/kp, ~134-400 MB live) with
recomputation. Forward runs the 5 projections with cuBLAS (transient, freed
immediately, never saved) and the EXISTING Triton scan kernels; saves only
(x, keep, h_seq, a_seq). Backward recomputes the projections and runs the
EXISTING verified Triton bwd kernel, then standard dx/dW.

CPU/GPU dispatch: on CUDA the Triton kernels run; on CPU a pure-torch
adjoint loop runs (local verification only — validates the Function's
backward math against torch autograd of the reference).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels import (
    SelectiveSegmentedStateV8A,
    _delta_scan_fwd_kernel,
    _delta_scan_bwd_kernel,
    _delta_scan_cpu,
    BLOCK_S,
)


def _adjoint_torch(dh, x, keep, h_seq, a_seq,
                   w_in, w_a, b_a, floor, w_b, b_b, w_k, b_k):
    """CPU-only adjoint: mirrors _delta_scan_bwd_kernel's math in torch.

    Returns (du, dap, dbp, dkp, dx, dW_in, dW_a, db_a, d_floor,
             dW_b, db_b, dW_k, db_k). Used for local verification only.
    """
    B, T, S = x.shape
    # recompute projections (same as forward)
    u = F.linear(x, w_in)
    ap = F.linear(x, w_a, b_a) + floor
    bp = F.linear(x, w_b, b_b)
    kp = F.linear(x, w_k, b_k)
    kr = keep.unsqueeze(-1)
    alpha = torch.sigmoid(ap)
    beta = torch.sigmoid(bp)
    kk = torch.sigmoid(kp)
    su = torch.sigmoid(u)
    v = u * su
    silu_p = su * (1.0 + u * (1.0 - su))
    k2 = kk * kk
    omb = 1.0 - beta * k2
    du = torch.empty_like(u)
    dap = torch.empty_like(ap)
    dbp = torch.empty_like(bp)
    dkp = torch.empty_like(kp)
    D = torch.zeros(B, S, dtype=torch.float32)
    for ti in range(T):
        t = T - 1 - ti
        a_next = a_seq[:, t + 1] if t + 1 < T else torch.zeros(B, S)
        D = dh[:, t] + a_next * D
        h_prev = h_seq[:, t - 1] if t > 0 else torch.zeros(B, S)
        du[:, t] = D * kr[:, t] * beta[:, t] * kk[:, t] * silu_p[:, t]
        dap[:, t] = (D * h_prev * kr[:, t] * alpha[:, t]
                     * (1.0 - alpha[:, t]) * omb[:, t])
        dbp[:, t] = (D * kr[:, t] * beta[:, t] * (1.0 - beta[:, t])
                     * (kk[:, t] * v[:, t]
                        - alpha[:, t] * k2[:, t] * h_prev))
        dkp[:, t] = (D * kr[:, t] * beta[:, t] * kk[:, t]
                     * (1.0 - kk[:, t])
                     * (v[:, t] - 2.0 * alpha[:, t] * kk[:, t] * h_prev))
    # NOTE: F.linear computes x @ W.T, so dL/dx = dL/du @ W (NOT W.t()).
    # (PyTorch Linear backward: grad_input = grad_output @ weight.)
    dx = du @ w_in + dap @ w_a + dbp @ w_b + dkp @ w_k
    # flat single-GEMM over B*T — PyTorch Linear backward's exact
    # reduction order (per-batch einsum diverged at TF32 precision)
    xr = x.reshape(-1, x.shape[-1])
    dW_in = du.reshape(-1, du.shape[-1]).t() @ xr
    dW_a = dap.reshape(-1, dap.shape[-1]).t() @ xr
    dW_b = dbp.reshape(-1, dbp.shape[-1]).t() @ xr
    dW_k = dkp.reshape(-1, dkp.shape[-1]).t() @ xr
    db_a = dap.sum(dim=(0, 1))
    db_b = dbp.sum(dim=(0, 1))
    db_k = dkp.sum(dim=(0, 1))
    d_floor = dap.sum()
    return du, dap, dbp, dkp, dx, dW_in, dW_a, db_a, d_floor, dW_b, db_b, dW_k, db_k


class _FusedProjScanFn(torch.autograd.Function):
    """Save-x-only Function: projections recomputed, never saved."""

    @staticmethod
    def forward(ctx, x, w_in, w_a, b_a, floor, w_b, b_b, w_k, b_k, keep):
        B, T, S = x.shape
        # projections: cuBLAS, transient — NOT saved
        u = F.linear(x, w_in)
        ap = F.linear(x, w_a, b_a) + floor
        bp = F.linear(x, w_b, b_b)
        kp = F.linear(x, w_k, b_k)
        if x.is_cuda:
            h = torch.empty(B, T, S, device=x.device, dtype=torch.float32)
            a = torch.empty(B, T, S, device=x.device, dtype=torch.float32)
            _delta_scan_fwd_kernel[(B,)](
                u.contiguous(), ap.contiguous(), bp.contiguous(),
                kp.contiguous(), keep.contiguous(), h, a, T,
                BLOCK=BLOCK_S, num_warps=4)
        else:
            kr = keep.unsqueeze(-1)
            alpha = torch.sigmoid(ap)
            beta = torch.sigmoid(bp)
            kk = torch.sigmoid(kp)
            a = kr * alpha * (1.0 - beta * kk * kk)
            h = _delta_scan_cpu(u, ap, bp, kp, keep)
        # saved: x + keep + scan outputs. u/ap/bp/kp deliberately absent.
        ctx.save_for_backward(x, keep, h, a,
                              w_in, w_a, b_a, floor, w_b, b_b, w_k, b_k)
        return h

    @staticmethod
    def backward(ctx, dh):
        (x, keep, h, a, w_in, w_a, b_a, floor,
         w_b, b_b, w_k, b_k) = ctx.saved_tensors
        B, T, S = x.shape
        dh = dh.contiguous()
        if x.is_cuda:
            # recompute projections (cuBLAS, transient)
            u = F.linear(x, w_in)
            ap = F.linear(x, w_a, b_a) + floor
            bp = F.linear(x, w_b, b_b)
            kp = F.linear(x, w_k, b_k)
            hs = torch.zeros(B, T + 1, S, device=x.device,
                             dtype=torch.float32)
            hs[:, 1:] = h                      # hs[t] = h_{t-1}
            apn = torch.zeros(B, T + 1, S, device=x.device,
                              dtype=torch.float32)
            apn[:, :T] = a                     # apn[t+1] = a_{t+1}
            du = torch.empty_like(u)
            dap = torch.empty_like(ap)
            dbp = torch.empty_like(bp)
            dkp = torch.empty_like(kp)
            _delta_scan_bwd_kernel[(B,)](
                dh, hs, apn, u.contiguous(), ap.contiguous(),
                bp.contiguous(), kp.contiguous(), keep.contiguous(),
                du, dap, dbp, dkp, T, BLOCK=BLOCK_S, num_warps=4)
            # NOTE: F.linear computes x @ W.T, so dL/dx = dL/du @ W (NOT W.t()).
            # (PyTorch Linear backward: grad_input = grad_output @ weight.)
            dx = du @ w_in + dap @ w_a + dbp @ w_b + dkp @ w_k
            # flat single-GEMM over B*T — PyTorch Linear backward's exact
            # reduction order (per-batch einsum diverged at TF32 precision)
            xr = x.reshape(-1, x.shape[-1])
            dW_in = du.reshape(-1, du.shape[-1]).t() @ xr
            dW_a = dap.reshape(-1, dap.shape[-1]).t() @ xr
            dW_b = dbp.reshape(-1, dbp.shape[-1]).t() @ xr
            dW_k = dkp.reshape(-1, dkp.shape[-1]).t() @ xr
            db_a = dap.sum(dim=(0, 1))
            db_b = dbp.sum(dim=(0, 1))
            db_k = dkp.sum(dim=(0, 1))
            d_floor = dap.sum()
        else:
            (du, dap, dbp, dkp, dx, dW_in, dW_a, db_a, d_floor,
             dW_b, db_b, dW_k, db_k) = _adjoint_torch(
                dh, x, keep, h, a,
                w_in, w_a, b_a, floor, w_b, b_b, w_k, b_k)
        return (dx, dW_in, dW_a, db_a, d_floor,
                dW_b, db_b, dW_k, db_k, None)


class SelectiveSegmentedStateV8AFused(SelectiveSegmentedStateV8A):
    """V8-A with fused projection+scan (Path B). Same params, same math.

    __init__ inherited unchanged -> identical params under the same seed.
    out_gate_proj stays a separate linear (its output is consumed after
    the scan; fusing it saves nothing).
    """

    def forward(self, x, reset=None):
        if reset is not None:
            keep = (1.0 - reset).float()
        else:
            keep = torch.ones(x.shape[0], x.shape[1],
                              device=x.device, dtype=torch.float32)
        h_seq = _FusedProjScanFn.apply(
            x, self.in_proj.weight,
            self.gate_proj.weight, self.gate_proj.bias, self.forget_floor,
            self.write_proj.weight, self.write_proj.bias,
            self.key_proj.weight, self.key_proj.bias,
            keep)
        self.last_state = h_seq.detach()  # stability logging only, as V8-A
        og = self.out_gate_proj(x).float()
        h_norm = h_seq / torch.sqrt(
            (h_seq ** 2).mean(dim=-1, keepdim=True) + self.eps)
        y = self.out_proj((torch.sigmoid(og) * h_norm).to(x.dtype))
        return self.dropout(y)
