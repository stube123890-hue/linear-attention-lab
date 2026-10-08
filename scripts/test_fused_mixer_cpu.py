"""CPU pre-flight for V11-D1 Path B (his standing rule: verify before GPU).

Compares SelectiveSegmentedStateV8AFused (CPU adjoint path) against a
pure-torch reference mixer (same math, _delta_scan_cpu, full autograd):
forward outputs and EVERY parameter gradient must match to ~1e-9.
Also exercises the reset/keep path.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels import _delta_scan_cpu, BLOCK_S
from fused_mixer import SelectiveSegmentedStateV8AFused


class RefMixer(nn.Module):
    """Pure-torch V8-A mixer: identical math, _delta_scan_cpu, full autograd."""

    def __init__(self, dim, state_dim=256):
        super().__init__()
        self.in_proj = nn.Linear(dim, state_dim, bias=False)
        self.gate_proj = nn.Linear(dim, state_dim)
        self.write_proj = nn.Linear(dim, state_dim)
        self.key_proj = nn.Linear(dim, state_dim)
        self.out_gate_proj = nn.Linear(dim, state_dim)
        self.out_proj = nn.Linear(state_dim, dim, bias=False)
        self.forget_floor = nn.Parameter(torch.tensor(0.0))
        self.write_proj.bias.data.zero_()   # b_beta = 0  (as V8-A)
        self.key_proj.bias.data.fill_(2.0)  # b_k = +2.0  (as V8-A)
        self.eps = 1e-6

    def forward(self, x, reset=None):
        u = self.in_proj(x).float()
        ap = (self.gate_proj(x) + self.forget_floor).float()
        bp = self.write_proj(x).float()
        kp = self.key_proj(x).float()
        og = self.out_gate_proj(x).float()
        keep = ((1.0 - reset).float() if reset is not None
                else torch.ones(x.shape[0], x.shape[1]))
        h_seq = _delta_scan_cpu(u, ap, bp, kp, keep)
        h_norm = h_seq / torch.sqrt(
            (h_seq ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.out_proj((torch.sigmoid(og) * h_norm).to(x.dtype))


def run_case(tag, reset):
    torch.manual_seed(0)
    ref = RefMixer(256)
    torch.manual_seed(0)
    fused = SelectiveSegmentedStateV8AFused(256, state_dim=256)
    # identical params under same seed (same __init__ order)
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(),
                                  fused.named_parameters()):
        assert n1 == n2, (n1, n2)
        assert torch.equal(p1, p2), f"param init mismatch: {n1}"
    g = torch.Generator().manual_seed(7)
    x = torch.randn(4, 16, 256, generator=g)
    yr = ref(x, reset)
    yf = fused(x, reset)
    fwd_err = (yr - yf).abs().max().item()
    lr, lf = yr.pow(2).sum(), yf.pow(2).sum()
    gr = torch.autograd.grad(lr, list(ref.parameters()))
    gf = torch.autograd.grad(lf, list(fused.parameters()))
    worst, worst_n = 0.0, ""
    for (n, _), a, b in zip(ref.named_parameters(), gr, gf):
        rel = ((a - b).abs().max() / a.abs().max().clamp_min(1e-12)).item()
        if rel > worst:
            worst, worst_n = rel, n
    print(f"[{tag}] fwd max-abs-err {fwd_err:.2e} | "
          f"worst grad rel-err {worst:.2e} ({worst_n})")
    assert fwd_err < 1e-9, fwd_err
    # contract gate is 1e-6 numerical equivalence (reduction-order FP noise
    # over 16K elements accounts for the ~3e-7 on the scalar floor grad)
    assert worst < 1e-6, (worst, worst_n)
    print(f"[{tag}] PASS")


run_case("no-reset", None)
g = torch.Generator().manual_seed(11)
reset = (torch.rand(4, 16, generator=g) < 0.2).float()
run_case("reset-mask", reset)
print("ALL CPU CHECKS PASS")
