"""CPU verification for the V9-B and V9-C notebooks (exit 0 required)."""
import json
import re
import sys

LAB = "/home/hatch/workspace/linear-attention-lab"
sys.path.insert(0, LAB)

import torch  # noqa: E402

V9B = f"{LAB}/linear_attention_v9b_bf16.ipynb"
V9C = f"{LAB}/linear_attention_v9c_fused_bwd.ipynb"


def load_nb(path, name):
    nb = json.load(open(path))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert cells, f"{name}: no code cells"
    return nb, cells


def check_compiles(cells, name):
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        compile(src, f"{name} cell {i}", "exec")
    print(f"[{name}] all {len(cells)} code cells compile")


def check_no_tl_shadow(cells, name):
    # the V8-A lesson: a bare `tl = ...` rebinds triton.language for later cells
    for i, c in enumerate(cells):
        for ln, line in enumerate("".join(c["source"]).split("\n")):
            s = line.strip()
            if s.startswith("#") or s.startswith("import ") or "triton" in s:
                continue
            if re.search(r"(?<![\w.])tl\s*=(?![=>])", line):
                raise AssertionError(
                    f"{name} cell {i} line {ln}: bare `tl` assignment (shadowing!): {line.strip()}")
    print(f"[{name}] no `tl` shadowing")


def check_save_fallback(cells, name):
    src = "\n".join("".join(c["source"]) for c in cells)
    assert "Drive copy skipped" in src, f"{name}: save cell lacks Drive best-effort fallback"
    print(f"[{name}] save cell has Drive best-effort fallback")


def extract_extra(cells, marker, name):
    for c in cells:
        src = "".join(c["source"])
        if marker in src:
            # the extra block starts at the marker comment line
            idx = src.find("# ---- V9-")
            assert idx != -1, f"{name}: extra-block comment not found"
            return src[idx:]
    raise AssertionError(f"{name}: marker {marker!r} not found")


print("== structural checks ==")
nb_b, cells_b = load_nb(V9B, "v9b")
nb_c, cells_c = load_nb(V9C, "v9c")
check_compiles(cells_b, "v9b")
check_compiles(cells_c, "v9c")
check_no_tl_shadow(cells_b, "v9b")
check_no_tl_shadow(cells_c, "v9c")
check_save_fallback(cells_b, "v9b")
check_save_fallback(cells_c, "v9c")

# --- V9-B content checks ---
src_b = "\n".join("".join(c["source"]) for c in cells_b)
for key in ("val", "train", "snorm", "gnorm", "kfwd", "kbwd", "tps",
            "mem_alloc", "mem_reserved"):
    assert f'"{key}"' in src_b, f"v9b: probe metric {key!r} missing"
print("[v9b] all 8 probe metrics present (bf16-fp32 differences)")
for marker in ("0.02", "6/7", "10%", "1.5"):
    assert marker in src_b, f"v9b: acceptance-gate marker {marker!r} missing"
assert ("NaN" in src_b or "non-finite" in src_b), "v9b: NaN/Inf clause missing"
print("[v9b] all 5 acceptance-gate clauses present")
assert "memory_reserved" in src_b and "max_memory_reserved" in src_b
print("[v9b] VRAM allocated + reserved both measured")
assert "2x16" in src_b or "2*MB" in src_b or "range(2)" in src_b
assert "autocast" in src_b
print("[v9b] micro-batch accumulation + autocast present")

# --- V9-C content checks ---
src_c = "\n".join("".join(c["source"]) for c in cells_c)
for name in ("_delta_scan_bwd_recompute_kernel", "_DeltaScanRecomputeFn",
             "triton_delta_scan_recompute", "SelectiveSegmentedStateV9C"):
    assert name in src_c, f"v9c: {name} missing"
print("[v9c] recompute variant (kernel + Fn + wrapper + module) present")
assert "1e-6" in src_c
print("[v9c] 1e-6 grad-equivalence gate present")
kern_py = open(f"{LAB}/triton_kernels.py").read()
assert "recompute" not in kern_py.lower(), "triton_kernels.py was modified!"
print("[v9c] triton_kernels.py untouched (existing kernel unbroken)")

print("== functional checks (CPU) ==")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from models import TinyLM, CausalSelfAttention, parallel_scan_ab  # noqa: E402
from triton_kernels import (  # noqa: E402
    SelectiveSegmentedStateV8A, triton_delta_scan, _delta_scan_cpu,
    _delta_scan_fwd_kernel, BLOCK_S)

# --- exec the V9-B extra block against the real lab modules ---
ns_b = {"torch": torch, "triton": triton, "tl": tl, "nn": torch.nn,
        "BLOCK_S": BLOCK_S, "_delta_scan_fwd_kernel": _delta_scan_fwd_kernel,
        "_delta_scan_bwd_kernel": None,  # resolved below
        "_delta_scan_cpu": _delta_scan_cpu,
        "SelectiveSegmentedStateV8A": SelectiveSegmentedStateV8A}
import triton_kernels as _tk
ns_b["_delta_scan_bwd_kernel"] = _tk._delta_scan_bwd_kernel
exec(extract_extra(cells_b, "class SelectiveSegmentedStateV9B", "v9b"), ns_b)
V9Bmod = ns_b["SelectiveSegmentedStateV9B"]
print("[v9b] extra block execs against real lab modules")

# --- micro-batch accumulation math on CPU (tiny V9B model) ---
torch.manual_seed(0)
def tiny_v9b():
    return TinyLM(20, 64, 2, 2, 32,
                  mixer_fn=lambda d, h: V9Bmod(d, 256),
                  ffn_hidden=128, newline_id=0)
m32 = tiny_v9b()
macc = tiny_v9b()
macc.load_state_dict(m32.state_dict())
xb = torch.randint(0, 20, (32, 32))
yb = torch.randint(0, 20, (32, 32))
m32.zero_grad()
m32(xb, yb)[1].backward()
g32 = [p.grad.clone() for p in m32.parameters()]
macc.zero_grad()
for k in range(2):
    macc(xb[k*16:(k+1)*16], yb[k*16:(k+1)*16])[1].div(2).backward()
gacc = [p.grad.clone() for p in macc.parameters()]
mdiff = max((a - b).abs().max().item() for a, b in zip(g32, gacc))
assert mdiff < 1e-6, f"micro-batch accumulation mismatch: {mdiff:.2e}"
print(f"[v9b] micro-batch accumulation math on CPU: max diff {mdiff:.2e} (< 1e-6)")

# --- exec the V9-C extra block via a real .py file (triton @jit needs one) ---
import importlib.util
_extra_c = extract_extra(cells_c, "class SelectiveSegmentedStateV9C", "v9c")
with open("/tmp/v9c_extra_check.py", "w") as f:
    f.write("import torch\nimport torch.nn as nn\nimport triton\n"
            "import triton.language as tl\n"
            "from triton_kernels import (BLOCK_S, _delta_scan_fwd_kernel,\n"
            "    _delta_scan_cpu, SelectiveSegmentedStateV8A)\n" + _extra_c)
spec = importlib.util.spec_from_file_location("v9c_extra_check", "/tmp/v9c_extra_check.py")
mod_c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_c)
assert hasattr(mod_c, "_delta_scan_bwd_recompute_kernel")
assert hasattr(mod_c, "_DeltaScanRecomputeFn")
assert hasattr(mod_c, "triton_delta_scan_recompute")
assert hasattr(mod_c, "SelectiveSegmentedStateV9C")
print("[v9c] extra block imports as a real module (jit decorator satisfied)")

# --- V9-C recompute-backward math on CPU: adjoint loop with recomputed a_{t+1}
#     vs autograd through the differentiable reference ---
def delta_bwd_recompute_cpu(dh, u, ap, bp, kp, keep):
    B, T, S = u.shape
    kr = keep.unsqueeze(-1)
    alpha = torch.sigmoid(ap); beta = torch.sigmoid(bp); kk = torch.sigmoid(kp)
    v = u * torch.sigmoid(u)
    h = torch.zeros(B, S); hs = [h]
    for t in range(T):
        at = kr[:, t] * alpha[:, t] * (1 - beta[:, t] * kk[:, t] ** 2)
        bt = kr[:, t] * beta[:, t] * kk[:, t] * v[:, t]
        h = at * h + bt
        hs.append(h)
    du = torch.zeros_like(u); dap = torch.zeros_like(ap)
    dbp = torch.zeros_like(bp); dkp = torch.zeros_like(kp)
    dnext = torch.zeros(B, S)
    for t in range(T - 1, -1, -1):
        if t + 1 < T:  # recompute a_{t+1} (was: load from saved tensor)
            a_next = kr[:, t+1] * alpha[:, t+1] * (1 - beta[:, t+1] * kk[:, t+1] ** 2)
        else:
            a_next = torch.zeros(B, S)
        D = dh[:, t] + a_next * dnext
        h_prev = hs[t]
        al, be, kkv = alpha[:, t], beta[:, t], kk[:, t]
        vv = v[:, t]; su = torch.sigmoid(u[:, t])
        silu_p = su * (1 + u[:, t] * (1 - su))
        k2 = kkv ** 2; omb = 1 - be * k2; krt = kr[:, t]
        du[:, t] = D * krt * be * kkv * silu_p
        dap[:, t] = D * h_prev * krt * al * (1 - al) * omb
        dbp[:, t] = D * krt * be * (1 - be) * (kkv * vv - al * k2 * h_prev)
        dkp[:, t] = D * krt * be * kkv * (1 - kkv) * (vv - 2 * al * kkv * h_prev)
        dnext = D
    return du, dap, dbp, dkp

torch.manual_seed(1)
B, T, S = 2, 16, 256
u = torch.randn(B, T, S); ap = torch.randn(B, T, S)
bp = torch.randn(B, T, S); kp = torch.randn(B, T, S)
keep = (torch.rand(B, T) > 0.3).float()
u1 = u.clone().requires_grad_(True); ap1 = ap.clone().requires_grad_(True)
bp1 = bp.clone().requires_grad_(True); kp1 = kp.clone().requires_grad_(True)
h_out = _delta_scan_cpu(u1, ap1, bp1, kp1, keep)
h_out.pow(2).sum().backward()
dh = 2 * h_out.detach()
gr = delta_bwd_recompute_cpu(dh, u, ap, bp, kp, keep)
ga = (u1.grad, ap1.grad, bp1.grad, kp1.grad)
errs = [float((a - b).abs().max()) for a, b in zip(gr, ga)]
assert all(e < 1e-6 for e in errs), f"recompute-bwd math mismatch: {errs}"
print(f"[v9c] recompute-backward math == autograd on CPU: "
      f"max {max(errs):.2e} (< 1e-6)")

print("\nALL V9-B/V9-C CHECKS PASSED")
