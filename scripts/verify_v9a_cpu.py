"""V9-A CPU verification (fail-fast gates before any Colab run).

No GPU here: verifies
  (1) V8A math on CPU: _delta_scan_cpu == direct manual loop (< 1e-6 fwd);
      autograd through the CPU scan == autograd through the manual loop (< 1e-5 bwd);
  (2) full V8A module == manual pipeline, inits, contractive at init;
  (3) the per-dataset FFN-width solver (same code path as the notebook)
      returns diff <= 3000 for vocabs 104 / 256 / 27 at SEQ_MAX=2048;
  (4) the generated notebook JSON is valid, every code cell compiles, and no
      cell reintroduces the V8-A `tl`-shadowing bug (bare `tl =` assignment).
"""
import sys, types, math, json, re
import torch

# ---- stub triton (CPU-only env; kernel definitions are inert here) ----
triton = types.ModuleType("triton")
triton.jit = lambda fn: fn
_tl = types.ModuleType("triton.language")
_tl.constexpr = int  # annotation placeholder only; kernels inert on CPU
triton.language = _tl
sys.modules["triton"] = triton
sys.modules["triton.language"] = _tl

sys.path.insert(0, "/home/hatch/workspace/linear-attention-lab")
from models import TinyLM, CausalSelfAttention, count_params
from triton_kernels import SelectiveSegmentedStateV8A, _delta_scan_cpu

torch.manual_seed(0)
ok = True
def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    ok = ok and cond

# ------------------------------------------------------- 1. forward (bar 1e-6)
def manual_delta_loop(u, ap, bp, kp, keep):
    B, T, S = u.shape
    h = torch.zeros(B, S, dtype=u.dtype)
    hs = []
    for t in range(T):
        kr = keep[:, t:t+1]
        alpha = torch.sigmoid(ap[:, t]); beta = torch.sigmoid(bp[:, t])
        kk = torch.sigmoid(kp[:, t])
        v = u[:, t] * torch.sigmoid(u[:, t])
        a = kr * alpha * (1.0 - beta * kk * kk)
        bb = kr * beta * kk * v
        h = a * h + bb
        hs.append(h)
    return torch.stack(hs, dim=1)

B, T, S = 4, 128, 256
u  = torch.randn(B, T, S); ap = torch.randn(B, T, S)
bp = torch.randn(B, T, S); kp = torch.randn(B, T, S)
keep = (torch.rand(B, T) > 0.3).float()
e_fwd = float((_delta_scan_cpu(u, ap, bp, kp, keep)
               - manual_delta_loop(u, ap, bp, kp, keep)).abs().max())
check("forward: cpu scan == direct loop", e_fwd < 1e-6, f"max delta {e_fwd:.2e}")

# ------------------------------------------------------ 2. backward (bar 1e-5)
def run_bwd(scan_fn):
    uu = u.clone().requires_grad_(True); aa = ap.clone().requires_grad_(True)
    bb_ = bp.clone().requires_grad_(True); kk_ = kp.clone().requires_grad_(True)
    h = scan_fn(uu, aa, bb_, kk_, keep)
    return torch.autograd.grad(h.pow(2).sum(), (uu, aa, bb_, kk_))

g_scan = run_bwd(_delta_scan_cpu)
g_loop = run_bwd(manual_delta_loop)
e_bwd = max(float((g - m).abs().max()) for g, m in zip(g_scan, g_loop))
check("backward: scan grads == loop grads", e_bwd < 1e-5, f"max delta {e_bwd:.2e}")

# ------------------------------------------------------- 3. full V8A module
DIM, LAYERS, HEADS, SEQ_MAX = 256, 8, 8, 2048
mix = SelectiveSegmentedStateV8A(DIM, state_dim=256)
check("init: key_proj bias == +2.0", bool((mix.key_proj.bias == 2.0).all()))
check("init: write_proj bias == 0",
      float(mix.write_proj.bias.abs().max()) == 0.0)

torch.manual_seed(1)
x = torch.randn(3, 16, DIM)
reset = (torch.rand(3, 16) < 0.2).float()
mix.eval()
with torch.no_grad():
    y_mod = mix(x, reset)
    uu = mix.in_proj(x); ap2 = mix.gate_proj(x) + mix.forget_floor
    bp2 = mix.write_proj(x); kp2 = mix.key_proj(x); og = mix.out_gate_proj(x)
    h_seq = manual_delta_loop(uu.float(), ap2.float(), bp2.float(), kp2.float(),
                              (1 - reset).float())
    h_norm = h_seq / torch.sqrt((h_seq ** 2).mean(-1, keepdim=True) + 1e-6)
    y_man = mix.out_proj((torch.sigmoid(og) * h_norm))
e_mod = float((y_mod - y_man).abs().max())
check("module == manual full pipeline", e_mod < 1e-6, f"max delta {e_mod:.2e}")

with torch.no_grad():
    alpha = torch.sigmoid(ap2); beta = torch.sigmoid(bp2); kk = torch.sigmoid(kp2)
    aa_ = alpha * (1 - beta * kk * kk)
check("strictly contractive at init", bool(((aa_ > 0) & (aa_ < 1)).all()))

# --------------------------------- 4. per-dataset FFN solver (notebook's path)
def solve_ffn(vocab, newline_id):
    torch.manual_seed(0)
    a = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
               mixer_fn=lambda d, h: CausalSelfAttention(d, h),
               newline_id=newline_id)
    v = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
               mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
               ffn_hidden=1024, newline_id=newline_id)
    d1024 = count_params(a) - count_params(v)
    del a, v
    return int(round(1024 + d1024 / (8 * 513)))

for vocab, tag in [(104, "A-corpus"), (256, "B-enwik8"), (27, "C-text8")]:
    h = solve_ffn(vocab, 0)
    torch.manual_seed(0)
    a = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
               mixer_fn=lambda d, hh: CausalSelfAttention(d, hh), newline_id=0)
    v = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ_MAX,
               mixer_fn=lambda d, hh: SelectiveSegmentedStateV8A(d, state_dim=256),
               ffn_hidden=h, newline_id=0)
    pa, pv = count_params(a), count_params(v)
    diff = abs(pa - pv)
    check(f"solver {tag} (vocab {vocab}): diff <= 3000", diff <= 3000,
          f"ffn={h} attn={pa} v8a={pv} diff={diff}")
    del a, v

# --------------------------------- 5. notebook JSON validity + cell checks
NB_PATH = "/home/hatch/workspace/linear-attention-lab/linear_attention_v9a_replication.ipynb"
nb = json.load(open(NB_PATH))
check("notebook: valid nbformat 4", nb.get("nbformat") == 4)
n_code = 0
tl_shadow = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    compile(src, f"cell{i}", "exec")  # raises SyntaxError on failure
    n_code += 1
    for ln, line in enumerate(src.split("\n")):
        if re.match(r"^\s*tl\s*=(?![=>])", line):
            tl_shadow.append((i, ln, line.strip()[:60]))
check(f"notebook: {n_code} code cells all compile", n_code == 11, f"found {n_code}")
check("notebook: no bare `tl =` shadowing (V8-A lesson)", not tl_shadow,
      f"hits={tl_shadow[:3]}")
heads = [ "".join(c["source"]).split("\n")[0]
          for c in nb["cells"] if c["cell_type"] == "markdown" ]
check("notebook: V9-A header cell present",
      heads and heads[0].startswith("# V9-A"), heads[0][:40] if heads else "")
need = ["KERNEL GATE", "Param-diff gate", "Multi-seed", "Dataset",
        "Zero-shot", "T=512", "Verdict", "Save everything"]
missing = [k for k in need if not any(k in h for h in heads)]
check("notebook: all spec sections present", not missing, f"missing={missing}")

print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
