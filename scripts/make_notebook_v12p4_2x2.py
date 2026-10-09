"""Generate V12 Phase 4 2x2 matched control notebook.

Design (his spec):
- 4 arms: A=vision-only/ATTN, B=vision-only/V8-A, C=joint/ATTN, D=joint/V8-A
- Same Flickr8k data, preprocessing, init, optimizer, 500 steps.
- Vision loss definition IDENTICAL across all four arms.
- Init verification: tensor-level equality on shared components.
- Predeclared: I = Δ_joint - Δ_vision-only, where Δ = L_V8-A - L_ATTN.
  I<0: rescue; I≈0: no interaction; I>0: joint hurts V8-A.
"""
import json
import subprocess
import tempfile
import os

WS = "/home/hatch/workspace/linear-attention-lab"
os.environ.setdefault("PYTHONPATH", os.path.expanduser("~/workspace/.pylibs"))


def inline(path, old_doc, new_doc, old_imp, new_imp=""):
    src = open(f"{WS}/{path}").read().replace(old_doc, new_doc)
    assert old_imp in src, path
    return src.replace(old_imp, new_imp)


cell_models = "\n\n".join([
    inline("models.py",
           '"""Linear-recurrent replacement for Transformer self-attention.',
           '"""(inlined) Linear-recurrent replacement for Transformer self-attention.',
           "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
           "import math\nimport torch.nn as nn\nimport torch.nn.functional as F\n"),
    inline("triton_kernels.py",
           '"""V6: Triton-fused selective scan for the gated segmented state space.',
           '"""(inlined) V6/V7/V8-A: Triton-fused scans.',
           "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
           "import triton\nimport triton.language as tl\n"),
    inline("fused_mixer.py",
           '"""V11-D1 Path B: fused projection+scan via save-x-only autograd Function.',
           '"""(inlined) V11-D1 Path B: fused projection+scan.',
           """import torch
import torch.nn as nn
import torch.nn.functional as F

from triton_kernels import (
    SelectiveSegmentedStateV8A,
    _delta_scan_fwd_kernel,
    _delta_scan_bwd_kernel,
    _delta_scan_cpu,
    BLOCK_S,
)
""",
           "# (imports stripped)\n"),
    inline("v12_multimodal.py",
           '"""V12 multimodal: AR sequence modeling over vision/audio on the frozen V8-A stack.',
           '"""(inlined) V12 multimodal helpers.',
           """import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Block  # frozen
""",
           "# (imports stripped)\n"),
])


def build_notebook():
    NB = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"}},
          "cells": []}

    def md(src):
        NB["cells"].append({"cell_type": "markdown", "metadata": {},
                            "source": src.splitlines(keepends=True)})

    def code(src):
        NB["cells"].append({"cell_type": "code", "metadata": {},
                            "source": src.splitlines(keepends=True),
                            "outputs": [], "execution_count": None})

    md("""# V12 Phase 4 2x2 — matched vision-only vs joint (Flickr8k)

4 arms × 500 steps. A=vis/ATTN, B=vis/V8-A, C=joint/ATTN, D=joint/V8-A.
Same data, init (tensor-verified), optimizer. Vision loss identical across arms.
I = Δ_joint − Δ_vis-only; I<0 → rescue.
""")

    code('''import os
import torch
assert torch.cuda.is_available(), "needs GPU"
device = "cuda"
def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(0)
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
''')

    # Data cell (same as pilot)
    code('''import urllib.request, zipfile, glob
from PIL import Image
import re
from collections import Counter

def dl(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("downloading", url.split("/")[-1], flush=True)
        urllib.request.urlretrieve(url, path)
    return path

img_zip = dl("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
             "data/Flickr8k_Dataset.zip")
txt_zip = dl("https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
             "data/Flickr8k_text.zip")
IMG_DIR = "data/Flicker8k_Dataset"
if not os.path.exists(IMG_DIR):
    print("extracting images...", flush=True)
    with zipfile.ZipFile(img_zip) as z:
        z.extractall("data")
_img_dirs = glob.glob("data/**/Flicker8k_Dataset", recursive=True)
if _img_dirs:
    IMG_DIR = _img_dirs[0]
print(f"image dir: {IMG_DIR}")

TXT_DIR = "data/Flickr8k_text"
if not os.path.exists(TXT_DIR):
    print("extracting text...", flush=True)
    with zipfile.ZipFile(txt_zip) as z:
        z.extractall("data")
_tok_paths = glob.glob("data/**/Flickr8k.token.txt", recursive=True)
assert _tok_paths, "Flickr8k.token.txt not found"
TOKEN_PATH = _tok_paths[0]
print(f"token file: {TOKEN_PATH}")

pairs = {}
with open(TOKEN_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_cap, caption = line.split("\\t")
        img_name = img_cap.split("#")[0]
        pairs.setdefault(img_name, []).append(caption)
items = [(n, cs[0]) for n, cs in pairs.items()
         if os.path.exists(f"{IMG_DIR}/{n}")]
print(f"paired items: {len(items)}")

def tok(s):
    return re.findall(r"[a-z]+", s.lower())
cnt = Counter()
for _, c in items:
    cnt.update(tok(c))
vocab = {"<pad>": 0, "<unk>": 1}
for w, _ in cnt.most_common(8000):
    vocab[w] = len(vocab)
VOCAB = len(vocab)
print(f"vocab: {VOCAB}")

def encode_caption(s, T):
    ids = [vocab.get(w, 1) for w in tok(s)][:T]
    ids += [0] * (T - len(ids))
    return ids

def load_image(name):
    return Image.open(f"{IMG_DIR}/{name}").convert("RGB")
''')

    code(cell_models)

    code('''import time, gc, random, statistics
import torch.nn as nn
import torch.utils.checkpoint as _ckpt

DIM, LAYERS, HEADS, FFN_V8A = 256, 8, 8, 766
T_TXT, PATCH, IMG, BS = 32, 8, 128, 32
T_VIS = (IMG // PATCH) ** 2  # 256
T_JOINT = T_TXT + T_VIS  # 288

_orig_block_forward = Block.forward
def _block_forward_ckpt(self, x, reset=None):
    x = x + self.mix(self.ln1(x), reset)
    x = x + _ckpt.checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
    return x
Block.forward = _block_forward_ckpt

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ---- Vision-only model (arms A/B): same as Phase 1, Flickr8k data ----
class VisAR(nn.Module):
    def __init__(self, dim, n_layers, n_heads, seq_len, mixer_fn, ffn_hidden):
        super().__init__()
        self.encoder = VisionEncoder(dim, patch=PATCH)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mixer_fn(dim, n_heads), 0.0, ffn_hidden)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 3*PATCH*PATCH, bias=False)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, img):
        x = self.encoder(img)
        x = x + self.pos_emb(torch.arange(x.shape[1], device=x.device))[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
    def loss(self, img):
        pred = self(img)
        with torch.no_grad():
            tgt = self.encoder.elements(img)
        # IDENTICAL to joint vision loss: MSE(pred[:, :-1], tgt[:, 1:])
        return F.mse_loss(pred[:, :-1], tgt[:, 1:])

# ---- Joint model (arms C/D): same as pilot ----
class JointEncoder(nn.Module):
    def __init__(self, dim, vocab, patch=8):
        super().__init__()
        self.txt_emb = nn.Embedding(vocab, dim)
        self.patch = patch
        self.elem_dim_vis = 3 * patch * patch
        self.proj_vis = nn.Linear(self.elem_dim_vis, dim)
    def elements_vis(self, img):
        B, C, H, W = img.shape
        p = self.patch
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(B, -1, self.elem_dim_vis)
    def forward(self, img, cap_ids):
        t = self.txt_emb(cap_ids)
        v = self.proj_vis(self.elements_vis(img))
        return torch.cat([t, v], dim=1)

class JointAR(nn.Module):
    def __init__(self, dim, n_layers, n_heads, seq_len, mixer_fn, ffn_hidden):
        super().__init__()
        self.encoder = JointEncoder(dim, VOCAB, PATCH)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mixer_fn(dim, n_heads), 0.0, ffn_hidden)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head_vis = nn.Linear(dim, 3*PATCH*PATCH, bias=False)
        self.head_txt = nn.Linear(dim, dim, bias=False)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, img, cap_ids):
        x = self.encoder(img, cap_ids)
        x = x + self.pos_emb(torch.arange(T_JOINT, device=x.device))[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        pv = self.head_vis(x[:, T_TXT:T_JOINT-1])
        pt = self.head_txt(x[:, :T_TXT-1])
        return pv, pt
    def vis_loss(self, img, cap_ids):
        # IDENTICAL definition: MSE on patch predictions
        pv, _ = self(img, cap_ids)
        with torch.no_grad():
            tv = self.encoder.elements_vis(img)[:, 1:]
        return F.mse_loss(pv, tv)

def build_vis(mixer, seed, ffn_hidden):
    set_seed(seed)
    m = VisAR(DIM, LAYERS, HEADS, T_VIS, mixer, ffn_hidden).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        if hasattr(_blk.mix, "forget_floor"):
            _blk.mix.forget_floor.data.fill_(_floors[_i])
    return m, torch.optim.AdamW(m.parameters(), lr=3e-4)

def build_joint(mixer, seed, ffn_hidden):
    set_seed(seed)
    m = JointAR(DIM, LAYERS, HEADS, T_JOINT, mixer, ffn_hidden).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        if hasattr(_blk.mix, "forget_floor"):
            _blk.mix.forget_floor.data.fill_(_floors[_i])
    return m, torch.optim.AdamW(m.parameters(), lr=3e-4)

# param matching
torch.manual_seed(0)
_m0 = VisAR(DIM, 1, HEADS, T_VIS,
            lambda d,h: SelectiveSegmentedStateV8AFused(d, state_dim=256),
            ffn_hidden=FFN_V8A)
_pv = sum(p.numel() for p in _m0.blocks[0].mix.parameters())
_ma = VisAR(DIM, 1, HEADS, T_VIS,
            lambda d,h: CausalSelfAttention(d, h), ffn_hidden=FFN_V8A)
_pa = sum(p.numel() for p in _ma.blocks[0].mix.parameters())
FFN_ATTN = match_ffn_hidden(FFN_V8A, DIM, _pv, _pa)
del _m0, _ma
print(f"mixer params/layer: v8a {_pv} vs attn {_pa} -> ffn {FFN_V8A} vs {FFN_ATTN}")

# FIXED data (seed 42 train, 43 val) — same for all four arms
frng = random.Random(42)
idx_tr = [frng.randrange(len(items)) for _ in range(BS)]
vrng = random.Random(43)
idx_va = [vrng.randrange(len(items)) for _ in range(BS)]
def load_pair(i):
    name, caption = items[i]
    im = load_image(name).resize((IMG, IMG), Image.BILINEAR)
    t = torch.from_numpy(__import__("numpy").array(im)).permute(2,0,1).float()/255.0
    cap = encode_caption(caption, T_TXT)
    return t, torch.tensor(cap, dtype=torch.long)
raw_tr = [load_pair(i) for i in idx_tr]
raw_va = [load_pair(i) for i in idx_va]
img_tr = ((torch.stack([r[0] for r in raw_tr]) - IMAGENET_MEAN) / IMAGENET_STD).to(device)
cap_tr = torch.stack([r[1] for r in raw_tr]).to(device)
img_va = ((torch.stack([r[0] for r in raw_va]) - IMAGENET_MEAN) / IMAGENET_STD).to(device)
cap_va = torch.stack([r[1] for r in raw_va]).to(device)
print(f"train: img {tuple(img_tr.shape)} cap {tuple(cap_tr.shape)}")
print(f"val:   img {tuple(img_va.shape)} cap {tuple(cap_va.shape)}")

# INIT VERIFICATION (his requirement): tensor-level equality on shared comps
def verify_init(m1, m2, label):
    # compare encoder + pos_emb (shared architecture components)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if "mix" in n1 or "mix" in n2:
            continue  # mixer differs by design
        if n1 != n2:
            continue  # different names (e.g., head vs head_vis)
        if not torch.equal(p1.cpu(), p2.cpu()):
            print(f"INIT MISMATCH {label}: {n1}", flush=True)
            return False
    print(f"INIT OK {label}: shared components match", flush=True)
    return True

# Build all four arms
print("Building 4 arms...", flush=True)
mA, oA = build_vis(lambda d,h: CausalSelfAttention(d,h), 0, FFN_ATTN)      # A: vis/attn
mB, oB = build_vis(lambda d,h: SelectiveSegmentedStateV8AFused(d, state_dim=256), 0, FFN_V8A)  # B: vis/v8a
mC, oC = build_joint(lambda d,h: CausalSelfAttention(d,h), 0, FFN_ATTN)    # C: joint/attn
mD, oD = build_joint(lambda d,h: SelectiveSegmentedStateV8AFused(d, state_dim=256), 0, FFN_V8A)  # D: joint/v8a

verify_init(mA, mB, "A/B vision encoders")
verify_init(mC, mD, "C/D joint encoders")

def train_vis(m, o, steps=500):
    m.train()
    for _ in range(5):  # warmup
        o.zero_grad(set_to_none=True)
        m.loss(img_tr).backward(); o.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        o.zero_grad(set_to_none=True)
        m.loss(img_tr).backward(); o.step()
    torch.cuda.synchronize()
    ms = 1000*(time.perf_counter()-t0)/steps
    m.eval()
    with torch.no_grad():
        lv = m.loss(img_va)
    return lv.item(), ms

def train_joint(m, o, steps=500):
    m.train()
    for _ in range(5):
        o.zero_grad(set_to_none=True)
        tot, _, _ = m.loss(img_tr, cap_tr)
        tot.backward(); o.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        o.zero_grad(set_to_none=True)
        tot, _, _ = m.loss(img_tr, cap_tr)
        tot.backward(); o.step()
    torch.cuda.synchronize()
    ms = 1000*(time.perf_counter()-t0)/steps
    m.eval()
    with torch.no_grad():
        lv = m.vis_loss(img_va, cap_va)
    return lv.item(), ms

# JointAR needs loss() returning (total, lv, lt) for training; add it
_orig_joint_loss = JointAR.vis_loss
def _joint_total_loss(self, img, cap_ids):
    pv, pt = self(img, cap_ids)
    with torch.no_grad():
        tv = self.encoder.elements_vis(img)[:, 1:]
        tt = self.encoder.txt_emb(cap_ids)[:, 1:].detach()
    lv = F.mse_loss(pv, tv)
    lt = F.mse_loss(pt, tt)
    return (lv+lt)/2, lv.detach(), lt.detach()
JointAR.loss = _joint_total_loss

results = {}
print("Training A (vis/attn)...", flush=True)
torch.cuda.reset_peak_memory_stats()
lvA, msA = train_vis(mA, oA)
paA = torch.cuda.max_memory_allocated()/1e6; prA = torch.cuda.max_memory_reserved()/1e6
results["A"] = (lvA, msA, paA, prA)
del mA, oA; gc.collect(); torch.cuda.empty_cache()

print("Training B (vis/v8a)...", flush=True)
torch.cuda.reset_peak_memory_stats()
lvB, msB = train_vis(mB, oB)
paB = torch.cuda.max_memory_allocated()/1e6; prB = torch.cuda.max_memory_reserved()/1e6
results["B"] = (lvB, msB, paB, prB)
del mB, oB; gc.collect(); torch.cuda.empty_cache()

print("Training C (joint/attn)...", flush=True)
torch.cuda.reset_peak_memory_stats()
lvC, msC = train_joint(mC, oC)
paC = torch.cuda.max_memory_allocated()/1e6; prC = torch.cuda.max_memory_reserved()/1e6
results["C"] = (lvC, msC, paC, prC)
del mC, oC; gc.collect(); torch.cuda.empty_cache()

print("Training D (joint/v8a)...", flush=True)
torch.cuda.reset_peak_memory_stats()
lvD, msD = train_joint(mD, oD)
paD = torch.cuda.max_memory_allocated()/1e6; prD = torch.cuda.max_memory_reserved()/1e6
results["D"] = (lvD, msD, paD, prD)
del mD, oD; gc.collect(); torch.cuda.empty_cache()

# Predeclared interaction
d_vis = results["B"][0] - results["A"][0]    # Δ vision-only
d_joint = results["D"][0] - results["C"][0]  # Δ joint
I = d_joint - d_vis

lines = []
lines.append(f"A vis/attn:  val_vis {results['A'][0]:.4f} | {results['A'][1]:.1f} ms/step alloc {results['A'][2]:.0f} res {results['A'][3]:.0f}")
lines.append(f"B vis/v8a:   val_vis {results['B'][0]:.4f} | {results['B'][1]:.1f} ms/step alloc {results['B'][2]:.0f} res {results['B'][3]:.0f}")
lines.append(f"C joint/attn: val_vis {results['C'][0]:.4f} | {results['C'][1]:.1f} ms/step alloc {results['C'][2]:.0f} res {results['C'][3]:.0f}")
lines.append(f"D joint/v8a:  val_vis {results['D'][0]:.4f} | {results['D'][1]:.1f} ms/step alloc {results['D'][2]:.0f} res {results['D'][3]:.0f}")
lines.append(f"Δ_vis-only = {d_vis:.4f} | Δ_joint = {d_joint:.4f} | I = {I:.4f}")
lines.append("I<0: rescue | I≈0: no interaction | I>0: joint hurts V8-A")
for L in lines:
    print(L, flush=True)
with open("/content/v12p4_2x2.txt", "w") as f:
    f.write("\\n".join(lines) + "\\n")
from google.colab import files
files.download("/content/v12p4_2x2.txt")
print("SAVE 2x2 DONE")
''')

    path = f"{WS}/linear_attention_v12p4_2x2.ipynb"
    with open(path, "w") as f:
        json.dump(NB, f)
    return path, len([c for c in NB["cells"] if c["cell_type"] == "code"])


if __name__ == "__main__":
    path, n = build_notebook()
    nb = json.load(open(path))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(code_cells):
        compile("".join(c["source"]), f"<2x2#{i}>", "exec")
    src = "\n".join("".join(c["source"]) for c in code_cells)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); tpath = f.name
    r = subprocess.run(["python3", "-m", "pyflakes", tpath],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.expanduser("~/workspace/.pylibs")})
    os.unlink(tpath)
    out = (r.stdout + r.stderr).strip()
    crit = [l for l in out.splitlines()
            if "undefined name" in l or "SyntaxError" in l or "invalid syntax" in l]
    assert not crit, crit
    print(f"{n} code cells, verified clean -> {path}")
    print("2X2 NOTEBOOK VERIFIED")
