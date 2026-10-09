"""Generate V12 Phase 4 joint multimodal notebook.

Design (his spec):
- PAIRED image-caption data (Flickr8k via HuggingFace), not independent samples.
- Sequence order: [caption tokens, image patches] — vision predictions see
  the full caption as context (tests: does text context compensate V8-A's
  spatial weakness?).
- Loss normalization: loss = (vision_mse + text_mse) / 2, where each is a
  mean over its own elements. Neither modality dominates by element count.
- Reports vision/text/total separately. 1500 steps, both arms.

Useful outcomes (his frame):
- V8-A vision improves vs Phase 1 vision-only → cross-modal compensation.
- Attention retains vision lead → weakness persists.
- V8-A faster but worse quality → efficiency-quality trade-off.
- Both degrade → investigate joint setup, not the mixer.
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

    md("""# V12 Phase 4 — joint image+caption (Flickr30k, paired)

- PAIRED data: each caption describes its image (not independent samples).
- Order: [caption tokens, image patches] — vision sees text context.
- Loss = (vision_mse + text_mse)/2, each a mean over its own elements.
- 500-step PILOT (extend only if justified). Train + held-out VAL loss.
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

    code('''import urllib.request, zipfile, glob
from PIL import Image
import re
from collections import Counter

# Flickr8k from source zips (Hub script-based datasets unsupported on this
# Colab image). Paired: each caption describes its image.
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
# find image dir robustly
_img_dirs = glob.glob("data/**/Flicker8k_Dataset", recursive=True)
if _img_dirs:
    IMG_DIR = _img_dirs[0]
print(f"image dir: {IMG_DIR}")
TXT_DIR = "data/Flickr8k_text"
if not os.path.exists(TXT_DIR):
    print("extracting text...", flush=True)
    with zipfile.ZipFile(txt_zip) as z:
        z.extractall("data")
# find Flickr8k.token.txt robustly (zip internal structure varies)
_tok_paths = glob.glob("data/**/Flickr8k.token.txt", recursive=True)
assert _tok_paths, "Flickr8k.token.txt not found after extraction"
TOKEN_PATH = _tok_paths[0]
print(f"token file: {TOKEN_PATH}")

# parse Flickr8k.token.txt: "1000268201_693b08cb0e.jpg#0\tA child in a pink dress..."
pairs = {}  # img_name -> [captions]
with open(TOKEN_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        img_cap, caption = line.split("\t")
        img_name = img_cap.split("#")[0]
        pairs.setdefault(img_name, []).append(caption)
items = [(n, cs[0]) for n, cs in pairs.items()
         if os.path.exists(f"{IMG_DIR}/{n}")]
print(f"paired items: {len(items)}")
print("sample:", items[0][1][:80])

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
T = T_TXT + T_VIS  # 288

_orig_block_forward = Block.forward
def _block_forward_ckpt(self, x, reset=None):
    x = x + self.mix(self.ln1(x), reset)
    x = x + _ckpt.checkpoint(self.mlp, self.ln2(x), use_reentrant=False)
    return x
Block.forward = _block_forward_ckpt

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

class JointEncoder(nn.Module):
    """Paired image+caption -> [text emb (T_TXT), patch emb (T_VIS)]."""
    def __init__(self, dim, vocab, patch=8):
        super().__init__()
        self.txt_emb = nn.Embedding(vocab, dim)
        self.patch = patch
        self.elem_dim_vis = 3 * patch * patch
        self.proj_vis = nn.Linear(self.elem_dim_vis, dim)
    def elements_vis(self, img):
        # raw patches (B, T_VIS, 192) — vision targets
        B, C, H, W = img.shape
        p = self.patch
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        return x.view(B, -1, self.elem_dim_vis)
    def forward(self, img, cap_ids):
        B = img.shape[0]
        t = self.txt_emb(cap_ids)  # (B, T_TXT, dim)
        v = self.proj_vis(self.elements_vis(img))  # (B, T_VIS, dim)
        return torch.cat([t, v], dim=1)  # (B, T, dim)

class JointAR(nn.Module):
    def __init__(self, dim, n_layers, n_heads, seq_len, mixer_fn, ffn_hidden):
        super().__init__()
        self.encoder = JointEncoder(dim, VOCAB, PATCH)
        self.seq_len = seq_len
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, mixer_fn(dim, n_heads), 0.0, ffn_hidden)
             for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.head_vis = nn.Linear(dim, 3 * PATCH * PATCH, bias=False)
        self.head_txt = nn.Linear(dim, dim, bias=False)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, img, cap_ids):
        x = self.encoder(img, cap_ids)
        x = x + self.pos_emb(torch.arange(T, device=x.device))[None]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        # text preds: positions 0..T_TXT-2 predict tokens 1..T_TXT-1
        # vision preds: positions T_TXT..T-2 predict patches
        pv = self.head_vis(x[:, T_TXT:T-1])      # (B, T_VIS-1, 192)
        pt = self.head_txt(x[:, :T_TXT-1])       # (B, T_TXT-1, dim)
        return pv, pt
    def loss(self, img, cap_ids):
        pv, pt = self(img, cap_ids)
        with torch.no_grad():
            tv = self.encoder.elements_vis(img)[:, 1:]          # (B, T_VIS-1, 192)
            tt = self.encoder.txt_emb(cap_ids)[:, 1:].detach()   # (B, T_TXT-1, dim)
        lv = F.mse_loss(pv, tv)  # mean over vision elements
        lt = F.mse_loss(pt, tt)  # mean over text elements
        return (lv + lt) / 2, lv.detach(), lt.detach()  # normalized!

def build(mixer, seed, ffn_hidden):
    set_seed(seed)
    m = JointAR(DIM, LAYERS, HEADS, T, mixer, ffn_hidden).to(device)
    import math as _math
    _floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
    for _i, _blk in enumerate(m.blocks):
        if hasattr(_blk.mix, "forget_floor"):
            _blk.mix.forget_floor.data.fill_(_floors[_i])
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    return m, o

# param matching (mixer-only)
torch.manual_seed(0)
_m0 = JointAR(DIM, 1, HEADS, T,
              lambda d, h: SelectiveSegmentedStateV8AFused(d, state_dim=256),
              ffn_hidden=FFN_V8A)
_pv = sum(p.numel() for p in _m0.blocks[0].mix.parameters())
_ma = JointAR(DIM, 1, HEADS, T,
              lambda d, h: CausalSelfAttention(d, h), ffn_hidden=FFN_V8A)
_pa = sum(p.numel() for p in _ma.blocks[0].mix.parameters())
FFN_ATTN = match_ffn_hidden(FFN_V8A, DIM, _pv, _pa)
del _m0, _ma
print(f"mixer params/layer: v8a {_pv} vs attn {_pa} -> ffn {FFN_V8A} vs {FFN_ATTN}")
print(f"T_txt={T_TXT} T_vis={T_VIS} T={T}")

# FIXED paired batch (seed 42): same (image, caption) pairs for both arms
# VALIDATION batch (seed 43): held-out pairs, never trained on
frng = random.Random(42)
idx_train = [frng.randrange(len(items)) for _ in range(BS)]
vrng = random.Random(43)
idx_val = [vrng.randrange(len(items)) for _ in range(BS)]
def load_pair(i):
    name, caption = items[i]
    im = load_image(name).resize((IMG, IMG), Image.BILINEAR)
    t = torch.from_numpy(__import__("numpy").array(im)).permute(2,0,1).float()/255.0
    cap = encode_caption(caption, T_TXT)
    return t, torch.tensor(cap, dtype=torch.long)
imgs, caps = zip(*[load_pair(i) for i in idx_train])
# stack first, THEN normalize (avoids (1,3,1,1) broadcast on single images)
img_fixed = ((torch.stack(imgs) - IMAGENET_MEAN) / IMAGENET_STD).to(device)
cap_fixed = torch.stack(caps).to(device)
vimgs, vcaps = zip(*[load_pair(i) for i in idx_val])
img_val = ((torch.stack(vimgs) - IMAGENET_MEAN) / IMAGENET_STD).to(device)
cap_val = torch.stack(vcaps).to(device)
print(f"fixed paired batch: img {tuple(img_fixed.shape)} cap {tuple(cap_fixed.shape)}")
print(f"validation batch:   img {tuple(img_val.shape)} cap {tuple(cap_val.shape)}")
assert tuple(img_fixed.shape) == (BS, 3, IMG, IMG)
assert tuple(cap_fixed.shape) == (BS, T_TXT)
assert tuple(img_val.shape) == (BS, 3, IMG, IMG)
assert tuple(cap_val.shape) == (BS, T_TXT)

def run_arm(mixer_fn, ffn_hidden, label, steps=500):
    """500-step pilot. Reports train + VALIDATION (held-out) per-modality loss."""
    m, o = build(mixer_fn, 0, ffn_hidden)
    # warmup (timed separately)
    warm = []
    for _ in range(5):
        o.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        tot, _, _ = m.loss(img_fixed, cap_fixed)
        tot.backward(); o.step()
        torch.cuda.synchronize()
        warm.append(1000*(time.perf_counter()-t0))
    compile_ms, warmup_med = warm[0], statistics.median(warm[1:])
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        o.zero_grad(set_to_none=True)
        tot, lv, lt = m.loss(img_fixed, cap_fixed)
        tot.backward(); o.step()
    torch.cuda.synchronize()
    train_ms = 1000*(time.perf_counter()-t0)/steps
    # final: train + VALIDATION per-modality loss (no grad)
    m.eval()
    with torch.no_grad():
        tot_tr, lv_tr, lt_tr = m.loss(img_fixed, cap_fixed)
        tot_va, lv_va, lt_va = m.loss(img_val, cap_val)
    peak_a = torch.cuda.max_memory_allocated()/1e6
    peak_r = torch.cuda.max_memory_reserved()/1e6
    del m, o; gc.collect(); torch.cuda.empty_cache()
    return {"train_total": f"{tot_tr.item():.4f}", "train_vis": f"{lv_tr.item():.4f}",
            "train_txt": f"{lt_tr.item():.4f}",
            "val_total": f"{tot_va.item():.4f}", "val_vis": f"{lv_va.item():.4f}",
            "val_txt": f"{lt_va.item():.4f}", "ms": f"{train_ms:.1f}",
            "alloc": f"{peak_a:.0f}", "res": f"{peak_r:.0f}",
            "compile_ms": f"{compile_ms:.0f}",
            "warmup_med_ms": f"{warmup_med:.1f}"}

results = {}
results["v8a"] = run_arm(lambda d,h: SelectiveSegmentedStateV8AFused(d, state_dim=256),
                         FFN_V8A, "V8-A")
results["attn"] = run_arm(lambda d,h: CausalSelfAttention(d, h), FFN_ATTN, "ATTN")

def fmt(r):
    return (f"train total {r['train_total']} (vis {r['train_vis']} txt {r['train_txt']}) | "
            f"VAL total {r['val_total']} (vis {r['val_vis']} txt {r['val_txt']}) | "
            f"{r['ms']} ms/step alloc {r['alloc']} res {r['res']} "
            f"[compile {r['compile_ms']} ms | warmup {r['warmup_med_ms']} ms]")
line1 = f"V8-A: {fmt(results['v8a'])}"
line2 = f"ATTN: {fmt(results['attn'])}"
print(line1, flush=True)
print(line2, flush=True)
with open("/content/v12p4.txt", "w") as f:
    f.write(line1 + "\\n" + line2 + "\\n")
from google.colab import files
files.download("/content/v12p4.txt")
print("SAVE p4 DONE")
''')

    path = f"{WS}/linear_attention_v12p4.ipynb"
    with open(path, "w") as f:
        json.dump(NB, f)
    return path, len([c for c in NB["cells"] if c["cell_type"] == "code"])


if __name__ == "__main__":
    path, n = build_notebook()
    nb = json.load(open(path))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(code_cells):
        compile("".join(c["source"]), f"<p4#{i}>", "exec")
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
    print("PHASE 4 NOTEBOOK VERIFIED")
