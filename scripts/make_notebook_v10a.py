"""Generate Colab notebook V10-E0: memory autopsy.

V9 closed two VRAM levers experimentally (bf16 storage, recompute-a) and showed
the scan state (~67 MB) was never the problem. V10-E0 measures where the ~943 MB
peak actually lives BEFORE any new lever is built.

His methodological requirement: record peak *live* memory by tensor/category
(bytes alive at the moment of peak), NOT cumulative allocation. Lifetimes decide
the map, not sizes. E0.3 implements this via _record_memory_history snapshots
dumped at fwd/bwd/opt points of one step, with allocations attributed by Python
stack -> source line -> category (LINEMAP baked in at generation time from the
exact inlined source via ast).

Gates: named live-at-peak categories >= 85% of snapshot live total; accounted
(live + fragmentation gap) within 10% of peak reserved. Falsifies any lever
whose target tensor isn't in the top-3 live-at-peak.
"""
import ast
import json

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


# ----------------------------------------------------------------------------
# Generator-side: build LINEMAP {lineno: category} from the exact cell-2 source
# ----------------------------------------------------------------------------
models_src = open("/home/hatch/workspace/linear-attention-lab/models.py").read().replace(
    '"""Linear-recurrent replacement for Transformer self-attention.',
    '"""(inlined) Linear-recurrent replacement for Transformer self-attention.').replace(
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import math\nimport torch.nn as nn\nimport torch.nn.functional as F\n")
kern_src = open("/home/hatch/workspace/linear-attention-lab/triton_kernels.py").read().replace(
    '"""V6: Triton-fused selective scan for the gated segmented state space.',
    '"""(inlined) V6/V7/V8-A: Triton-fused scans.').replace(
    "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
    "import triton\nimport triton.language as tl\n")
cell2_src = models_src + "\n\n" + kern_src


def categorize_line(text, cls):
    t = text.lower()
    if cls == "_DeltaScanFn" or "scanfn" in cls.lower():
        return "scan-state"
    if "mlp" in t or "gelu" in t:
        return "ffn"
    if any(k in t for k in ("in_proj", "gate_proj", "write_proj", "key_proj",
                            "out_gate_proj", "out_proj")):
        return "mixer-proj"
    if "triton_delta_scan" in t:
        return "scan-state"
    if ("layernorm" in t or "rmsnorm" in t or "rms_norm" in t or "ln1" in t
            or "ln2" in t or "ln_f" in t or "sqrt" in t):
        return "norms"
    if "embedding" in t or "tok_emb" in t or "pos_emb" in t:
        return "embeddings"
    if "cross_entropy" in t or "self.head" in t:
        return "logits/loss"
    if cls in ("SelectiveSegmentedStateV8A", "Block", "TinyLM",
               "CausalSelfAttention", "LinearRunningState"):
        return cls + "-misc"
    return "other"


def build_linemap(src):
    tree = ast.parse(src)
    spans = []  # (start, end, kind, name, parent_class)

    def visit(node, cur_cls=None):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                spans.append((ch.lineno, ch.end_lineno, "class", ch.name, None))
                visit(ch, ch.name)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((ch.lineno, ch.end_lineno, "func", ch.name, cur_cls))
                visit(ch, cur_cls)
            else:
                visit(ch, cur_cls)

    visit(tree)
    lmap = {}
    for i, text in enumerate(src.splitlines(), start=1):
        func_match = cls_match = None
        for (s, e, kind, name, pc) in spans:
            if s <= i <= e:
                if kind == "func":
                    func_match = (name, pc)
                elif kind == "class" and cls_match is None:
                    cls_match = name
        if func_match:
            cat = categorize_line(text, func_match[1] or "")
        elif cls_match:
            cat = categorize_line(text, cls_match)
        else:
            cat = "other"
        if cat != "other":
            lmap[i] = cat
    return lmap


LINEMAP = build_linemap(cell2_src)
print(f"LINEMAP entries: {len(LINEMAP)}")
assert len(LINEMAP) > 50, "LINEMAP suspiciously small"

# ----------------------------------------------------------------------------
# Notebook cells
# ----------------------------------------------------------------------------
md("""# V10-E0 — Memory autopsy: where exactly are the ~943 MB?

V9 killed two VRAM levers (bf16 storage, recompute-`a`) and proved the scan
state (~67 MB) was never the problem. This notebook measures the peak before
any new lever is designed.

**Methodological rule for this autopsy:** report peak *live* memory by
tensor/category — bytes alive at the moment of peak — never cumulative
allocation. A large tensor freed before the peak contributes zero to it;
lifetimes decide the map, not sizes.

Analytical prediction (to confirm or kill):
FFN ≈ 232 MB | mixer projections ≈ 134 MB | norms/residuals ≈ 134 MB |
scan state ≈ 67 MB | remainder (workspace/transients/fragmentation) ≈ 440 MB.

**E0 gate:** named live-at-peak categories ≥ 85% of snapshot live total, and
accounted (live + fragmentation gap) within 10% of peak reserved. Any lever
whose target tensor isn't in the top-3 live-at-peak is falsified before it's
built. No GitHub push.
""")

md("## 0. Setup")
code("""import torch
import triton
print("torch", torch.__version__, "| triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda"
assert torch.cuda.is_available(), "V10-E0 needs the T4 GPU"

def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed(0)
""")

md("## 1. Data — campaign corpus A only (vocab 104, same as V8-A Run 1)")
code("""import os, urllib.request
def dl(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("downloading", url, flush=True)
        urllib.request.urlretrieve(url, path)
    return path

URLS = {
    "tinyshakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "alice.txt":           "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "frankenstein.txt":    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "pride.txt":           "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
}
parts = []
for name, url in URLS.items():
    p = dl(url, f"data/{name}")
    with open(p, encoding="utf-8", errors="ignore") as f:
        parts.append(f.read())
text_a = "\\n".join(parts)
chars_a = sorted(set(text_a)); vocab_a = len(chars_a)
stoi_a = {c: i for i, c in enumerate(chars_a)}
data_a = torch.tensor([stoi_a[c] for c in text_a], dtype=torch.long)
nl_a = stoi_a.get("\\n")
assert vocab_a == 104, f"vocab drift: {vocab_a}"

def make_get_batch(d, seq_len):
    def get_batch(split="train", batch_size=32):
        dd = d[:int(0.95 * len(d))] if split == "train" else d[int(0.95 * len(d)):]
        i = torch.randint(0, len(dd) - seq_len - 1, (batch_size,))
        x = torch.stack([dd[j:j+seq_len] for j in i]).to(device)
        y = torch.stack([dd[j+1:j+seq_len+1] for j in i]).to(device)
        return x, y
    return get_batch

get_batch = make_get_batch(data_a, 128)
print(f"vocab {vocab_a} | tokens {len(data_a)/1e6:.1f}M | nl_id={nl_a}")
""")

md("## 2. Models + Triton kernels (inlined, no imports)")
code(cell2_src)

md("## 2b. Line->category map (baked at generation from the exact source above)")
code(f"_LINEMAP = {LINEMAP!r}\n"
     "print(f\"LINEMAP loaded: {len(_LINEMAP)} source lines categorized\")\n"
     "from collections import Counter\n"
     "print(Counter(_LINEMAP.values()))\n")

md("## 3. Build V8-A (identical to Run 1) + train_step")
code("""DIM, LAYERS, HEADS, SEQ, FFN_H = 256, 8, 8, 128, 766
v8a = TinyLM(vocab_a, DIM, LAYERS, HEADS, SEQ,
             mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
             ffn_hidden=FFN_H, newline_id=nl_a).to(device)
import math as _math
_floors = torch.linspace(_math.log(0.3/0.7), _math.log(0.9/0.1), LAYERS)
for _i, _blk in enumerate(v8a.blocks):
    _blk.mix.forget_floor.data.fill_(_floors[_i])
n_params = sum(p.numel() for p in v8a.parameters())
print(f"params: {n_params} (expect ~6.37M)")

opt = torch.optim.AdamW(v8a.parameters(), lr=3e-4)

def train_step(bs=32):
    opt.zero_grad(set_to_none=True)
    xb, yb = get_batch("train", batch_size=bs)
    _, loss = v8a(xb, yb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(v8a.parameters(), 1.0)
    opt.step()
    return float(loss)

train_step()  # warmup: Triton compile + allocator warm
print("warmup done")
""")

md("## E0.1 — Coarse peak split (allocated vs reserved)")
code("""torch.cuda.reset_peak_memory_stats()
for _ in range(3):
    train_step()
peak_alloc = torch.cuda.max_memory_allocated() / 1e6
peak_res = torch.cuda.max_memory_reserved() / 1e6
frag_gap = peak_res - peak_alloc
print(f"peak alloc    {peak_alloc:.0f} MB")
print(f"peak reserved {peak_res:.0f} MB")
print(f"fragmentation gap (reserved-alloc): {frag_gap:.0f} MB  <- first free target (E2)")
""")

md("## E0.2 — Allocator counters")
code("""s = torch.cuda.memory_stats()
for k in ("allocated_bytes.all.peak", "reserved_bytes.all.peak",
          "inactive_split_bytes.all.peak", "num_alloc_retries", "num_ooms"):
    v = s[k]
    print(f"{k:32s}: {v/1e6:.1f} MB" if "bytes" in k else f"{k:32s}: {v}")
""")

md("""## E0.3 — Peak-LIVE attribution by category (the centerpiece)

Snapshots dumped at three points of one step (post-forward, post-backward,
post-optimizer). Live blocks at each dump are attributed via Python stack ->
source line -> category. The max-live dump is the peak composition.
""")
code('''import pickle

def snapshot_live(tag):
    """Dump a memory snapshot; return {category: live_bytes} at dump time."""
    path = f"/content/snap_{tag}.pickle"
    torch.cuda.memory._dump_snapshot(path)
    snap = pickle.load(open(path, "rb"))
    cats, total = {}, 0
    n_blocks = 0
    _dbg = {"n_frames_total": 0, "blocks_with_user_frame": 0}
    for seg in snap["segments"]:
        for b in seg.get("blocks", []):
            if b.get("state") != "active_allocated":
                continue
            n_blocks += 1
            sz = b["size"]
            total += sz
            cat = "other"
            frames = b.get("frames", []) or []
            _dbg["n_frames_total"] += len(frames)
            hit_user = False
            # torch orders frames outermost->innermost; scan innermost-first so a
            # block attributes to the module line that created it, not the
            # training-loop line that called the model.
            for fr in reversed(frames):
                if isinstance(fr, (list, tuple)) and len(fr) >= 2:
                    fn, ln = str(fr[0]), fr[1]
                elif isinstance(fr, dict):
                    fn, ln = str(fr.get("filename", "")), fr.get("line", 0)
                else:
                    continue
                if "torch/optim" in fn or "adamw" in fn.lower():
                    cat = "optimizer"
                    break
                if "ipython-input" in fn:
                    hit_user = True
                    try:
                        c = _LINEMAP.get(int(ln), "other")
                    except (TypeError, ValueError):
                        c = "other"
                    if c != "other":
                        cat = c
                        break
                    # line categorized as other: keep scanning outward for a named frame
            if hit_user:
                _dbg["blocks_with_user_frame"] += 1
            cats[cat] = cats.get(cat, 0) + sz
    print(f"[debug] frames seen: {_dbg['n_frames_total']}, "
          f"blocks with >=1 user frame: {_dbg['blocks_with_user_frame']}/{n_blocks}")
    return cats, total, n_blocks

torch.cuda.memory._record_memory_history(enabled=True)
opt.zero_grad(set_to_none=True)
xb, yb = get_batch("train")
_, loss = v8a(xb, yb)
cats_fwd, tot_fwd, nb_fwd = snapshot_live("fwd")
loss.backward()
cats_bwd, tot_bwd, nb_bwd = snapshot_live("bwd")
torch.nn.utils.clip_grad_norm_(v8a.parameters(), 1.0)
opt.step()
cats_opt, tot_opt, nb_opt = snapshot_live("opt")
torch.cuda.memory._record_memory_history(enabled=None)

_snaps = {"post-forward": (cats_fwd, tot_fwd, nb_fwd),
          "post-backward": (cats_bwd, tot_bwd, nb_bwd),
          "post-optimizer": (cats_opt, tot_opt, nb_opt)}
for tag, (cats, tot, nb) in _snaps.items():
    print(f"--- live at {tag}: {tot/1e6:.0f} MB across {nb} blocks ---")
    for c, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {c:28s} {v/1e6:7.1f} MB ({100*v/tot:4.1f}%)")
    print()
''')

md("## E0.4 — Per-op attribution (profiler, complement to E0.3)")
code("""from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True, record_shapes=True, with_stack=True) as prof:
    for _ in range(2):
        train_step()
print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=12))
""")

md("""## E0.5 — Owner-by-elimination (live-at-peak deltas)

Peak with the FFN / mixer ablated. The delta vs full *is* that component's
live-at-peak contribution, independent of the snapshot attribution.
""")
code('''def peak_of(make_fn, steps=2):
    m = make_fn().to(device)
    o = torch.optim.AdamW(m.parameters(), lr=3e-4)
    m(get_batch("train", batch_size=32)[0])  # warmup fwd (Triton compile)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        o.zero_grad(set_to_none=True)
        _, loss = m(*get_batch("train", batch_size=32))
        loss.backward()
        o.step()
    p = torch.cuda.max_memory_allocated() / 1e6
    del m, o
    import gc; gc.collect()
    return p

def build_variant(mode):
    m = TinyLM(vocab_a, DIM, LAYERS, HEADS, SEQ,
               mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
               ffn_hidden=FFN_H, newline_id=nl_a)
    if mode == "noffn":
        for blk in m.blocks:
            blk.mlp = torch.nn.Identity()
    if mode == "nomix":
        class Pass(torch.nn.Module):
            def forward(self, x, reset=None):
                return torch.zeros_like(x)
        for blk in m.blocks:
            blk.mix = Pass()
    return m

p_full = peak_of(lambda: build_variant("full"))
p_noffn = peak_of(lambda: build_variant("noffn"))
p_nomix = peak_of(lambda: build_variant("nomix"))
print(f"full     {p_full:.0f} MB")
print(f"no-FFN   {p_noffn:.0f} MB  (FFN live-at-peak ~ {p_full-p_noffn:.0f} MB)")
print(f"no-mixer {p_nomix:.0f} MB  (mixer live-at-peak ~ {p_full-p_nomix:.0f} MB)")
''')

md("## E0.6 — Batch-dim ablation (isolate batch-linear activations)")
code("""res = {}
for bs in (32, 16, 8, 4):
    train_step(bs)  # warm shapes
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        train_step(bs)
    res[bs] = torch.cuda.max_memory_allocated() / 1e6
    print(f"batch {bs:2d}: peak {res[bs]:.0f} MB", flush=True)
print("batch-linear slope ~", f"{(res[32]-res[4])/28:.1f} MB per sample")
""")

md("""## E0.7 — VERDICT: the optimization map

Gate: named live-at-peak categories >= 85% of snapshot live total, and
accounted (live + fragmentation gap) within 10% of peak reserved.
""")
code('''# Recompute the coarse peak for the gate (fresh stats, same protocol)
torch.cuda.reset_peak_memory_stats()
for _ in range(3):
    train_step()
g_alloc = torch.cuda.max_memory_allocated() / 1e6
g_res = torch.cuda.max_memory_reserved() / 1e6
g_frag = g_res - g_alloc

best_tag, (best_cats, best_tot, best_nb) = max(
    _snaps.items(), key=lambda kv: kv[1][1])
named = sum(v for k, v in best_cats.items() if k != "other")
other = best_cats.get("other", 0)
accounted = best_tot / 1e6 + g_frag

print(f"peak composition from: {best_tag} ({best_tot/1e6:.0f} MB live)")
print(f"peak reserved (coarse): {g_res:.0f} MB | frag gap: {g_frag:.0f} MB")
print(f"named categories: {named/1e6:.0f} MB ({100*named/best_tot:.1f}% of live)")
print(f"uncategorized 'other': {other/1e6:.0f} MB ({100*other/best_tot:.1f}% of live)")
print(f"accounted (live+frag) vs reserved: {accounted:.0f} vs {g_res:.0f} MB")
print()
print("=== ranked live-at-peak map ===")
ranked = sorted(best_cats.items(), key=lambda kv: -kv[1])
for i, (c, v) in enumerate(ranked, 1):
    print(f"  {i}. {c:28s} {v/1e6:7.1f} MB")
print()
print("=== analytical prediction (confirm or kill) ===")
for c, v in [("ffn", 232), ("mixer-proj", 134), ("norms", 134),
             ("scan-state", 67), ("remainder", 440)]:
    print(f"  {c:28s} ~{v} MB predicted")
print()

g1 = named / best_tot >= 0.85
g2 = abs(accounted - g_res) / g_res <= 0.10
print(f"GATE named>=85%: {'PASS' if g1 else 'FAIL'}")
print(f"GATE accounted within 10% of reserved: {'PASS' if g2 else 'FAIL'}")
if g1 and g2:
    top3 = [c for c, _ in ranked[:3]]
    print(f"E0 VERDICT: MAP CONFIRMED — top-3 live-at-peak: {top3}")
    print("Levers green-lit in this order: " +
          ", ".join(f"E{i}" for i, c in
                     enumerate(["ffn", "mixer-proj", "norms"], 1)
                     if c in top3) or "re-examine")
else:
    print("E0 VERDICT: MAP INCOMPLETE — no lever gets built on this. "
          "Inspect 'other' and re-run.")
''')

md("## 8. Save")
code('''lines = []
lines.append(f"peak_alloc_MB={g_alloc:.0f}")
lines.append(f"peak_reserved_MB={g_res:.0f}")
lines.append(f"frag_gap_MB={g_frag:.0f}")
for tag, (cats, tot, nb) in _snaps.items():
    lines.append(f"--- live_at_{tag} total_MB={tot/1e6:.0f} blocks={nb} ---")
    for c, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        lines.append(f"{tag} {c} MB={v/1e6:.1f}")
lines.append(f"ablation no-FFN delta_MB={p_full-p_noffn:.0f}")
lines.append(f"ablation no-mixer delta_MB={p_full-p_nomix:.0f}")
for bs, v in res.items():
    lines.append(f"batch_{bs} peak_MB={v:.0f}")
open("/content/v10a_results.txt", "w").write("\\n".join(lines))
print(open("/content/v10a_results.txt").read())
from google.colab import files
files.download("/content/v10a_results.txt")
print("SAVE DONE")
''')

with open("/home/hatch/workspace/linear-attention-lab/linear_attention_v10a_autopsy.ipynb", "w") as f:
    json.dump(NB, f)
print("notebook written:",
      "/home/hatch/workspace/linear-attention-lab/linear_attention_v10a_autopsy.ipynb",
      f"({len(NB['cells'])} cells)")
