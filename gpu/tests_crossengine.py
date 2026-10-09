"""跨引擎数值对照：同一权重、同一局面，PyTorch 与 C 引擎的输出必须一致"""
import os, subprocess, sys, re, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from goai_gpu.model import Net
from goai_gpu.export import export_c_bin
from goai_gpu.rules import Game, BLACK, WHITE

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOAI = os.path.join(PROJ, "build", "GoAI")

torch.manual_seed(20241001)
size, ch, vh = 9, 32, 64
m = Net(size=size, channels=ch, vhidden=vh)
# 稍微扰动一下，避免初始化恰好对称
with torch.no_grad():
    for t in m.c_param_tensors():
        t.add_(torch.randn_like(t) * 0.05)
path = os.path.join(tempfile.gettempdir(), "goai_cross_check.bin")
export_c_bin(m, path)

def py_eval(kind):
    g = Game(size)
    col = WHITE if kind == 1 else BLACK
    if kind in (0, 1, 4):
        g.cells[:, 0:4] = col
    g.to_move = WHITE if kind >= 3 else BLACK
    x = torch.from_numpy(g.features()).unsqueeze(0)
    pol, val = m.policy_value(x)
    return float(val.item()), int(pol.argmax()), float(pol.max().item())

out = subprocess.run([GOAI, "diag", "--weights", path], capture_output=True, text=True)
vals_c, moves_c = [], []
for line in out.stdout.splitlines():
    mm = re.search(r"value ([+-][0-9.]+)\s+top policy move (\S+)", line)
    if mm:
        vals_c.append(float(mm.group(1)))
        moves_c.append(mm.group(2))

def vertex(idx):
    if idx == size * size:
        return "pass"
    return "%c%d" % (chr(ord('A') + (idx % size) + (1 if idx % size >= 8 else 0)), size - idx // size)

print("  局面                    PyTorch 价值   C 引擎价值    差值    PyTorch首选  C首选")
worst = 0.0
ok = True
for kind in range(5):
    v_py, best_py, p_py = py_eval(kind)
    v_c = vals_c[kind]
    mv_c = moves_c[kind]
    diff = abs(v_py - v_c)
    worst = max(worst, diff)
    same_move = vertex(best_py) == mv_c
    ok = ok and diff < 2e-4
    print("  kind=%d                  %+9.5f    %+9.5f   %.5f   %-6s      %-6s %s"
          % (kind, v_py, v_c, diff, vertex(best_py), mv_c, "" if same_move else "(首选不同)"))

print("\n  最大价值偏差 = %.6f （应为浮点误差量级）" % worst)
print("跨引擎一致性", "通过" if ok else "失败")
sys.exit(0 if ok else 1)
