import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/gpu")
from goai_gpu.rules import Game, PASS

g = Game(9)
rng = random.Random(1)
# 走到中盘
for _ in range(40):
    mv = rng.choice(g.legal_moves())
    g.play(mv)

t0 = time.perf_counter()
N = 2000
for _ in range(N):
    g.legal_moves()
t1 = time.perf_counter()
print("legal_moves (9x9 中盘): %.1f us/次" % ((t1-t0)/N*1e6))

t0 = time.perf_counter()
for _ in range(N*5):
    g.clone()
t1 = time.perf_counter()
print("clone: %.2f us/次" % ((t1-t0)/(N*5)*1e6))

t0 = time.perf_counter()
for _ in range(N*20):
    g.features()
t1 = time.perf_counter()
print("features: %.2f us/次" % ((t1-t0)/(N*20)*1e6))

# 模拟一次完整自对弈的成本（4 盘 x 8 sims）
import torch
from goai_gpu.model import Net
from goai_gpu.mcts import Search
net = Net(9, 4, 8, 32)
net.eval()
t0 = time.perf_counter()
searches = [Search(9, 7.0) for _ in range(4)]
s = searches[0]
for _ in range(32):
    node, path = s.descend()
    n = s.nodes[node]
    if n.terminal:
        s.backup(path, n.tval); continue
    import numpy as np
    x = torch.from_numpy(np.stack([n.board.features()])).float()
    with torch.inference_mode():
        pol, val = net.policy_value(x)
    s.expand(node, pol[0].numpy(), float(val[0]))
    s.backup(path, float(val[0]))
t1 = time.perf_counter()
print("32 次 MCTS 模拟（含 1 次批量推理/次）: %.1f ms => %.2f ms/模拟" % ((t1-t0)*1e3, (t1-t0)*1e3/32))
print("预计 4 盘 x 8 sims x 100 手 = 3200 模拟 ≈ %.1f 秒" % ((t1-t0)/32*3200))
