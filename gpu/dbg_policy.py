import sys, os, random
sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
from goai_gpu.model import Net
import train as T
class C: pass
cfg = C(); cfg.size=9; cfg.channels=8; cfg.vhidden=32; cfg.games=3; cfg.sims=8
cfg.komi=7.0; cfg.temp_moves=12; cfg.max_moves=243; cfg.amp="none"; cfg.device="cpu"; cfg.save_sgf=0
net = Net(9,4,8,32); net.eval()
pool = T.SelfPlayPool(cfg, net, "cpu", random.Random(1))
xs, pis, zs = pool.run()
zeros = int((pis.sum(axis=1) == 0).sum())
print("样本数 =", len(zs))
print("pi 全零行 =", zeros, "/", len(pis), " pi 最大值 = %.3f" % float(pis.max()))
print("每盘手数 =", [len(m) for m in pool.moves], " 胜负 =", pool.winner)
print("z 分布:", {float(v): int((zs==v).sum()) for v in set(zs.tolist())})
ok = zeros == 0 and float(pis.max()) > 0.05
print("策略目标自检", "通过" if ok else "失败")
sys.exit(0 if ok else 1)
