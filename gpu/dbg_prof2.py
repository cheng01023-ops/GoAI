import sys, os, random, cProfile, pstats, io
sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
from goai_gpu.model import Net
import train as T
class C: pass
cfg = C(); cfg.size=9; cfg.channels=8; cfg.vhidden=32; cfg.games=1; cfg.sims=4
cfg.komi=7.0; cfg.temp_moves=12; cfg.max_moves=243; cfg.amp="none"; cfg.device="cpu"; cfg.save_sgf=0
net = Net(9,4,8,32); net.eval()
pool = T.SelfPlayPool(cfg, net, "cpu", random.Random(1))
pr = cProfile.Profile(); pr.enable()
xs, pis, zs = pool.run()
pr.disable()
print("样本数", len(zs), " 手数", len(pool.moves[0]))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
print(s.getvalue()[:2000])
