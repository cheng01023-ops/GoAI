import sys, os, random, time
sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
from goai_gpu.model import Net
from goai_gpu.mcts import Search
class C: pass
cfg = C(); cfg.size=9; cfg.channels=8; cfg.vhidden=32; cfg.games=1; cfg.sims=4
cfg.komi=7.0; cfg.temp_moves=12; cfg.max_moves=243; cfg.amp="none"; cfg.device="cpu"; cfg.save_sgf=0
net = Net(9,4,8,32); net.eval()

def eval_batch(boards):
    x = np.stack([b.features() for b in boards]).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.inference_mode():
        pol, val = net.policy_value(xt)
    return pol.float().cpu().numpy(), val.float().cpu().numpy()

s = Search(9, 7.0)
t_desc = t_eval = t_exp = t_back = t_play = 0.0
moves = 0
rng = random.Random(1)
while moves < 30:
    sim_step = 0
    for _ in range(cfg.sims):
        sim_step += 1
        t0 = time.perf_counter(); node, path = s.descend(); t_desc += time.perf_counter()-t0
        n = s.nodes[node]
        if n.terminal:
            t0 = time.perf_counter(); s.backup(path, n.tval); t_back += time.perf_counter()-t0
            continue
        t0 = time.perf_counter(); pols, vals = eval_batch([n.board]); t_eval += time.perf_counter()-t0
        t0 = time.perf_counter(); s.expand(node, pols[0], float(vals[0])); t_exp += time.perf_counter()-t0
        t0 = time.perf_counter(); s.backup(path, float(vals[0])); t_back += time.perf_counter()-t0
    t0 = time.perf_counter()
    mv = s.choose(0.0, rng)
    board = s.nodes[s.root].board
    board.play_known_legal(mv)
    s.advance(mv)
    t_play += time.perf_counter()-t0
    moves += 1
print("30 手 (每手 %d 次模拟) 耗时分解:" % cfg.sims)
print("  descend %.3fs | eval %.3fs | expand %.3fs | backup %.3fs | 落子 %.3fs" % (t_desc, t_eval, t_exp, t_back, t_play))
print("  合计 %.3fs => 每手 %.1f ms, 每模拟 %.3f ms" % (t_desc+t_eval+t_exp+t_back+t_play, (t_desc+t_eval+t_exp+t_back+t_play)/30*1000, (t_desc+t_eval+t_exp+t_back+t_play)/(30*cfg.sims)*1000))
print("  节点数 %d, 根 N=%d" % (len(s.nodes), s.nodes[s.root].N))
