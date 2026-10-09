import sys, os, random, time
sys.path.insert(0, os.path.abspath("."))
import numpy as np, torch
from goai_gpu.model import Net
import train as T

class C: pass
cfg = C(); cfg.size=9; cfg.channels=8; cfg.vhidden=32; cfg.games=1; cfg.sims=4
cfg.komi=7.0; cfg.temp_moves=12; cfg.max_moves=243; cfg.amp="none"; cfg.device="cpu"; cfg.save_sgf=0
net = Net(9,4,8,32); net.eval()

pool = T.SelfPlayPool(cfg, net, "cpu", random.Random(1))
# 打补丁：给 run 里的落子步骤加打印
orig_run = T.SelfPlayPool.run
s = pool.searches[0]
t0 = time.time()
active = [0]
sim_step = 0
while active:
    sim_step += 1
    batch_boards, batch_meta = [], []
    for g in active:
        se = pool.searches[g]
        node, path = se.descend()
        n = se.nodes[node]
        if n.terminal:
            se.backup(path, n.tval); continue
        batch_boards.append(n.board); batch_meta.append((g, node, path))
    if batch_boards:
        pols, vals = pool.eval_batch(batch_boards)
        for (g, node, path), pol, val in zip(batch_meta, pols, vals):
            pool.searches[g].expand(node, pol, float(val))
            pool.searches[g].backup(path, float(val))
    if sim_step % cfg.sims: continue
    for g in active:
        se = pool.searches[g]
        mv = se.choose(1.0 if len(pool.moves[g]) < cfg.temp_moves else 0.0, pool.rng)
        se.advance(mv)
        board = se.nodes[se.root].board
        pool.moves[g].append(mv)
        if len(pool.moves[g]) % 25 == 0:
            print("  第 %3d 手: nmoves=%3d passes=%d to_move=%d 用时 %.2fs" % (
                len(pool.moves[g]), board.nmoves, board.passes, board.to_move, time.time()-t0)); sys.stdout.flush()
        if board.is_terminal(cfg.max_moves):
            print("  终局于第 %d 手 (nmoves=%d passes=%d)" % (len(pool.moves[g]), board.nmoves, board.passes))
            active = []
            break
print("总用时 %.2fs, 手数 %d, 样本 %d" % (time.time()-t0, len(pool.moves[0]), len(pool.examples[0])))
