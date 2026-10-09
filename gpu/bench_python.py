import sys, os, time, random
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from goai_gpu.mcts import Search

# 纯 Python 侧成本：搜索树 + 规则（不包含网络推理）
for size, sims in ((9, 64), (19, 64)):
    s = Search(size, 7.0, pass_min_move=2 * size)
    rng = random.Random(1)
    t0 = time.perf_counter()
    n = 0
    for _ in range(sims):
        node, path = s.descend()
        s.expand(node, np.full(size * size + 1, 1.0 / (size * size + 1), dtype=np.float32), 0.0)
        s.backup(path, 0.0)
        n += 1
    dt = time.perf_counter() - t0
    print("%2d 路：纯 Python 侧 %.0f us/次模拟（%d 次共 %.2fs）" % (size, dt / n * 1e6, n, dt))
