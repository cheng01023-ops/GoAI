"""规则自检：与 C 引擎 tests/test_board.c 中的用例一一对应"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from goai_gpu.rules import Game, PASS, BLACK, WHITE, EMPTY

checks = failures = 0


def check(cond, msg):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
        print("  FAIL:", msg)


def P(n, x, y):
    return y * n + x


def put(g, x, y, c):
    g.cells[y, x] = c


print("== Python 规则自检 ==")

# 1) 单子提子
g = Game(9)
put(g, 4, 4, WHITE); put(g, 3, 4, BLACK); put(g, 5, 4, BLACK); put(g, 4, 3, BLACK)
g.to_move = BLACK
check(g.play(P(9, 4, 5)), "吃子着法合法")
check(g.cells[4, 4] == EMPTY, "被吃的子被移除")
check(g.cells[4, 5] == BLACK, "落子保留")

# 2) 禁入点
g = Game(9)
for x, y in ((3, 4), (5, 4), (4, 3), (4, 5)):
    put(g, x, y, WHITE)
g.to_move = BLACK
check(not g.is_legal(P(9, 4, 4)), "自杀点判为非法")
check(not g.play(P(9, 4, 4)), "自杀被拒绝")

# 3) 整块提子
g = Game(9)
put(g, 3, 3, WHITE); put(g, 4, 3, WHITE)
for x, y in ((3, 2), (4, 2), (2, 3), (3, 4), (4, 4)):
    put(g, x, y, BLACK)
g.to_move = BLACK
check(g.play(P(9, 5, 3)), "填最后一气合法")
check(int(np.count_nonzero(g.cells == WHITE)) == 0, "两块白子被提")

# 4) 简单劫
g = Game(9)
put(g, 4, 3, BLACK); put(g, 5, 3, WHITE)
put(g, 3, 4, BLACK); put(g, 4, 4, WHITE); put(g, 6, 4, WHITE)
put(g, 4, 5, BLACK); put(g, 5, 5, WHITE)
g.to_move = BLACK
check(g.play(P(9, 5, 4)), "劫争提子合法")
check(g.ko == P(9, 4, 4), "劫点正确")
check(not g.is_legal(P(9, 4, 4)), "立即回提被禁止")
check(g.play(P(9, 0, 0)), "他处落子合法")
check(g.ko == -1 and g.is_legal(P(9, 4, 4)), "他处落子后劫解除")

# 5) 数子
g = Game(9)
put(g, 4, 4, BLACK)
check(abs(g.score(0.0) - 81.0) < 1e-9, f"单黑子=81 目(得到 {g.score(0.0)})")
check(abs(g.score(7.0) - 74.0) < 1e-9, "贴目被扣除")
g = Game(9)
put(g, 4, 4, BLACK); put(g, 0, 0, WHITE)
check(abs(g.score(0.0)) < 1e-9, "双方共有的空区是单官")

# 6) 终局判定
g = Game(9)
g.play(PASS); g.play(PASS)
check(g.is_terminal(243), "连续两次停着终局")

# 7) 随机对局：合法性、终止、分数一致
rng = random.Random(20241001)
bad = 0
for game_i in range(30):
    g = Game(9)
    steps = 0
    while not g.is_terminal(243) and steps < 400:
        moves = g.legal_moves()
        if not moves:
            break
        m = rng.choice(moves)
        before = g.cells.copy()
        if not g.play(m):
            bad += 1
            break
        if m != PASS and np.array_equal(before, g.cells):
            pass
        steps += 1
    # 终局后棋盘上的块都必须有气（合法局面）
    for color in (BLACK, WHITE):
        ys, xs = np.where(g.cells == color)
        for y, x in zip(ys, xs):
            _, libs = g._group(y * 9 + x)
            if not libs:
                bad += 1
    s = g.score(7.0)
    if not (isinstance(s, float) and -400 < s < 400):
        bad += 1
check(bad == 0, f"30 局随机对局未发现非法状态（bad={bad}）")

# 8) 特征平面与 C 版一致
g = Game(9)
g.play(P(9, 4, 4))
x = g.features()
check(x.shape == (4, 9, 9), "特征形状 4x9x9")
check(x[0].sum() + x[1].sum() == 1.0, "自己/对手平面各有一子")
check(x[3].sum() == 81.0, "全 1 平面")
check(x[1, 4, 4] == 1.0, "轮白时黑子在“对手”平面")   # 特征始终以当前行棋方视角

print(f"\n{checks} 项检查，{failures} 项失败")
print("规则自检通过 ✅" if failures == 0 else "规则自检失败 ❌")
sys.exit(1 if failures else 0)
