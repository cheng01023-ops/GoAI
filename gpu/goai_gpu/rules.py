"""goai_gpu.rules —— 与 C 引擎 board.c 规则一致的围棋实现（纯 Python，供批量自对弈使用）。

设计要点：
  * 规则与 C 版逐条对应：禁入点（自杀）、整块提子、简单劫、连续停着终局、Tromp-Taylor 数子
  * 每个 Game 对象只维护自己的棋盘；自对弈时同时在跑几百盘，互不干扰
  * 足够快：9 路一次合法性判断约几十微秒，瓶颈在网络推理（那才是 GPU 的活）
"""
from __future__ import annotations

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
PASS = -1

# 邻居表缓存：n -> [(n0,n1,...), ...]，避免反复构造生成器
_NBR = {}


def neighbors_of(n: int):
    t = _NBR.get(n)
    if t is None:
        t = []
        for p in range(n * n):
            y, x = divmod(p, n)
            lst = []
            if y > 0: lst.append(p - n)
            if y < n - 1: lst.append(p + n)
            if x > 0: lst.append(p - 1)
            if x < n - 1: lst.append(p + 1)
            t.append(tuple(lst))
        _NBR[n] = t
    return t


class Game:
    """一盘棋。move 用 0..N*N-1 表示落子，PASS 表示停着。"""

    __slots__ = ("n", "cells", "to_move", "ko", "passes", "nmoves", "_legal_cache",
                 "_cells_flat", "_flat_ver", "_ver")

    def __init__(self, n: int = 9):
        self.n = n
        self.cells = np.zeros((n, n), dtype=np.int8)
        self.to_move = BLACK
        self.ko = -1
        self.passes = 0
        self.nmoves = 0
        self._legal_cache = None
        self._cells_flat = None
        self._flat_ver = -1
        self._ver = 0

    # ---------- 基本工具 ----------
    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.n = self.n
        g.cells = self.cells.copy()
        g.to_move = self.to_move
        g.ko = self.ko
        g.passes = self.passes
        g.nmoves = self.nmoves
        g._legal_cache = None
        g._cells_flat = None
        g._flat_ver = -1
        g._ver = self._ver
        return g

    def neighbors(self, p: int):
        return neighbors_of(self.n)[p]

    def _group(self, p: int):
        """返回 (同色整块的点的集合, 气的集合)"""
        flat = self._flat()
        nbrs = neighbors_of(self.n)
        color = flat[p]
        stack = [p]
        seen = {p}
        libs = set()
        while stack:
            q = stack.pop()
            for r in nbrs[q]:
                v = flat[r]
                if v == EMPTY:
                    libs.add(r)
                elif v == color and r not in seen:
                    seen.add(r)
                    stack.append(r)
        return seen, libs

    def _flat(self):
        f = getattr(self, "_cells_flat", None)
        if f is None or self._flat_ver != self._ver:
            f = self.cells.reshape(-1)
            self._cells_flat = f
            self._flat_ver = self._ver
        return f

    # ---------- 规则 ----------
    def is_legal(self, move: int) -> bool:
        if move == PASS:
            return True
        n = self.n
        p = int(move)
        if p < 0 or p >= n * n:
            return False
        flat = self._flat()
        if flat[p] != EMPTY or p == self.ko:
            return False
        color = self.to_move
        opp = BLACK + WHITE - color
        nbrs = neighbors_of(n)[p]
        for r in nbrs:
            if flat[r] == EMPTY:
                return True                      # 有空邻居 -> 一定有气
        for r in nbrs:
            if flat[r] == color:
                stones, libs = self._group(r)
                for l in libs:
                    if l != p:
                        return True
        for r in nbrs:
            if flat[r] == opp:
                _, libs = self._group(r)
                if len(libs) == 1:
                    return True                  # 提掉对方
        return False

    def play_known_legal(self, move: int) -> None:
        """已知合法时直接落子（跳过合法性检查，搜索里省时间）"""
        n = self.n
        if move == PASS:
            self.passes += 1
            self.ko = -1
            self.to_move = BLACK + WHITE - self.to_move
            self.nmoves += 1
            self._ver += 1
            return
        p = int(move)
        flat = self._flat()
        color = self.to_move
        opp = BLACK + WHITE - color
        flat[p] = color
        captured = 0
        single = False
        for r in neighbors_of(n)[p]:
            if flat[r] == opp:
                stones, libs = self._group(r)
                if not libs:
                    if len(stones) == 1:
                        single = True
                    captured += len(stones)
                    for s2 in stones:
                        flat[s2] = EMPTY
        _, my_libs = self._group(p)
        self.ko = -1
        if captured == 1 and single and len(my_libs) == 1:
            self.ko = next(iter(my_libs))
        self.passes = 0
        self.to_move = opp
        self.nmoves += 1
        self._ver += 1

    def legal_moves(self, allow_pass: bool = True) -> list:
        n = self.n
        out = [p for p in range(n * n) if self.is_legal(p)]
        if allow_pass:
            out.append(PASS)
        return out

    def play(self, move: int) -> bool:
        """落子；非法返回 False（棋盘不变）"""
        if not self.is_legal(move):
            return False
        n = self.n
        if move == PASS:
            self.passes += 1
            self.ko = -1
            self.to_move = BLACK + WHITE - self.to_move
            self.nmoves += 1
            self._legal_cache = None
            self._ver += 1
            return True

        p = int(move)
        color = self.to_move
        opp = BLACK + WHITE - color
        self.cells[p // n, p % n] = color
        captured = 0
        captured_single = False
        for r in list(self.neighbors(p)):
            if self.cells[r // n, r % n] == opp:
                stones, libs = self._group(r)
                if not libs:
                    if len(stones) == 1:
                        captured_single = True
                    captured += len(stones)
                    for s in stones:
                        self.cells[s // n, s % n] = EMPTY
        # 自杀（提子后仍无气）
        _, my_libs = self._group(p)
        if not my_libs:                      # 理论上不会发生（已验证过合法性）
            self.cells[p // n, p % n] = EMPTY
            return False
        self.ko = -1
        if captured == 1 and captured_single and len(my_libs) == 1:
            self.ko = next(iter(my_libs))     # 简单劫
        self.passes = 0
        self.to_move = opp
        self.nmoves += 1
        self._legal_cache = None
        self._ver += 1
        return True

    # ---------- 终局与数子 ----------
    def is_terminal(self, max_moves: int) -> bool:
        return self.passes >= 2 or self.nmoves >= max_moves

    def score(self, komi: float) -> float:
        """Tromp-Taylor 数子，返回黑 - 白 - 贴目"""
        n = self.n
        black = int(np.count_nonzero(self.cells == BLACK))
        white = int(np.count_nonzero(self.cells == WHITE))
        seen = np.zeros((n, n), dtype=bool)
        for sy in range(n):
            for sx in range(n):
                if self.cells[sy, sx] != EMPTY or seen[sy, sx]:
                    continue
                stack = [(sy, sx)]
                seen[sy, sx] = True
                region = 0
                touch_b = touch_w = False
                while stack:
                    y, x = stack.pop()
                    region += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < n and 0 <= xx < n:
                            v = self.cells[yy, xx]
                            if v == EMPTY:
                                if not seen[yy, xx]:
                                    seen[yy, xx] = True
                                    stack.append((yy, xx))
                            elif v == BLACK:
                                touch_b = True
                            else:
                                touch_w = True
                if touch_b and not touch_w:
                    black += region
                elif touch_w and not touch_b:
                    white += region
        return float(black - white) - komi

    def winner(self, komi: float) -> int:
        s = self.score(komi)
        return BLACK if s > 0 else (WHITE if s < 0 else 0)

    def features(self) -> np.ndarray:
        """输入特征：4 个平面 [自己/对手/劫点/全1]，与 C 版 net_features 完全一致"""
        n = self.n
        me, opp = self.to_move, BLACK + WHITE - self.to_move
        x = np.zeros((4, n, n), dtype=np.float32)
        x[0] = (self.cells == me)
        x[1] = (self.cells == opp)
        if self.ko >= 0:
            x[2].flat[self.ko] = 1.0
        x[3] = 1.0
        return x

    def to_sgf_moves(self) -> list:
        return []
