"""goai_gpu.mcts - 批量 PUCT 搜索（为 GPU 批量推理优化）。

关键优化：
  1) 惰性建子节点：展开时只记录合法着法与先验，真被走到的子节点才克隆棋盘
     （否则一次展开要克隆 80 个棋盘，实测占总耗时 92%）
  2) 用 numpy 一次算子节点选择，避开 Python 循环
  3) 同时跑 B 盘棋，每轮把所有叶子拼成一个 batch 送 GPU，喂满显卡
"""
from __future__ import annotations

import math
import random

import numpy as np

from .rules import Game, PASS, BLACK, WHITE


class Node:
    __slots__ = ("N", "W", "board", "children", "moves", "priors",
                 "edge_N", "edge_W", "parent", "expanded", "terminal", "tval")

    def __init__(self, board, prior=1.0, parent=-1):
        self.N = 0
        self.W = 0.0
        self.board = board
        self.children = []      # 子节点 id，-1 表示尚未创建
        self.moves = []
        self.priors = []
        self.edge_N = []
        self.edge_W = []
        self.parent = parent
        self.expanded = False
        self.terminal = False
        self.tval = 0.0


class Search:
    """一盘棋的搜索树（落子后把对应子树提为新根，实现树复用）"""

    def __init__(self, size: int, komi: float, c_puct: float = 1.5,
                 pass_min_move: int = 0, noise_alpha: float = 0.0, noise_eps: float = 0.0,
                 rng=None):
        self.size = size
        self.komi = komi
        self.c_puct = c_puct
        self.pass_min_move = pass_min_move   # 这手之前不允许停着（防止乱停把棋局草草结束）
        self.noise_alpha = noise_alpha       # 根节点 Dirichlet 噪声（自对弈探索用）
        self.noise_eps = noise_eps
        self.rng = rng
        self.nodes = []
        self.root = self.new_node(Game(size))

    def new_node(self, board, prior=1.0, parent=-1) -> int:
        self.nodes.append(Node(board, prior, parent))
        return len(self.nodes) - 1

    # ------------------------------------------------ 选择
    def descend(self):
        """走到一个未展开（或终局）的节点，返回 (node_id, path)"""
        node = self.root
        path = []
        while True:
            n = self.nodes[node]
            if n.terminal or not n.expanded or not n.moves:
                return node, path
            en = np.asarray(n.edge_N, dtype=np.float64)
            ew = np.asarray(n.edge_W, dtype=np.float64)
            q = np.where(en > 0, -(ew / np.maximum(en, 1.0)), 0.0)
            sqrt_n = math.sqrt(n.N) if n.N > 0 else 1.0
            u = self.c_puct * np.asarray(n.priors, dtype=np.float64) * sqrt_n / (1.0 + en)
            best = int(np.argmax(q + u))
            child = n.children[best]
            if child < 0:                       # 惰性创建
                board = n.board.clone()
                board.play_known_legal(n.moves[best])   # 停着也要推进（否则棋局永远不会终局）
                child = self.new_node(board, n.priors[best], node)
                n.children[best] = child
            path.append((node, best, child))
            node = child

    def expand(self, node: int, policy: np.ndarray, value: float) -> float:
        n = self.nodes[node]
        n.N += 1
        n.W += value
        n.expanded = True
        moves = n.board.legal_moves()
        if n.board.nmoves < self.pass_min_move:
            no_pass = [m for m in moves if m != PASS]
            if no_pass:
                moves = no_pass
        if not moves:
            n.terminal = True
            n.tval = 0.0
            return 0.0
        nn_ = self.size * self.size
        pri = np.array([policy[m if m != PASS else nn_] for m in moves], dtype=np.float64)
        s = pri.sum()
        pri = pri / s if s > 0 else np.full(len(moves), 1.0 / len(moves))
        if node == self.root and self.noise_eps > 0.0 and self.noise_alpha > 0.0 and self.rng is not None:
            noise = np.random.default_rng(self.rng.randrange(1 << 30)).dirichlet(
                [self.noise_alpha] * len(pri))
            pri = (1.0 - self.noise_eps) * pri + self.noise_eps * noise
        pri = pri / pri.sum()
        n.moves = list(moves)
        n.priors = pri.tolist()
        n.children = [-1] * len(moves)
        n.edge_N = [0] * len(moves)
        n.edge_W = [0.0] * len(moves)
        return value

    def backup(self, path, value: float) -> None:
        v = value
        for (parent, edge, _child) in reversed(path):
            pn = self.nodes[parent]
            pn.edge_N[edge] += 1
            pn.edge_W[edge] += v
            pn.N += 1
            pn.W += -v
            v = -v

    # ------------------------------------------------ 选点
    def root_policy(self) -> np.ndarray:
        n = self.nodes[self.root]
        nn_ = self.size * self.size
        pol = np.zeros(nn_ + 1, dtype=np.float32)
        for m, cN in zip(n.moves, n.edge_N):
            pol[m if m != PASS else nn_] = cN
        t = pol.sum()
        if t > 0:
            pol /= t
        return pol

    def choose(self, temperature: float, rng: random.Random) -> int:
        n = self.nodes[self.root]
        if not n.moves:
            return PASS
        visits = np.asarray(n.edge_N, dtype=np.float64)
        if temperature <= 1e-3:
            return n.moves[int(np.argmax(visits))]
        w = visits ** (1.0 / temperature)
        s = w.sum()
        if s <= 0:
            return rng.choice(n.moves)
        r = rng.random() * s
        acc = 0.0
        for m, v in zip(n.moves, w):
            acc += v
            if r < acc:
                return m
        return n.moves[-1]

    def advance(self, move: int) -> None:
        """落子后把对应子树提为新根；没有子树就重建"""
        n = self.nodes[self.root]
        for i, m in enumerate(n.moves):
            if m == move:
                child = n.children[i]
                if child < 0:
                    board = n.board.clone()
                    board.play_known_legal(move)
                    child = self.new_node(board, n.priors[i], -1)
                self.nodes[child].parent = -1
                self.root = child
                return
        board = n.board.clone()
        board.play_known_legal(move)
        self.nodes = []
        self.root = self.new_node(board)
