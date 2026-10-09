"""goai_gpu.model - 与 C 引擎 net.c 结构逐层一致的策略/价值网络。

结构（必须与 C 版完全一致，否则权重无法互通）：
    输入 4xNxN  ->  conv3x3(4->C) ReLU  ->  conv3x3(C->C) ReLU
    策略头: conv1x1(C->2) -> flatten(2*N*N) -> Linear(-> N*N+1)
    价值头: conv1x1(C->1) -> flatten(N*N)   -> Linear(-> H) ReLU -> Linear(-> 1) -> tanh
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, size: int = 9, planes: int = 4, channels: int = 32, vhidden: int = 64):
        super().__init__()
        self.size = size
        self.planes = planes
        self.channels = channels
        self.vhidden = vhidden
        nn_ = size * size
        self.c1 = nn.Conv2d(planes, channels, 3, padding=1)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pw = nn.Conv2d(channels, 2, 1)
        self.pfc = nn.Linear(2 * nn_, nn_ + 1)
        self.vw = nn.Conv2d(channels, 1, 1)
        self.vfc1 = nn.Linear(nn_, vhidden)
        self.vfc2 = nn.Linear(vhidden, 1)

    def forward(self, x: torch.Tensor):
        """x: (B, planes, N, N) -> (logits (B, N*N+1), value (B,))"""
        h = F.relu(self.c1(x))
        h = F.relu(self.c2(h))
        logits = self.pfc(self.pw(h).flatten(1))
        v = torch.tanh(self.vfc2(F.relu(self.vfc1(self.vw(h).flatten(1)))).squeeze(1))
        return logits, v

    @torch.no_grad()
    def policy_value(self, x: torch.Tensor):
        logits, v = self.forward(x)
        return F.softmax(logits, dim=1), v

    def c_param_tensors(self):
        """顺序与 C 版 net.c 的 layout() 完全一致"""
        return [
            self.c1.weight, self.c1.bias,
            self.c2.weight, self.c2.bias,
            self.pw.weight, self.pw.bias,
            self.pfc.weight, self.pfc.bias,
            self.vw.weight, self.vw.bias,
            self.vfc1.weight, self.vfc1.bias,
            self.vfc2.weight, self.vfc2.bias,
        ]

    def param_count(self) -> int:
        return sum(int(t.numel()) for t in self.c_param_tensors())
