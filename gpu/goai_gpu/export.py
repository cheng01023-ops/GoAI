"""goai_gpu.export - C 引擎权重格式(.bin) 与 PyTorch 互转。

C 版 net.c 的文件格式（小端）：
    uint32 magic = 0x474F4149
    int32  size, planes, channels, vhidden, n_params
    float32 params[n_params]   顺序见 net.c 的 layout()
"""
from __future__ import annotations

import struct

import numpy as np
import torch

from .model import Net

MAGIC = 0x474F4149


def _layout(size: int, planes: int, channels: int, vhidden: int):
    nn_ = size * size
    return [
        (channels, planes, 3, 3), (channels,),
        (channels, channels, 3, 3), (channels,),
        (2, channels, 1, 1), (2,),
        (nn_ + 1, 2 * nn_), (nn_ + 1,),
        (1, channels, 1, 1), (1,),
        (vhidden, nn_), (vhidden,),
        (1, vhidden), (1,),
    ]


def export_c_bin(model: Net, path: str) -> int:
    """把 PyTorch 网络写成 C 引擎可读的 .bin，返回参数个数"""
    shapes = _layout(model.size, model.planes, model.channels, model.vhidden)
    tensors = model.c_param_tensors()
    assert len(tensors) == len(shapes)
    parts, total = [], 0
    for t, shp in zip(tensors, shapes):
        flat = t.detach().to("cpu", torch.float32).reshape(-1).numpy().astype("<f4")
        assert flat.size == int(np.prod(shp)), (tuple(t.shape), shp)
        parts.append(flat.tobytes())
        total += flat.size
    with open(path, "wb") as f:
        f.write(struct.pack("<Iiiiii", MAGIC, model.size, model.planes,
                            model.channels, model.vhidden, total))
        for p in parts:
            f.write(p)
    return total


def import_c_bin(model: Net, path: str) -> Net:
    """读取 C 引擎的 .bin 到 PyTorch 网络（可用于续训）"""
    with open(path, "rb") as f:
        head = f.read(24)
        magic, size, planes, channels, vhidden, n_params = struct.unpack("<Iiiiii", head)
        if magic != MAGIC:
            raise ValueError("not a GoAI weights file: magic=0x%08X" % magic)
        raw = np.frombuffer(f.read(n_params * 4), dtype="<f4")
    if (size, planes, channels, vhidden) != (model.size, model.planes, model.channels, model.vhidden):
        raise ValueError("shape mismatch: file %dx%d/%dch vs model %dx%d/%dch" %
                         (size, size, channels, model.size, model.size, model.channels))
    shapes = _layout(size, planes, channels, vhidden)
    off = 0
    with torch.no_grad():
        for t, shp in zip(model.c_param_tensors(), shapes):
            n = int(np.prod(shp))
            t.copy_(torch.from_numpy(raw[off:off + n].copy()).reshape(t.shape))
            off += n
    return model
