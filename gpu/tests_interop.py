"""权重互通自检：PyTorch -> C .bin -> PyTorch 必须逐位一致，并让 C 引擎真正加载它"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from goai_gpu.model import Net
from goai_gpu.export import export_c_bin, import_c_bin
from goai_gpu.rules import Game

torch.manual_seed(7)
ok = True
for (size, ch, vh) in [(9, 8, 16), (9, 32, 64), (13, 16, 32), (19, 8, 16)]:
    m = Net(size=size, channels=ch, vhidden=vh)
    path = os.path.join(tempfile.gettempdir(), "goai_rt_%d_%d.bin" % (size, ch))
    n = export_c_bin(m, path)
    m2 = Net(size=size, channels=ch, vhidden=vh)
    import_c_bin(m2, path)
    same = all(torch.equal(a, b) for a, b in zip(m.c_param_tensors(), m2.c_param_tensors()))
    print("  %2dx%-2d ch=%-3d  参数 %6d  往返一致=%s" % (size, size, ch, n, same))
    ok = ok and same

# 给 C 引擎准备一个真实文件：9 路 32 通道，取值确定（便于人工核对）
torch.manual_seed(1234)
m = Net(size=9, channels=32, vhidden=64)
p = "/tmp/goai_cross_9x9_32ch.bin"
export_c_bin(m, p)
g = Game(9)
x = torch.from_numpy(g.features()).unsqueeze(0)
pol, val = m.policy_value(x)
print("  交叉验证文件: %s" % p)
print("  PyTorch 空盘价值 = %+.6f, 首选点 = %d" % (val.item(), int(pol.argmax())))
print("权重互通自检", "通过" if ok else "失败")
sys.exit(0 if ok else 1)
