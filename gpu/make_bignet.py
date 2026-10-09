"""模拟"朋友练完发给我"的场景：用 PyTorch 造一个大网络导出 .bin，验证 C 引擎能吃"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from goai_gpu.model import Net
from goai_gpu.export import export_c_bin, import_c_bin

os.makedirs("/tmp/his_net", exist_ok=True)
for (size, ch, vh, name) in [(19, 256, 128, "his_19x19_256ch.bin"),
                             (19, 128, 128, "his_19x19_128ch.bin"),
                             (9, 128, 64,  "his_9x9_128ch.bin")]:
    torch.manual_seed(42)
    m = Net(size=size, channels=ch, vhidden=vh)
    path = os.path.join("/tmp/his_net", name)
    n = export_c_bin(m, path)
    # 往返校验
    m2 = Net(size=size, channels=ch, vhidden=vh)
    import_c_bin(m2, path)
    same = all(torch.equal(a, b) for a, b in zip(m.c_param_tensors(), m2.c_param_tensors()))
    mb = os.path.getsize(path) / 1e6
    print("%-22s 参数 %8d  文件 %.1f MB  往返一致=%s" % (name, n, mb, same))
