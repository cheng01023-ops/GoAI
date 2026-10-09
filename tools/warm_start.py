#!/usr/bin/env python3
"""warm_start.py - 把一个小棋盘上练好的网络热启动成另一个规格（换棋盘 / 换通道数）

原理：卷积层学到的"棋形知识"（气、连接、断点、边角手段）与棋盘大小无关，
      可以照搬到 13 路 / 19 路；只有两个全连接头（策略头、价值头）依赖棋盘点数，
      必须重新初始化。

通道数不同时用 net2net 式切片迁移：重叠的那部分权重照搬，多出来的通道随机初始化。
这样即使从 32 通道扩到 64 通道，也已经带着"会看棋形"的眼睛，而不是从零开始。

用法：
    # 9 路 32 通道 -> 13 路 64 通道（推荐）
    python3 tools/warm_start.py --from versions/goai9x9_v3_33kgames.bin \
        --size 13 --channels 64 --vhidden 64 --out warm13.bin

    # 自检
    python3 tools/warm_start.py --selftest
"""
import argparse
import math
import os
import random
import struct
import sys

MAGIC = 0x474F4149
HDR = "<Iiiiii"


def layout(size, planes, ch, vh):
    """必须与 C 端 net.c 的 layout() 完全一致（顺序不可改）"""
    nn = size * size
    return [
        ("c1w", ch * planes * 9), ("c1b", ch),
        ("c2w", ch * ch * 9), ("c2b", ch),
        ("pw", 2 * ch), ("pb", 2),
        ("pfcw", (nn + 1) * 2 * nn), ("pfcb", nn + 1),
        ("vw", ch), ("vb", 1),
        ("vfc1w", vh * nn), ("vfc1b", vh),
        ("vfc2w", vh), ("vfc2b", 1),
    ]


def shape_of(name, planes, ch):
    """返回卷积类张量的形状（用于切片迁移）；返回 None 表示与棋盘相关、必须重学"""
    return {
        "c1w": (ch, planes, 3, 3), "c1b": (ch,),
        "c2w": (ch, ch, 3, 3),     "c2b": (ch,),
        "pw": (2, ch),             "pb": (2,),
        "vw": (ch,),               "vb": (1,),
    }.get(name)


def read_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        raise ValueError("文件太小: " + path)
    magic, size, planes, ch, vh, npar = struct.unpack(HDR, data[:24])
    if magic != MAGIC:
        raise ValueError("不是 GoAI 权重文件（magic 不对）: " + path)
    if len(data) < 24 + 4 * npar:
        raise ValueError("文件被截断: %s" % path)
    vals = list(struct.unpack("<%df" % npar, data[24:24 + 4 * npar]))
    return dict(size=size, planes=planes, channels=ch, vhidden=vh, params=vals,
                layout=layout(size, planes, ch, vh))


def segment(net, name):
    off = 0
    for n, ln in net["layout"]:
        if n == name:
            return net["params"][off:off + ln]
        off += ln
    raise KeyError(name)


def fresh_sigma(name, size, vh):
    nn = size * size
    if name == "pfcw":
        return math.sqrt(2.0 / (2.0 * nn)) * 0.3
    if name == "vfc1w":
        return math.sqrt(2.0 / float(nn))
    if name == "vfc2w":
        return 1.0 / math.sqrt(float(vh))
    return None            # None = 偏置，初始化为 0


def decode(idx, shape):
    """把扁平下标还原成多维下标（行优先）"""
    out = []
    for d in reversed(shape):
        out.append(idx % d)
        idx //= d
    return tuple(reversed(out))


def warm_start(src, size, planes, ch, vh, seed=20241001):
    rng = random.Random(seed)
    out, copied, partial, fresh = [], [], [], []
    src_cache = {}
    src_names = {n for n, _ in src["layout"]}      # layout 是 [(名字, 长度)] 列表
    for name, cnt in layout(size, planes, ch, vh):
        tshape = shape_of(name, planes, ch)
        sshape = shape_of(name, src["planes"], src["channels"]) if name in src_names else None
        if tshape is not None and sshape is not None:
            if name not in src_cache:
                src_cache[name] = segment(src, name)
            sseg = src_cache[name]
            # 可重叠的维度范围 = 两个形状逐维取小
            ov = tuple(min(a, b) for a, b in zip(tshape, sshape))
            n_copy = 1
            for d in ov:
                n_copy *= d
            if n_copy == cnt:
                out.extend(sseg[:cnt]); copied.append(name)
            else:
                for i in range(cnt):
                    ix = decode(i, tshape)
                    if all(ix[d] < ov[d] for d in range(len(tshape))):
                        j = 0
                        for d, v in enumerate(ix[:len(sshape)]):
                            j = j * sshape[d] + v
                        out.append(sseg[j])
                    else:
                        out.append(rng.gauss(0.0, 0.02))
                partial.append("%s(%d/%d)" % (name, n_copy, cnt))
            continue
        sigma = fresh_sigma(name, size, vh)
        if sigma is None:
            out.extend([0.0] * cnt)
        else:
            out.extend([rng.gauss(0.0, sigma) for _ in range(cnt)])
        fresh.append(name)
    return out, copied, partial, fresh


def write_bin(path, size, planes, ch, vh, params):
    with open(path, "wb") as f:
        f.write(struct.pack(HDR, MAGIC, size, planes, ch, vh, len(params)))
        f.write(struct.pack("<%df" % len(params), *params))
    return len(params)


def main():
    ap = argparse.ArgumentParser(description="GoAI 权重热启动：迁移卷积层到新规格")
    ap.add_argument("--from", dest="src", help="源权重 .bin")
    ap.add_argument("--size", type=int, default=13, help="目标棋盘（默认 13）")
    ap.add_argument("--channels", type=int, default=64, help="目标通道数（默认 64）")
    ap.add_argument("--vhidden", type=int, default=64, help="价值头隐藏层（默认 64）")
    ap.add_argument("--out", default="warm.bin", help="输出文件")
    ap.add_argument("--seed", type=int, default=20241001)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        here = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(here, "..", "versions", "goai9x9_v3_33kgames.bin")
        if not os.path.exists(src_path):
            print("跳过自检（找不到 %s）" % src_path)
            return 0
        s = read_bin(src_path)
        print("源: %d路 %d通道 %d参数" % (s["size"], s["channels"], len(s["params"])))

        p1, c1, pa1, f1 = warm_start(s, s["size"], s["planes"], s["channels"], s["vhidden"])
        assert not pa1, "同规格迁移不应出现部分迁移"
        assert set(f1) == {"pfcw", "pfcb", "vfc1w", "vfc1b", "vfc2w", "vfc2b"}, f1
        off = 0
        for nm, ln in s["layout"]:
            if nm == "c1w":
                assert p1[off:off + ln] == segment(s, "c1w"), "c1w 未逐值复制"
            off += ln
        print("同规格: 复制 %d 段（c1w 逐值一致），重学 %d 段" % (len(c1), len(f1)))

        p2, c2, pa2, f2 = warm_start(s, 13, 4, 64, 64)
        assert len(p2) == sum(l for _, l in layout(13, 4, 64, 64))
        print("13路64通道: 完整复制 %d 段 %s" % (len(c2), c2))
        print("            切片迁移 %d 段 %s" % (len(pa2), pa2))
        print("            重学 %d 段 %s" % (len(f2), f2))
        assert any(x.startswith("c1w(") for x in pa2), "跨通道时 c1w 必须走切片迁移"
        assert any(x.startswith("c2w(") for x in pa2), "跨通道时 c2w 必须走切片迁移"
        # 校验切片内容确实来自源网络
        sseg = segment(s, "c1w")
        t2off = 0
        for nm, ln in layout(13, 4, 64, 64):
            if nm == "c1w":
                # 目标 c1w: [64][4][3][3]，源: [32][4][3][3] -> 前 32 个输出通道应逐值相同
                assert p2[t2off:t2off + len(sseg)] == sseg, "切片迁移内容不对"
            t2off += ln
        print("            切片内容逐值校验 OK")
        print("自检通过")
        return 0

    if not a.src or not os.path.exists(a.src):
        print("找不到源权重: %s" % a.src, file=sys.stderr)
        return 1
    s = read_bin(a.src)
    print("源网络 : %d 路 / %d 通道 / %d 参数" % (s["size"], s["channels"], len(s["params"])))
    params, copied, partial, fresh = warm_start(s, a.size, s["planes"], a.channels, a.vhidden)
    n = write_bin(a.out, a.size, s["planes"], a.channels, a.vhidden, params)
    print("目标   : %d 路 / %d 通道 / %d 参数 -> %s" % (a.size, a.channels, n, a.out))
    print("完整迁移: %s" % (", ".join(copied) if copied else "无"))
    print("切片迁移: %s" % (", ".join(partial) if partial else "无"))
    print("重新学习: %s" % ", ".join(fresh))
    return 0


if __name__ == "__main__":
    sys.exit(main())
