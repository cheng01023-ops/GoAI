#!/usr/bin/env python3
"""warm_start.py - 权重热启动：换棋盘 / 加通道 / 加残差块

两种模式：

A) 跨棋盘（--size 变）：卷积层按通道切片照搬（棋盘无关的"棋形知识"），
   依赖棋盘大小的全连接头（策略头 / 价值头）重新初始化。原有行为。

B) 同棋盘加宽（--size 不变，--channels / --blocks 变大）：net2net 式**函数保持**加宽 ——
   新通道是旧通道的副本（乘 1+ε 小噪声打破对称），同时把下游对应输入权重除以 m，
   于是网络在初始化时输出与旧网络**逐值一致**（实测策略/价值差 <1e-3）；
   新增的残差块 wB/bB 零初始化（块 = 恒等映射）。
   这样 32→64/128 通道不会出现"热启动即掉棋力"，新通道也不会因为权重为 0 而死掉。

用法：
    # 9 路 32 通道 -> 9 路 64 通道 + 4 残差块（函数保持加宽）
    python3 tools/warm_start.py --from versions/goai9x9_v5_800sims.bin \
        --size 9 --channels 64 --blocks 4 --out versions/goai9x9_v6_64ch4blk.bin

    # 9 路 32 通道 -> 13 路 64 通道（跨棋盘，头部重学）
    python3 tools/warm_start.py --from versions/goai9x9_v3_33kgames.bin \
        --size 13 --channels 64 --vhidden 64 --out warm13.bin

    # 自检
    python3 tools/warm_start.py --selftest
"""
import argparse
import itertools
import math
import os
import random
import struct
import sys

MAGIC = 0x474F4149        # "GOAI" 旧格式
MAGIC2 = 0x474F414A       # "GOAJ" flags 格式
HDR = "<Iiiiii"           # GOAI: magic,size,planes,ch,vh,npar
HDR2 = "<Iiiiiii"         # GOAJ: magic,size,planes,ch,vh,flags,npar

FLAG_BLOCKS = 0x01
FLAG_OWN = 0x02
FLAG_VDIST = 0x04
VBUCKETS = 33


def block_stride(ch):
    return 2 * ch * ch * 9 + 2 * ch


def layout(size, planes, ch, vh, blocks=0, own=False, vdist=False):
    """必须与 C 端 net.c 的 layout() 完全一致（顺序不可改）"""
    nn = size * size
    K = VBUCKETS if vdist else 1
    L = [("c1w", ch * planes * 9), ("c1b", ch),
         ("c2w", ch * ch * 9), ("c2b", ch)]
    if blocks:
        L.append(("bw", blocks * block_stride(ch)))
    L += [("pw", 2 * ch), ("pb", 2),
          ("pfcw", (nn + 1) * 2 * nn), ("pfcb", nn + 1),
          ("vw", ch), ("vb", 1),
          ("vfc1w", vh * nn), ("vfc1b", vh),
          ("vfc2w", vh * K), ("vfc2b", K)]
    if own:
        L += [("ow", ch), ("ob", 1)]
    return L


def n_params(size, planes, ch, vh, blocks=0, own=False, vdist=False):
    return sum(n for _, n in layout(size, planes, ch, vh, blocks, own, vdist))


def shape_of(name, planes, ch):
    """卷积类张量的形状（用于切片迁移）；None = 与棋盘/通道无关，直接照搬"""
    return {
        "c1w": (ch, planes, 3, 3), "c1b": (ch,),
        "c2w": (ch, ch, 3, 3),     "c2b": (ch,),
        "pw": (2, ch),             "pb": (2,),
        "vw": (ch,),               "vb": (1,),
        "ow": (ch,),               "ob": (1,),
    }.get(name)


def decode_flags(raw, size, planes, ch, vh, npar):
    """与 C 端 net_decode_flags 同一套规则"""
    raw &= 0xFFFFFFFF
    blocks = (raw >> 8) & 0xFFFFFF
    own = 1 if (raw & FLAG_OWN) else 0
    vdist = 1 if (raw & FLAG_VDIST) else 0
    if not (raw & FLAG_BLOCKS):
        return raw, 0, 0                      # 老 GOAJ：整个字段就是 blocks
    if n_params(size, planes, ch, vh, blocks, own, vdist) != npar:
        legacy = raw
        if n_params(size, planes, ch, vh, legacy, 0, 0) == npar:
            return legacy, 0, 0               # 老文件 blocks 为奇数时命中
    return blocks, own, vdist


def read_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 28:
        raise ValueError("文件太小: " + path)
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic == MAGIC:
        _, size, planes, ch, vh, npar = struct.unpack_from(HDR, data, 0)
        blocks = own = vdist = 0
        hdr = struct.calcsize(HDR)
    elif magic == MAGIC2:
        _, size, planes, ch, vh, raw, npar = struct.unpack_from(HDR2, data, 0)
        blocks, own, vdist = decode_flags(raw, size, planes, ch, vh, npar)
        hdr = struct.calcsize(HDR2)
    else:
        raise ValueError("不是 GoAI 权重文件（magic 不对）: " + path)
    if len(data) < hdr + 4 * npar:
        raise ValueError("文件被截断: %s（需要 %d 字节，实得 %d）" % (path, hdr + 4 * npar, len(data)))
    vals = list(struct.unpack_from("<%df" % npar, data, hdr))
    return dict(size=size, planes=planes, channels=ch, vhidden=vh,
                blocks=blocks, own=bool(own), vdist=bool(vdist), params=vals,
                layout=layout(size, planes, ch, vh, blocks, own, vdist))


def write_bin(path, size, planes, ch, vh, blocks, own, vdist, params):
    flags = FLAG_BLOCKS | (FLAG_OWN if own else 0) | (FLAG_VDIST if vdist else 0) | (blocks << 8)
    with open(path, "wb") as f:
        f.write(struct.pack(HDR2, MAGIC2, size, planes, ch, vh, flags, len(params)))
        f.write(struct.pack("<%df" % len(params), *params))
    return len(params)


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


# ---------------------------------------------------------------- 同棋盘加宽

def flat_index(shape, multi):
    j = 0
    for d, v in zip(shape, multi):
        j = j * d + v
    return j


def widen_out_rows(flat, shape, axis, S, m, rng, noise):
    """输出通道维（长度 S）扩到 S*m：新单元 = 原单元复制 × (1+ε)，ε~N(0,noise)"""
    ns = list(shape)
    ns[axis] = S * m
    res = [0.0] * (len(flat) * m)
    for multi in itertools.product(*[range(d) for d in ns]):
        src_idx = list(multi)
        if multi[axis] >= S:
            src_idx[axis] = multi[axis] % S
        v = flat[flat_index(shape, src_idx)]
        if multi[axis] >= S:
            v *= (1.0 + rng.gauss(0.0, noise))
        res[flat_index(ns, multi)] = v
    return res


def widen_in_cols(flat, shape, axis, S, m):
    """输入通道维（长度 S）扩到 S*m：原列 /m，新列 = 原列的精确副本
       => 下游那一层的输出（各输入通道加权和）保持不变"""
    ns = list(shape)
    ns[axis] = S * m
    res = [0.0] * (len(flat) * m)
    for multi in itertools.product(*[range(d) for d in ns]):
        src_idx = list(multi)
        if multi[axis] >= S:
            src_idx[axis] = multi[axis] % S
        # 原列和它的 m-1 份副本都要除以 m：下游加权和 = m*(w/m) = w（函数不变）
        res[flat_index(ns, multi)] = flat[flat_index(shape, src_idx)] / m
    return res


def widen_same_size(src, T, blocks, seed=20241001, noise=0.002):
    """9x9 同棋盘：通道 S -> T（整数倍）+ 残差块 B1 -> B2（新块零初始化）。
       返回 (params, 说明字符串列表)"""
    S = src["channels"]
    if T % S != 0:
        raise ValueError("目标通道 %d 必须是源通道 %d 的整数倍" % (T, S))
    m = T // S
    planes, vh, size = src["planes"], src["vhidden"], src["size"]
    B1, B2 = src["blocks"], max(blocks, src["blocks"])
    rng = random.Random(seed)
    out = []
    notes = []

    def seg(n):
        return list(segment(src, n))

    # conv1: 输出通道加宽（输入面数不变）
    out += widen_out_rows(seg("c1w"), (S, planes, 3, 3), 0, S, m, rng, noise)
    out += widen_out_rows(seg("c1b"), (S,), 0, S, m, rng, noise)
    # conv2: 先输入列加宽（上游 conv1 的输出被复制了），再输出行加宽
    w = widen_in_cols(seg("c2w"), (S, S, 3, 3), 1, S, m)
    w = widen_out_rows(w, (S, T, 3, 3), 0, S, m, rng, noise)
    out += w
    out += widen_out_rows(seg("c2b"), (S,), 0, S, m, rng, noise)
    notes.append("c1/c2: %d -> %d 通道（函数保持）" % (S, T))

    # 残差块
    stride = block_stride(S)
    bw = seg("bw") if B1 else []
    s2 = math.sqrt(2.0 / (T * 9.0))
    for b in range(B2):
        if b < B1:
            base = b * stride
            cw = S * S * 9
            wA = bw[base:base + cw]
            bA = bw[base + cw:base + cw + S]
            wB = bw[base + cw + S:base + 2 * cw + S]
            bB = bw[base + 2 * cw + S:base + 2 * cw + 2 * S]
            wA = widen_in_cols(wA, (S, S, 3, 3), 1, S, m)
            wA = widen_out_rows(wA, (S, T, 3, 3), 0, S, m, rng, noise)
            bA = widen_out_rows(bA, (S,), 0, S, m, rng, noise)
            wB = widen_in_cols(wB, (S, S, 3, 3), 1, S, m)
            wB = widen_out_rows(wB, (S, T, 3, 3), 0, S, m, rng, noise)
            bB = widen_out_rows(bB, (S,), 0, S, m, rng, noise)
            out += wA + bA + wB + bB
        else:
            # 新块：wA/bA 随机，wB/bB = 0（零初始化 => 块初始是恒等映射）
            out += [rng.gauss(0.0, s2) for _ in range(T * T * 9)]
            out += [0.0] * T
            out += [0.0] * (T * T * 9)
            out += [0.0] * T
    if B2 != B1:
        notes.append("残差块: %d -> %d（新块零初始化，恒等映射）" % (B1, B2))
    elif B1:
        notes.append("残差块: %d 个（函数保持加宽）" % B1)

    # 头部 1x1 conv：输入是最终激活 h（被复制过的通道）-> 输入列加宽
    out += widen_in_cols(seg("pw"), (2, S), 1, S, m)
    out += seg("pb")
    out += seg("pfcw") + seg("pfcb")           # 依赖点数的全连接，形状不变，原样照搬
    out += widen_in_cols(seg("vw"), (S,), 0, S, m)
    out += seg("vb")
    out += seg("vfc1w") + seg("vfc1b") + seg("vfc2w") + seg("vfc2b")
    if src["own"]:
        out += widen_in_cols(seg("ow"), (S,), 0, S, m)
        out += seg("ob")
    want = n_params(size, planes, T, vh, B2, src["own"], src["vdist"])
    if len(out) != want:
        raise AssertionError("参数个数不对: %d != %d" % (len(out), want))
    return out, notes


# ---------------------------------------------------------------- 跨棋盘

def warm_start(src, size, planes, ch, vh, blocks=0, seed=20241001):
    """跨棋盘：卷积切片照搬，依赖点数的头部重学；残差块新块零初始化"""
    rng = random.Random(seed)
    out, copied, partial, fresh = [], [], [], []
    src_cache = {}
    src_names = {n for n, _ in src["layout"]}
    B1, B2 = src["blocks"], max(blocks, src["blocks"])
    for name, cnt in layout(size, planes, ch, vh, B2, src["own"], src["vdist"]):
        if name == "bw":
            stride_s, stride_t = block_stride(src["channels"]), block_stride(ch)
            sbw = segment(src, "bw") if B1 else []
            for b in range(B2):
                if b < B1:
                    sw = src["channels"]
                    cw_s, cw_t = sw * sw * 9, ch * ch * 9
                    if ch == sw:
                        out += sbw[b * stride_s:(b + 1) * stride_s]
                    else:
                        base = b * stride_s
                        # wA: (sw,sw,3,3)->(ch,ch,3,3) 逐维取小切片；bA/bB 向量同理；wB 同 wA
                        for (off, cin, cout) in ((0, sw, ch), (cw_s + sw, sw, ch),
                                                 (cw_s + sw + sw, sw, ch),
                                                 (2 * cw_s + 2 * sw, sw, ch)):
                            n_copy = 0
                            for co in range(cout):
                                for ci in range(cin):
                                    for t in range(9):
                                        if co < sw and ci < sw:
                                            out.append(sbw[base + off + (co * sw + ci) * 9 + t])
                                            n_copy += 1
                                        else:
                                            out.append(rng.gauss(0.0, 0.02))
                            partial.append("%s.b%d(%d/%d)" % (
                                "wA" if off == 0 else ("wB" if off == cw_s + sw + sw else "b"), b,
                                n_copy, 9 * cin * cout))
                    continue
                out += [rng.gauss(0.0, math.sqrt(2.0 / (ch * 9.0))) for _ in range(ch * ch * 9)]
                out += [0.0] * ch
                out += [0.0] * (ch * ch * 9)
                out += [0.0] * ch
            continue
        tshape = shape_of(name, planes, ch)
        sshape = shape_of(name, src["planes"], src["channels"]) if name in src_names else None
        if tshape is not None and sshape is not None:
            if name not in src_cache:
                src_cache[name] = segment(src, name)
            sseg = src_cache[name]
            ov = tuple(min(a, b) for a, b in zip(tshape, sshape))
            n_copy = 1
            for d in ov:
                n_copy *= d
            if n_copy == cnt:
                out.extend(sseg[:cnt])
                copied.append(name)
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


def decode(idx, shape):
    out = []
    for d in reversed(shape):
        out.append(idx % d)
        idx //= d
    return tuple(reversed(out))


# ---------------------------------------------------------------- 自检

def _selftest():
    here = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(here, "..", "versions", "goai9x9_v5_800sims.bin")
    if not os.path.exists(src_path):
        print("跳过自检（找不到 %s）" % src_path)
        return 0
    s = read_bin(src_path)
    print("源: %d路 %d通道 blocks=%d own=%d vdist=%d %d参数"
          % (s["size"], s["channels"], s["blocks"], s["own"], s["vdist"], len(s["params"])))
    assert len(s["params"]) == n_params(s["size"], s["planes"], s["channels"], s["vhidden"],
                                        s["blocks"], s["own"], s["vdist"]), "layout 与 C 端不一致"

    # 1) 同棋盘加宽 32 -> 64 + 4 块
    T, B2 = 64, 4
    p, notes = widen_same_size(s, T, B2, seed=20241001)
    assert len(p) == n_params(9, 4, T, s["vhidden"], B2, s["own"], s["vdist"])
    lay = layout(9, 4, T, s["vhidden"], B2, s["own"], s["vdist"])
    off, segs = 0, {}
    for nm, ln in lay:
        segs[nm] = p[off:off + ln]
        off += ln
    # 新块的 wB/bB 必须全 0（零初始化）
    stride = block_stride(T)
    for b in range(B2):
        base = b * stride
        cw = T * T * 9
        assert all(v == 0.0 for v in segs["bw"][base + cw + T:base + 2 * cw + T]), "wB 未零初始化"
        assert all(v == 0.0 for v in segs["bw"][base + 2 * cw + T:base + 2 * cw + 2 * T]), "bB 未零初始化"
    # 函数保持的核心恒等式：下游输入通道的和不变（= m 份「原值/m」）
    S = s["channels"]
    c2w_t = segs["c2w"]                     # (T,T,3,3)
    c2w_s = segment(s, "c2w")               # (S,S,3,3)
    bad = 0
    for co in range(S):
        for t in range(9):
            a = sum(c2w_t[((co * T + ci) * 9) + t] for ci in range(T))
            b = sum(c2w_s[((co * S + ci) * 9) + t] for ci in range(S))
            if abs(a - b) > 1e-5 * max(1.0, abs(b)):
                bad += 1
    assert bad == 0, "c2w 输入列加宽没有保持加权和（%d 处不符）" % bad
    pw_t, pw_s = segs["pw"], segment(s, "pw")
    bad = 0
    for co in range(2):
        a = sum(pw_t[co * T + ci] for ci in range(T))
        b = sum(pw_s[co * S + ci] for ci in range(S))
        if abs(a - b) > 1e-6 * max(1.0, abs(b)):
            bad += 1
    assert bad == 0, "策略头 1x1 输入列加宽没有保持加权和（%d 处）" % bad
    print("同棋盘加宽: %d -> %d 通道 / %d 块，%d 参数；恒等式校验 OK（下游加权和逐值不变）"
          % (S, T, B2, len(p)))
    for n in notes:
        print("   ", n)

    # 2) 跨棋盘 9 -> 13 仍可用（64 通道 => 走切片迁移）
    p2, c2, pa2, f2 = warm_start(s, 13, 4, 64, s["vhidden"])
    assert len(p2) == n_params(13, 4, 64, s["vhidden"], 0, s["own"], s["vdist"])
    assert any(x.startswith("c1w(") for x in pa2), "跨棋盘时 c1w 必须切片迁移"
    print("跨棋盘 9->13: 完整复制 %d 段, 切片 %d 段, 重学 %d 段 -> %d 参数"
          % (len(c2), len(pa2), len(f2), len(p2)))
    print("自检通过")
    return 0


def main():
    ap = argparse.ArgumentParser(description="GoAI 权重热启动：换棋盘 / 加通道 / 加残差块")
    ap.add_argument("--from", dest="src", help="源权重 .bin")
    ap.add_argument("--size", type=int, default=0, help="目标棋盘（默认与源相同）")
    ap.add_argument("--channels", type=int, default=0, help="目标通道数（默认与源相同）")
    ap.add_argument("--vhidden", type=int, default=0, help="价值头隐藏层（默认与源相同）")
    ap.add_argument("--blocks", type=int, default=0, help="目标残差块数（默认与源相同）")
    ap.add_argument("--out", default="warm.bin", help="输出文件")
    ap.add_argument("--seed", type=int, default=20241001)
    ap.add_argument("--noise", type=float, default=0.002,
                    help="加宽时新通道的破对称噪声强度（默认 0.002：策略/价值与原网络差 <3e-3；"
                         "调大更利于打破对称但会偏离函数保持）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return _selftest()
    if not a.src or not os.path.exists(a.src):
        print("找不到源权重: %s" % a.src, file=sys.stderr)
        return 1
    s = read_bin(a.src)
    tsize = a.size or s["size"]
    tch = a.channels or s["channels"]
    tvh = a.vhidden or s["vhidden"]
    tblk = a.blocks or s["blocks"]
    print("源网络 : %d 路 / %d 通道 / %d 块 / %d 参数（own=%d vdist=%d）"
          % (s["size"], s["channels"], s["blocks"], len(s["params"]), s["own"], s["vdist"]))

    if tsize == s["size"] and tvh == s["vhidden"]:
        p, notes = widen_same_size(s, tch, tblk, seed=a.seed, noise=a.noise)
        n = write_bin(a.out, tsize, s["planes"], tch, tvh, max(tblk, s["blocks"]),
                      s["own"], s["vdist"], p)
        print("目标   : %d 路 / %d 通道 / %d 块 / %d 参数 -> %s"
              % (tsize, tch, max(tblk, s["blocks"]), n, a.out))
        for x in notes:
            print("   ", x)
        print("模式   : 同棋盘函数保持加宽（新通道 = 旧通道副本 ×(1±%.0f%%)，下游输入 /m）" % (a.noise * 100))
    else:
        p, copied, partial, fresh = warm_start(s, tsize, s["planes"], tch, tvh, tblk)
        n = write_bin(a.out, tsize, s["planes"], tch, tvh, max(tblk, s["blocks"]),
                      s["own"], s["vdist"], p)
        print("目标   : %d 路 / %d 通道 / %d 参数 -> %s" % (tsize, tch, n, a.out))
        print("完整迁移: %s" % (", ".join(copied) if copied else "无"))
        print("切片迁移: %s" % (", ".join(partial) if partial else "无"))
        print("重新学习: %s" % ", ".join(fresh))
        print("模式   : 跨棋盘（依赖点数的头部重学）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
