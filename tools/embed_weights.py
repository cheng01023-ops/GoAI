#!/usr/bin/env python3
"""embed_weights.py - 把训练好的权重 .bin 编译进 C 程序（生成 include/weights_builtin.h）

生成的头文件被 src/goboard.c 包含，于是人机对弈程序 GoBoard
**单个可执行文件就是完整的 AI**（不需要额外的 .bin，拷给别人也不会丢）。

用法：
    # 把当前最强的存档权重写进程序
    python3 tools/embed_weights.py --from versions/goai9x9_v5_800sims.bin \
        --out include/weights_builtin.h

    # 自动挑最新的训练输出（runs9_s800/best.bin -> runs_step5/latest.bin -> ...）
    python3 tools/embed_weights.py --latest --out include/weights_builtin.h

    # 看看会挑到哪个文件（不写文件）
    python3 tools/embed_weights.py --latest --dry-run

然后重新编译：make goboard   （或 VS Code 里 F5 选 “▶ 对弈：GoBoard”）
"""
import argparse
import datetime
import os
import struct
import sys

# --latest 的候选（按 mtime 取最新；可自行增删）
CANDIDATES = [
    "runs9_s800/best.bin",
    "runs9_s800/latest.bin",
    "runs_step5/latest.bin",
    "runs_v4/latest.bin",
    "runs_live/best.bin",
    "runs_live/latest.bin",
    "runs_v2/latest.bin",
    "versions/goai9x9_v5_800sims.bin",
    "versions/goai9x9_v4_lr001.bin",
    "versions/goai9x9_v3_33kgames.bin",
]

MAGIC = 0x474F4149
MAGIC2 = 0x474F414A
FLAG_OWN, FLAG_VDIST, FLAG_BLOCKS = 0x02, 0x04, 0x01
VBUCKETS = 33


def describe(data):
    """读权重头，返回一句人类可读的说明（顺带做一次格式校验）"""
    if len(data) < 28:
        raise ValueError("文件太小，不像 GoAI 权重")
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic == MAGIC:
        _, size, planes, ch, vh, npar = struct.unpack_from("<Iiiiii", data, 0)
        extra = ""
    elif magic == MAGIC2:
        _, size, planes, ch, vh, flags, npar = struct.unpack_from("<Iiiiiii", data, 0)
        blocks = (flags >> 8) & 0xFFFFFF
        own = bool(flags & FLAG_OWN)
        vdist = bool(flags & FLAG_VDIST)
        extra = " · %d 残差块%s%s" % (blocks, " · 领地头" if own else "",
                                      " · 值分布(%d桶)" % VBUCKETS if vdist else "")
    else:
        raise ValueError("magic 不对（%08x），不是 GoAI 权重文件" % magic)
    hdr = 24 if magic == MAGIC else 28
    if len(data) != hdr + 4 * npar:
        raise ValueError("文件长度不对：%d 字节，按头部应为 %d" % (len(data), hdr + 4 * npar))
    return "%d 路棋盘 · %d 通道%s · %d 参数 · %d 字节" % (size, ch, extra, npar, len(data))


def pick_latest(root=".", verbose=True):
    best, best_t = None, None
    for rel in CANDIDATES:
        p = os.path.join(root, rel)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            t = os.path.getmtime(p)
            if best_t is None or t > best_t:
                best, best_t = p, t
    if best and verbose:
        print("最新训练输出: %s（%s）" % (best, datetime.datetime.fromtimestamp(best_t)))
    return best


def emit(src_path, out_path, root="."):
    with open(src_path, "rb") as f:
        data = f.read()
    info = describe(data)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rel = os.path.relpath(src_path, root)
    lines = []
    lines.append("/* weights_builtin.h - 由 tools/embed_weights.py 自动生成，请勿手改")
    lines.append(" *")
    lines.append(" * 训练好的神经网络权重直接编译进程序，因此：")
    lines.append(" *   · 单个可执行文件就是完整的 AI，不需要额外的 .bin 文件")
    lines.append(" *   · 发给别人 / 换电脑 / 拷到 U 盘都不会丢")
    lines.append(" *")
    lines.append(" * 来源: %s" % rel)
    lines.append(" * 规格: %s" % info)
    lines.append(" * 生成: %s（重新生成：python3 tools/embed_weights.py --latest）" % ts)
    lines.append(" */")
    lines.append("#ifndef GOAI_WEIGHTS_BUILTIN_H")
    lines.append("#define GOAI_WEIGHTS_BUILTIN_H")
    lines.append("")
    lines.append("#define GOAI_BUILTIN_NET_LEN %d" % len(data))
    lines.append("")
    lines.append("static const unsigned char GOAI_BUILTIN_NET[GOAI_BUILTIN_NET_LEN] = {")
    per = 16
    for i in range(0, len(data), per):
        chunk = data[i:i + per]
        lines.append("    " + ",".join(str(b) for b in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("#endif /* GOAI_WEIGHTS_BUILTIN_H */")
    text = "\n".join(lines) + "\n"
    if out_path == "-":
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
    return len(data), info


def main():
    ap = argparse.ArgumentParser(description="把权重 .bin 生成 C 头文件（编译进程序）")
    ap.add_argument("--from", dest="src", help="源权重 .bin")
    ap.add_argument("--latest", action="store_true", help="自动挑最新的训练输出（%s）"
                    % "/".join(CANDIDATES[:4]) + " ...")
    ap.add_argument("--out", default="include/weights_builtin.h", help="输出头文件")
    ap.add_argument("--dry-run", action="store_true", help="只显示会挑哪个文件，不写")
    a = ap.parse_args()

    src = a.src
    if a.latest or not src:
        src = pick_latest()
        if not src:
            print("找不到候选权重（试过 %s）" % ", ".join(CANDIDATES), file=sys.stderr)
            return 1
    if not os.path.exists(src):
        print("找不到权重文件: %s" % src, file=sys.stderr)
        return 1
    if a.dry_run:
        with open(src, "rb") as f:
            print("%s -> %s" % (src, describe(f.read())))
        return 0
    n, info = emit(src, a.out)
    print("已写入 %s" % a.out)
    print("  来源: %s" % src)
    print("  规格: %s" % info)
    print("  下一步: make goboard   （或 VS Code 里按 F5 选 “▶ 对弈：GoBoard”）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
