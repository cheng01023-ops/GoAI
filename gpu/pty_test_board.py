"""用伪终端驱动 GoBoard（显式设置 TERM），验证界面渲染与按键"""
import os, pty, select, sys, time, re

def run(keys, wait_after=3.0):
    pid, fd = pty.fork()
    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.environ["LANG"] = "en_US.UTF-8"
        os.execv("./build/GoBoard", ["./build/GoBoard", "--size", "9", "--sims", "30",
                                     "--weights", "runs_v2/latest.bin"])
        os._exit(1)
    buf = b""
    def drain(t):
        nonlocal buf
        end = time.time() + t
        while time.time() < end:
            r, _, _ = select.select([fd], [], [], 0.1)
            if r:
                try:
                    d = os.read(fd, 65536)
                except OSError:
                    return
                if not d:
                    return
                buf += d
    drain(1.5)
    for k in keys:
        os.write(fd, k)
        drain(0.4)
    drain(wait_after)
    os.write(fd, b"q")
    drain(1.0)
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return buf.decode("utf-8", "replace")

raw = run([b"\x1b[C", b"\x1b[C", b"\x1b[B", b"\x1b[B", b"\r"])
clean = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", raw).replace("\x1b(B", "").replace("\x1b=", "")
lines = clean.split("\n")
print("原始字节数:", len(raw))
show = [l for l in lines if l.strip()]
print("---- 最后一屏 ----")
print("\n".join(show[-22:]))
print("---- 判定 ----")
print("  坐标行:", "OK" if "A B C D E F G H J" in clean else "失败")
print("  黑子 X:", "OK" if "X" in clean else "失败")
print("  白子 O:", "OK" if "O" in clean else "失败")
print("  状态栏:", "OK" if ("轮到" in clean or "AI" in clean or "提子" in clean) else "失败")
print("  错误信息:", [l for l in lines if "Error" in l or "error" in l][:3])
