"""GoAI GPU 服务端 —— 批量推理 + 训练，权重由服务端持有（客户端无需本地网络）

架构：
    C 引擎（几百盘棋并行搜索，规则/搜索都是 C，微秒级）
        │ 每个叶子 → 一次批量评估请求
        ▼
    本服务（把多条连接、多个请求攒成一个 batch → GPU 一次前向 → 广播回去）
        │ 自对弈样本通过 upload 通道回传
        ▼
    训练线程在 GPU 上训练 → 定期导出 C 兼容 .bin 并热切换推理权重

用法：
    python server.py --size 19 --channels 256 --games-batch 256 --device cuda --amp bf16
"""
from __future__ import annotations

import argparse
import os
import queue
import selectors
import socket
import struct
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goai_gpu.export import export_c_bin, import_c_bin
from goai_gpu.model import Net

MAGIC = 0x49414F47  # "GOAI"
VERSION = 1


def localtime():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Conn:
    __slots__ = ("sock", "rbuf", "mode", "need", "hdr_done", "count", "acc",
                 "policy", "value", "pending")

    def __init__(self, sock):
        self.sock = sock
        self.rbuf = bytearray()
        self.mode = None          # 'eval' / 'upload'
        self.need = 0
        self.hdr_done = False
        self.count = 0
        self.acc = bytearray()
        self.policy = None
        self.value = None
        self.pending = None       # (count, ndarray features)


class Trainer(threading.Thread):
    """在 GPU 上训练；训练好的权重定期同步给推理侧（无锁双缓冲切换）"""

    def __init__(self, cfg, models, state):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.models = models          # [m0, m1]
        self.state = state            # {"idx": 0, ...}
        self.q = queue.Queue(maxsize=200)
        self.n_positions = 0
        self.n_steps = 0
        self.loss_pi = 0.0
        self.loss_v = 0.0
        self.stop_flag = False
        nn_ = cfg.size * cfg.size
        self.bx = torch.zeros(cfg.buffer, 4, cfg.size, cfg.size, dtype=torch.uint8, device=cfg.device)
        self.bpi = torch.zeros(cfg.buffer, nn_ + 1, dtype=torch.float16, device=cfg.device)
        self.bz = torch.zeros(cfg.buffer, dtype=torch.float16, device=cfg.device)
        self.count = self.head = 0

    def submit(self, x, pi, z):
        try:
            self.q.put_nowait((x, pi, z))
        except queue.Full:
            pass

    def run(self):
        cfg = self.cfg
        opt = torch.optim.AdamW(self.models[0].parameters(), lr=cfg.lr, weight_decay=1e-4)
        last_export = time.time()
        while not self.stop_flag:
            try:
                x, pi, z = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            n = len(z)
            for i in range(n):
                k = self.head
                self.bx[k].copy_(torch.from_numpy(x[i]).to(torch.uint8))
                self.bpi[k].copy_(torch.from_numpy(pi[i]).to(torch.float16))
                self.bz[k] = float(z[i])
                self.head = (k + 1) % cfg.buffer
                self.count = min(self.count + 1, cfg.buffer)
            self.n_positions += n
            if self.count < cfg.train_batch:
                continue
            model = self.models[self.state["idx"]]          # 训练当前推理用的那份
            model.train()
            for _ in range(cfg.steps_per_chunk):
                idx = torch.randint(0, self.count, (cfg.train_batch,), device=cfg.device)
                xb = self.bx[idx].float()
                pib = self.bpi[idx].float()
                zb = self.bz[idx].float()
                # 8 重对称增强（整批用同一个变换，够用且便宜）
                r = np.random.randint(8)
                if r >= 4:
                    xb = torch.flip(xb, dims=[3])
                    pb = torch.flip(pib[:, :-1].reshape(-1, cfg.size, cfg.size), dims=[2]).reshape(-1, cfg.size * cfg.size)
                    pib = torch.cat([pb, pib[:, -1:]], dim=1)
                rot = r % 4
                if rot:
                    xb = torch.rot90(xb, rot, dims=[2, 3])
                    pb = torch.rot90(pib[:, :-1].reshape(-1, cfg.size, cfg.size), rot, dims=[1, 2]).reshape(-1, cfg.size * cfg.size)
                    pib = torch.cat([pb, pib[:, -1:]], dim=1)
                opt.zero_grad(set_to_none=True)
                if cfg.device == "cuda" and cfg.amp != "none":
                    dt = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
                    with torch.autocast("cuda", dtype=dt):
                        logits, v = model(xb)
                        loss = -(pib * F.log_softmax(logits, dim=1)).sum(1).mean() + ((v - zb) ** 2).mean()
                    loss.backward()
                else:
                    logits, v = model(xb)
                    loss = -(pib * F.log_softmax(logits, dim=1)).sum(1).mean() + ((v - zb) ** 2).mean()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    self.loss_pi = float(-(pib * F.log_softmax(logits.float(), dim=1)).sum(1).mean())
                    self.loss_v = float(((v.float() - zb) ** 2).mean())
                self.n_steps += 1
            self.state["batches"] = self.state.get("batches", 0) + 1
            # 热切换：把刚训练好的权重同步给推理侧（另一份缓冲，索引翻转即可）
            if time.time() - last_export > cfg.export_seconds:
                other = 1 - self.state["idx"]
                with torch.no_grad():
                    self.models[other].load_state_dict(model.state_dict())
                self.state["idx"] = other
                try:
                    export_c_bin(self.models[other], os.path.join(cfg.out, "latest.bin"))
                    export_c_bin(self.models[other], os.path.join(cfg.out, "best.bin"))
                except Exception as e:  # noqa: BLE001
                    print("导出权重失败:", e)
                last_export = time.time()
                print("[训练] 局面 %d | 步数 %d | 策略损失 %.4f | 价值损失 %.4f | %s 已热更新推理权重"
                      % (self.n_positions, self.n_steps, self.loss_pi, self.loss_v, localtime()))
                sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="GoAI GPU 推理+训练服务")
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--vhidden", type=int, default=64)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--max-batch", type=int, default=1024, help="一次前向最多合并多少局面")
    ap.add_argument("--batch-wait-ms", type=float, default=3.0, help="攒批等待时间")
    ap.add_argument("--train-batch", type=int, default=512)
    ap.add_argument("--steps-per-chunk", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--buffer", type=int, default=300000)
    ap.add_argument("--export-seconds", type=float, default=60.0)
    ap.add_argument("--out", default="runs_gpu")
    ap.add_argument("--resume", default="")
    ap.add_argument("--stats-seconds", type=float, default=10.0)
    cfg = ap.parse_args()

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 没有 CUDA，改用 CPU")
        cfg.device = "cpu"
    os.makedirs(cfg.out, exist_ok=True)
    dev_name = torch.cuda.get_device_name(0) if cfg.device == "cuda" else cfg.device.upper()

    models = [Net(cfg.size, 4, cfg.channels, cfg.vhidden).to(cfg.device) for _ in range(2)]
    if cfg.resume and os.path.exists(cfg.resume):
        import_c_bin(models[0], cfg.resume)
        print("继续训练：已载入", cfg.resume)
    for m in models:
        m.eval()
    state = {"idx": 0}
    nn_ = cfg.size * cfg.size
    planes = 4

    trainer = Trainer(cfg, models, state)
    trainer.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((cfg.host, cfg.port))
    srv.listen(64)
    sel = selectors.DefaultSelector()
    sel.register(srv, selectors.EVENT_READ)
    conns = {}

    print("=== GoAI GPU 服务端 ===")
    print("  设备 %s | 棋盘 %dx%d | %d 通道 | 端口 %d | 攒批上限 %d" %
          (dev_name, cfg.size, cfg.size, cfg.channels, cfg.port, cfg.max_batch))
    print("  客户端用法： ./build/GoAI gtrain --remote %s:%d ..." % (cfg.host, cfg.port))
    sys.stdout.flush()

    n_eval_batches = n_eval_positions = n_upload = 0
    t_busy = 0.0
    t_last = time.time()

    def handle_message(c: Conn):
        """解析完一条消息；返回待评估的 (conn, count, features) 或 None"""
        nonlocal n_upload
        if c.mode == "upload":
            total = c.count * planes * nn_ + c.count * (nn_ + 1) * 4 + c.count * 4
            if len(c.acc) < total:
                return None
            off = 0
            x = np.frombuffer(c.acc, dtype=np.uint8, count=c.count * planes * nn_, offset=off)
            off += c.count * planes * nn_
            pi = np.frombuffer(c.acc, dtype=np.float32, count=c.count * (nn_ + 1), offset=off)
            off += c.count * (nn_ + 1) * 4
            z = np.frombuffer(c.acc, dtype=np.float32, count=c.count, offset=off)
            trainer.submit(x.reshape(c.count, planes, cfg.size, cfg.size).copy(), pi.reshape(c.count, nn_ + 1).copy(), z.copy())
            n_upload += c.count
            c.mode = None
            c.acc = bytearray()
            return None
        # eval
        need = c.count * planes * nn_ * 4
        if len(c.acc) < need:
            return None
        x = np.frombuffer(c.acc, dtype=np.float32, count=c.count * planes * nn_).reshape(c.count, planes, cfg.size, cfg.size).copy()
        c.mode = None
        c.acc = bytearray()
        return (c, c.count, x)

    while True:
        events = sel.select(timeout=cfg.batch_wait_ms / 1000.0)
        for key, _ in events:
            if key.fileobj is srv:
                sock, _addr = srv.accept()
                sock.setblocking(False)
                sel.register(sock, selectors.EVENT_READ)
                conns[sock] = Conn(sock)
                continue
            sock = key.fileobj
            c = conns.get(sock)
            if c is None:
                continue
            try:
                data = sock.recv(1 << 20)
            except BlockingIOError:
                continue
            if not data:
                sel.unregister(sock)
                sock.close()
                conns.pop(sock, None)
                continue
            c.rbuf += data
            if not c.hdr_done:
                if len(c.rbuf) < 8:
                    continue
                magic, ver = struct.unpack_from("<II", c.rbuf, 0)
                if magic != MAGIC:
                    print("协议错误，断开一个连接")
                    sel.unregister(sock); sock.close(); conns.pop(sock, None)
                    continue
                del c.rbuf[:8]
                sock.sendall(struct.pack("<6i", cfg.size, planes, cfg.channels, cfg.vhidden, nn_, cfg.max_batch))
                c.hdr_done = True
            # 解析消息头
            if c.mode is None:
                if len(c.rbuf) < 4:
                    continue
                c.count = struct.unpack_from("<I", c.rbuf, 0)[0]
                del c.rbuf[:4]
                c.mode = "upload" if c.count >= 0x80000000 else "eval"
                if c.mode == "upload":
                    c.count -= 0x80000000      # 上传用最高位标记
            c.acc += c.rbuf
            c.rbuf = bytearray()
            res = handle_message(c)
            if res is not None:
                c.pending = res

        # 攒批
        items, owners = [], []
        for c in list(conns.values()):
            if c.pending is None:
                continue
            _, cnt, x = c.pending
            take = min(cnt, cfg.max_batch - len(items))
            if take <= 0:
                break
            if take < cnt:      # 一条消息超过上限：只处理一部分（客户端消息不会太大，这里保险）
                items.append(x[:take]); owners.append((c, take))
                c.pending = (c, cnt - take, x[take:])
            else:
                items.append(x); owners.append((c, cnt))
                c.pending = None
        if items:
            xb = torch.from_numpy(np.concatenate(items, axis=0)).to(cfg.device, non_blocking=True)
            idx = state["idx"]
            model = models[idx]
            t0 = time.time()
            with torch.inference_mode():
                if cfg.device == "cuda" and cfg.amp != "none":
                    dt = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
                    with torch.autocast("cuda", dtype=dt):
                        logits, v = model(xb)
                else:
                    logits, v = model(xb)
                pol = F.softmax(logits.float(), dim=1).cpu().numpy()
                val = v.float().cpu().numpy()
            t_busy += time.time() - t0
            off = 0
            for (c, cnt) in owners:
                p = pol[off:off + cnt].astype("<f4", copy=False).tobytes()
                vv = val[off:off + cnt].astype("<f4", copy=False).tobytes()
                try:
                    c.sock.sendall(p + vv)
                except OSError:
                    pass
                off += cnt
            n_eval_batches += 1
            n_eval_positions += off
        if time.time() - t_last >= cfg.stats_seconds:
            el = time.time() - t_last
            gpu_pct = 100.0 * t_busy / max(1e-9, el)
            util = ""
            if cfg.device == "cuda":
                try:
                    util = " | 显卡利用率 %d%%" % int(torch.cuda.utilization())
                except Exception:  # noqa: BLE001
                    util = ""
            print("[服务] 连接 %d | 评估 %.0f 局面/秒 | 训练批次 %d (%d 步) | 上传样本 %d | 推理忙碌 %0.1f%%%s"
                  % (len(conns), n_eval_positions / max(1e-9, el), state.get("batches", 0),
                     trainer.n_steps, n_upload, gpu_pct, util))
            sys.stdout.flush()
            n_eval_batches = n_eval_positions = 0
            t_busy = 0.0
            t_last = time.time()


if __name__ == "__main__":
    main()
