"""GoAI GPU 训练器 —— 批量自对弈 + GPU 训练，权重自动导出为 C 引擎可热加载的 .bin

用法（Windows/Linux/macOS 通用）：
    python train.py --size 19 --channels 128 --games 256 --sims 64 --batch 512 --device cuda --amp bf16
    python train.py --size 9  --channels 32  --games 64  --sims 40  --device cpu        # 小规模自检

设计目标：把 5080 跑满
  * B 盘棋同时进行，每轮把所有叶子拼成一个 batch 送 GPU（单样本推理喂不饱显卡）
  * 回放缓冲直接常驻显存，训练时零拷贝
  * 混合精度（bf16/fp16）+ AdamW + 8 重对称增强
  * 每轮结束导出 latest.bin / best.bin（C 版棋手可实时热加载）
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from goai_gpu.export import export_c_bin, import_c_bin
from goai_gpu.mcts import Search
from goai_gpu.model import Net
from goai_gpu.rules import Game, PASS, BLACK, WHITE


# ----------------------------------------------------------------- 工具
def localtime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_status(path, **kv):
    try:
        with open(path, "w", encoding="utf-8") as f:
            for k, v in kv.items():
                f.write("%s=%s\n" % (k, v))
    except OSError:
        pass


def stop_requested(path) -> bool:
    return os.path.exists(path)


def sgf_open(path, size, komi, comment):
    f = open(path, "w", encoding="utf-8")
    f.write("(;GM[1]FF[4]CA[UTF-8]AP[GoAI-GPU:1.0]SZ[%d]KM[%.1f]PB[Black]PW[White]RU[Chinese]C[%s]\n"
            % (size, komi, comment))
    return f


def sgf_move(f, move, color, size):
    if move == PASS:
        f.write(";%s[]" % ("B" if color == BLACK else "W"))
    else:
        f.write(";%s[%c%c]" % ("B" if color == BLACK else "W", ord('a') + move % size, ord('a') + move // size))


def sgf_close(f, winner):
    f.write("RE[%s])\n" % ("B+R" if winner == BLACK else ("W+R" if winner == WHITE else "0")))
    f.close()


# ----------------------------------------------------------------- 批量自对弈
class SelfPlayPool:
    def __init__(self, cfg, net, device, rng):
        self.cfg = cfg
        self.net = net
        self.device = device
        self.rng = rng
        self.searches = [Search(cfg.size, cfg.komi,
                                 pass_min_move=cfg.pass_min_move,
                                 noise_alpha=cfg.dirichlet, noise_eps=cfg.noise_eps,
                                 rng=rng)
                          for _ in range(cfg.games)]
        self.examples = [[] for _ in range(cfg.games)]   # [(features, policy)]
        self.done = [False] * cfg.games
        self.winner = [0] * cfg.games
        self.sgfs = [None] * cfg.games
        self.moves = [[] for _ in range(cfg.games)]
        self.t_infer = 0.0      # 花在推理上的时间（在 GPU 上就是显卡干活的时间）
        self.n_infer = 0
        self.n_boards = 0       # 送进网络的局面总数

    def eval_batch(self, boards):
        t0 = time.perf_counter()
        x = np.stack([b.features() for b in boards]).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device, non_blocking=True)
        with torch.inference_mode():
            if self.cfg.amp == "bf16" and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pol, val = self.net.policy_value(xt)
            elif self.cfg.amp == "fp16" and self.device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    pol, val = self.net.policy_value(xt)
            else:
                pol, val = self.net.policy_value(xt)
        out = (pol.float().cpu().numpy(), val.float().cpu().numpy())
        self.t_infer += time.perf_counter() - t0
        self.n_infer += 1
        self.n_boards += len(boards)
        return out

    def run(self):
        """跑完 cfg.games 盘自对弈，返回 (x, pi, z)"""
        cfg = self.cfg
        active = list(range(cfg.games))
        sim_step = 0
        while active:
            sim_step += 1
            # 每轮：每盘活跃棋局各做一次模拟，所有叶子拼成一个 batch 送 GPU
            batch_boards, batch_meta = [], []
            for g in active:
                s = self.searches[g]
                node, path = s.descend()
                n = s.nodes[node]
                if n.terminal:
                    s.backup(path, n.tval)
                    continue
                batch_boards.append(n.board)
                batch_meta.append((g, node, path))
            if batch_boards:
                pols, vals = self.eval_batch(batch_boards)
                for (g, node, path), pol, val in zip(batch_meta, pols, vals):
                    self.searches[g].expand(node, pol, float(val))
                    self.searches[g].backup(path, float(val))
            if sim_step % cfg.sims != 0:
                continue
            # 每盘棋落一子
            still = []
            for g in active:
                s = self.searches[g]
                temp = 1.0 if len(self.moves[g]) < cfg.temp_moves else 0.0
                mv = s.choose(temp, self.rng)
                pol_target = s.root_policy()
                self.examples[g].append((s.nodes[s.root].board.features().copy(), pol_target))
                board = s.nodes[s.root].board
                if self.sgfs[g] is not None:
                    sgf_move(self.sgfs[g], mv, board.to_move, cfg.size)
                self.moves[g].append(mv)
                # 只由 advance 负责“落子”：新根的棋盘 = 原局面 + 这一手
                # （若自己也落一手，会与子节点里提前克隆的棋盘不一致，导致棋局状态回退）
                s.advance(mv)
                board = s.nodes[s.root].board
                if board.is_terminal(cfg.max_moves):
                    self.done[g] = True
                    self.winner[g] = board.winner(cfg.komi)
                    if self.sgfs[g] is not None:
                        sgf_close(self.sgfs[g], self.winner[g])
                    continue
                still.append(g)
            active = still
        xs, pis, zs = [], [], []
        for g in range(cfg.games):
            w = self.winner[g]
            for i, (x, pi) in enumerate(self.examples[g]):
                player = BLACK if i % 2 == 0 else WHITE
                z = 0.0 if w == 0 else (1.0 if w == player else -1.0)
                xs.append(x)
                pis.append(pi)
                zs.append(z)
        if not xs:
            return (np.zeros((0, 4, cfg.size, cfg.size), np.float32),
                    np.zeros((0, cfg.size * cfg.size + 1), np.float32), np.zeros(0, np.float32))
        return np.stack(xs), np.stack(pis).astype(np.float32), np.array(zs, np.float32)


# ----------------------------------------------------------------- 评估
def _mcts_move(cfg, net, device, search, board, sims, rng):
    """单盘棋的 MCTS 选点（每次模拟都立即展开，保证访问数正确累计）"""
    for _ in range(sims):
        node, path = search.descend()
        n = search.nodes[node]
        if n.terminal:
            search.backup(path, n.tval)
            continue
        x = torch.from_numpy(n.board.features()).unsqueeze(0).to(device)
        with torch.inference_mode():
            pol, val = net.policy_value(x)
        search.expand(node, pol[0].float().cpu().numpy(), float(val[0]))
        search.backup(path, float(val[0]))
    return search.choose(0.0, rng)


def play_pair(cfg, net_a, net_b, device, sims, games, rng):
    """net_a 与 net_b 对打（net_b=None 表示随机走子），返回 net_a 胜率"""
    wins = 0
    for gi in range(games):
        a_is_black = (gi % 2 == 0)
        board = Game(cfg.size)
        sa = Search(cfg.size, cfg.komi, pass_min_move=cfg.pass_min_move)
        sb = Search(cfg.size, cfg.komi, pass_min_move=cfg.pass_min_move)
        while not board.is_terminal(cfg.max_moves):
            a_turn = (board.to_move == BLACK) == a_is_black
            if a_turn:
                mv = _mcts_move(cfg, net_a, device, sa, board, sims, rng)
            elif net_b is None:
                mv = rng.choice(board.legal_moves())
            else:
                mv = _mcts_move(cfg, net_b, device, sb, board, sims, rng)
            if not board.play(mv):
                board.play(PASS)
                mv = PASS
            sa.advance(mv)
            sb.advance(mv)
        w = board.winner(cfg.komi)
        if w != 0 and (w == BLACK) == a_is_black:
            wins += 1
    return wins / max(1, games)


# ----------------------------------------------------------------- 主循环
def main():
    ap = argparse.ArgumentParser(description="GoAI GPU 训练器")
    ap.add_argument("--size", type=int, default=9)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--vhidden", type=int, default=64)
    ap.add_argument("--games", type=int, default=64, help="并行自对弈盘数（越大越吃满 GPU）")
    ap.add_argument("--sims", type=int, default=40, help="每步 MCTS 模拟次数")
    ap.add_argument("--iters", type=int, default=20, help="训练轮数，0 = 无限")
    ap.add_argument("--steps", type=int, default=200, help="每轮梯度步数")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--buffer", type=int, default=300000, help="回放缓冲（局面数，常驻显存）")
    ap.add_argument("--komi", type=float, default=7.0)
    ap.add_argument("--max-moves", type=int, default=0, help="单局最大手数（0 = 3*N*N）")
    ap.add_argument("--pass-min-move", type=int, default=0, help="这手之前不允许停着（0 = 自动）")
    ap.add_argument("--temp-moves", type=int, default=12)
    ap.add_argument("--dirichlet", type=float, default=0.3, help="根节点 Dirichlet 噪声 alpha")
    ap.add_argument("--noise-eps", type=float, default=0.25, help="噪声混合比例")
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--eval-sims", type=int, default=40)
    ap.add_argument("--gate-every", type=int, default=3)
    ap.add_argument("--gate-games", type=int, default=16)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--threads", type=int, default=0, help="CPU 线程数（自对弈的树操作）")
    ap.add_argument("--out", default="runs_gpu")
    ap.add_argument("--seed", type=int, default=20241001)
    ap.add_argument("--resume", default="")
    ap.add_argument("--save-sgf", type=int, default=4)
    ap.add_argument("--compile", action="store_true", help="torch.compile 加速（首次较慢）")
    cfg = ap.parse_args()
    if not cfg.max_moves:
        cfg.max_moves = 3 * cfg.size * cfg.size
    if not cfg.pass_min_move:
        cfg.pass_min_move = max(2 * cfg.size, (cfg.size * cfg.size * 2) // 3)

    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    os.makedirs(cfg.out, exist_ok=True)
    os.makedirs(os.path.join(cfg.out, "games"), exist_ok=True)
    stop_path = os.path.join(cfg.out, "STOP")
    if os.path.exists(stop_path):
        os.remove(stop_path)
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[警告] 没有检测到 CUDA，自动改用 CPU（速度会慢很多）")
        device = "cpu"
    cfg.device = device
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else device.upper()
    print("=== GoAI GPU 训练器 ===")
    print("  设备 %s | torch %s | 棋盘 %dx%d | 通道 %d | 并行 %d 盘 | 每步 %d 次模拟"
          % (dev_name, torch.__version__, cfg.size, cfg.size, cfg.channels, cfg.games, cfg.sims))
    if device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        print("  显存 %.1f / %.1f GB 可用" % (free_b / 1e9, total_b / 1e9))

    net = Net(cfg.size, 4, cfg.channels, cfg.vhidden).to(device)
    if cfg.resume and os.path.exists(cfg.resume):
        import_c_bin(net, cfg.resume)
        print("  继续训练：已载入 %s" % cfg.resume)
    net.train()
    if cfg.compile and device == "cuda":
        try:
            net = torch.compile(net)
            print("  已启用 torch.compile")
        except Exception as e:      # noqa: BLE001
            print("  torch.compile 不可用:", e)

    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=1e-4)
    nn_ = cfg.size * cfg.size
    bx = torch.zeros(cfg.buffer, 4, cfg.size, cfg.size, dtype=torch.uint8, device=device)
    bpi = torch.zeros(cfg.buffer, nn_ + 1, dtype=torch.float16, device=device)
    bz = torch.zeros(cfg.buffer, dtype=torch.float16, device=device)
    cap = cfg.buffer
    count = head = 0

    csv_path = os.path.join(cfg.out, "train_log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("iteration,games,positions,policy_loss,value_loss,winrate_vs_random,gate_winrate,elapsed_s\n")

    best_pi = -1.0
    t_start = time.time()
    it = 0
    while cfg.iters == 0 or it < cfg.iters:
        it += 1
        t0 = time.time()
        # ---------- 自对弈 ----------
        pool = SelfPlayPool(cfg, net, device, rng)
        for g in range(min(cfg.save_sgf, cfg.games)):
            path = os.path.join(cfg.out, "games", "iter%03d_game%02d.sgf" % (it, g + 1))
            pool.sgfs[g] = sgf_open(path, cfg.size, cfg.komi, "GoAI GPU self-play iteration %d" % it)
        xs, pis, zs = pool.run()
        n_new = len(zs)
        for i in range(n_new):
            k = head
            bx[k].copy_(torch.from_numpy(xs[i]).to(torch.uint8))
            bpi[k].copy_(torch.from_numpy(pis[i]).to(torch.float16))
            bz[k] = float(zs[i])
            head = (k + 1) % cap
            count = min(count + 1, cap)
        t_self = time.time() - t0

        # ---------- 训练 ----------
        pl = vl = 0.0
        if count >= cfg.batch:
            net.train()
            t1 = time.time()
            for _ in range(cfg.steps):
                idx = torch.randint(0, count, (cfg.batch,), device=device)
                x = bx[idx].float()
                pi = bpi[idx].float()
                z = bz[idx].float()
                k = rng.randrange(8)
                if k >= 4:
                    x = torch.flip(x, dims=[3])
                    pi_board = torch.flip(pi[:, :nn_].reshape(-1, cfg.size, cfg.size), dims=[2]).reshape(-1, nn_)
                    pi = torch.cat([pi_board, pi[:, nn_:]], dim=1)
                rot = k % 4
                if rot:
                    x = torch.rot90(x, rot, dims=[2, 3])
                    pi_board = torch.rot90(pi[:, :nn_].reshape(-1, cfg.size, cfg.size), rot, dims=[1, 2]).reshape(-1, nn_)
                    pi = torch.cat([pi_board, pi[:, nn_:]], dim=1)
                opt.zero_grad(set_to_none=True)
                if device == "cuda" and cfg.amp != "none":
                    dt = torch.bfloat16 if cfg.amp == "bf16" else torch.float16
                    with torch.autocast("cuda", dtype=dt):
                        logits, v = net(x)
                        loss = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean() + ((v - z) ** 2).mean()
                    loss.backward()
                else:
                    logits, v = net(x)
                    loss = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean() + ((v - z) ** 2).mean()
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    ce = -(pi * F.log_softmax(logits.float(), dim=1)).sum(1).mean().item()
                    mse = ((v.float() - z) ** 2).mean().item()
                pl += ce / cfg.steps
                vl += mse / cfg.steps
            t_train = time.time() - t1
        else:
            t_train = 0.0

        # ---------- 评估 ----------
        net.eval()
        wr = play_pair(cfg, net, None, device, cfg.eval_sims, cfg.eval_games, rng)
        gw = -1.0
        if it % cfg.gate_every == 0 and os.path.exists(best_path):
            best_net = Net(cfg.size, 4, cfg.channels, cfg.vhidden).to(device)
            import_c_bin(best_net, best_path)
            best_net.eval()
            gw = play_pair(cfg, net, best_net, device, cfg.eval_sims, cfg.gate_games, rng)
        best_path = os.path.join(cfg.out, "best.bin")
        export_c_bin(net, os.path.join(cfg.out, "latest.bin"))
        torch.save({"model": net.state_dict(), "cfg": vars(cfg), "iter": it},
                   os.path.join(cfg.out, "iter_%03d.pt" % it))
        promote = (not os.path.exists(best_path)) or (gw >= 0.55) or (gw < 0 and wr > best_pi)
        if promote:
            export_c_bin(net, best_path)
            best_pi = max(best_pi, wr)
            tag = "晋级 best.bin"
        else:
            tag = "保留旧 best.bin"
        el = time.time() - t0
        total = time.time() - t_start
        infer_pct = 100.0 * pool.t_infer / max(1e-9, t_self)
        batch_avg = pool.n_boards / max(1, pool.n_infer)
        print("          └ 自对弈细分：推理 %.1f%%（%d 次批量前向，平均 batch %.0f）｜树搜索+规则 %.1f%%"
              % (infer_pct, pool.n_infer, batch_avg, 100.0 - infer_pct))
        print("第 %d 轮 | 自对弈 %d 局/%d 局面 (%.1fs) | 训练 %.1fs | 策略损失 %.4f 价值损失 %.4f | "
              "对随机 %.1f%% [%s] | 累计 %.1f 分钟"
              % (it, cfg.games, n_new, t_self, t_train, pl, vl, wr * 100, tag, total / 60))
        sys.stdout.flush()
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write("%d,%d,%d,%.5f,%.5f,%.4f,%.4f,%.2f\n" % (it, cfg.games, n_new, pl, vl, wr, gw, el))
        write_status(os.path.join(cfg.out, "status.txt"),
                     state="running", iteration=it, game=0, games_this_iter=cfg.games,
                     total_games=it * cfg.games, elapsed_iter="%.1f" % el,
                     elapsed_total="%.1f" % total, threads=cfg.threads or 1, sims=cfg.sims,
                     channels=cfg.channels, pid=os.getpid(),
                     winrate_vs_random="%.4f" % wr, device=dev_name, updated=localtime())
        if stop_requested(stop_path):
            print("\n检测到 STOP 文件：本轮已保存，训练结束。")
            break
    write_status(os.path.join(cfg.out, "status.txt"), state="stopped", iteration=it,
                 elapsed_total="%.1f" % (time.time() - t_start), updated=localtime())
    print("\n训练结束，权重：%s/latest.bin（C 引擎可直接加载）" % cfg.out)


if __name__ == "__main__":
    main()
