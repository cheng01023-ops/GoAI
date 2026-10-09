# GoAI —— 用 C 语言从零写的围棋 AI（AlphaZero 风格 · 可自己训练 · 可对弈）

一个**不依赖任何深度学习框架**的围棋 AI：规则、蒙特卡洛树搜索（MCTS）、卷积神经网络的前向与反向传播
全部用 **C11** 手写。它可以在你自己的电脑上**自我对弈训练**，练强了直接跟人下棋。

- 🎯 **纯 C 实现**：不依赖 PyTorch/TensorFlow 也能训练（CPU 版）；有 NVIDIA 显卡时可选 PyTorch 加速
- 🧠 **AlphaZero 思路**：策略+价值双头 CNN、PUCT 搜索、Dirichlet 噪声探索、镜像数据增强、胜率晋级机制
- 🖥️ **三种用法**：终端全屏对弈（macOS/Linux）、纯文本对弈（Windows 零依赖）、GTP 协议（接 Sabaki 等图形界面）
- 📦 **开箱即玩**：`standalone/` 里的 `GoBoard.exe` 已把训练好的权重编译进程序，双击就能下
- ⚡ **榨干硬件**：多线程自对弈 + 可选 GPU 批量推理服务（C 引擎搜索 + Python/PyTorch 只做批量前向）

---

## 实测结果（9 路棋盘）

| 版本 | 训练量 | 对随机走子 | 对上一代 |
|---|---|---|---|
| v1 | 12 轮 / 360 局 / 5.8 分钟 | 78.8% | — |
| v2（32 通道） | 14 轮 / 560 局 / 11.9 分钟 | 92.5% | **100%**（40:0 胜 v1） |
| v3（整夜满速，8 核 10 线程） | 829 轮 / **35,320 局** / 8.6 小时 | **100%** | 98.8%（对 v2）、95%（对 8 小时前的自己） |

> v3 的棋力大约相当于"刚学会规则、下过几十盘"的人类：会吃子、会做眼、会贴身应手，
> 但没有定式概念。它已经进入平台期（策略损失 3.04 → 2.88），想更强需要加宽网络/加深搜索。

---

## 快速开始

### ① 只想下棋（最省事）

- **Windows**：解压 `standalone/`，双击 `PLAY_NOW.bat` —— 不用装编译器、不用装 Python
- **macOS / Linux**：`cd standalone && make goboard && ./build/GoBoard --sims 200`（全屏棋盘，方向键操作）

对弈程序默认用**编译进程序内部**的 v3 权重。想换网络：`--weights 你的.bin`。

### ② 自己训练（CPU，任何电脑都能跑）

**macOS / Linux**

```bash
make -s all && make -s test          # 编译 + 自检（69 项测试）
./build/GoAI train --size 9 --channels 32 --forever \
  --games 40 --sims 140 --steps 250 --threads 8 \
  --gate 1 --gategames 20 --eval-every 3 --out runs_live
```

**Windows**（需要 MSYS2 或 [w64devkit](https://github.com/skeeto/w64devkit/releases)）

```bat
mingw32-make -f Makefile.win all
build\GoAI.exe train --size 9 --channels 32 --forever --games 40 --sims 140 --steps 250 --threads 8 --out runs_live
```

- 训练会**一直跑**，每轮把最新权重写到 `runs_live/latest.bin`；`--gate 1` 打开晋级赛，
  只有明显更强的版本才会覆盖 `best.bin`
- 想看进度：`runs_live/status.txt`（每 2 秒刷新）、`runs_live/train_log.csv`（每轮一行）、
  `python3 tools/plot.py runs_live/train_log.csv`（画曲线）
- 停止：在输出目录建一个名为 `STOP` 的空文件即可（会在当前轮结束后保存权重退出）
- 随时对弈：`./build/GoAI eval --a 网络.bin --b random --games 40 --sims 100` 看胜率

### ③ 有 NVIDIA 显卡：榨干 GPU

CPU 版里搜索是并行瓶颈；想让显卡真正满载，用**推理服务**架构：

```bash
# 服务端（可跑在 5080 上）：批量推理 + GPU 训练 + 权重热更新
pip install -r gpu/requirements.txt
python gpu/server.py --size 19 --channels 256 --device cuda --amp bf16 --max-batch 2048 --out runs_gpu

# 对弈端（C 引擎跑搜索，几千盘并行）
./build/GoAI gtrain --remote 127.0.0.1:8899 --threads 16 --slots 64 --sims 64 --size 19
```

实测（CPU 服务端做下限参考）：服务端评估 **30,939 局面/秒**、推理忙碌 **94%**，
而早期"纯 Python 搜索"方案显卡只用到 8% 左右。

---

## 训练进阶：让曲线继续往下走

9 路 32 通道练到 3 万多局后会遇到平台期：策略损失在 2.7~2.9 之间来回震荡、晋级赛长期 50%。
原因是**学习信号饱和 + 固定学习率**，不是训练时间不够。程序内置了四个机制来突破它：

### 1. 学习率衰减（必开）

    --lr 0.005 --lr-decay-every 100 --lr-decay-factor 0.85 --lr-min 0.0005

第 k 轮的学习率 = `lr * factor^(k / every)`，夹在 [lr_min, lr] 之间。
按**绝对轮次**计算，所以中途停止再 `--resume` 不会打乱计划。
每轮的学习率会打印在控制台，并写进 `train_log.csv` 的 `lr` 列。

### 2. 固定锚点评估（判断"到底有没有变强"）

    --anchor versions/goai9x9_v3_33kgames.bin --anchor-every 10 --anchor-games 20

拿一个**永不再变**的网络当尺子，每隔 N 轮和它对战若干局，把胜率写进 CSV 的
`anchor_winrate` 列。因为它不随训练漂移，这条曲线比"对上一版胜率"干净得多 ——
**这是判断训练是否真的在进步的主指标**。

### 3. 退化回滚保护

    --rollback 1 --rollback-patience 3

连续 N 次晋级赛胜率低于 45%（判定练歪了），自动载入 `best.bin` 重新出发，
并把 Adam 的一阶/二阶动量清零，避免带着"坏记忆"继续跑。

### 4. 评估带置信区间

晋级赛和锚点评估都会打印 `±X%` 的 95% 置信区间（`1.96*sqrt(p(1-p)/n)`）。
**看进步时务必对照它**：12 局打出的 58% 和 42% 在统计上没有区别。

### 推荐配置

    ./build/GoAI train --size 9 --channels 32 --forever \
      --resume runs_v4/latest.bin \
      --games 40 --sims 140 --steps 250 --batch 64 \
      --lr 0.005 --lr-decay-every 100 --lr-decay-factor 0.85 --lr-min 0.0005 \
      --threads 10 --evalgames 12 --evalsims 20 \
      --gate 1 --gategames 40 --eval-every 5 \
      --anchor versions/goai9x9_v3_33kgames.bin --anchor-every 10 --anchor-games 20 \
      --rollback 1 --rollback-patience 3 --out runs_v4

### 训练效率测量（13 路 32 通道，160 模拟，10 线程）

每轮 40 局 195 秒的耗时分布：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 自对弈 | 162.6s | **84%** |
| 梯度训练 | 11.3s | 6% |
| 评估/晋级/锚点 | 19.6s | 10% |

结论：
- **瓶颈在自对弈**，不在训练（250 步梯度只占 6%）—— 所以"并行化反向传播"这类优化收益极小
- 自对弈的并行度：单线程 25.8 秒/局 → 10 线程 4.07 秒/局，**加速 6.4 倍**
  （M5 是 4 性能核 + 6 能效核，等效约 6 核，已接近硬件上限）
- 想再快只能让**每次模拟更便宜**（编译优化 / SIMD / 树复用），而不是加线程

### train_log.csv 的列

`iteration, games, positions, policy_loss, value_loss, param_norm, winrate_vs_random,`
`gate_winrate, elapsed_s, lr, anchor_winrate, selfplay_s, train_s, eval_s`

（`gate_winrate` = 对"当前最佳"的胜率，`-1` 表示这一轮没打；`anchor_winrate` 同理）

---

## 换大棋盘：用热启动省掉前期学习

9 路练得差不多了想上 13 路 / 19 路，不必从随机权重开始。
**卷积层学到的棋形知识（气、连接、断点、边角手段）与棋盘大小无关**，
只有策略头/价值头依赖棋盘点数。`tools/warm_start.py` 负责把前者搬过去：

    # 9 路 32 通道 -> 13 路 64 通道（通道数不同时用切片迁移）
    python3 tools/warm_start.py --from versions/goai9x9_v3_33kgames.bin \
        --size 13 --channels 64 --vhidden 64 --out warm13.bin

    # 自检（校验逐值复制是否正确）
    python3 tools/warm_start.py --selftest

输出会明确告诉你哪些层被完整迁移、哪些做了切片迁移、哪些必须重学：

    完整迁移: pb, vb
    切片迁移: c1w(1152/2304), c1b(32/64), c2w(9216/36864), c2b(32/64), pw(64/128), vw(32/64)
    重新学习: pfcw, pfcb, vfc1w, vfc1b, vfc2w, vfc2b

拿到 `warm13.bin` 就能直接开始训练：

    ./build/GoAI train --size 13 --channels 64 --forever \
      --resume warm13.bin --games 40 --sims 200 --steps 250 \
      --lr 0.004 --lr-decay-every 100 --lr-decay-factor 0.85 \
      --threads 10 --gate 1 --gategames 40 --eval-every 5 \
      --anchor warm13.bin --anchor-every 10 --anchor-games 20 --out runs13

> **实测开销（M5，10 线程）**：13 路每局约 180 手（9 路约 110 手），局面数 2.1 倍，
> 搜索树也更大 —— 每轮 40 局约 4.4 分钟，是 9 路（36 秒）的 **约 8 倍**。
> 也就是说 13 路一夜大约能跑 130 轮 / 5,000 局。用 --sims 控制强度，13 路建议 160~300。
> 用 `--sims` 控制强度：13 路建议 200~400。

---

## 性能优化记录

### 卷积前向重写（1.8× ~ 3.0× 提速）

单线程前向微基准（交替测量取稳定值）：

| 规模 | 优化前 | 优化后 | 加速 |
|---|---|---|---|
| 9 路 32 通道 | 0.272 ms | **0.191 ms** | 1.4× |
| 13 路 32 通道 | 0.593 ms | **0.322 ms** | 1.8× |
| 19 路 64 通道 | 6.221 ms | **2.076 ms** | **3.0×** |

**做法**：原来的实现是"逐像素 + 每个像素跳 9 次边界判断"；改成
按 (输出通道, 输入通道) 配对**累加整张输出平面**：

- 内部像素（13 路时占 72%）走**无分支快路径**，边界像素单独处理
- 累加平面只有 n² 个 float（13 路 = 676 字节），常驻 L1
- 内层循环对连续 x 访问，**可被编译器自动向量化**
- 卷积核的 9 个权重先读进局部常量，避免重复寻址

**正确性**：与旧实现输出**逐字节一致**（13 路与 9 路网络分别对拍），
91 项测试全部通过（含有限差分梯度校验）。

**生产环境实测**（13 路 32 通道，每轮 40 局，同一配置前后对照）：

| 阶段 | 优化前 | 优化后 | 加速 |
|---|---|---|---|
| 自对弈 | 162.6s | **88.3s** | **1.84×** |
| 梯度训练 | 11.3s | 6.0s | 1.9× |
| 评估 | 19.6s | 8.3s | 2.4× |
| **整轮合计** | **193.5s** | **102.6s** | **1.89×** |

### 三轮优化累计效果（13 路 32 通道，每轮 40 局，10 线程）

| 优化阶段 | 自对弈 | 加速 | 做法 |
|---|---|---|---|
| 最初（朴素卷积） | 162.6s | 1.00× | 逐像素 + 每像素 9 次边界判断 |
| 第一轮 | 88.3s | 1.84× | 按输出通道累加 + 内部区域无分支 |
| 第二轮 | 44.1s | 3.69× | 3×3 卷积拆成 9 个 tap，每个 tap 整行 axpy |
| 第三轮 | **34.2s** | **4.75×** | 手写 NEON intrinsics（4 宽 FMA） |

整轮耗时 193.5s → **40.5s（4.8×）**，梯度训练 11.3s → 3.0s。
三轮优化后输出与最初**数值完全一致**（最大绝对差 0），91 项测试全程通过。

### 单次模拟的成本构成（优化后）

| 部分 | 耗时 | 占比 |
|---|---|---|
| 网络前向 | 0.156 ms | ~55% |
| 树操作 + 特征编码 + 回传 | ~0.13 ms | ~45% |

下一步优化方向：进一步压网络前向（行宽补到 4 的倍数）、
树复用（同样的模拟次数换更深搜索）。

### 训练耗时分布（13 路 32 通道，10 线程）

| 阶段 | 占比 |
|---|---|
| 自对弈 | **84~90%** |
| 梯度训练（250 步） | 6~13% |
| 评估 + 晋级 + 锚点 | 2~10% |

推论：优化重点应该放在**每次模拟的成本**上（网络前向 + 树操作），
而不是并行化梯度训练 —— 后者最多只能拿回 13%。

### 编译选项实测

`-march=native` 在这台 M5 上**没有稳定收益**（A/B 交替测量落在噪声内）。
Makefile 仍保留 `NATIVE=1` 开关，但默认关闭以保证二进制可跨机器分发。

### A/B 实验：树复用（Reuse）反而有害

树复用是指走完一手后保留那棵子树，下次搜索接着用（Leela Zero 等实现常用）。
理论上同样模拟次数能搜得更深。实测（9 路，从 v4 出发各跑 20 轮，锚点都是 v4）：

| 组 | 第 10 轮锚点 | 第 20 轮锚点 | 策略损失 | 价值损失 |
|---|---|---|---|---|
| 复用关 | 30% | **40%** | 2.671 → 2.707 ↑ | 0.645 → 0.730 ↑ |
| 复用开 | 10% | **15%** | 2.588 → **2.207** ↓ | 0.852 → **0.612** ↓ |

**训练损失明显更好，但实际棋力明显更差。**

原因：复用得越深，搜索给出的策略目标越"尖锐"（集中在一两条最佳线），
网络学会的是**狭窄的确定性走法**而不是"多个好点的分布"，导致

- 策略塌缩：网络对自己的首选着法过度自信，低模拟次数下反而下得差
- 自对弈对局趋同：总是沿同一条线走，训练数据多样性下降

**结论：在本项目的配置（140~160 次模拟 + 小网络）下默认关闭树复用。**
代码与 `--reuse 1` 开关保留，等将来模拟次数大幅提高（如 800+）时可以再验证。

> 一个更普遍的教训：**训练损失不是棋力**。这轮 A/B 里，损失降得最多的那一组
> 棋力反而最差 —— 如果只看损失，会得出完全错误的结论。只有固定锚点对局才看得清。

---

## 优化日志（按时间顺序，全部有实测数据）

| # | 改动 | 结论 | 证据 |
|---|---|---|---|
| 1 | 审计探索机制 | Dirichlet 噪声 + 温度采样**本来就正确** | 代码审查 + 数值验证 |
| 2 | 学习率衰减（--lr-decay-*） | ✅ 采用 | 单元测试 15 项；实测 0.02→0.01→0.005 精确衰减 |
| 3 | 固定锚点评估（--anchor-*） | ✅ 采用，**最重要的工具** | 所有后续实验的判据 |
| 4 | 退化回滚（--rollback） | ✅ 采用 | 实测触发并正确恢复 best.bin |
| 5 | 评估置信区间 | ✅ 采用 | 单元测试 7 项 |
| 6 | 卷积热启动（warm_start.py） | ✅ 采用 | 9路→13路完整迁移卷积层，逐值校验 |
| 7 | 学习率 0.005 vs 0.001 | 0.001 才不退化 | A/B：0.005 锚点 35%、0.001 锚点 75% |
| 8 | 卷积前向重写（累加+无分支） | ✅ 采用 | 生产 162.6s→88.3s（1.84×），输出逐字节一致 |
| 9 | 9-tap axpy 卷积 | ✅ 采用 | 88.3s→44.1s（累计 3.69×） |
| 10 | NEON 手写向量化 | ✅ 采用 | 44.1s→34.2s（累计 4.75×） |
| 11 | 树复用（--reuse） | ❌ **有害，默认关闭** | A/B：损失降更多但锚点 15% vs 40% |
| 12 | 模拟次数 160 vs 320 | 见下节 | A/B 进行中 |

### 一条贯穿始终的教训

**训练损失不是棋力。** 第 11 项（树复用）里，损失降得最多的一组棋力反而最差。
只有"固定锚点 + 置信区间"的对局数据能反映真实强弱 —— 这也是本项目坚持
每项改动都做 A/B 的原因。

## 项目结构

```
src/board.c        围棋规则：气、提子、打劫、禁入点、中国规则数子
src/net.c          卷积神经网络：前向/反向手写实现，Adam 优化器
src/mcts.c         PUCT 蒙特卡洛树搜索（含批量推理的两阶段接口）
src/train.c        自对弈训练主循环、样本回放、胜率评估、晋级机制
src/apps.c         GTP 协议 / 终端对弈 / 棋谱导出 / 引擎对战
src/goboard.c      独立对弈程序（ncurses 全屏 + 纯文本双界面，权重可编译进程序）
src/compat.c       跨平台（文件/时间/线程/目录）
src/sockets.c      TCP 客户端（只有连 GPU 推理服务时才编译）
gpu/server.py      GPU 推理+训练服务端（攒批推理、权重热更新）
gpu/train.py       纯 Python/PyTorch 训练器（另一条路，适合快速试验）
gpu/goai_gpu/      规则镜像、模型定义、权重导入导出（与 C 端 .bin 格式互通）
standalone/        可直接发给别人玩的独立包（含预编译 Windows exe）
versions/          冻结的历史权重
tests/             69 项单元测试（make test）
tools/             打包、画图、Xcode 工程生成等脚本
```

## 权重文件格式

C 端与 Python 端互通的自定义格式，便于跨平台交换：

    uint32 magic('GOAI') | int32 size, planes, channels, vhidden, n_params | float32 参数...

小端 float32，**macOS / Windows / Linux 通用**。把别人练好的 `.bin` 拷过来就能用：
`./build/GoBoard --weights 别人的.bin`

## 常见问题

**Q：训练要多久才能赢我？**
9 路、8 线程，约 1 小时（3000 局左右）就能稳定赢随机走子；要有点棋形概念大概需要几万局。

**Q：为什么不用 PyTorch 训练？**
C 版不依赖任何框架，任何机器 clone 下来就能跑。PyTorch 版（`gpu/`）是给有显卡的人加速用的，两者权重互通。

**Q：能下 19 路吗？**
可以（`--size 19`），但仓库里的权重都是 9 路的。19 路需要自己训练（更慢，建议有 GPU）。

**Q：AI 什么水平？**
诚实地说：**初学者水平**。它能吃子、做眼、应手，但没有定式体系。作为一个"从零实现 + 能自我进化"的工程样本，它的价值在代码和训练流程本身。

---

## 自动同步到 GitHub

改代码后自动推送（macOS）：

   双击 启动自动同步.command        # 开一个终端窗口守着，每 60 秒检查一次，有改动就推送
   （关掉窗口或 Ctrl+C 停止）

手动同步：双击 sync-to-github.command，或运行 bash tools/sync_github.sh

> **为什么不用 macOS 后台任务（launchd）？**
> macOS 的安全机制（TCC）不允许后台任务读取「文档 / 桌面 / 下载」目录下的文件，
> 放在这些目录里的项目用 launchd 会报 Operation not permitted。
> 终端窗口有权限，所以自动同步做成终端守护进程。若把项目放在 `~/GoAI` 这类非受保护目录，
> 也可以用 `bash tools/autosync.sh install --force` 装成开机后台任务。
>
> Windows / Linux：把 `tools/sync_github.sh` 加进任务计划程序或 cron 即可，逻辑一样。

## 原来的自动同步到 GitHub

本仓库自带"改了代码就自动推送"的机制（macOS）：

    bash tools/autosync.sh install     # 安装后台任务（只需一次）
    bash tools/autosync.sh status      # 查看状态和最近日志
    bash tools/autosync.sh uninstall   # 卸载

安装后：`src/ include/ gpu/ tools/ standalone/ versions/` 任何文件一改动就会自动提交并推送，
另外每 10 分钟兜底检查一次。同步日志在 `tools/autosync.log`。

不想用后台任务也可以手动同步：双击 `sync-to-github.command`（macOS）或运行 `bash tools/sync_github.sh`。

> Windows / Linux 用户：把 `tools/sync_github.sh` 挂到任务计划程序 / cron 里即可，逻辑完全一样。

## 许可

MIT License，随便用。如果你用它训练出了更强的网络，欢迎分享权重文件。

---

## English

A from-scratch Go (Weiqi) AI in pure C11: board rules, PUCT MCTS, and a convolutional
policy/value network with hand-written forward/backward passes — no ML framework required.

- **Play now**: `standalone/PLAY_NOW.bat` (Windows, prebuilt exe, AI weights compiled in) or `make goboard`
- **Train yourself**: `./build/GoAI train --size 9 --channels 32 --forever --threads 8 --out runs_live`
- **With an NVIDIA GPU**: `python gpu/server.py --device cuda` + `./build/GoAI gtrain --remote 127.0.0.1:8899`
- **Measured**: 35,320 self-play games in 8.6 h on an 8-core laptop → 100% vs random, 98.8% vs the previous generation
- **Tests**: `make test` (69 checks). License: MIT.

Weights use a simple little-endian float32 `.bin` format shared between the C and Python sides.
