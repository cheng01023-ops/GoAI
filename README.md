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
