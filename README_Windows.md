# GoAI on Windows —— 给朋友的说明

这套东西是**一个自己会下围棋的 AI**（C 语言，蒙特卡洛树搜索 + 神经网络），
可以在这台 Windows 机器上**持续自我对弈、自我变强**，还能用 5080 加速。

只有三步：**装环境 → 训练 → 看进度**。下面按"最省事"到"最榨干显卡"排列。

---

## 方案 A：WSL2（推荐，10 分钟搞定，源码零改动）

WSL2 里是真正的 Linux，C 代码可以直接编译运行，多核全用上。

1. 以管理员身份打开 PowerShell：
   ```powershell
   wsl --install -d Ubuntu
   ```
   装完重启，按提示设置 Linux 用户名密码。
2. 进入 WSL（开始菜单搜 Ubuntu），安装编译工具：
   ```bash
   sudo apt update && sudo apt install -y build-essential python3
   ```
3. 把本文件夹拷进 WSL（在 Windows 里该文件夹路径假设是 `C:\\GoAI-Windows`）：
   ```bash
   cd /mnt/c/GoAI-Windows
   make -s all && make -s test        # 编译 + 跑 69 项单元测试
   ```
4. 开始持续训练（CPU 多线程，后台低优先级）：
   ```bash
   nice -n 10 ./build/GoAI train --size 9 --channels 32 --forever --resume runs_v2/latest.bin \
        --games 40 --sims 140 --steps 250 --batch 64 --lr 0.01 \
        --evalgames 12 --evalsims 40 --threads 8 --gate 1 --gategames 20 --eval-every 3 \
        --plot 1 --out runs_live
   ```
   想要一键脚本就用 `bash 开始训练.sh`（本包里也有 Linux/WSL 版脚本）。
5. 停止：`touch runs_live/STOP`（或跑 `bash 停止训练.sh`），会在当前一轮结束后保存权重。

**性能**：取决于 CPU 核数。8 线程 9 路 32 通道大约 **500 局/分钟**量级，
比 MacBook 上快 2～4 倍。**注意：这条路用不到 5080**，显卡会闲着。

---

## 方案 B：MSYS2 / MinGW-w64（原生 Windows 编译，不需要 WSL）

1. 安装 [MSYS2](https://www.msys2.org/)，打开 "MSYS2 MINGW64" 终端：
   ```bash
   pacman -S --noconfirm mingw-w64-x86_64-gcc make
   ```
2. 进入本目录（`cd /c/GoAI-Windows`），编译与测试：
   ```bash
   make -f Makefile.win -s all
   make -f Makefile.win test
   ```
3. 训练（Windows 原生进程，双击 `START_TRAINING.bat` 也可以）：
   ```bash
   ./build/GoAI.exe train --size 9 --channels 32 --forever --resume runs_v2/latest.bin \
        --games 40 --sims 140 --steps 250 --batch 64 --lr 0.01 \
        --evalgames 12 --evalsims 40 --threads 8 --gate 1 --gategames 20 --eval-every 3 \
        --plot 1 --out runs_live
   ```
4. 双击 `CHECK_PROGRESS.bat` 看实时状态；双击 `STOP_TRAINING.bat` 优雅停止（会先保存权重）。

> 说明：Windows 版的"一键停止"用的是 **STOP 文件**机制（`runs_live/STOP`），
> 因为 Windows 没有 POSIX 信号；引擎每局之间检查一次，所以通常几秒内停下。

---

## 方案 C：用 5080 训练（GPU 版，这才是显卡该干的活）

C 引擎是**纯 CPU**的，5080 在方案 A/B 里完全用不上。
`gpu/` 目录里是 **PyTorch 版训练器**：几百盘棋同时自对弈、每轮把几百个局面拼成
一个 batch 送上显卡，回放缓冲常驻显存、混合精度、8 重对称增强——专门为了喂饱 5080。

### 安装（一次性，约 10 分钟）

```bat
cd gpu
install_windows.bat
```
它会创建虚拟环境并安装 **CUDA 版 PyTorch**（默认 cu128，5080 需要较新的 CUDA；
如果装完 `python -c "import torch;print(torch.cuda.is_available())"` 是 False，
把 `install_windows.bat` 里的 cu128 换成 cu126/cu129 再跑一次）。

### 训练（推荐 19 路 + 大网络，才能吃满显卡）

```bat
cd gpu
train_windows.bat
```
默认参数（可在 bat 里改）：

| 参数 | 说明 |
|------|------|
| `--size 19 --channels 128` | 19 路、128 通道（9 路 32 通道对 5080 来说太小，跑不满） |
| `--games 256` | 同时自对弈 256 盘（**这是喂饱 GPU 的关键，显存够就往上加**） |
| `--sims 64` | 每步 MCTS 模拟次数（越多越强，也越慢） |
| `--batch 512` | 训练批大小（5080 32GB 显存可以到 1024～2048） |
| `--amp bf16` | 混合精度，速度翻倍（5080 的 BF16 很强） |
| `--buffer 300000` | 回放缓冲局面数，常驻显存（19 路 4 平面 30 万局面约占 1.7GB） |

```bat
python train.py --size 19 --channels 128 --games 256 --sims 64 ^
    --iters 0 --steps 400 --batch 512 --lr 2e-3 --amp bf16 --device cuda ^
    --eval-games 16 --gate-every 3 --out runs_gpu
```
`--iters 0` = 无限循环（一直练到你叫停）。

### 停止 / 看进度
- 停止：双击 `gpu\stop_windows.bat`（写 `runs_gpu\STOP`，当前轮结束后退出）
- 进度：`runs_gpu\status.txt`、`runs_gpu\train_log.csv`

### 和 C 引擎互通（重要）
GPU 版每轮都会导出 `runs_gpu/latest.bin` 和 `best.bin`，格式与 C 引擎**完全兼容**
（已做逐位往返测试）。所以在 5080 上练出来的网络，可以直接拿给 C 版棋手用：

```bash
./build/GoAI eval --a runs_gpu/latest.bin --b random --games 40 --sims 100
./build/GoAI play --weights runs_gpu/latest.bin --sims 400      # 人机对弈
```



---

## 方案 D（推荐，真正让 5080 干活）：C 搜索 + GPU 推理服务

前三套方案里，**搜索树和围棋规则要么在 CPU 的 C 里、要么在 Python 里**，显卡只能干等。
方案 D 把两者拆开，各干各擅长的：

```
   C 引擎（16 线程 × 64 盘 = 1024 盘并行自对弈）
   规则、提子、劫、搜索树 —— 全是 C，微秒级
        │  每一轮把这一批叶子（最多 2048 个局面）一次发给服务端
        ▼
   GPU 服务端（PyTorch）
   攒批 → 一次前向 → 策略/价值广播回去；同时用回传样本在 GPU 上训练，
   每 60 秒导出 latest.bin 并热切换推理权重（对弈端无需重启）
```

### 本机实测（CPU 服务端，作为下限参考）

| 指标 | 数值 |
|------|------|
| 服务端评估速度 | **7,960 局面/秒**（19 路 32 通道，CPU） |
| 推理忙碌占比 | **98.8%**（说明攒批成功，服务端几乎一直在算） |
| 并行棋局 | 64 盘（4 线程 × 16 盘） |
| 训练 | 1,899 步 / 125,880 样本，策略损失降到 2.35，权重已热更新 |
| 导出 | `latest.bin` / `best.bin`（C 兼容格式，随时可拿来下棋） |

**在 5080 上估算**：19 路 128 通道、batch 512 时，一次前向约 1.5 ms，
而服务端每批的 Python 开销约 2–4 ms → **显卡利用率约 30–60%**（旧方案只有 8–10%）。
网络越大、batch 越大，利用率越高 —— 所以 5080 上请用 19 路 + 128/256 通道 + 1024 盘并行。

### Windows 上怎么跑

```bat
:: 1) 装环境（一次性）
gpu\install_windows.bat

:: 2) 编译 C 引擎（MSYS2 MINGW64 终端）
make -f Makefile.win

:: 3) 起 GPU 服务端（双击或命令行）
gpu\start_server_windows.bat

:: 4) 另开一个终端跑自对弈（双击或命令行）
gpu\train_gpu_windows.bat
::    等价命令：
::    build\GoAI.exe gtrain --remote 127.0.0.1:8899 --threads 16 --slots 64 --sims 64 --size 19 --out runs_gpu

:: 5) 停止：双击 gpu\stop_windows.bat（写 runs_gpu\STOP，当前一轮结束后退出）
```

### 调参（想更榨干显卡就调这三个）

| 参数 | 位置 | 说明 |
|------|------|------|
| `--slots` × `--threads` | 对弈端 | 并行棋局数 = 每批送进 GPU 的局面数。**1024~2048 最好**（显存够就往上加） |
| `--max-batch` | 服务端 | 一次前向最多合并多少局面，设 2048 以上 |
| `--channels` | 服务端 | 128/256 —— 网络越大，显卡干活占比越高 |

服务端每 10 秒打印一行统计（评估速度、训练步数、上传样本、**推理忙碌占比**），
直接看这个百分比就知道显卡用了几成。GPU 上还可以开另一个窗口跑 `nvidia-smi dmon` 看真实利用率。

### 两种训练模式怎么选

| 模式 | 命令 | 适合 |
|------|------|------|
| **纯 GPU 版**（Python 搜索） | `python gpu/train.py --device cuda` | 快速试跑、网络很小的时候 |
| **方案 D**（C 搜索 + GPU 服务） | `gpu\start_server_windows.bat` + `gpu\train_gpu_windows.bat` | **正式训练，显卡利用率高 3~6 倍** |

---

## ⚠️ 关于"能不能榨干 5080"——先看实测数据

GPU 版确实把**每一次模拟的评估都批量送上了显卡**，但**它现在还不能榨干 5080**。
原因很直接：搜索树和围棋规则跑在 Python 里，而显卡每次只干很少的活。

实测（本机 Python 3.14 / torch 2.14，纯 CPU 侧成本）：

| 项目 | 9 路 | 19 路 |
|------|------|-------|
| Python 侧（搜索树 + 规则）每次模拟 | **38 µs** | **129 µs** |
| 5080 在 batch 256 时每次模拟分摊到的推理时间（估算） | ~5 µs | ~12 µs |
| **显卡利用率（估算）** | **≈ 10%** | **≈ 8%** |

也就是说：**现在跑起来，5080 大概只用上一成**，其余时间都在等 Python。
训练器每轮会打印一行实测值，你可以直接看：

```
└ 自对弈细分：推理 8.3%（1204 次批量前向，平均 batch 256）｜树搜索+规则 91.7%
```

### 想真正榨干，需要改架构（这是下一步该做的事）

把搜索从 Python 挪回 **C 引擎**（我们已经有一个很快的、69 项测试全过的 C 引擎），
再用一个 **GPU 推理服务**批量供货：

```
   C 引擎（几百盘棋并行搜索，规则/树用 C 跑，微秒级）
        │  每个叶子需要评估时，把 4×N×N 特征丢进队列
        ▼
   GPU 推理服务（攒够 256 个请求 → 一次前向 → 广播回策略/价值）
```

粗算：19 路 256 通道 + 256 盘并行时，每轮 = C 搜索 2.5 ms + GPU 前向 3 ms
→ 显卡利用率可到 **50% 左右**，整体吞吐比现在这版高 **20~50 倍**。

在改架构之前，**更实际的做法是先用 CPU 版**：包里的 C 引擎在 8 线程下
9 路 32 通道就有 ~500 局/分钟，Scaling 到 16 核台式机 CPU 大约能到 1000+ 局/分钟，
对 9 路来说已经够练出能陪你下棋的棋力了。

要我把上面的"C 引擎 + GPU 推理服务"做出来吗？那是真正让 5080 干活的做法。

---

## 常见问题

**Q: 编译报错 `__uint128_t` / `sysconf` / `pthread`？**
A: 那是旧版代码的问题；本包已经加了跨平台层（`include/compat.h`），
   MSYS2(gcc) / WSL(gcc/clang) / macOS(clang) 都能编。MSVC(`cl.exe`) 不支持，请用方案 A/B。

**Q: 训练跑满 CPU 会不会卡？**
A: 用的是低优先级 + 少留 2 个核（`--threads`）；WSL 下想更温和就 `nice -n 19` 并把
   `--threads` 调小。GPU 版基本不吃 CPU，但会占显存。

**Q: 怎么知道自己练得对不对？**
A: 看 `runs_live/train_log.csv`：`winrate_vs_random` 应该从 30~50% 涨到 80%+；
   不涨就先提高 `--sims`（比如 140→300）再练。

**Q: 能不能直接接着我朋友（Mac）练出来的权重继续练？**
A: 可以。包里 `runs_v2/latest.bin` 是 9 路 32 通道的已有权重：
   C 版用 `--resume runs_v2/latest.bin`；GPU 版用 `--resume runs_v2/latest.bin`
   （通道数要一致，即 `--channels 32`）。

**Q: 训练要多久才有意思？**
A: 9 路 32 通道在 CPU 上几小时能明显强于随机；想在 19 路跟人下得像样，
   建议 GPU 版跑 10⁵～10⁶ 局（5080 上大约几天），期间可以随时停下来试下棋。

---

## 目录说明

```
GoAI-Windows/
├── src/ include/ tests/      C 引擎源码（跨平台）
├── Makefile.win              MSYS2/MinGW 构建
├── START_TRAINING.bat STOP_TRAINING.bat CHECK_PROGRESS.bat   Windows 一键脚本
├── gpu/                      PyTorch GPU 训练器（5080 用这个）
│   ├── install_windows.bat   安装 CUDA 版 PyTorch
│   ├── train_windows.bat     一键开始 GPU 训练
│   ├── train.py              批量自对弈 + 训练 + 导出 C 兼容权重
│   └── goai_gpu/             规则/网络/MCTS/权重互转
├── runs_v2/latest.bin        已有的 9 路 32 通道权重（可继续训练）
├── tools/plot.py             画训练曲线（纯标准库）
└── README.md report.md       完整文档与实验报告
```
