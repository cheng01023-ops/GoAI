# GoBoard —— 一个自带 AI 的围棋程序（C 语言）

这是一个**单文件即可运行**的围棋人机对战程序：神经网络已经**编译进程序内部**，
不需要 Python、不需要显卡、不需要任何模型文件。编译出来一个 `GoBoard.exe`（约 220 KB）就能下棋。

- 棋盘：9 路（内置权重就是 9 路的）
- 规则：中国规则，贴目 7.0，禁入点 / 打劫 / 停着 / 数子全部正常
- AI：AlphaZero 风格（策略+价值神经网络 + 蒙特卡洛树搜索），**v5 权重：自我对弈 27,800 局、
  每步 800 次模拟**（目前最强存档，对上一代 v4 实测 81.7%）
- 想要更强的对手：`--sims 400`；想换弱一点的：`--weights weights/goai9x9_v3_33kgames.bin`
- 中文速查在这份：[怎么玩.md](怎么玩.md)（双击哪个文件、按键、怎么换权重、怎么改代码）

---

## Windows 上怎么跑（四选一）

### 方法 0：什么都不装，直接玩（最快）

**双击 PLAY_NOW.bat** —— 包里已经带了编译好的 GoBoard.exe（Windows 64 位），
不需要装编译器、不需要装 Python、不需要联网。解压 → 双击 → 开始下棋。

    GoBoard.exe      299 KB，AI 权重已编译进程序内部
    只依赖 Windows 自带的系统 DLL（KERNEL32 + UCRT），Win10/11 直接能跑

### 方法 1：从源码编译（改了代码就用这个）

### 方法 1：双击脚本（最简单）

1. **装编译器**（只需要一次，二选一）：
   - **免安装**：下载 [w64devkit](https://github.com/skeeto/w64devkit/releases)（约 60 MB 的 zip），
     解压到任意目录，双击里面的 `w64devkit.exe`，会开一个带 gcc 的命令行窗口
   - **或者装 MSYS2**：[msys2.org](https://www.msys2.org/) 装完后在 MSYS2 MINGW64 窗口里执行
     `pacman -S --noconfirm mingw-w64-x86_64-gcc`
2. 把本文件夹放到任意位置（路径**不要**有中文，避免踩坑）
3. **双击 `BUILD_AND_PLAY.bat`** —— 会自动编译并开始下棋

### 方法 2：VS Code（推荐给要改代码的人）

1. 用 VS Code 打开**本文件夹**（File → Open Folder）
2. 装扩展：**C/C++**（ms-vscode.cpptools）
3. 按 **F5**，选「▶ 对弈（Windows）」→ 自动编译并在终端里开始下棋
4. 想自己敲命令：VS Code 里按 `Ctrl+``` 开终端，然后
   ```
   gcc -O3 -ffast-math -std=c11 -DGOAI_NO_CURSES -Iinclude src/goboard.c src/board.c src/compat.c src/net.c src/mcts.c -o GoBoard.exe -lm
   GoBoard.exe --sims 200
   ```

### 方法 3：命令行手动编译

```bat
mingw32-make -f Makefile.win goboard
GoBoard.exe --sims 200
```

> 提示：Windows 版用的是**纯文本界面**，不需要 ncurses，也不需要 pthread，只要有 gcc 就能编译。
> macOS / Linux 上则可以用 `make goboard` 编译出带全屏棋盘（方向键操作）的版本。

---

## 怎么下

启动后会打印棋盘，黑棋是 `X`，白棋是 `O`，`[X]` 表示最后一手。

```
    A B C D E F G H J
 9  .  .  .  .  .  .  .  .  .  9
 8  .  .  .  .  .  .  .  .  .  8
 ...
  落子 >
```

在提示符后面输入坐标即可：

| 输入 | 作用 |
|---|---|
| `D4` | 在 D4 落子（列用字母 A-J，跳过 I；行用数字 1-9，1 在最下面） |
| `pass` | 停着（双方都停着就终局数子） |
| `undo` | 悔棋（退回你和 AI 各一手） |
| `hint` | 让 AI 告诉你它建议下哪里 |
| `save` | 保存棋谱（SGF 格式，可以拖进 Sabaki 等软件复盘） |
| `new` | 重新开一局 |
| `resign` | 认输 |
| `quit` | 退出 |

macOS/Linux 的全屏版用**方向键移动光标、回车落子**，其他按键一样（键盘上 p/u/i/s/n/r/q）。

---

## 常用参数

```bat
GoBoard.exe                     :: 默认：9 路，AI 每步思考 200 次
GoBoard.exe --sims 600          :: AI 更强（思考 600 次，慢一点）
GoBoard.exe --sims 60           :: AI 更弱更快（适合新手）
GoBoard.exe --white             :: 你执白，AI 先行
GoBoard.exe --handicap 4        :: 让你 4 子
GoBoard.exe --weights 别的.bin  :: 换一个训练好的网络（见下）
```

## 换成你自己训练的网络

程序内置的是 9 路 32 通道的网络。如果你自己训练出了新的 `.bin` 权重（本项目同门的训练器会导出这种格式）：

```bat
GoBoard.exe --weights runs_live\best.bin
```

只要棋盘大小一致（都是 9 路），通道数可以不同，程序会自动适配。

**注意**：内置权重只有 9 路的。如果你想下 13 路或 19 路，需要一个对应棋盘的网络文件，
并且用 `--size 13` 启动 —— 用 9 路网络下 13 路是不允许的（程序会拒绝加载）。

---

## 文件说明

```
src/goboard.c            对弈程序主体（界面 + 对局流程 + 权重热加载）
src/board.c              围棋规则（气、提子、劫、禁入点、数子）
src/net.c                神经网络前向推理（含内置权重加载）
src/mcts.c               蒙特卡洛树搜索（PUCT）
src/compat.c             跨平台小工具（文件、时间、线程）
include/weights_builtin.h  ⭐ AI 权重（编译进程序里，44 万行数字）
Makefile / Makefile.win   macOS/Linux 与 Windows 的编译脚本
BUILD_AND_PLAY.bat      Windows 双击即用
```


## 小提示：为什么脚本名是英文的

BUILD_AND_PLAY.bat / BUILD.bat / PLAY.bat 用英文命名，是为了避免中文文件名在压缩包跨系统传输时
乱码（Windows 解压后可能变成乱字符，导致双击没反应）。程序界面和输出全部是中文，不受影响。

如果双击 .bat 没反应，可能是 Windows 的"网络文件锁定"：右键 → 属性 → 勾选"解除锁定" → 确定。

## 常见问题

**Q：报错 "gcc 不是内部或外部命令"**
A：没装编译器，或者装了但没加进 PATH。回到「方法 1」装 w64devkit / MSYS2。

**Q：编译报错说找不到 pthread / ncurses**
A：Windows 版不需要它们。请确认编译命令里带了 `-DGOAI_NO_CURSES`（.bat 和 VS Code 配置里都已经带上了）。

**Q：AI 太慢**
A：把 `--sims` 调小（比如 60）。9 路上 200 次模拟通常在 0.1 秒内。

**Q：能下 19 路吗**
A：引擎支持 9/13/19，但需要对应棋盘的训练权重。内置的是 9 路。

---

祝你玩得开心。有任何问题把报错原文发我。
