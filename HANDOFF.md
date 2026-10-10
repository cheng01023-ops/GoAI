# 交接文档（给新会话用）

> 在新会话里说一句：**读 HANDOFF.md，按里面的优先级开始干活** —— 就够了。

## 一、当前成果（已存档，可复现）

| 模型 | 说明 | 实测 |
|---|---|---|
| versions/goai9x9_v5_800sims.bin | **当前最强**（800 模拟练 139 轮 / 27,800 局）| 对 v4 **81.7%**（60 局成对开局：47胜9负4和）|
| versions/goai9x9_v4_lr001.bin | 上一代 | 对 v3 66.2% |
| versions/goai9x9_v3_33kgames.bin | 原始基线 | — |

## 二、已验证的关键规律（别再踩）

1. **搜索质量决定自对弈成败**：模拟 140 → 800，同样网络同样数据量，锚点从 40%（退化）变成 70%（持续提升）
2. **训练损失 ≠ 棋力**：损失降得最多的那轮棋力最差（树复用实验）
3. **评估必须用成对开局**：A vs A 从 ±12.5 点波动变成稳定 50%
4. **A/B 必须独占机器 + 交替测量**：曾出现同一版本差 66% 的假测量
5. **换源码后要 rm -f build/net.o 强制重编**，并用 md5 确认两个二进制真的不同

## 三、下一步优先级

### 第一优先：批量推理 + MPS（本机算力最大杠杆）

单点送 GPU 一定更慢（一次前向 0.087ms vs GPU 启动开销 0.05~0.1ms）；
批量之后完全反过来：batch 64 → 每次前向约 3us ≈ **30 倍**。
而模拟次数是唯一验证有效的杠杆 → 前向快 30 倍就能跑 5000+ 模拟。

做法：
1. 先做 CPU 批量版（验证批量结果与单点结果逐值一致）
2. mcts.c 改成批量送叶子：各线程把待评估局面推进队列，攒够 N 个一起算
3. 接 PyTorch/MPS 服务端（仓库已有 gpu/server.py + socket 协议，加 batch 接口）
4. 实测加速比，再用高模拟次数跑锚点实验

### 第二优先：第 5 步 辅助目标（设计已写好）

见 STEP5_DESIGN.md：领地辅助头 + 值目标分布化。
特点：注入外部信息、不依赖搜索质量 —— 唯一能绕开"弱老师"死循环的路。

### 第三优先：第 4 步 网络规模

9 路 32 → 64 通道（+2 残差块）。用 tools/warm_start.py 做通道切片热启动。
注意：64 通道前向约 4 倍 → 配合批量推理才划算。

## 四、怎么控制训练（用户可随时启停）

- 停止：touch runs9_s800/STOP → 30 秒内停，且**不会被自动重启**
- 开始：rm runs9_s800/STOP → 看门狗 8 分钟内自动拉起；或直接跑下面的完整命令
- 桌面按钮（开始训练 / 停止训练）**已配好**指向本工作区与 800 模拟配置
- ⚠️ 不要再挂不尊重 STOP 的看门狗（上次害得用户关不掉训练）

完整命令：

    cd ~/Documents/deepseek-harness/default-workspace/GoAI && rm -f runs9_s800/STOP && \
    nohup nice -n 5 caffeinate -s -i ./build/GoAI train --size 9 --channels 32 --forever \
      --resume runs9_s800/best.bin --games 200 --sims 800 --steps 250 --batch 64 \
      --lr 0.0003 --lr-decay-every 30 --lr-decay-factor 0.9 --lr-min 0.00002 \
      --threads 10 --evalgames 12 --evalsims 200 --gate 1 --gategames 40 --eval-every 5 \
      --anchor versions/goai9x9_v4_lr001.bin --anchor-every 10 --anchor-games 40 \
      --rollback 1 --rollback-patience 3 --reuse 0 --open-plies 4 --sgf 2 --plot 1 \
      --out runs9_s800 >> runs9_s800_console.log 2>&1 &

## 五、当前机器状态

- 训练：已停止（runs9_s800/STOP 存在）—— 新会话可自行决定是否恢复
- 看门狗：/tmp/goai_stall2.sh（STOP 感知版）
- 测试：106 项全过 / GitHub 已同步
- 内存吃紧（16GB，曾换出 15GB）→ 变慢先查内存

## 六、关键文件

| 文件 | 内容 |
|---|---|
| README.md | 全部实验结论 + 优化日志 + 复现规范 |
| STATUS.md | 在跑实验的状态与排查方法 |
| STEP5_DESIGN.md | 第 5 步设计 + 10 步实施清单 |
| HANDOFF.md | 本文件 |
