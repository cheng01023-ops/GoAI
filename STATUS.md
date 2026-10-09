# 当前实验状态（自动更新文件，可安全修改）

> 这个文件记录**正在进行中**的实验，方便任何人/任何会话立刻接手。
> 历史结论见 README 的「核心结论汇总」与「优化日志」。

## 正在跑：残差块实验（runs_blk2）

**要回答的问题**：给网络加残差块（增加表达力）能否让棋力突破瓶颈？

**背景**：此前三次实验证明"已收敛网络继续训练会退化"（锚点 40% / 28% / 42%），
且与学习率无关、加宽数据只能缓解 → 推断根因是**网络只有 2 层卷积、表达力不足**。

**配置**：
    ./build/GoAI train --size 13 --channels 32 --blocks 2 --forever \
      --resume runs13/best.bin --games 100 --sims 160 --steps 250 --batch 64 \
      --lr 0.001 --lr-decay-every 40 --lr-decay-factor 0.9 --lr-min 0.00005 \
      --threads 10 --evalgames 12 --evalsims 20 --gate 1 --gategames 40 --eval-every 5 \
      --anchor versions/goai13x13_warm.bin --anchor-every 10 --anchor-games 30 \
      --rollback 1 --rollback-patience 3 --reuse 0 --open-plies 4 --sgf 2 --plot 1 \
      --out runs_blk2

**判据**：第 10 轮、第 20 轮的**锚点胜率**（对固定基准 `versions/goai13x13_warm.bin`）
- 突破 50% 并持续上升 → 容量假设成立，继续加大（更多块 / 更多通道）
- 仍在 50% 附近或更低 → 容量不是瓶颈，需要换方向（监督预训练 / 不同目标函数）

**关键前置修复**：残差块**必须零初始化**（第二个卷积置 0，使块初始为恒等映射）。
第一次实验（runs_blk/）没做这个，网络被打坏：对随机胜率 90% → 12%、锚点 0.0%。
修复后：对随机 91.7%、每局 196 手 ✓

**当前进度**：第 2 轮完成（每轮约 8 分钟）

**进度换算**（便于判断是不是卡住了）：
- 每局约 3 秒（10 线程聚合）、每轮 100 局约 7~8 分钟
- 第 5 轮出晋级赛、第 10/20 轮出锚点
- 注意：会话检查间隔可能只有十几秒，看不到变化是正常的，别误判卡死；
  判断是否在跑请看 runs_blk2/status.txt 的 updated 时间戳与 game 计数

**环境注意事项（实测）**：
- 系统内存吃紧（换出量超 1,000,000 页约 16GB，可用常低于 200MB）会让每轮从 8 分钟
  拉长到 9 分钟以上、CPU 从约 900% 降到约 690% —— 这是环境问题，不是程序 bug
- 训练进程本身只占约 65MB，内存大户是浏览器/微信/VS Code 等
- 若需要提速：减少 --threads 或 --games（但会改变实验条件，需重新评估）

## 怎么查

    # 学习曲线（策略/价值损失、对随机胜率、晋级、锚点）
    python3 -c "import csv;[print(r) for r in csv.DictReader(open('runs_blk2/train_log.csv'))]"

    # 实时状态
    cat runs_blk2/status.txt

    # 后台组件是否还在
    pgrep -fl "GoAI|goai_watchdog|mon_blk2|watch_sync"

## 已知故障：卡死（进程活着但某一局不再推进）

**症状**：`runs_blk2/status.txt` 的 `updated` 时间戳长期不变，进程 CPU 只有约 100%
（单线程）—— 说明 9 个线程已完成、最后 1 个卡在某局棋里。

**为什么普通看门狗没抓到**：它只检查进程是否存在，而进程一直活着。
**处理**：已加第二层「停滞看门狗」`/tmp/goai_stall_watch.sh`：
每 30 秒采样 `updated`，连续 12 次（6 分钟）不变就杀掉重启。

**如果再次发生**：
1. 看 `/tmp/goai_stall.log` 有没有重启记录
2. 若频繁卡死，可能是某类局面触发搜索异常 → 需要复现那一局（保存 SGF 可回放）
3. 临时缓解：降低 `--sims` 或给单局加时间上限

## 后台组件（都已 nohup 分离，可无人值守）

| 组件 | 作用 |
|---|---|
| `GoAI train --out runs_blk2` | 训练主进程 |
| `/tmp/goai_watchdog.sh` | 进程意外退出时按原配置自动重启 |
| `/tmp/mon_blk2.sh` | 跑到第 12 轮后把学习曲线写入 `/tmp/blk2_report.txt` |
| `tools/watch_sync.sh 60` | 每 60 秒提交并推送 GitHub |

## 已知注意事项

- 机器内存吃紧（16GB，可用约 150MB）→ 若变慢先减 `--games` 或 `--threads`
- **做 A/B 必须独占机器**，否则测量无效（曾出现同一版本差 66%）
- 换源码后构建要 `rm -f build/net.o`，并用 md5 确认两个二进制不同

