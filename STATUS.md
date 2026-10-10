# 当前实验状态（自动更新文件，可安全修改）

> 这个文件记录**正在进行中**的实验，方便任何人/任何会话立刻接手。
> 历史结论见 README 的「核心结论汇总」与「优化日志」。

## 刚做完：第 ⑤ 步（领地辅助头 + 值分布化）—— runs_step5

**结论：实现与校验全部通过；在"从已收敛 s800 继续微调"的口径下棋力没有提升。**

**配置**（200 局/轮，800 模拟/手；锚点每轮 40 局、成对开局）：

    ./build/GoAI train --size 9 --channels 32 --forever \
      --resume versions/step5_hotstart_9x9.bin --own-weight 0.15 --vdist 1 \
      --games 200 --sims 800 --steps 250 --batch 64 \
      --lr 0.0002 --lr-decay-every 30 --lr-decay-factor 0.9 --lr-min 0.00002 \
      --threads 10 --evalgames 12 --evalsims 200 --gate 1 --gategames 40 --eval-every 5 \
      --anchor versions/goai9x9_v5_800sims.bin --anchor-every 1 --anchor-games 40 \
      --rollback 1 --rollback-patience 3 --open-plies 4 --sgf 0 --plot 1 --out runs_step5

**结果**（5 轮 / 1000 局，已优雅停止，权重 `runs_step5/latest.bin`）：

| 轮 | 价值 CE | 领地 BCE | 锚点(对 s800，40 局成对开局) |
|---|---|---|---|
| 1 | 2.515 | 0.6735 | 45.0% |
| 2 | 1.994 | 0.6438 | 47.5% |
| 3 | 1.654 | 0.6250 | 40.0% |
| 4 | 1.346 | 0.6102 | 35.0% |
| 5 | 1.168 | 0.5973 | 40.0% |

成对开局对拉（200 模拟）：训练后 vs 基线 **48.8%**（120 局，±8.9）；
训练后 vs 起点 37.5%（60 局）；起点 vs 基线 45.0%（60 局）。

**下一步（建议）**：不要再在 s800 上微调来验证第 ⑤ 步 ——
① 从更早检查点（`versions/goai9x9_v4_lr001.bin`）或从零开始，让辅助目标参与表示学习；
② 扫 `--own-weight 0.3`（领地损失是 81 点平均，0.15 的有效梯度可能偏弱）；
③ A/B 两臂必须**独占机器**，同一批开局（`--open-plies 4`，默认 open_seed 固定）。

**起点权重**：`versions/step5_hotstart_9x9.bin` = 从 s800 权重**标定迁移**出分布头
（不标定的话起点锚点只有 2.5%，标定后 45%，公式见 README「第 ⑤ 步」实测二）。
冒烟记录：`runs_step5_smoke/`（2 轮，未标定热启动：损失正常但锚点 2.5%~5%）。

## 已暂停：s800 持续训练（runs9_s800）

为了给第 ⑤ 步的 A/B 腾出机器（同机并行会让锚点测量失真），已暂停：

- `runs9_s800/STOP` 存在 → 停滞看门狗不会重启它
- **恢复方法**：`rm runs9_s800/STOP`；停滞看门狗（`/tmp/goai_stall2.sh`，已重新启动）
  会在 status 停滞 8 分钟后自动重启训练，也可以直接按 README 的命令重跑

## 怎么查

    # 学习曲线（含 own_loss / anchor_winrate）
    python3 -c "import csv;[print(r) for r in csv.DictReader(open('runs_step5/train_log.csv'))]"

    # 成对开局 A/B（不训练，直接对拉）
    ./build/GoAI eval --a runs_step5/latest.bin --b versions/goai9x9_v5_800sims.bin \
      --games 120 --sims 200 --open-plies 4

## 环境注意事项（实测）

- 系统内存吃紧（16GB，可用常低于 200MB）会让每轮变慢；这是环境问题，不是程序 bug
- **做 A/B 必须独占机器**（历史教训：同一版本曾差 66%）
- `/tmp/goai_stall2.sh` 在 `runs9_s800/STOP` 存在时会 `pkill -x GoAI`：
  跑别的实验前记得先停掉它（或改脚本），否则会误杀
