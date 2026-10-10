# 当前实验状态（自动更新文件，可安全修改）

> 这个文件记录**正在进行中**的实验，方便任何人/任何会话立刻接手。
> 历史结论见 README 的「核心结论汇总」与「优化日志」。

## 正在跑：第 ⑤ 步（领地辅助头 + 值分布化）—— runs_step5

**要回答的问题**：加上领地辅助目标和值分布头，棋力能否相对 step ④ 的 s800 基线
（`versions/goai9x9_v5_800sims.bin`）突破 50% 并上升？

**配置**（200 局/轮，800 模拟/手；锚点每轮 40 局、成对开局）：

    ./build/GoAI train --size 9 --channels 32 --forever \
      --resume versions/step5_hotstart_9x9.bin --own-weight 0.15 --vdist 1 \
      --games 200 --sims 800 --steps 250 --batch 64 \
      --lr 0.0002 --lr-decay-every 30 --lr-decay-factor 0.9 --lr-min 0.00002 \
      --threads 10 --evalgames 12 --evalsims 200 --gate 1 --gategames 40 --eval-every 5 \
      --anchor versions/goai9x9_v5_800sims.bin --anchor-every 1 --anchor-games 40 \
      --rollback 1 --rollback-patience 3 --open-plies 4 --sgf 0 --plot 1 --out runs_step5

**起点**：`versions/step5_hotstart_9x9.bin` —— 从 s800 权重**标定迁移**出分布头：
不标定（分布头随机）时起点锚点只有 2.5%，标定后 45%（60 局成对开局）。
标定公式见 README「第 ⑤ 步」实测二。

**判据**：锚点胜率（对 s800、成对开局）**突破 50% 并上升** → 第 ⑤ 步有效；
停在 40% 附近 → 辅助目标没帮上忙。
（值分布化之后 `value_loss` 是交叉熵，与历史 MSE 不可比，不要用损失判断。）

**进度**：`runs_step5/train_log.csv` 的 `anchor_winrate` 列，或 `runs_step5/status.txt`。
每轮约 5 分钟（自对弈 3.6 分 + 训练 + 评估/锚点 1.2 分）。

**冒烟（已完成）**：`runs_step5_smoke/`，2 轮 200 局，对随机 100%、每局约 100 手、
领地 BCE 0.695→0.657、值 CE 3.50→1.05；但**未标定**热启动下锚点掉到 2.5%~5%
（价值头被重置成均匀分布），这也是上面要标定迁移的原因。

## 已暂停：s800 持续训练（runs9_s800）

为了给第 ⑤ 步的 A/B 腾出机器（同机并行会让锚点测量失真），已暂停：

- `runs9_s800/STOP` 存在 → 停滞看门狗不会重启它
- 停滞看门狗 `/tmp/goai_stall2.sh` 已停（它在 STOP 存在时会 `pkill -x GoAI`，
  会误杀其它实验的进程）
- **恢复方法**：`rm runs9_s800/STOP`，然后按 README 的命令重跑训练；
  需要停滞看门狗的话重新 `nohup /tmp/goai_stall2.sh &`

## 怎么查

    # 学习曲线（含 own_loss / anchor_winrate）
    python3 -c "import csv;[print(r) for r in csv.DictReader(open('runs_step5/train_log.csv'))]"

    # 实时状态
    cat runs_step5/status.txt

    # 成对开局 A/B（不训练，直接对拉）
    ./build/GoAI eval --a runs_step5/latest.bin --b versions/goai9x9_v5_800sims.bin \
      --games 60 --sims 200 --open-plies 4

## 环境注意事项（实测）

- 系统内存吃紧（16GB，可用常低于 200MB）会让每轮变慢；这是环境问题，不是程序 bug
- **做 A/B 必须独占机器**（历史教训：同一版本曾差 66%）
- 换源码后构建要 `rm -f build/net.o` 并用 md5 确认二进制变了（make 的依赖已覆盖）
