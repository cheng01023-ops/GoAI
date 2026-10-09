#!/bin/bash
# 训练监控：每 60 秒把关键指标追加到 runs_v4/monitor.log，异常也会记下来
cd "$HOME/Documents/deepseek-harness/default-workspace/GoAI" || exit 1
LOG=runs_v4/monitor.log
while true; do
  if ! pgrep -x GoAI >/dev/null; then
    echo "[$(date '+%H:%M:%S')] ⚠️ 训练进程不在了" >> "$LOG"
    exit 0
  fi
  LINE=$(python3 -c "
import csv
rows=list(csv.DictReader(open('runs_v4/train_log.csv')))
if not rows: raise SystemExit
r=rows[-1]
a=float(r['anchor_winrate']); g=float(r['gate_winrate'])
print('第 %s 轮 | 策略 %s | 价值 %s | lr %s | 锚点 %s | 晋级 %s' % (
  r['iteration'], r['policy_loss'], r['value_loss'], r['lr'],
  '-' if a<0 else '%.0f%%'%(a*100), '-' if g<0 else '%.0f%%'%(g*100)))
" 2>/dev/null)
  echo "[$(date '+%H:%M:%S')] $LINE" >> "$LOG"
  sleep 60
done
