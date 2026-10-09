#!/bin/bash
# 训练监控：每 60 秒把关键指标追加到 <输出目录>/monitor.log，进程消失也会记录
# 用法: bash tools/monitor_train.sh [输出目录]   默认 runs13
OUT="${1:-runs13}"
cd "$HOME/Documents/deepseek-harness/default-workspace/GoAI" || exit 1
LOG="$OUT/monitor.log"
while true; do
  if ! pgrep -x GoAI >/dev/null; then
    echo "[$(date '+%H:%M:%S')] 训练进程已退出" >> "$LOG"
    exit 0
  fi
  LINE=$(python3 -c "
import csv,sys
try:
    rows=list(csv.DictReader(open('$OUT/train_log.csv')))
except Exception:
    sys.exit()
if not rows: raise SystemExit
r=rows[-1]
def pct(k):
    v=float(r.get(k,-1) or -1)
    return '-' if v<0 else '%.0f%%'%(v*100)
print('第 %s 轮 | 策略 %s | 价值 %s | lr %s | 锚点 %s | 晋级 %s' % (
  r['iteration'], r['policy_loss'], r['value_loss'], r['lr'], pct('anchor_winrate'), pct('gate_winrate')))
" 2>/dev/null)
  [ -n "$LINE" ] && echo "[$(date '+%H:%M:%S')] $LINE" >> "$LOG"
  sleep 60
done
