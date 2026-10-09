#!/bin/bash
# GoAI 停止训练（Linux / WSL 版）：写 STOP 文件，当前一轮结束后安全退出
cd "$(dirname "$0")" || exit 1
OUT="${GOAI_OUT:-runs_live}"
mkdir -p "$OUT"
echo "=== GoAI 停止训练 ==="
echo stop > "$OUT/STOP"
for i in $(seq 1 180); do
  pgrep -f "build/GoAI train" >/dev/null || { echo "✅ 已安全停止；权重 $OUT/latest.bin"; exit 0; }
  sleep 1
done
echo "等待超时，强制结束"; pkill -f "build/GoAI train"; echo "已强制结束"
