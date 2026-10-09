#!/bin/bash
# GoAI 一键开始训练（Linux / WSL 版）
cd "$(dirname "$0")" || exit 1
OUT="${GOAI_OUT:-runs_live}"
mkdir -p "$OUT"
rm -f "$OUT/STOP"
echo "=== GoAI 一键开始训练（Linux/WSL）==="
make -s all || { echo "编译失败"; exit 1; }
CPUS=$(nproc 2>/dev/null || echo 4)
THREADS=$(( CPUS > 8 ? 8 : CPUS ))
[ "$THREADS" -lt 1 ] && THREADS=1
if [ -f /sys/class/power_supply/AC0/online ] && [ "$(cat /sys/class/power_supply/AC0/online 2>/dev/null)" = "0" ]; then
  THREADS=$(( THREADS / 2 )); [ "$THREADS" -lt 1 ] && THREADS=1; echo "电池供电，降为 $THREADS 线程"
fi
echo "线程数 $THREADS；持续自我迭代，随时可停"
nohup nice -n 10 ./build/GoAI train --size 9 --channels 32 --forever --resume "$OUT/latest.bin" \
  --games 40 --sims 140 --steps 250 --batch 64 --lr 0.01 --evalgames 12 --evalsims 40 \
  --threads "$THREADS" --sgf 2 --gate 1 --gategames 20 --eval-every 3 --plot 1 --out "$OUT" \
  < /dev/null >> "$OUT/train_console.log" 2>&1 &
sleep 3
if pgrep -f "build/GoAI train" >/dev/null; then
  echo "✅ 已启动（日志 $OUT/train_console.log）"
  echo "   看进度： bash 查看进度.sh     停止： bash 停止训练.sh"
else
  echo "❌ 启动失败："; tail -5 "$OUT/train_console.log"
fi
