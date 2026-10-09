#!/bin/bash
# GoAI 查看进度（Linux / WSL 版）
cd "$(dirname "$0")" || exit 1
OUT="${GOAI_OUT:-runs_live}"
echo "=== 当前状态 ==="
if [ -f "$OUT/status.txt" ]; then
  while IFS='=' read -r k v; do
    case "$k" in
      state) case "$v" in running) v="训练中";; stopped) v="已停止";; finished) v="已完成";; esac;;
      elapsed_total) v="$(awk -v s="$v" 'BEGIN{printf "%.1f 分钟", s/60}')";;
    esac
    printf "  %-16s %s\n" "$k" "$v"
  done < "$OUT/status.txt"
else
  echo "  （还没有训练记录）"
fi
echo
echo "=== 最近 8 轮 ==="
[ -f "$OUT/train_log.csv" ] && { head -1 "$OUT/train_log.csv"; tail -8 "$OUT/train_log.csv"; }
echo
if command -v python3 >/dev/null 2>&1; then
  python3 tools/plot.py "$OUT/train_log.csv" "$OUT/training_curve.png" >/dev/null 2>&1 && echo "曲线已更新：$OUT/training_curve.png"
fi
