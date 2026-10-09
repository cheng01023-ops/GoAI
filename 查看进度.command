#!/bin/bash
# 查看训练进度：状态 + 最近几轮 + 刷新训练曲线
cd "$(dirname "$0")" || exit 1
source "tools/goai_env.sh"
clear
goai_dashboard
echo
echo "=== 最近 10 轮（CSV）==="
if [ -f "$GOAI_OUT/train_log.csv" ]; then
  head -1 "$GOAI_OUT/train_log.csv"
  tail -10 "$GOAI_OUT/train_log.csv"
else
  echo "（还没有训练记录）"
fi
echo
( cd "$GOAI_PROJ" && python3 tools/plot.py "$GOAI_OUT/train_log.csv" "$GOAI_OUT/training_curve.png" >/dev/null 2>&1 )
if [ -f "$GOAI_OUT/training_curve.png" ]; then
  open "$GOAI_OUT/training_curve.png"
  echo "已打开训练曲线图：$GOAI_OUT/training_curve.png（Preview 会自动跟随刷新）"
fi
echo
if [ "${1:-}" != "--no-wait" ]; then printf "按回车关闭窗口…"; read -r _; fi
