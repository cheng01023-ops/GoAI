#!/bin/bash
# 一键开始 / 继续 GoAI 自我迭代训练（后台运行，关掉窗口也不停）
cd "$(dirname "$0")" || exit 1
source "tools/goai_env.sh"
clear
echo "=== GoAI 一键开始训练 ==="
echo
goai_start
if [ "${1:-}" = "--start" ]; then
  echo
  echo "（训练已在后台运行，本窗口可以关闭）"
  exit 0
fi
echo
echo "提示："
echo "  · 训练在后台持续自我迭代，关掉这个窗口也不会停"
echo "  · 看进度：双击『查看进度.command』或『GoAI 控制台.command』"
echo "  · 要停下：双击『停止训练.command』（会先保存权重）"
echo
printf "按回车关闭窗口…"
read -r _
