#!/bin/bash
# 一键停止 GoAI 训练（优雅停止：跑完当前这一轮再退出并保存权重）
cd "$(dirname "$0")" || exit 1
source "tools/goai_env.sh"
clear
echo "=== GoAI 停止训练 ==="
echo
goai_stop
if [ "${1:-}" = "--stop" ]; then exit 0; fi
echo
printf "按回车关闭窗口…"
read -r _
