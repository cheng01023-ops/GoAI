#!/bin/bash
# 双击开始人机对弈（macOS）。权重已编译在程序里，无需其它文件。
cd "$(dirname "$0")" || exit 1
if [ ! -x build/GoBoard ] || [ src/goboard.c -nt build/GoBoard ] || [ include/weights_builtin.h -nt build/GoBoard ]; then
  echo "正在编译对弈程序…"
  make -s goboard || { echo "编译失败，请先运行 make goboard 看错误"; read -r _; exit 1; }
fi
clear
exec ./build/GoBoard --sims "${1:-200}"
