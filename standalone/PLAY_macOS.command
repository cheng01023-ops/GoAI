#!/bin/bash
# macOS / Linux：编译并开始对弈（全屏棋盘，方向键操作）
cd "$(dirname "$0")" || exit 1
echo "正在编译 GoBoard（全屏版）..."
make -s goboard || { echo "编译失败"; read -r _; exit 1; }
./build/GoBoard --sims 200 --weights builtin
