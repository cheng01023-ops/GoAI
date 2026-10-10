#!/bin/bash
# macOS / Linux：双击开始对弈
#   · 包里带了 macOS(arm64) 预编译版 -> 直接用，不需要编译器
#   · 其它平台 / 没有预编译版 -> 从源码编译（macOS 需 Xcode 命令行工具）
cd "$(dirname "$0")" || exit 1
BIN=./GoBoard-macOS
if [ "$(uname -s)" = "Darwin" ] && [ -x "$BIN" ]; then
  exec "$BIN" --sims 200 --weights builtin
fi
echo "正在从源码编译 GoBoard（全屏版）…"
make -s goboard || {
  echo "编译失败：macOS 请先装 Xcode 命令行工具（xcode-select --install）"
  read -r _
  exit 1
}
exec ./build/GoBoard --sims 200 --weights builtin
