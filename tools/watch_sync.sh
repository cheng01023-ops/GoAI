#!/bin/bash
# GoAI 自动同步守护进程（在终端里运行，靠终端的文件权限，绕开 macOS 对 文档/桌面 目录的限制）
# 每 60 秒检查一次：有改动就提交并推送到 GitHub。关闭窗口或 Ctrl+C 停止。
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
cd "$REPO" || exit 1
INTERVAL="$1"
if [ -z "$INTERVAL" ]; then INTERVAL=60; fi

clear
echo "================================================================"
echo "  GoAI - 自动同步守护进程"
echo "================================================================"
echo "  仓库目录: $REPO"
echo "  检查间隔: $INTERVAL 秒"
echo "  同步日志: tools/autosync.log"
echo
echo "  关闭这个窗口 或 按 Ctrl+C 即可停止。"
echo "================================================================"
echo

n=0
while true; do
  n=$((n + 1))
  OUT=$(bash "$REPO/tools/sync_github.sh" 2>&1 | tail -2)
  case "$OUT" in
    *没有新改动*) printf "\r  [%s] 第 %d 次检查：无改动      " "$(date '+%H:%M:%S')" "$n" ;;
    *) echo; echo "$OUT" | sed 's/^/  /' ;;
  esac
  sleep "$INTERVAL"
done
