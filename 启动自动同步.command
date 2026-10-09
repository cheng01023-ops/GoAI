#!/bin/bash
# 双击：启动自动同步（开一个终端窗口守着，改了东西就推送到 GitHub）
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE" || exit 1
bash tools/watch_sync.sh 60
