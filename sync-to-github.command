#!/bin/bash
# 双击：把改动同步到 GitHub
cd "$(dirname "$0")" || exit 1
bash tools/sync_github.sh
echo
echo "（关掉这个窗口即可；想自动同步请看 README 的「自动同步」一节）"
sleep 2
