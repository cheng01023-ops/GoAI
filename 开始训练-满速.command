#!/bin/bash
# GoAI 满速训练：用满 CPU、不加后台 QoS（会跟你正常用电脑抢性能）
cd "$(dirname "$0")" || exit 1
export GOAI_FULLSPEED=1
source "tools/goai_env.sh"
clear
echo "=== GoAI 满速训练 ==="
echo "  · 用满 CPU（$(sysctl -n hw.ncpu) 线程），不加后台 QoS → 训练快 4~5 倍"
echo "  · 代价：编译、看视频、开网页会明显变慢；插电时更合适"
echo
goai_start
if [ "${1:-}" = "--start" ]; then echo "（后台运行，可关闭窗口）"; exit 0; fi
echo
echo "提示：要停下双击『停止训练.command』；想看进度双击『查看进度.command』"
printf "按回车关闭窗口…"
read -r _
