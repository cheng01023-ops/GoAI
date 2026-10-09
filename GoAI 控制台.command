#!/bin/bash
# GoAI 一键控制台 —— 开始 / 监控 / 停止 自我迭代训练
# 双击即可运行（macOS 会用「终端」打开）
cd "$(dirname "$0")" || exit 1
source "tools/goai_env.sh"

for arg in "$@"; do
  case "$arg" in
    --start) clear; goai_start; echo; echo "训练在后台运行，可直接关闭本窗口。"; exit 0 ;;
    --stop)  clear; goai_stop;  echo; printf "按回车关闭…"; read -r _; exit 0 ;;
  esac
done

while true; do
  clear
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║            🤖  GoAI 围棋 AI · 一键控制台                      ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  goai_dashboard
  echo
  echo "  ─────────────────────────────────────────────────────────────"
  echo "   1) 开始 / 继续训练（后台运行，可关窗口）"
  echo "   2) 实时监控面板（看它一盘盘下棋）"
  echo "   3) 停止训练（会先保存权重）"
  echo "   4) 打开训练曲线图（自动刷新）"
  echo "   5) 查看训练日志"
  echo "   6) 调整训练强度（当前：$(goai_intensity_name)）"
  echo "   7) 打开训练产物目录"
  echo "   8) 退出"
  echo "  ─────────────────────────────────────────────────────────────"
  printf "  请选择 [1-8]: "
  read -r choice
  case "$choice" in
    1) clear; goai_start; echo; printf "按回车返回菜单…"; read -r _ ;;
    2) goai_watch ;;
    3) clear; goai_stop; echo; printf "按回车返回菜单…"; read -r _ ;;
    4) goai_refresh_plot >/dev/null 2>&1 || true; echo "已打开 $GOAI_OUT/training_curve.png"; sleep 1 ;;
    5) clear; if [ -f "$GOAI_LOGFILE" ]; then tail -40 "$GOAI_LOGFILE"; else echo "还没有日志（先选 1 开始训练）"; fi; echo; printf "按回车返回菜单…"; read -r _ ;;
    6) clear; goai_set_intensity; echo; printf "按回车返回菜单…"; read -r _ ;;
    7) open "$GOAI_OUT" 2>/dev/null || true ;;
    8) clear; echo "再见 👋  训练如仍在后台运行，可用『停止训练.command』停止。"; exit 0 ;;
    *) ;;
  esac
done
