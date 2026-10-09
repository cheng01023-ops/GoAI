#!/bin/bash
# 安装/卸载「改动自动同步到 GitHub」的后台任务（macOS launchd）
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC="$REPO/tools/com.goai.autosync.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.goai.autosync.plist"
LABEL="com.goai.autosync"

case "${1:-install}" in
  install|"")
    mkdir -p "$HOME/Library/LaunchAgents"
    sed "s|__REPO__|$REPO|g" "$PLIST_SRC" > "$PLIST_DST"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || launchctl load "$PLIST_DST"
    echo "✅ 已安装自动同步：源码目录一有改动就推送，另外每 10 分钟检查一次"
    echo "   配置：$PLIST_DST"
    echo "   日志：tools/autosync.log"
    ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || launchctl unload "$PLIST_DST" 2>/dev/null
    rm -f "$PLIST_DST"
    echo "已卸载自动同步"
    ;;
  status)
    launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | head -12 || echo "未安装"
    echo "最近日志："; tail -5 "$REPO/tools/autosync.log" 2>/dev/null || echo "  （还没有日志）"
    ;;
  *)
    echo "用法: $0 [install|uninstall|status]"
    ;;
esac
