#!/bin/bash
# 安装/卸载「改动自动同步到 GitHub」的后台任务（macOS launchd）
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SRC="$REPO/tools/com.goai.autosync.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.goai.autosync.plist"
LABEL="com.goai.autosync"

# macOS 的安全机制：文档 / 桌面 / 下载 目录下的文件，后台任务无权读取
# （会报 Operation not permitted）。这种情况请用「启动自动同步.command」。
REPO_PROTECTED=0
case "$REPO" in
  "$HOME/Documents"/*|"$HOME/Desktop"/*|"$HOME/Downloads"/*) REPO_PROTECTED=1 ;;
esac

case "${1:-install}" in
  install|"")
    if [ "$REPO_PROTECTED" = "1" ]; then
      echo "项目在 $REPO"
      echo "macOS 不允许后台任务读取 文档/桌面/下载 目录，launchd 方案会失败。"
      echo "请改用：双击「启动自动同步.command」（在终端里跑，有权限）。"
      echo "若仍要装 launchd 版本（例如把项目移到 ~/GoAI），加 --force。"
      if [ "$2" != "--force" ]; then exit 0; fi
    fi
    mkdir -p "$HOME/Library/LaunchAgents"
    sed "s|__REPO__|$REPO|g" "$PLIST_SRC" > "$PLIST_DST"
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null
    launchctl bootstrap "gui/$(id -u)" "$PLIST_DST" 2>/dev/null || launchctl load "$PLIST_DST"
    echo "✅ 已安装自动同步：源码目录一有改动就推送，另外每 2 分钟检查一次"
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
