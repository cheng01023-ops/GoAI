#!/bin/bash
# 把当前改动提交并推送到 GitHub（手动双击 或 launchd 自动调用都可）
# 用法: bash tools/sync_github.sh [提交说明]
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
LOG="$REPO/tools/autosync.log"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

command -v git >/dev/null 2>&1 || { say "没有 git"; exit 1; }
git rev-parse --git-dir >/dev/null 2>&1 || { say "这里还不是 git 仓库"; exit 1; }
git remote get-url origin >/dev/null 2>&1 || { say "还没配置远程仓库，先双击 setup-github.command"; exit 1; }

git add -A || exit 1
if git diff --cached --quiet; then
  # 没有新改动时，也要检查是否有"已提交但没推送"的内容
  BRANCH_NOW=$(git rev-parse --abbrev-ref HEAD)
  AHEAD=0
  if git rev-parse --verify --quiet "@{upstream}" >/dev/null 2>&1; then
    AHEAD=$(git rev-list --count "@{upstream}..HEAD" 2>/dev/null || echo 0)
  else
    AHEAD=1   # 没有上游分支，按需要推送处理
  fi
  if [ "$AHEAD" -gt 0 ] 2>/dev/null; then
    say "有 $AHEAD 个已提交但未推送的提交，正在推送"
    if git push -q origin "$BRANCH_NOW" 2>>"$LOG"; then
      say "已推送"
    else
      say "推送失败（检查网络或 gh auth status）"; exit 1
    fi
  else
    say "没有新改动，跳过"
  fi
  exit 0
fi
MSG="${1:-自动同步: $(date '+%Y-%m-%d %H:%M')}"
git commit -q -m "$MSG" || exit 1
say "已提交: $MSG"

# 顺带把仓库内容镜像到桌面那份（方便你在桌面文件夹里直接用最新代码）
MIRROR="$HOME/Desktop/新建文件夹/GoAI"
if [ -d "$MIRROR" ]; then
  rsync -a --exclude 'build/' --exclude 'runs_live/' --exclude 'runs/' --exclude '.git/'         --exclude '__pycache__/' --exclude '.venv/' --exclude 'GoAI-Windows/'         "$REPO/src/" "$MIRROR/src/" 2>/dev/null
  rsync -a --exclude '__pycache__/' "$REPO/include/" "$MIRROR/include/" 2>/dev/null
  rsync -a --exclude '__pycache__/' "$REPO/gpu/" "$MIRROR/gpu/" 2>/dev/null
  rsync -a --exclude '__pycache__/' "$REPO/tools/" "$MIRROR/tools/" 2>/dev/null
  for f in Makefile Makefile.win README.md README_Windows.md report.md LICENSE .gitignore; do
    [ -f "$REPO/$f" ] && cp -f "$REPO/$f" "$MIRROR/$f" 2>/dev/null
  done
  say "已把最新代码镜像到桌面文件夹"
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if git push -q origin "$BRANCH" 2>>"$LOG"; then
  say "已推送到 GitHub（$BRANCH）"
else
  say "推送失败（检查网络或登录状态：~/.local/bin/gh auth status）"
  exit 1
fi
