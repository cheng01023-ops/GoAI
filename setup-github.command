#!/bin/bash
# 一次性设置：登录 GitHub -> 创建仓库 -> 首次推送
# 双击本文件即可（会打开浏览器让你授权）
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
GH="$HOME/.local/bin/gh"
[ -x "$GH" ] || GH="$(command -v gh || true)"
clear
echo "================================================"
echo "  GoAI - 开源到 GitHub（一次性设置）"
echo "================================================"
echo
if [ -z "$GH" ]; then
  echo "❌ 没找到 GitHub CLI（gh）"
  echo "   请重新运行安装，或手动执行： brew install gh"
  read -r _; exit 1
fi

# ---- 1) 登录 ----
if ! "$GH" auth status >/dev/null 2>&1; then
  echo "第 1 步：登录 GitHub"
  echo "  · 会显示一个 8 位代码，并提示按回车打开浏览器"
  echo "  · 在浏览器里粘贴代码、点授权即可"
  echo
  "$GH" auth login --hostname github.com --git-protocol https --web || { echo "登录失败"; read -r _; exit 1; }
  echo
else
  echo "第 1 步：已登录 GitHub（$(  "$GH" api user -q .login 2>/dev/null || echo '?')）"
fi

# ---- 2) 仓库名 ----
DEFAULT_NAME="GoAI"
printf "第 2 步：仓库名 [默认 %s]: " "$DEFAULT_NAME"
read -r NAME
NAME="${NAME:-$DEFAULT_NAME}"
USER="$("$GH" api user -q .login 2>/dev/null)"
echo "  将在 github.com/$USER/$NAME 创建公开仓库"
printf "  确认？(y/n) "
read -r OK
[ "$OK" = "y" ] || [ "$OK" = "Y" ] || { echo "已取消"; read -r _; exit 0; }

# ---- 3) 初始化 git 并提交 ----
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  git init -q -b main 2>/dev/null || { git init -q; git checkout -q -b main 2>/dev/null; }
  echo "  已初始化 git 仓库"
fi
git add -A
git diff --cached --quiet || git commit -q -m "GoAI: C 语言实现的围棋 AI（规则/MCTS/神经网络全手写）" && echo "  已提交"

# ---- 4) 创建远程仓库并推送 ----
if git remote get-url origin >/dev/null 2>&1; then
  echo "  远程仓库已存在，直接推送"
  git push -u origin main || git push -u origin HEAD
else
  "$GH" repo create "$NAME" --public --source=. --remote=origin --push \
    --description "C 语言从零实现的围棋 AI：规则 + PUCT MCTS + 手写 CNN 训练（AlphaZero 风格）" \
    || { echo "创建仓库失败（可能名字已被占用，换个名字重跑）"; read -r _; exit 1; }
fi

echo
echo "✅ 完成！仓库地址： https://github.com/$USER/$NAME"
echo
printf "第 4 步：安装「改了就自动同步」后台任务？(y/n) "
read -r AUTO
if [ "$AUTO" = "y" ] || [ "$AUTO" = "Y" ]; then
  bash tools/autosync.sh install
fi
echo
echo "以后想手动同步：双击 2-同步更新到GitHub（双击）.command"
printf "按回车关闭…"; read -r _
