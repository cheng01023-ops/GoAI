#!/bin/bash
# 生成"可以直接发给别人"的 Windows 包（GoAI-Windows/ + zip）
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="$SRC/GoAI-Windows"
cd "$SRC"

echo "[1/4] 清理旧包"
rm -rf "$DST" "$SRC/GoAI-Windows.zip"

echo "[2/4] 复制文件（排除构建产物与调试脚本）"
mkdir -p "$DST"
rsync -a \
  --exclude 'build/' --exclude 'runs/' --exclude 'runs_live/' --exclude 'runs_gpu/' \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'dbg_*.py' --exclude 'bench_rules.py' \
  --exclude 'GoAI-Windows/' --exclude '*.zip' \
  --exclude '.DS_Store' --exclude '*.xcodeproj/' \
  ./ "$DST/"

# 保留一份已有权重，方便对方续训
mkdir -p "$DST/runs_v2"
cp -f runs_v2/latest.bin "$DST/runs_v2/latest.bin" 2>/dev/null || true
cp -f runs_v2/train_log.csv "$DST/runs_v2/train_log.csv" 2>/dev/null || true

echo "[3/4] 生成校验清单"
{
  echo "GoAI Windows 包"
  echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "文件数: $(find "$DST" -type f | wc -l | tr -d ' ')"
  echo "大小: $(du -sh "$DST" | cut -f1)"
  echo
  echo "顶层内容:"
  ls -1 "$DST"
} > "$DST/PACKAGE_INFO.txt"

echo "[4/4] 打包 zip"
(cd "$SRC" && zip -qr GoAI-Windows.zip GoAI-Windows -x '*.DS_Store')
echo "完成："
du -sh "$DST" "$SRC/GoAI-Windows.zip"
