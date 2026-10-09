@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal
if not exist .venv\Scripts\python.exe (
  echo [错误] 还没装环境，请先双击 install_windows.bat
  pause & exit /b 1
)
set OUT=runs_gpu
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\STOP" del "%OUT%\STOP"

echo ==============================================
echo   GoAI GPU 训练（榨干 5080）
echo ==============================================
echo   - 19 路 128 通道，256 盘并行自对弈
echo   - bf16 混合精度，回放缓冲常驻显存
echo   - 每轮自动导出 latest.bin / best.bin（C 引擎可直接用）
echo   - 停止：双击 stop_windows.bat
echo.

rem 显存不够就把 --games 调小（32GB 显存可以到 512）
set ARGS=--size 19 --channels 128 --games 256 --sims 64 --iters 0 --steps 400 --batch 512 --lr 2e-3 --amp bf16 --device cuda --eval-games 16 --eval-sims 64 --gate-every 3 --buffer 300000 --save-sgf 4 --out %OUT%

echo 参数：%ARGS%
echo.
.venv\Scripts\python -u train.py %ARGS%
pause
