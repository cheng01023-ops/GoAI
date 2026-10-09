@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [错误] 还没装环境，请先双击 install_windows.bat
  pause & exit /b 1
)
set OUT=runs_gpu
if not exist "%OUT%" mkdir "%OUT%"

echo ================================================================
echo   GoAI GPU 服务端（批量推理 + GPU 训练 + 权重热更新）
echo ================================================================
echo   C 引擎负责搜索和规则（微秒级），显卡只做它最擅长的事：
echo   攒够几百个局面 -> 一次前向 -> 结果广播回去
echo.
echo   19 路 256 通道 / 单次最多合并 2048 个局面 / bf16
echo   每 60 秒导出 latest.bin 并热切换到推理权重
echo ================================================================
echo.
echo  启动后，另开一个终端运行对弈端：
echo     build\GoAI.exe gtrain --remote 127.0.0.1:8899 --threads 8 --slots 64 --sims 64 --size 19 --out runs_gpu
echo  （或双击 gpu\train_gpu_windows.bat）
echo.

.venv\Scripts\python -u server.py --size 19 --channels 256 --vhidden 128 --device cuda --amp bf16 ^
  --port 8899 --max-batch 2048 --batch-wait-ms 3 --train-batch 512 --steps-per-chunk 20 ^
  --lr 2e-3 --buffer 300000 --export-seconds 60 --stats-seconds 10 --out %OUT%
pause
