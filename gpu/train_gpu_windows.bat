@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set OUT=runs_gpu
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\STOP" del "%OUT%\STOP"
if not exist build\GoAI.exe (
  echo [错误] 还没编译 C 引擎。请在 MSYS2 MINGW64 里运行:  make -f Makefile.win
  pause & exit /b 1
)
echo ================================================================
echo   GoAI 对弈/自对弈端（C 搜索 + 远程 GPU 推理）
echo ================================================================
echo   16 线程 x 每线程 64 盘 = 1024 盘并行
echo   端口 8899（先双击 gpu\start_server_windows.bat 把服务端起起来）
echo ================================================================
echo.
build\GoAI.exe gtrain --remote 127.0.0.1:8899 --threads 16 --slots 64 --sims 64 ^
  --size 19 --out %OUT% --pass-min-move 240 --max-moves 1000 --sgf 2
pause
