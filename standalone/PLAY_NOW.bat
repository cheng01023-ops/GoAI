@echo off
rem GoBoard - 双击就能下（不需要编译器；AI 权重已在包里）
cd /d "%~dp0"
chcp 65001 >nul
if not exist GoBoard.exe (
  echo 找不到 GoBoard.exe，请先运行 BUILD_AND_PLAY.bat
  pause
  exit /b 1
)
set W=weights\goai9x9_v5_800sims.bin
if exist "%W%" (
  echo 启动 GoBoard（最强权重 v5：%W%）...
  echo.
  GoBoard.exe --sims 200 --weights "%W%"
) else (
  echo 启动 GoBoard（使用 exe 内置权重）...
  echo.
  GoBoard.exe --sims 200
)
pause
