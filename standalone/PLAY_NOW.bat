@echo off
rem GoBoard - play right now (no compiler needed, uses the prebuilt exe)
cd /d "%~dp0"
chcp 65001 >nul
if not exist GoBoard.exe (
  echo GoBoard.exe not found. Please run BUILD_AND_PLAY.bat instead.
  pause
  exit /b 1
)
echo Starting GoBoard (the AI is already inside the exe)...
echo.
GoBoard.exe --sims 200
pause
