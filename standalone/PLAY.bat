@echo off
rem GoBoard - play (uses the existing GoBoard.exe)
cd /d "%~dp0"
chcp 65001 >nul
if not exist GoBoard.exe (
  echo GoBoard.exe not found. Run BUILD_AND_PLAY.bat first.
  pause
  exit /b 1
)
echo Type a coordinate to play ^(for example D4^).
echo Commands: pass / undo / hint / save / new / resign / quit
echo.
GoBoard.exe %*
pause
