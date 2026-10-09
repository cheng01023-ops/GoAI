@echo off
rem GoBoard - build only
cd /d "%~dp0"
chcp 65001 >nul
where gcc >nul 2>nul
if errorlevel 1 (
  echo [ERROR] gcc not found. See README.md or just use PLAY_NOW.bat
  pause
  exit /b 1
)
echo Compiling...
gcc -O3 -ffast-math -std=c11 -DGOAI_NO_CURSES -Iinclude src\goboard.c src\board.c src\compat.c src\net.c src\mcts.c -o GoBoard.exe -lm
if errorlevel 1 ( echo Compile failed. & pause & exit /b 1 )
echo OK: GoBoard.exe
pause
