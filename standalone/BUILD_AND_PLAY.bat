@echo off
rem GoBoard - build from source and start playing
cd /d "%~dp0"
chcp 65001 >nul
echo ================================================================
echo   GoBoard - build and play  (the AI is built into the program)
echo ================================================================
echo.
where gcc >nul 2>nul
if errorlevel 1 goto nogcc
echo [1/2] Compiling (about 10 seconds)...
gcc -O3 -ffast-math -std=c11 -DGOAI_NO_CURSES -Iinclude src\goboard.c src\board.c src\compat.c src\net.c src\mcts.c -o GoBoard.exe -lm
if errorlevel 1 goto buildfail
echo       OK: GoBoard.exe
echo.
echo [2/2] Starting. Type a coordinate like D4 to play, "quit" to exit.
echo.
GoBoard.exe --sims 200
echo.
pause
exit /b 0

:nogcc
echo [ERROR] gcc compiler not found.
echo.
echo   Option A (recommended, no installer): download w64devkit
echo       https://github.com/skeeto/w64devkit/releases
echo       Unzip it, run w64devkit.exe, then cd to this folder and run this file again.
echo.
echo   Option B: install MSYS2  https://www.msys2.org/
echo       Then in the MSYS2 MINGW64 window run:
echo         pacman -S --noconfirm mingw-w64-x86_64-gcc
echo.
echo   Option C: do not compile at all - just double-click PLAY_NOW.bat
echo       (a prebuilt GoBoard.exe is already included)
echo.
pause
exit /b 1

:buildfail
echo.
echo [ERROR] Compile failed. Please send the messages above to the author.
pause
exit /b 1
