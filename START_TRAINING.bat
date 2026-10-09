@echo off
rem GoAI - start continuous self-play training (Windows)
rem ASCII only on purpose: Chinese text in .bat files breaks on some Windows codepages.
cd /d "%~dp0"
chcp 65001 >nul
setlocal enabledelayedexpansion
set OUT=runs_live
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\STOP" del "%OUT%\STOP"

echo ==============================================
echo   GoAI - start training (Windows)
echo ==============================================
echo.

where gcc >nul 2>&1
if errorlevel 1 (
  echo [ERROR] gcc not found. Install MSYS2 + mingw-w64 first ^(see README_Windows.md^),
  echo         or run the equivalent command under WSL.
  pause
  exit /b 1
)

echo [1/3] Building the engine...
if not exist build mkdir build
make -f Makefile.win -s all
if errorlevel 1 (
  echo [ERROR] Build failed. Please send the messages above to the author.
  pause
  exit /b 1
)

rem ---- threads = cores - 2, capped at 8; halved on battery ----
set /a THREADS=%NUMBER_OF_PROCESSORS%-2
if %THREADS% LSS 1 set THREADS=1
if %THREADS% GTR 8 set THREADS=8
set POWER=AC power
for /f "tokens=2 delims==" %%b in ('wmic path Win32_Battery get BatteryStatus /value 2^>nul ^| findstr "="') do (
  if "%%b"=="1" ( set POWER=battery & set /a THREADS=THREADS/2 )
)
if %THREADS% LSS 1 set THREADS=1
echo [2/3] Machine: %NUMBER_OF_PROCESSORS% cores, %POWER%, self-play threads %THREADS%

echo [3/3] Starting in background at low priority (uses only idle CPU)
start "GoAI training" /low /b cmd /c "build\GoAI.exe train --size 9 --channels 32 --forever --resume %OUT%\latest.bin --games 40 --sims 140 --steps 250 --batch 64 --lr 0.01 --evalgames 12 --evalsims 40 --threads %THREADS% --sgf 2 --gate 1 --gategames 20 --eval-every 3 --plot 1 --out %OUT% > %OUT%\train_console.log 2>&1"

timeout /t 3 >nul
echo.
echo Training is running in the background ^(closing this window will NOT stop it^).
echo   - check progress : double-click CHECK_PROGRESS.bat
echo   - stop training  : double-click STOP_TRAINING.bat
echo.
echo For MAXIMUM speed ^(uses all cores, expects to be plugged in^):
echo   build\GoAI.exe train --size 9 --channels 32 --forever --resume %OUT%\latest.bin ^
echo     --games 40 --sims 140 --steps 250 --batch 64 --lr 0.01 --threads %NUMBER_OF_PROCESSORS% ^
echo     --evalgames 8 --evalsims 20 --gate 1 --gategames 12 --eval-every 10 --out %OUT%
echo.
pause
