@echo off
rem GoAI - show training progress
cd /d "%~dp0"
chcp 65001 >nul
set OUT=runs_live
echo ==============================================
echo   GoAI - training progress
echo ==============================================
if exist "%OUT%\status.txt" (
  type "%OUT%\status.txt"
) else (
  echo No training record yet.
)
echo.
echo ---- last 8 rounds ----
if exist "%OUT%\train_log.csv" (
  powershell -NoProfile -Command "Get-Content '%OUT%\train_log.csv' -Tail 9"
)
echo.
echo ---- training curve ----
where python >nul 2>&1 && (
  python tools\plot.py "%OUT%\train_log.csv" "%OUT%\training_curve.png" 2>nul
  if exist "%OUT%\training_curve.png" start "" "%OUT%\training_curve.png"
) || echo ^(no Python found, skipping the plot^)
echo.
pause
