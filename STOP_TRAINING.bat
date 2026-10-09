@echo off
rem GoAI - stop training gracefully (saves weights first)
cd /d "%~dp0"
chcp 65001 >nul
set OUT=runs_live
echo ==============================================
echo   GoAI - stop training
echo ==============================================
if not exist "%OUT%\status.txt" (
  echo No training record found ^(%OUT%\status.txt missing^).
  pause
  exit /b 0
)
echo Requesting stop ^(weights are saved at the end of the current round^)...
echo stop > "%OUT%\STOP"
for /l %%i in (1,1,180) do (
  tasklist /fi "imagename eq GoAI.exe" 2>nul | find /i "GoAI.exe" >nul || goto stopped
  timeout /t 1 >nul
)
echo [WARN] Timed out, killing the process.
taskkill /im GoAI.exe /f >nul 2>&1
:stopped
echo.
echo Stopped. Weights: %OUT%\latest.bin  ^(best: %OUT%\best.bin^)
pause
