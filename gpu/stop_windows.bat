@echo off
chcp 65001 >nul
cd /d "%~dp0"
set OUT=runs_gpu
echo ==============================================
echo   GoAI GPU 训练 - 停止
echo ==============================================
if not exist "%OUT%" mkdir "%OUT%"
echo stop > "%OUT%\STOP"
echo 已发出停止请求（写入了 %OUT%\STOP），当前这一轮结束后会保存权重并退出。
echo 权重：%OUT%\latest.bin（最佳：%OUT%\best.bin）
timeout /t 3 >nul
