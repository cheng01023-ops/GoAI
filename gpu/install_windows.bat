@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==============================================
echo   GoAI GPU 训练器 - 环境安装（RTX 5080 用 CUDA 版）
echo ==============================================
where python >nul 2>&1
if errorlevel 1 (
  echo [错误] 没找到 python，请先安装 Python 3.10+ 并勾选 Add to PATH
  pause & exit /b 1
)
if not exist .venv (
  echo [1/3] 创建虚拟环境 .venv ...
  python -m venv .venv || (echo 创建失败 & pause & exit /b 1)
)
echo [2/3] 升级 pip ...
.venv\Scripts\python -m pip install -q --upgrade pip
echo [3/3] 安装 CUDA 版 PyTorch（cu128，约 2-3 GB，请耐心等待）...
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install numpy
echo.
echo 自检：
.venv\Scripts\python -c "import torch;print('torch',torch.__version__,'cuda可用:',torch.cuda.is_available());print('显卡:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
echo.
echo 如果上面显示 cuda可用: False，请把本文件里的 cu128 改成 cu126 或 cu129 再运行一次。
pause
