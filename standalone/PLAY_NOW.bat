@echo off
rem GoBoard launcher. IMPORTANT: keep every line ABOVE "chcp 65001" ASCII-only,
rem because cmd reads this file in the OEM codepage (cp936 on Chinese Windows)
rem until chcp runs -- non-ASCII bytes before it can break the parser.
rem (This file must also keep CRLF line endings, or cmd mis-parses blocks.)
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set "W=weights\goai9x9_v5_800sims.bin"
if not exist GoBoard.exe goto noexe
if not exist "%W%" goto builtin
echo 启动 GoBoard（最强权重 v5）...
echo.
GoBoard.exe --sims 200 --weights "%W%"
if errorlevel 1 goto fallback
goto done
:builtin
echo 启动 GoBoard（使用程序内置权重）...
echo.
GoBoard.exe --sims 200
if errorlevel 1 goto fallback
goto done
:fallback
echo.
echo [提示] GoBoard.exe 启动失败（最常见原因：缺少 libwinpthread-1.dll）。
echo        改用自带的备用版本（不依赖额外 DLL，加载同一份 v5 权重）...
if not exist GoBoard-fallback.exe goto buildhint
GoBoard-fallback.exe --sims 200 --weights "%W%"
if errorlevel 1 goto buildhint
goto done
:buildhint
echo.
echo [错误] 两个版本都启动不了。请运行 BUILD_AND_PLAY.bat 在本机重新编译
echo        （需要 gcc / w64devkit），编译出的 exe 不再依赖额外 DLL。
goto done
:noexe
echo 找不到 GoBoard.exe，请先运行 BUILD_AND_PLAY.bat 编译。
:done
chcp 936 >nul
echo.
pause
