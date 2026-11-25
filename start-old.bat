@echo off
rem ============================================================
rem  ONNX Player launcher (ASCII only on purpose: cmd + UTF-8 batch
rem  files are unreliable, keep this file pure ASCII).
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
title ONNX Player - X2 / A3

rem ---- locate Python 3.12 ----
set "PYCMD="
if exist "%~dp0..\python312\python.exe" set "PYCMD="%~dp0..\python312\python.exe""
if defined PYCMD goto :have_py
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3.12"
if defined PYCMD goto :have_py
python -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYCMD=python"

:have_py
if not defined PYCMD (
  echo [ERROR] Python not found.
  echo         Install Python 3.12, or keep the bundled "python312" folder
  echo         next to the "onnx-player" folder.
  echo.
  pause
  exit /b 1
)

rem ---- first run: install dependencies ----
if not exist "%~dp0.pylibs\onnxruntime" (
  echo [First run] Installing Python dependencies into .pylibs ...
  call "%~dp0setup_env.bat"
  if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed. See messages above.
    pause
    exit /b 1
  )
)

set "PYTHONPATH=%~dp0.pylibs"
set "PYTHONUTF8=1"
set "OPENARG="
if defined ONNX_PLAYER_NO_OPEN set "OPENARG=--no-open"
chcp 65001 >nul

echo ============================================================
echo   ONNX Player - Lingxi X2 / Expedition A3
echo   URL: http://127.0.0.1:8766/
echo   Close this window or press Ctrl+C to stop.
echo ============================================================

%PYCMD% -X utf8 -m backend.server --port 8766 %OPENARG%
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" echo [ERROR] Server exited with code %RC%. Read the messages above.
echo Server stopped. Press any key to close this window.
pause >nul
endlocal & exit /b %RC%
