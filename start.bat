@echo off
rem ============================================================
rem  ONNX Player launcher (on-demand mjswan player).
rem  Opens the library page; opening one ONNX only prepares that
rem  one file (cached), no full-site rebuild.
rem  Pure ASCII on purpose: cmd + UTF-8 batch files are unreliable.
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
title ONNX Player - Lingxi X2 / Expedition A3

rem ---- locate Python 3.12 ----
set "PYCMD="
if exist "%~dp0..\python312\python.exe" set "PYCMD="%~dp0..\python312\python.exe""
if defined PYCMD goto :have_python
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3.12"
if defined PYCMD goto :have_python
python -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYCMD=python"
goto :no_python

:no_python
echo [ERROR] Python not found.
echo         Install Python 3.12, or keep the bundled "python312" folder
echo         next to the "onnx-player" folder.
pause
exit /b 1

:have_python
rem ---- first run: install dependencies into .pylibs ----
if exist "%~dp0.pylibs\onnxruntime" goto :deps_ok
echo [First run] Installing Python dependencies into .pylibs ...
call "%~dp0setup_env.bat"
if errorlevel 1 goto :deps_failed
if not exist "%~dp0.pylibs\onnxruntime" goto :deps_failed
goto :deps_ok

:deps_failed
echo.
echo [ERROR] Dependency installation failed. See messages above.
pause
exit /b 1

:deps_ok
set "PYTHONPATH=%~dp0.pylibs"
set "PYTHONUTF8=1"
set "OPENARG="
if defined ONNX_PLAYER_NO_OPEN set "OPENARG=--no-open"

echo ============================================================
echo   ONNX Player - Lingxi X2 / Expedition A3
echo   Open a file page: http://127.0.0.1:8767/library
echo   Player:          http://127.0.0.1:8767/
echo   Old player fallback: start-old.bat (http://127.0.0.1:8766/)
echo   Close this window or press Ctrl+C to stop.
echo ============================================================
echo(

%PYCMD% -X utf8 "%~dp0mjswan\player\server.py" --port 8767 %OPENARG%
set "RC=%ERRORLEVEL%"

echo(
if not "%RC%"=="0" echo [ERROR] Server exited with code %RC%. Read the messages above.
echo Server stopped. Press any key to close this window.
pause >nul
endlocal & exit /b %RC%
