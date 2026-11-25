@echo off
rem ============================================================
rem  Rebuild mjswan static site (mjswan\site).
rem  Requires a Python environment with mjswan + torch + onnxruntime:
rem    - set MJSWAN_PYTHON to that python.exe, or
rem    - create mjswan\.venv (see mjswan\README.md), or
rem    - use the local python312 if it already has mjswan installed.
rem  Pure ASCII on purpose.
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul

set "PYEXE="
if defined MJSWAN_PYTHON set "PYEXE=%MJSWAN_PYTHON%"
if not defined PYEXE if exist "%~dp0mjswan\.venv\Scripts\python.exe" set "PYEXE=%~dp0mjswan\.venv\Scripts\python.exe"
if not defined PYEXE if exist "%~dp0..\python312\python.exe" set "PYEXE=%~dp0..\python312\python.exe"

if not defined PYEXE goto :no_python
goto :check_mjswan

:no_python
echo [ERROR] No Python candidate found.
echo         Set MJSWAN_PYTHON to a python.exe that has mjswan installed,
echo         or create mjswan\.venv first (see mjswan\README.md).
pause
exit /b 1

:check_mjswan
"%PYEXE%" -c "import mjswan" >nul 2>nul
if not errorlevel 1 goto :build
echo [ERROR] This Python has no mjswan installed:
echo         %PYEXE%
echo(
echo         Quick setup (Windows, CPU only):
echo           "%PYEXE%" -m pip install mjswan onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
echo           "%PYEXE%" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
echo         Or set MJSWAN_PYTHON to an existing mjswan environment.
pause
exit /b 1

:build
rem npm cache/registry workarounds for custom-TS plugin build
set "npm_config_cache=%~dp0mjswan\npm-cache"
set "npm_config_registry=https://registry.npmmirror.com"

echo Building mjswan site from mjswan\resources.json ...
"%PYEXE%" -X utf8 "%~dp0mjswan\build_site.py" %*
set "RC=%ERRORLEVEL%"
echo(
if not "%RC%"=="0" goto :failed
echo Done. Run start-static.bat to view the baked site.
pause
endlocal & exit /b 0

:failed
echo [ERROR] Build failed with code %RC%. Read the messages above.
pause
endlocal & exit /b %RC%
