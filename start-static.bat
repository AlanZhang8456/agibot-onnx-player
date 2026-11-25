@echo off
rem ============================================================
rem  ONNX Player (mjswan static site) launcher.
rem  Pure ASCII on purpose: cmd + UTF-8 batch files are unreliable.
rem  Static site is served by Python's stdlib http.server.
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "PORT=8769"
if defined MJSWAN_PORT set "PORT=%MJSWAN_PORT%"

rem ---- locate Python (any 3.x is enough for http.server) ----
set "PYEXE="
set "PYARGS="
if exist "%~dp0..\python312\python.exe" goto :py_bundled
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 goto :py_py
python -c "import sys" >nul 2>nul
if not errorlevel 1 goto :py_python
goto :no_python

:py_bundled
set "PYEXE=%~dp0..\python312\python.exe"
goto :have_python

:py_py
set "PYEXE=py"
set "PYARGS=-3.12"
goto :have_python

:py_python
set "PYEXE=python"
goto :have_python

:no_python
echo [ERROR] Python not found.
echo         Install Python 3.12, or keep the bundled "python312" folder
echo         next to the "onnx-player" folder.
pause
exit /b 1

:have_python
if exist "%~dp0mjswan\site\BUILD_INFO.json" goto :site_ok
echo [ERROR] mjswan site has not been built yet: mjswan\site\BUILD_INFO.json missing.
echo         Run rebuild_mjswan.bat first (needs a Python env with mjswan installed).
pause
exit /b 1

:site_ok
echo ============================================================
echo   ONNX Player (mjswan static site) - Lingxi X2 / Expedition A3
echo   URL: http://127.0.0.1:%PORT%/
echo   On-demand player: start.bat (8767); old player: start-old.bat (http://127.0.0.1:8766/)
echo   Close this window or press Ctrl+C to stop the server.
echo ============================================================
echo(

if not defined ONNX_PLAYER_NO_OPEN start "" "http://127.0.0.1:%PORT%/"

"%PYEXE%" %PYARGS% -X utf8 -m http.server %PORT% --bind 127.0.0.1 --directory "%~dp0mjswan\site"
set "RC=%ERRORLEVEL%"

echo(
if not "%RC%"=="0" echo [ERROR] Server exited with code %RC%. Read the messages above.
echo Server stopped. Press any key to close this window.
pause >nul
endlocal & exit /b %RC%
