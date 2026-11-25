@echo off
rem ============================================================
rem  Install Python dependencies into .pylibs (ASCII only).
rem ============================================================
setlocal EnableExtensions
cd /d "%~dp0"

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
  echo [ERROR] Python not found. Install Python 3.12 first.
  exit /b 1
)

echo Using Python: %PYCMD%
%PYCMD% -c "import sys; print('Python', sys.version.split()[0], sys.executable)"

%PYCMD% -m pip install --disable-pip-version-check --no-input --upgrade --target "%~dp0.pylibs" -r "%~dp0requirements.txt"
if not errorlevel 1 goto :done

echo.
echo [INFO] Direct PyPI failed, retrying with a mirror ...
%PYCMD% -m pip install --disable-pip-version-check --no-input --upgrade ^
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple ^
  --target "%~dp0.pylibs" -r "%~dp0requirements.txt"
if errorlevel 1 (
  echo [ERROR] Dependency installation failed.
  exit /b 1
)

:done
echo Dependencies are ready.
endlocal
exit /b 0
