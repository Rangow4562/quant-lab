@echo off
setlocal
cd /d "%~dp0"
title quant-lab

echo.
echo ============================ quant-lab ============================
where python >nul 2>nul
if errorlevel 1 (
  echo   [missing] python is not on PATH
  goto :fail
)
python --version
python -c "import numpy" >nul 2>nul
if errorlevel 1 (
  echo   [missing] numpy - install it with: pip install numpy
  goto :fail
)
echo   numpy  : ok

set STARTED=%time%
echo.
echo --------------------------- self-checks ---------------------------
python selfcheck.py
if errorlevel 1 goto :fail

echo.
echo -------------------------- experiments ----------------------------
python experiments.py all
if errorlevel 1 goto :fail

echo.
echo [ok] all done - started %STARTED%, finished %time%
exit /b 0

:fail
echo.
echo [failed] quant-lab stopped - see the message above
exit /b 1
