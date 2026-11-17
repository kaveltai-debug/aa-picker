@echo off
setlocal
cd /d "%~dp0"

set "PY_CMD="
python -c "import sys" >nul 2>nul && set "PY_CMD=python"
if not defined PY_CMD (
    py -3 -c "import sys" >nul 2>nul && set "PY_CMD=py -3"
)
if not defined PY_CMD (
    echo Python 3 not found. Please install Python 3 from python.org.
    pause
    exit /b 1
)

start "" /b powershell -NoProfile -Command "for($i=0;$i -lt 60;$i++){try{$c=New-Object Net.Sockets.TcpClient('127.0.0.1',8742);$c.Close();Start-Process 'http://127.0.0.1:8742';break}catch{Start-Sleep -Milliseconds 250}}"

%PY_CMD% -m aa_picker.web
if errorlevel 1 (
    echo.
    echo Server exited with an error.
    pause
)
