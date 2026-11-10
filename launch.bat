@echo off
REM ScribeScan launcher - double-click to start the app.
REM   * First run: installs everything automatically (one time, ~3 GB download).
REM   * Then: starts the server for this PC + your Wi-Fi and opens your browser.
REM   * Keep this window open while you use the app; close it (or Ctrl+C) to stop.
REM Optional: "launch.bat nobrowser" starts the server without opening a browser.
setlocal
title ScribeScan
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: installing ScribeScan. This downloads about 3 GB and can take 10-20 minutes.
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup.ps1"
    if errorlevel 1 (
        echo.
        echo Setup failed - scroll up to see the error.
        pause
        exit /b 1
    )
    echo.
)

set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if "%AIDET_PORT%"=="" set "AIDET_PORT=8000"

REM Open the browser as soon as the server answers (waits up to 60 s in the background).
if /i not "%~1"=="nobrowser" (
    start "" /min powershell -NoProfile -WindowStyle Hidden -Command "$u='http://localhost:%AIDET_PORT%'; for ($i = 0; $i -lt 60; $i++) { try { Invoke-WebRequest \"$u/api/health\" -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; break } catch { Start-Sleep -Seconds 1 } }"
)

echo ==============================================================
echo  ScribeScan is starting. Keep this window open while you use it.
echo  To stop it: close this window or press Ctrl+C.
echo  Use the "Network:" address below on your phone / other devices.
echo ==============================================================
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port %AIDET_PORT%

echo.
echo ScribeScan has stopped.
pause
