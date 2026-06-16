@echo off
setlocal
cd /d "%~dp0"

echo [Site Design Platform] cleaning old 8088 listeners...
for /f "tokens=5" %%p in ('netstat -ano -p tcp ^| findstr /R /C:":8088 .*LISTENING"') do (
  taskkill /PID %%p /F >nul 2>nul
)

echo [Site Design Platform] starting python app...
python apps\site_design_platform\start.py --host 127.0.0.1 --port 8088
if errorlevel 1 (
  echo.
  echo [ERROR] startup failed. Check Python environment and traceback above.
  pause
)

endlocal
