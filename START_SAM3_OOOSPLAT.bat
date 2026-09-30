@echo off
setlocal
cd /d "%~dp0"
if not exist "F:\SAM3\.venv_desktop\Scripts\python.exe" (
  echo Cannot find F:\SAM3\.venv_desktop\Scripts\python.exe
  pause
  exit /b 1
)
"F:\SAM3\.venv_desktop\Scripts\pythonw.exe" app.py
