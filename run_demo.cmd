@echo off
setlocal
cd /d "%~dp0"

rem Use the project venv if there is one, otherwise the system Python.
set "PY=python"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"

if not exist "DTI\dist\index.html" (
  echo [ERROR] Frontend not built. Run once:  cd DTI ^&^& npm install ^&^& npm run build
  exit /b 1
)

echo Starting AgroNiddan (LeafX) at http://127.0.0.1:5000
echo Press Ctrl+C to stop.
rem Open the browser after the model has had a few seconds to load.
start "" /min cmd /c "timeout /t 8 >nul & start http://127.0.0.1:5000"
"%PY%" app.py

endlocal
