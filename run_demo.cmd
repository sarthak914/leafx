@echo off
setlocal
cd /d "%~dp0"

rem Use the project venv if there is one, otherwise the first installed Python
rem that has the project's packages (several Python versions may be installed).
set "PY="
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY (
  for %%V in (3.13 3.12 3.11 3.10) do if not defined PY (
    py -%%V -c "import ultralytics, flask" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY set "PY=python"

if not exist "DTI\dist\index.html" (
  echo [ERROR] Frontend not built. Run once:  cd DTI ^&^& npm install ^&^& npm run build
  exit /b 1
)

echo Starting AgroNiddan (LeafX) at http://127.0.0.1:5000
echo Press Ctrl+C to stop.
rem Open the browser after the model has had a few seconds to load.
start "" /min cmd /c "timeout /t 8 >nul & start http://127.0.0.1:5000"
echo Using: %PY%
%PY% app.py

endlocal
