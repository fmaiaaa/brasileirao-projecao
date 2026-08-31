@echo off
REM Wrapper usado pelo Agendador de Tarefas (segundas 03:00).
REM Evita caminhos com acentos via junction em %LOCALAPPDATA%\brasileirao-retrain\repo
setlocal
cd /d "%~dp0\.."
if not exist "scripts\weekly_retrain.py" (
  echo ERRO: repo nao encontrado em %CD%>> artifacts\prob_ml\weekly_retrain.log
  exit /b 1
)

set PYTHONUNBUFFERED=1
set LOG=artifacts\prob_ml\weekly_retrain.log

echo.>> "%LOG%"
echo ===== INICIO %DATE% %TIME% =====>> "%LOG%"

where git >nul 2>&1
if %ERRORLEVEL%==0 (
  git fetch origin main >> "%LOG%" 2>&1
  git pull --ff-only origin main >> "%LOG%" 2>&1
)

"C:\Users\kaleb\AppData\Local\Programs\Python\Python313\python.exe" -m pip install -r requirements.txt -q >> "%LOG%" 2>&1

"C:\Users\kaleb\AppData\Local\Programs\Python\Python313\python.exe" scripts\weekly_retrain.py --budget fast --no-backtest >> "%LOG%" 2>&1
set EXIT=%ERRORLEVEL%
echo ===== FIM %DATE% %TIME% exit=%EXIT% =====>> "%LOG%"
exit /b %EXIT%
