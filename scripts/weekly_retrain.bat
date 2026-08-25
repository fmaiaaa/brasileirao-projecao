@echo off
REM Segunda 03:00 — FPT + overlay Sheets + regressão + probabilístico + XLSX entrega
cd /d "%~dp0\.."
set PYTHONUNBUFFERED=1
"C:\Users\kaleb\AppData\Local\Programs\Python\Python313\python.exe" scripts\weekly_retrain.py --budget fast >> artifacts\prob_ml\weekly_retrain.log 2>&1
