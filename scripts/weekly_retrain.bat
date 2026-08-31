@echo off
REM Segunda 03:00 — use scripts\setup_weekly_scheduler.ps1 para registrar no Windows
cd /d "%~dp0\.."
call scripts\run_weekly_retrain_scheduled.bat
