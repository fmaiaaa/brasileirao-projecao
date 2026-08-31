# Instala/atualiza a tarefa Windows BrasileiraoProbMLWeeklyRetrain (segundas 03:00).
# Job: FPT + prob_ml + 7 modelos × 4 seções → brasileirao_modelos.xlsx
# Uso (PowerShell):
#   cd scripts
#   .\setup_weekly_scheduler.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WrapDir = Join-Path $env:LOCALAPPDATA "brasileirao-retrain"
$Junction = Join-Path $WrapDir "repo"
$RunBat = Join-Path $WrapDir "run.bat"
$TaskName = "BrasileiraoProbMLWeeklyRetrain"
$Python = "C:\Users\kaleb\AppData\Local\Programs\Python\Python313\python.exe"
$StartTime = "03:00"

New-Item -ItemType Directory -Force -Path $WrapDir | Out-Null

if (Test-Path $Junction) {
    cmd /c "rmdir `"$Junction`"" | Out-Null
}
cmd /c "mklink /J `"$Junction`" `"$ProjectRoot`"" | Out-Null

$bat = @"
@echo off
call "$Junction\scripts\run_weekly_retrain_scheduled.bat"
"@
Set-Content -Path $RunBat -Value $bat -Encoding ASCII

schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
schtasks /Create /TN $TaskName `
    /TR "`"$RunBat`"" `
    /SC WEEKLY /D MON /ST $StartTime `
    /RL LIMITED `
    /F | Out-Null

Write-Host "Tarefa: $TaskName"
Write-Host "Quando: toda segunda as $StartTime"
Write-Host "Wrapper: $RunBat"
Write-Host "Repo (junction): $Junction -> $ProjectRoot"
schtasks /Query /TN $TaskName /FO LIST /V | Select-String -Pattern "Proxima|Status|Tipo de Agendamento|Hora de in|Dias:|Tarefa a ser executada"
