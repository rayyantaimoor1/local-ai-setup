# Registers the daily long-memory summarization job (default 3 AM) in Task Scheduler.
# Ollama must already be running at that time; the task does not start it.
# Usage:  powershell -ExecutionPolicy Bypass -File install_scheduled_task.ps1 [-At "03:00"]
param([string]$At = "03:00")

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Missing $python - run setup.bat first." }

$action  = New-ScheduledTaskAction -Execute $python -Argument "summarize_memory.py" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "LocalAI-SummarizeMemory" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Daily long-memory summary for the local AI system" -Force | Out-Null
Write-Host "Registered LocalAI-SummarizeMemory (daily at $At)."
