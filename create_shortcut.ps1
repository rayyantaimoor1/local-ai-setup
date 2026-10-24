# Creates a "Local AI" shortcut on your Desktop that runs start.bat.
# Usage:  powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$link = Join-Path $desktop "Local AI.lnk"

$shell = New-Object -ComObject WScript.Shell
$sc = $shell.CreateShortcut($link)
$sc.TargetPath = Join-Path $root "start.bat"
$sc.WorkingDirectory = $root
$sc.Description = "Start the offline local AI assistant"
$sc.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
$sc.Save()
Write-Host "Created $link"
