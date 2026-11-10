# Start ScribeScan on all network interfaces so phones/laptops on your Wi-Fi can reach it.
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Port 9000 -Token "my-secret"
param(
    [int]$Port = 8000,
    [string]$Token = ""
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:AIDET_PORT = "$Port"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if ($Token) { $env:AIDET_ACCESS_TOKEN = $Token }

.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port $Port
