# One-time setup: creates .venv and installs CUDA PyTorch + app dependencies.
# Usage (from the project folder):
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cuda cu126   # other CUDA build
param(
    [string]$Cuda = "cu124",
    [switch]$Dev
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"

& $py -m pip install --upgrade pip
Write-Host "Installing PyTorch ($Cuda build)..." -ForegroundColor Cyan
& $py -m pip install torch --index-url "https://download.pytorch.org/whl/$Cuda"

Write-Host "Installing app requirements..." -ForegroundColor Cyan
if ($Dev) { & $py -m pip install -r requirements-dev.txt } else { & $py -m pip install -r requirements.txt }

& $py -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else 'CPU only - check NVIDIA driver / CUDA build')"
Write-Host "`nDone. Start the server with:  powershell -ExecutionPolicy Bypass -File scripts\run.ps1" -ForegroundColor Green
