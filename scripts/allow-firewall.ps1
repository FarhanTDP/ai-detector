# Allow other devices on your PRIVATE network to reach the server.
# Must be run from an *Administrator* PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\allow-firewall.ps1
# Remove later with:
#   Remove-NetFirewallRule -DisplayName "ScribeScan (TCP 8000)"
param([int]$Port = 8000)

New-NetFirewallRule `
    -DisplayName "ScribeScan (TCP $Port)" `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
    -Profile Private
Write-Host "Inbound TCP $Port allowed on Private networks." -ForegroundColor Green
Write-Host "Make sure your Wi-Fi is set to 'Private' (Settings > Network & internet > Wi-Fi > your network)."
