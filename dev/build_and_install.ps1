
$ErrorActionPreference = "Stop"

Set-Location -Path "lads_opcua_client"
py -m build
pip install "dist\lads_opcua_client-0.0.1.tar.gz"

Set-Location -Path "..\lads_opcua_viewer"
py -m build
pip install "dist\lads_opcua_viewer-0.0.1.tar.gz"

Set-Location -Path ".."
Write-Host "`nInstallation complete!" -ForegroundColor Green
