
$ErrorActionPreference = "Stop"

Set-Location -Path "lads_opcua_client"

Write-Host "Uninstalling the non-editable lads_opcua_client package (if installed)..."
pip uninstall -y lads_opcua_client

Write-Host "Installing the package in editable mode..."
pip install -e .

Write-Host "Uninstalling the non-editable lads_opcua_viewer package (if installed)..."
pip uninstall -y lads_opcua_viewer

Set-Location -Path "..\lads_opcua_viewer"

Write-Host "Installing the package in editable mode..."
pip install -e .

Set-Location -Path ".."
Write-Host "Installation complete. The package is now installed in editable mode."
