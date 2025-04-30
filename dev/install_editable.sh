#!/usr/bin/env bash
# install_editable.sh
# Installs lads_opcua_client and lads_opcua_viewer in editable (-e) mode.

set -euo pipefail       # stop on first error, prevent unset vars, trace pipe errors

# Always work relative to the location of this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Uninstalling any existing non-editable lads_opcua_client package (if present)…"
pip uninstall -y lads_opcua_client || true   # ignore error if not installed

echo "Installing lads_opcua_client in editable mode…"
cd "$ROOT_DIR/lads_opcua_client"
pip install -e .

echo "Uninstalling any existing non-editable lads_opcua_viewer package (if present)…"
pip uninstall -y lads_opcua_viewer || true

echo "Installing lads_opcua_viewer in editable mode…"
cd "$ROOT_DIR/lads_opcua_viewer"
pip install -e .

echo "Installation complete. Both packages are now installed in editable mode."
