#!/usr/bin/env bash
set -euo pipefail

# Navigate to lads_opcua_client
cd lads_opcua_client

# Build and install package
python3 -m build
pip install dist/lads_opcua_client-0.0.1.tar.gz

# Navigate to lads_opcua_viewer
cd ../lads_opcua_viewer

# Build and install package
python3 -m build
pip install dist/lads_opcua_viewer-0.0.1.tar.gz

# Return to project root
echo -e "
Installation complete!"
