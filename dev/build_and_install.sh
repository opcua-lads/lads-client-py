#!/usr/bin/env bash
set -euo pipefail

# Always work relative to the location of this script
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to lads_opcua_client
cd "$ROOT_DIR/../lads_opcua_client"

# Build and install package
python3 -m build
pip install dist/lads_opcua_client-0.0.1.tar.gz

# Navigate to lads_opcua_viewer
cd "$ROOT_DIR/../lads_opcua_viewer"

# Build and install package
python3 -m build
pip install dist/lads_opcua_viewer-0.0.1.tar.gz

# Return to project root
echo -e "
Installation complete!"