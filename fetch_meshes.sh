#!/bin/bash
# Download the 60km Galewsky-jet base mesh shared by all three
# experiments (simple, jax-simple CPU, jax-simple GPU).
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p meshes
cd meshes

URL="https://github.com/dengwirda/periscope/releases/download/data-v1/mesh_w_elev_cvt_7.zip"

if [ -f mesh_w_elev_cvt_7.nc ]; then
    echo "mesh_w_elev_cvt_7.nc already present, skipping download"
    exit 0
fi

wget -O mesh_w_elev_cvt_7.zip "$URL"
unzip -o mesh_w_elev_cvt_7.zip
rm -f mesh_w_elev_cvt_7.zip

echo "done: $(pwd)/mesh_w_elev_cvt_7.nc"
