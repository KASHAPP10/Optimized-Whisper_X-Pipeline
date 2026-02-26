#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw data/manifest experiments/results
if [ "$#" -eq 1 ]; then
  ZIP="$1"
  echo "Unpacking $ZIP into data/raw/"
  unzip -o "$ZIP" -d data/raw/dataset
  echo "Listing files:"
  ls -la data/raw/dataset | sed -n '1,200p'
else
  echo "Usage: $0 path/to/dataset.zip"
fi
