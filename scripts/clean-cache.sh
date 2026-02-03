#!/bin/bash
# Clean all cache and temporary files from the project
# Can be run manually or used in git hooks

set -e

cd "$(dirname "$0")/.."

echo "Cleaning cache files..."

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove pytest cache
rm -rf .pytest_cache 2>/dev/null || true

# Remove ruff cache
rm -rf .ruff_cache 2>/dev/null || true

# Remove coverage files
rm -f .coverage 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true

# Remove egg-info
rm -rf *.egg-info 2>/dev/null || true

# Remove GDAL auxiliary files
find . -name "*.aux.xml" -type f -delete 2>/dev/null || true

# Remove Jupyter checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# Remove macOS files
find . -name ".DS_Store" -type f -delete 2>/dev/null || true

echo "Done."
