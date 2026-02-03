#!/bin/bash
# Install git hooks for the project
# Run this once after cloning the repository

set -e

cd "$(dirname "$0")/.."

echo "Installing git hooks..."

# Create hooks directory
mkdir -p .git/hooks

# Create pre-push hook
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Pre-push hook to clean cache files before pushing
./scripts/clean-cache.sh
EOF

chmod +x .git/hooks/pre-push

echo "Git hooks installed successfully."
echo "The pre-push hook will automatically clean cache files before each push."
