#!/usr/bin/env bash
set -euo pipefail

# Script to set up Lean 4 + Mathlib on Linux host (Ubuntu/Debian)

echo "=== Installing Lean 4 (elan) and Mathlib ==="

# 1. Install prerequisites
sudo apt-get update && sudo apt-get install -y curl git build-essential ca-certificates

# 2. Install elan
if ! command -v elan &> /dev/null; then
    echo "Installing elan..."
    curl -sSf 'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh' | sh -s -- -y
    source "$HOME/.elan/env"
else
    echo "elan is already installed."
fi

# 3. Create Mathlib workspace cache
LEAN_WORKSPACE="${HOME}/lean_mathlib_workspace"
mkdir -p "${LEAN_WORKSPACE}"
cd "${LEAN_WORKSPACE}"

echo "Configuring Lean 4 project in ${LEAN_WORKSPACE}..."
lake init LeanSandbox math

# Fetch precompiled Mathlib build cache
lake exe cache get

# Verify compilation
echo "Building Lean workspace..."
lake build

echo "=== Lean 4 setup successfully finished! ==="
echo "Lake path: $(which lake)"
echo "Lean path: $(which lean)"
