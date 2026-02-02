#!/bin/bash
# =============================================================================
# Run All Tests
# =============================================================================
#
# This script runs unit and integration tests for the project.
#
# Test categories:
#   - Unit tests: Individual components (filters, models, metrics, utils)
#   - Integration tests: End-to-end pipelines
#
# Usage:
#   bash scripts/run_tests.sh           # Run all tests
#   bash scripts/run_tests.sh -v        # Verbose output
#   bash scripts/run_tests.sh -f        # Stop on first failure
#
# =============================================================================

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Add project to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Parse arguments
VERBOSE=""
FAILFAST=""
while getopts "vf" opt; do
    case $opt in
        v) VERBOSE="-v" ;;
        f) FAILFAST="--failfast" ;;
    esac
done

echo "=============================================="
echo "Running Tests"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# -----------------------------------------------------------------------------
# Run all tests
# -----------------------------------------------------------------------------
echo "Discovering and running tests in tests/..."
echo ""

python -m unittest discover -s tests $VERBOSE $FAILFAST

echo ""
echo "=============================================="
echo "All Tests Passed"
echo "=============================================="
