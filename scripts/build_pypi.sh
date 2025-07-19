#!/bin/bash
# Script to build and publish to PyPI

set -e

echo "🏗️  Building tail-chasing-detector for PyPI..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Install build tools
pip install --upgrade pip setuptools wheel twine

# Build source distribution and wheel
python -m build

# Check the package
echo "📦 Checking package with twine..."
twine check dist/*

echo "✅ Package built successfully!"
echo ""
echo "Files created:"
ls -la dist/

echo ""
echo "To upload to TestPyPI (for testing):"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI (for production):"
echo "  twine upload dist/*"
echo ""
echo "Make sure you have configured your PyPI credentials in ~/.pypirc"