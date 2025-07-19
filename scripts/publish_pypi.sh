#!/bin/bash
# Build and publish tail-chasing-detector to PyPI

set -e

echo "🏗️  Building tail-chasing-detector package..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Install build dependencies
pip install build twine

# Build the package
python -m build

echo "📦 Package built successfully!"
echo ""
echo "Files created:"
ls -la dist/

echo ""
echo "📋 Checking package with twine..."
twine check dist/*

echo ""
echo "🚀 Ready to upload to PyPI!"
echo ""
echo "To upload to TestPyPI (for testing):"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI (for production):"
echo "  twine upload dist/*"
echo ""
echo "Note: You'll need to set up PyPI credentials first:"
echo "  1. Create account at https://pypi.org"
echo "  2. Create API token at https://pypi.org/manage/account/token/"
echo "  3. Create ~/.pypirc with your credentials"