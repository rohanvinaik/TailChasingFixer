#!/bin/bash
# Build and package VS Code extension

set -e

cd vscode-extension

echo "📦 Building VS Code Extension..."

# Install dependencies
npm install

# Compile TypeScript
npm run compile

# Install vsce if not already installed
if ! command -v vsce &> /dev/null; then
    echo "Installing vsce..."
    npm install -g vsce
fi

# Package extension
echo "Creating .vsix package..."
vsce package

echo "✅ Extension packaged successfully!"
echo ""
echo "Files created:"
ls -la *.vsix

echo ""
echo "To install locally:"
echo "  code --install-extension tail-chasing-detector-*.vsix"
echo ""
echo "To publish to VS Code Marketplace:"
echo "  1. Create publisher at https://marketplace.visualstudio.com/manage"
echo "  2. Get Personal Access Token from Azure DevOps"
echo "  3. Run: vsce publish"