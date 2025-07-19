#!/bin/bash

# Run demo of tail-chasing detector

echo "Running Tail-Chasing Detector on demo code..."
echo "========================================="

cd "$(dirname "$0")/.."

# Run on the example directory
echo "Analyzing examples/demo_target..."
python -m tailchasing examples/demo_target --verbose

echo ""
echo "Generating JSON report..."
python -m tailchasing examples/demo_target --json > demo_report.json

echo ""
echo "Report saved to demo_report.json"
echo ""
echo "You can also try:"
echo "  tailchasing . --html           # Generate HTML report"
echo "  tailchasing . --fail-on 20     # Fail if score > 20"
echo "  tailchasing . --exclude vendor # Exclude vendor directory"
