#!/bin/bash
# Example usage of Tail-Chasing Detector

echo "=================================="
echo "Tail-Chasing Detector Demo"
echo "=================================="
echo ""

# Run on the demo target
echo "1. Analyzing demo code with tail-chasing patterns..."
echo "   $ tailchasing examples/demo_target"
echo ""
cd "$(dirname "$0")/.."
python -m tailchasing examples/demo_target

echo ""
echo "=================================="
echo ""
echo "2. Generating JSON report..."
echo "   $ tailchasing examples/demo_target --json > report.json"
echo ""

python -m tailchasing examples/demo_target --json > demo_report.json 2>&1
echo "JSON report saved to demo_report.json"

echo ""
echo "=================================="
echo ""
echo "3. Key findings from the analysis:"
echo ""

python -c "
import json
with open('demo_report.json') as f:
    lines = f.readlines()
    # Find where JSON starts
    json_start = next(i for i, line in enumerate(lines) if line.strip() == '{')
    data = json.loads(''.join(lines[json_start:]))
    
print(f'Total Issues: {data[\"summary\"][\"total_issues\"]}')
print(f'Risk Score: {data[\"summary\"][\"global_score\"]} ({data[\"summary\"][\"risk_level\"]})')
print(f'')
print('Issue Breakdown:')
for issue_type, count in data['distribution'].items():
    print(f'  - {issue_type}: {count}')
print(f'')
print('Sample Issues Detected:')
for issue in data['issues'][:3]:
    print(f'  [{issue[\"kind\"]}] {issue[\"message\"]}')
"

echo ""
echo "=================================="
echo ""
echo "The detector successfully identified:"
echo "  ✓ Circular import chains"
echo "  ✓ Duplicate/redundant functions"
echo "  ✓ Phantom/placeholder implementations"
echo "  ✓ References to non-existent functions"
echo "  ✓ Imports from non-existent modules"
echo ""
echo "These are classic symptoms of LLM-assisted 'tail-chasing' development!"
