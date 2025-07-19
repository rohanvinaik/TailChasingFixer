#!/bin/bash
# Demonstrate semantic hypervector analysis

echo "=== Tail Chasing Detector - Semantic Analysis Demo ==="
echo ""
echo "This demo shows how the semantic analyzer detects:"
echo "- Functions with same behavior but different code"
echo "- Fragmented implementations of the same concept"
echo "- Semantic stagnation in placeholder functions"
echo ""

# Create a temporary config with semantic analysis enabled
cat > .tailchasing_demo.yml << EOF
paths:
  include:
    - .
  exclude:
    - __pycache__

semantic:
  enable: true
  hv_dim: 8192
  min_functions: 5  # Lower threshold for demo
  z_threshold: 2.0  # Slightly lower for demo
  
  channel_weights:
    NAME_TOKENS: 1.0
    DOC_TOKENS: 0.8
    CALLS: 1.2
    CONTROL: 0.7
    ARG_SIG: 0.9

scoring_weights:
  semantic_duplicate_function: 3
  prototype_fragmentation: 3
  semantic_stagnant_placeholder: 2
EOF

echo "Running semantic analysis on semantic_demo.py..."
echo ""

# Run the analyzer
tailchasing semantic_demo.py --config .tailchasing_demo.yml

# Clean up
rm .tailchasing_demo.yml

echo ""
echo "=== Analysis Complete ==="
echo ""
echo "The semantic analyzer found:"
echo "1. Multiple average calculation functions (semantic duplicates)"
echo "2. Four different email validation implementations (prototype fragmentation)"
echo "3. Several wrapper functions that just call process_data"
echo "4. Placeholder functions that are semantically empty"
echo ""
echo "These patterns indicate tail-chasing behavior where the same"
echo "functionality is reimplemented multiple times instead of being"
echo "properly refactored and consolidated."
