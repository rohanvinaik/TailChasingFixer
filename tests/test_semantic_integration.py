"""Integration test for semantic analyzer."""

import ast
import pytest
from pathlib import Path

from tailchasing.core.loader import parse_files
from tailchasing.core.symbols import SymbolTable
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.analyzers.semantic_hv import SemanticHVAnalyzer


def test_semantic_analyzer_integration():
    """Test the semantic analyzer on example code."""
    
    # Configuration
    config = {
        'semantic': {
            'enable': True,
            'hv_dim': 1024,  # Smaller for testing
            'min_functions': 5,
            'z_threshold': 2.0,
            'fdr_q': 0.05,
            'channel_weights': {
                'NAME_TOKENS': 1.0,
                'CALLS': 1.2,
                'DOC_TOKENS': 0.8
            }
        }
    }
    
    # Parse example code
    example_code = '''
# Semantic duplicates
def calculate_sum(numbers):
    """Calculate total of numbers."""
    total = 0
    for n in numbers:
        total += n
    return total

def compute_total(values):
    """Compute sum of values."""
    result = 0
    for val in values:
        result = result + val
    return result

def add_all(items):
    """Add all items together."""
    s = 0
    for i in items:
        s += i
    return s

# Different function
def multiply_all(items):
    """Multiply all items."""
    product = 1
    for i in items:
        product *= i
    return product

# Placeholder
def analyze_data():
    """Analyze the data."""
    pass
'''
    
    # Create temporary file
    test_file = Path("test_semantic.py")
    test_file.write_text(example_code)
    
    try:
        # Parse and build symbol table
        ast_index = parse_files([test_file])
        symbol_table = SymbolTable()
        
        for file, tree in ast_index.items():
            symbol_table.ingest(file, tree, example_code)
        
        # Create context
        ctx = AnalysisContext(
            config=config,
            files=[test_file],
            ast_index=ast_index,
            symbol_table=symbol_table,
            cache={}
        )
        
        # Run analyzer
        analyzer = SemanticHVAnalyzer()
        issues = analyzer.run(ctx)
        
        # Check results
        assert len(issues) > 0
        
        # Should find semantic duplicates
        semantic_dups = [i for i in issues if i.kind == "semantic_duplicate_function"]
        assert len(semantic_dups) >= 1
        
        # Should find placeholder
        placeholders = [i for i in issues if i.kind == "semantic_stagnant_placeholder"]
        assert len(placeholders) >= 1
        
        # Verify the duplicate involves our sum functions
        found_sum_duplicate = False
        for issue in semantic_dups:
            evidence = issue.evidence
            pair = evidence.get('pair', [])
            if pair:
                names = [p['name'] for p in pair]
                if any('sum' in n or 'total' in n or 'add' in n for n in names):
                    found_sum_duplicate = True
                    break
        
        assert found_sum_duplicate, "Should detect sum/total/add functions as semantic duplicates"
        
    finally:
        # Cleanup
        test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    test_semantic_analyzer_integration()
    print("Semantic analyzer integration test passed!")
