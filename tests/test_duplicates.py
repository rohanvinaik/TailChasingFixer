"""Test for duplicate detection."""

import pytest
from pathlib import Path
import ast

from tailchasing.analyzers.duplicates import DuplicateFunctionAnalyzer, FunctionCollector


def test_structural_hash_detects_duplicates():
    """Test that structural hashing identifies duplicate functions."""
    
    code = '''
def func1(x, y):
    total = x + y
    return total * 2

def func2(a, b):
    sum_val = a + b
    return sum_val * 2
    
def func3(x, y):
    return x + y
'''
    
    tree = ast.parse(code)
    analyzer = DuplicateFunctionAnalyzer()
    
    # Extract functions
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            hash_val = analyzer._structural_hash(node)
            functions.append((node.name, hash_val))
            
    # func1 and func2 should have same hash (structurally identical)
    assert functions[0][1] == functions[1][1]
    
    # func3 should have different hash
    assert functions[2][1] != functions[0][1]


def test_function_collector():
    """Test function collection from AST."""
    
    code = '''
class MyClass:
    def method1(self):
        pass
        
    def method2(self):
        return 42

def standalone_func():
    pass
'''
    
    tree = ast.parse(code)
    collector = FunctionCollector("test.py")
    collector.visit(tree)
    
    assert len(collector.functions) == 3
    
    # Check function names
    names = [f["name"] for f in collector.functions]
    assert "method1" in names
    assert "method2" in names
    assert "standalone_func" in names
    
    # Check full names
    full_names = [f["full_name"] for f in collector.functions]
    assert "MyClass.method1" in full_names
    assert "MyClass.method2" in full_names
    assert "standalone_func" in full_names
