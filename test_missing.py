#!/usr/bin/env python3
from pathlib import Path
import ast
from tailchasing.analyzers.missing_symbols import MissingSymbolAnalyzer, SymbolCollector, ReferenceVisitor
from collections import defaultdict

# Load handler.py
handler_file = Path('examples/demo_target/handler.py')
code = handler_file.read_text()
tree = ast.parse(code)

# Collect symbols defined
collector = SymbolCollector(str(handler_file))
collector.visit(tree)
print("Symbols defined in handler.py:")
for name, locs in collector.symbols.items():
    print(f"  {name}: {locs}")

# Collect references
ref_visitor = ReferenceVisitor(str(handler_file))
ref_visitor.visit(tree)
print("\nReferences in handler.py:")
missing_refs = ['apply_advanced_transform', 'validate_with_ml_model', 'preprocess_data']
for ref in ref_visitor.references:
    if ref['name'] in missing_refs:
        print(f"  {ref['name']} at line {ref['line']}")
