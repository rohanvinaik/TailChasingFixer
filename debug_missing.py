#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from pathlib import Path
from tailchasing.core.loader import collect_files, parse_files
from tailchasing.core.symbols import SymbolTable
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.analyzers.missing_symbols import MissingSymbolAnalyzer

# Setup
root = Path('examples/demo_target')
files = collect_files(root)
ast_index = parse_files(files)
symbol_table = SymbolTable()

for file, tree in ast_index.items():
    source = Path(file).read_text()
    symbol_table.ingest_file(file, tree, source)

# Create context
ctx = AnalysisContext(
    config={},
    root_dir=root,
    file_paths=files,
    ast_index=ast_index,
    symbol_table=symbol_table,
    source_cache={}
)

# Run analyzer
analyzer = MissingSymbolAnalyzer()
issues = list(analyzer.run(ctx))

print(f"Found {len(issues)} missing symbol issues:")
for issue in issues:
    print(f"  [{issue.kind}] {issue.file}:{issue.line} - {issue.message}")
