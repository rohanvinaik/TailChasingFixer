# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TailChasingFixer is a Python package that detects LLM-assisted tail-chasing anti-patterns in codebases. It uses advanced semantic analysis including hypervector computing to identify duplicate functions, circular imports, phantom implementations, and other code quality issues that commonly arise from LLM-generated code.

## Common Development Commands

### Installation
```bash
# Install in development/editable mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install specific feature sets
pip install -e ".[dev]"          # Development tools
pip install -e ".[enhanced]"     # Enhanced analysis features
pip install -e ".[visualization]" # Visualization capabilities
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=tailchasing

# Run specific test file
pytest tests/test_duplicates.py

# Run tests with verbose output
pytest -v
```

### Code Quality Tools
```bash
# Format code with black
black tailchasing/ tests/

# Check code style with ruff
ruff check tailchasing/

# Fix auto-fixable issues with ruff
ruff check --fix tailchasing/

# Type checking with mypy
mypy tailchasing/

# Run all checks (typical before committing)
black tailchasing/ tests/ && ruff check tailchasing/ && mypy tailchasing/
```

### Building and Distribution
```bash
# Build the package
python -m build

# Build for PyPI distribution
bash scripts/build_pypi.sh

# Publish to PyPI (requires credentials)
bash scripts/publish_pypi.sh
```

### Running the Tool
```bash
# Basic analysis on current directory
tailchasing .

# Enhanced analysis with semantic features
tailchasing-enhanced . --enhanced --semantic-multimodal

# Generate HTML report
tailchasing . --html report.html

# Auto-fix with explanations
tailchasing-enhanced . --auto-fix --explain
```

## Architecture

### Core Components

1. **Analyzers** (`tailchasing/analyzers/`): Individual analysis modules
   - `circular_import.py`: Detects circular dependencies
   - `duplicate_functions.py`: Finds structurally identical functions
   - `phantom_functions.py`: Identifies placeholder/stub functions
   - `missing_symbol.py`: Finds references to non-existent symbols
   - `semantic_duplicate.py`: Semantic similarity detection
   - `hallucination_cascade.py`: Detects fictional subsystems
   - `context_window_thrashing.py`: Finds reimplementations
   - `import_anxiety.py`: Identifies over-importing patterns

2. **Semantic Analysis** (`tailchasing/semantic/`): Advanced pattern detection
   - `hypervector.py`: High-dimensional semantic encoding
   - `multimodal.py`: Multi-modal semantic analysis
   - `smart_filter.py`: Reduces false positives
   - `prototype_detector.py`: Finds multiple implementations of same concept

3. **Core Engine** (`tailchasing/core/`):
   - `detector.py`: Main detection orchestrator
   - `ast_analyzer.py`: AST parsing and traversal
   - `import_graph.py`: Import dependency analysis
   - `issue.py`: Issue representation and risk scoring

4. **CLI Interfaces**:
   - `cli.py`: Standard command-line interface
   - `cli_enhanced.py`: Enhanced interface with advanced features

5. **Fixers** (`tailchasing/fixers/`): Auto-fix generation
   - `suggestion_generator.py`: Creates fix suggestions
   - `fix_applier.py`: Applies fixes to code

### Key Design Patterns

- **Plugin Architecture**: Analyzers implement common interface for extensibility
- **Visitor Pattern**: AST traversal using Python's ast module
- **Configuration-Driven**: YAML configs control behavior (`.tailchasing.yml`)
- **Multi-Format Output**: Supports text, JSON, and HTML reports

### Dependencies

Core dependencies include:
- `click`: CLI framework
- `rich`: Terminal formatting
- `numpy/scipy`: Numerical computations for semantic analysis
- `networkx`: Graph analysis for import cycles
- `pyyaml`: Configuration parsing

## Important Implementation Details

1. **Semantic Analysis Threshold**: Uses z-score threshold (default 2.5) to determine statistical significance of semantic duplicates

2. **Risk Scoring**: Each issue type has configurable weights that contribute to overall risk score

3. **Smart Filtering**: Automatically excludes legitimate patterns like `__init__.py` files, test boilerplate, and protocol implementations to reduce false positives

4. **File Traversal**: Respects `.gitignore` patterns and configuration exclusions when scanning directories

5. **AST-Based Analysis**: Uses Python's ast module for accurate code parsing rather than regex matching

## VS Code Extension

A companion VS Code extension is available in `vscode-extension/` that provides:
- Real-time detection of tail-chasing patterns
- Inline suggestions and quick fixes
- Integration with the CLI tool