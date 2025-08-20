# CLAUDE.md - CRITICAL INSTRUCTIONS FOR CLAUDE

**⚠️ READ THIS FIRST - This file contains critical instructions for working with the TailChasingFixer codebase.**

## 🚨 CRITICAL: Never Create Mock Analyzers

When users ask for new analysis features:
1. **USE THE EXISTING FRAMEWORK** - All analyzers must inherit from `BaseAnalyzer`
2. **FOLLOW THE PATTERN** - Check existing analyzers in `tailchasing/analyzers/`
3. **NO SIMPLIFIED VERSIONS** - The tool must perform real AST analysis, not mock detection
4. **MAINTAIN PERFORMANCE** - Target <5 seconds for 1000 files

## Project Overview

TailChasingFixer is a Python package that detects LLM-assisted tail-chasing anti-patterns in codebases. It uses advanced semantic analysis including hypervector computing to identify duplicate functions, circular imports, phantom implementations, and other code quality issues that commonly arise from LLM-generated code.

### Mission Statement
This tool addresses a critical problem in modern software development: the accumulation of redundant, incomplete, and inconsistent code patterns that arise when Large Language Models (LLMs) lose context during extended coding sessions. By detecting and fixing these patterns, TailChasingFixer helps maintain code quality and reduce technical debt in AI-assisted development.

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

6. **Hypervector Computing**: Uses 1024-dimensional bipolar vectors for semantic encoding, enabling robust similarity detection even with structural variations

7. **Statistical Background Model**: Maintains a background distribution of random function pairs to establish statistical significance of detected patterns

## Common Anti-Patterns to Watch For

When modifying this codebase, be aware of these patterns that the tool itself is designed to detect:

### 1. Context Window Thrashing
- **Symptom**: Reimplementing similar functionality in different parts of the codebase
- **Example**: The original `auto_fix_engine.py` was 1108 lines and has been modularized
- **Prevention**: Keep files under 400 lines, use clear module boundaries

### 2. Type Definition Duplication
- **Symptom**: Multiple definitions of the same types (e.g., FunctionRecord)
- **Fixed**: Consolidated into `core/types.py`
- **Prevention**: Always check `core/types.py` before defining new types

### 3. Import Anxiety
- **Symptom**: Importing modules "just in case" or importing the same module multiple ways
- **Prevention**: Use explicit imports, remove unused imports immediately

### 4. Phantom Implementations
- **Symptom**: Stub functions with `pass`, `raise NotImplementedError`, or TODO comments
- **Prevention**: Implement functions immediately or document why they're stubs

## Critical Files and Their Roles

### Core Type System (`core/types.py`)
- **FunctionRecord**: Central type for function metadata
- **ImportInfo**: Import relationship tracking
- **IssueKind**: Enum of all detected anti-patterns
- **CRITICAL**: All new analyzers must use these types, never redefine

### Semantic Index (`semantic/index.py`)
- **Purpose**: Maintains hypervector representations of all functions
- **Key Methods**: `add_function()`, `find_similar()`, `compute_z_scores()`
- **Memory**: Caches up to 10,000 hypervectors by default
- **NOTE**: Fixed indentation issues at lines 314-349 in August 2024

### Context Thrashing Detector (`analyzers/advanced/context_thrashing.py`)
- **Thresholds**: min_line_distance=500, similarity_threshold=0.75
- **Weights**: structure=45%, semantic=30%, literal=15%, imports=10%
- **IMPORTANT**: Must call `super().__init__(self.name)` not just `super().__init__()`

### Auto-Fix Engine (`fixers/auto_fix/`)
- **Modularized**: Split from single 1108-line file into 7 modules
- **Components**: base, backup, dependency, safety, registry, engine, factory
- **Compatibility**: `auto_fix_engine_compat.py` maintains backward compatibility

## Common Pitfalls and Solutions

### Pitfall 1: BaseAnalyzer Initialization
```python
# WRONG - Missing name parameter
class MyAnalyzer(BaseAnalyzer):
    def __init__(self, config=None):
        super().__init__()  # ERROR!

# CORRECT
class MyAnalyzer(BaseAnalyzer):
    name = "my_analyzer"
    def __init__(self, config=None):
        super().__init__(self.name)  # Pass name to parent
```

### Pitfall 2: Semantic Encoder Config
```python
# WRONG - Missing config parameter
encoder.encode_function(func_info)

# CORRECT - Always pass config (even if empty)
encoder.encode_function(func_info, config={})
```

### Pitfall 3: Module Boundaries
```python
# WRONG - Forgetting __all__ exports
# my_module.py
def public_function(): ...
def _private_helper(): ...

# CORRECT - Define public interface
# my_module.py
__all__ = ['public_function']  # Controls what's exported
def public_function(): ...
def _private_helper(): ...
```

## Performance Optimization Tips

1. **Semantic Index**: Call `optimize_cache()` periodically for long-running processes
2. **Large Codebases**: Use `--parallel` flag with worker count matching CPU cores
3. **Memory Constraints**: Reduce `max_pairs_sample` from 1000 to 500
4. **Speed vs Accuracy**: Use `--quick` flag for faster but less thorough analysis

## Testing Strategy

### Unit Tests
- Each analyzer has dedicated test file: `test_<analyzer_name>.py`
- Semantic modules have comprehensive tests: ~60% coverage achieved
- Use pytest fixtures for common test data

### Integration Tests
- `tests/integration/` contains end-to-end tests
- Test real code patterns from `test_data/` directory
- Verify fix generation doesn't break code

### Performance Tests
- Profile with `python -m cProfile` for bottlenecks
- Monitor memory with `memory_profiler` for leaks
- Benchmark semantic index with various cache sizes

## VS Code Extension

A companion VS Code extension is available in `vscode-extension/` that provides:
- Real-time detection of tail-chasing patterns
- Inline suggestions and quick fixes
- Integration with the CLI tool

## 🎯 Key Success Metrics

When validating or testing, these are the targets:
- **Detection Rate**: >85% (currently 92%)
- **False Positive Rate**: <10% (currently 7.4%)
- **Analysis Speed**: <5s for 1000 files
- **Memory Usage**: <500MB for typical codebases
- **Hypervector Dimension**: 1024 (fixed)
- **Statistical Significance**: z-score > 2.5

## Recent Changes (August 2024)

1. **Context Window Thrashing Improvements**
   - Raised similarity threshold from 0.6 to 0.75
   - Increased min_line_distance from 200 to 500
   - Added semantic similarity methods for deeper analysis
   - Reduced false positives by ~60%

2. **Modularization of Large Files**
   - Split 1108-line auto_fix_engine.py into 7 modules
   - Each module now under 400 lines
   - Maintained 100% backward compatibility

3. **Type Consolidation**
   - Unified 5 duplicate FunctionRecord definitions
   - Centralized in core/types.py
   - Added comprehensive type hints

4. **Module Boundary Enforcement**
   - Added __all__ exports to control public interfaces
   - Prevents accidental internal API usage
   - Improves code maintainability

5. **Import Cleanup**
   - Reduced unused imports from 429+ to 18 (95.8% reduction)
   - Fixed circular import issues
   - Standardized import ordering