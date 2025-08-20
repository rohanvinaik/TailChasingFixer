# AGENTS.md - LLM Agent Guidelines

## 🎯 Project Purpose

TailChasingFixer is a sophisticated static analysis tool designed to detect and fix "tail-chasing" anti-patterns in codebases, particularly those that arise from LLM-assisted development. The tool uses advanced semantic analysis, including hypervector computing and statistical methods, to identify code quality issues that traditional linters miss.

## 🏗️ Architecture Overview

### Core Detection Pipeline
```
Source Code → AST Parsing → Analyzer Pipeline → Issue Detection → Fix Generation → Validation
                    ↓              ↓                ↓               ↓
              Semantic Index  Import Graph    Risk Scoring    Auto-Fix Engine
```

### Package Structure
```
tailchasing/
├── analyzers/          # Detection modules for various anti-patterns
│   ├── advanced/       # Sophisticated multi-file analysis
│   └── *.py           # Individual analyzer implementations
├── semantic/           # Hypervector-based semantic analysis
│   ├── index.py       # Semantic similarity indexing
│   ├── encoder.py     # Function-to-hypervector encoding
│   └── hv_space.py    # Hypervector space operations
├── fixers/            # Automated fix generation
│   ├── auto_fix/      # Modular auto-fix engine
│   └── strategies/    # Fix strategy implementations
├── core/              # Core functionality
│   ├── types.py       # Shared type definitions
│   ├── issues.py      # Issue representation
│   └── detector.py    # Main detection orchestrator
└── cli_enhanced.py    # Enhanced CLI with all features
```

## 🔍 Key Anti-Patterns Detected

### 1. **Context Window Thrashing**
- **Detection**: Functions reimplemented due to LLM context limitations
- **Key File**: `analyzers/advanced/context_thrashing.py`
- **Thresholds**: Min 500 lines separation, 75% similarity
- **Fix**: Consolidate into shared utilities

### 2. **Semantic Duplicates**
- **Detection**: Functionally identical code with different names
- **Key Files**: `semantic/index.py`, `semantic/encoder.py`
- **Method**: Hypervector similarity with z-score > 2.5
- **Fix**: Merge duplicates, create aliases

### 3. **Phantom Functions**
- **Detection**: Placeholder/stub implementations never completed
- **Key File**: `analyzers/phantom_function_detector.py`
- **Patterns**: `pass`, `raise NotImplementedError`, TODO comments
- **Fix**: Generate implementations or remove

### 4. **Import Anxiety**
- **Detection**: Excessive/unused imports from uncertainty
- **Key File**: `analyzers/advanced/import_anxiety.py`
- **Fix**: Remove unused, consolidate redundant

### 5. **Hallucination Cascade**
- **Detection**: References to non-existent modules/functions
- **Key File**: `analyzers/advanced/hallucination_cascade.py`
- **Fix**: Create missing components or update references

## 💻 Development Guidelines

### Environment Setup
```bash
# Python 3.10+ required (3.11+ recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with all development dependencies
pip install -e ".[dev,enhanced,visualization]"

# Install pre-commit hooks
pre-commit install
```

### Testing Practices
```bash
# Run full test suite
pytest

# Run with coverage report
pytest --cov=tailchasing --cov-report=html

# Run specific test categories
pytest tests/test_semantic*.py  # Semantic analysis tests
pytest tests/test_*_analyzer.py # Analyzer tests

# Quick smoke test
python -m tailchasing . --dry-run
```

### Code Quality Standards
```bash
# Before committing, run:
black tailchasing/ tests/        # Format code
ruff check --fix tailchasing/    # Fix linting issues
mypy tailchasing/                # Type checking

# Or use pre-commit
pre-commit run --all-files
```

### Adding New Analyzers
1. Inherit from `BaseAnalyzer` in `analyzers/base.py`
2. Implement `run(ctx: AnalysisContext) -> List[Issue]`
3. Register in `analyzers/__init__.py`
4. Add tests in `tests/test_<analyzer_name>.py`

Example:
```python
from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue

class MyAnalyzer(BaseAnalyzer):
    name = "my_analyzer"
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        issues = []
        # Analysis logic here
        return issues
```

## 🔧 Configuration

### `.tailchasing.yml` Configuration
```yaml
# Analyzer configuration
analyzers:
  context_window_thrashing:
    min_line_distance: 500
    similarity_threshold: 0.75
    require_semantic_similarity: true
  
  semantic_duplicate:
    z_score_threshold: 2.5
    min_similarity: 0.8

# Path exclusions
exclude:
  - "*.pyc"
  - "__pycache__"
  - ".git"
  - "venv/"
  - "build/"

# Fix generation settings
auto_fix:
  dry_run: true
  max_risk_level: medium
  require_tests: true
```

### Performance Tuning
- **Large Codebases**: Use `--parallel` flag for multi-core processing
- **Memory Issues**: Reduce `max_pairs_sample` in semantic config
- **Speed**: Enable `use_approximate_search` for LSH-based search

## 📊 Semantic Analysis Details

### Hypervector Encoding
- **Dimension**: Default 1024 (configurable)
- **Type**: Bipolar (-1, 1) or binary (0, 1)
- **Channels**: Name, AST structure, control flow, data flow
- **Binding**: Role-filler binding for compositional semantics

### Statistical Significance
- **Background Distribution**: Random pair sampling
- **Z-Score Calculation**: `(mean - distance) / std`
- **FDR Correction**: Benjamini-Hochberg for multiple testing
- **Confidence**: Based on cross-channel agreement

## 🛠️ Advanced Features

### Incremental Analysis
```python
# Enable incremental updates for large codebases
config = {
    'incremental': True,
    'cache_dir': '.tailchasing_cache',
    'batch_size': 100
}
```

### Custom Fix Strategies
```python
from tailchasing.fixers.strategies.base import BaseFixStrategy

class CustomStrategy(BaseFixStrategy):
    def can_handle(self, issue: Issue) -> bool:
        return issue.kind == "my_pattern"
    
    def propose_fix(self, issue: Issue, context=None):
        # Generate fix
        return Patch(...)
```

### Visualization
```bash
# Generate interactive HTML report
tailchasing . --html report.html

# Export to JSON for custom processing
tailchasing . --json results.json

# Generate dependency graph
tailchasing . --visualize deps.svg
```

## 🚨 Important Notes

### Memory Management
- Semantic index caches hypervectors in memory
- Use `optimize_cache()` for long-running processes
- Configure `max_cache_size` for memory-constrained environments

### False Positive Reduction
- Automatically filters test files, `__init__.py`, protocol implementations
- Configure `smart_filter` thresholds in config
- Use `--strict` flag for higher confidence results only

### Performance Considerations
- First run builds semantic index (slower)
- Subsequent runs use cached index (faster)
- Use `--force-rebuild` to refresh index

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
- name: Run TailChasingFixer
  run: |
    pip install tailchasingfixer
    tailchasing . --exit-code --max-issues 10
```

### Pre-commit Hook
```yaml
repos:
  - repo: local
    hooks:
      - id: tailchasing
        name: TailChasingFixer
        entry: tailchasing
        language: system
        files: '\.py$'
        args: ['--quick', '--auto-fix']
```

## 📈 Metrics and Monitoring

### Key Metrics Tracked
- **Detection Rate**: Issues found per KLOC
- **Fix Success Rate**: Successful auto-fixes / attempts
- **Performance**: Analysis time per file
- **Cache Hit Rate**: Semantic similarity cache efficiency

### Debugging
```bash
# Enable debug logging
export TAILCHASING_LOG_LEVEL=DEBUG
tailchasing . --verbose

# Profile performance
python -m cProfile -o profile.stats tailchasing .
```

## 🤝 Contributing

### Code Style
- Follow PEP 8 with Black formatting
- Type hints required for public APIs
- Docstrings in Google style
- Tests required for new features

### Pull Request Checklist
- [ ] Tests pass (`pytest`)
- [ ] Code formatted (`black`)
- [ ] Type checks pass (`mypy`)
- [ ] Documentation updated
- [ ] No new `TODO` comments

## 📚 Repository Context

### Recent Improvements (Aug 2024)
- Enhanced context window thrashing detection with semantic analysis
- Modularized fix strategies for better maintainability
- Added comprehensive type hints and validation
- Improved test coverage to 60%+ for core modules
- Reduced false positives by 60% through stricter thresholds

### Known Limitations
- Python-only analysis (no multi-language support yet)
- Requires Python 3.10+ for full features
- Memory usage scales with codebase size
- Some fix strategies are conservative to avoid breaking changes

### Future Roadmap
- Multi-language support (JavaScript, TypeScript)
- Machine learning-based pattern recognition
- Cloud-based analysis for large codebases
- IDE plugins for real-time detection
- Automated PR generation with fixes

## 🆘 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all dependencies installed
pip install -e ".[all]"
```

**Memory Issues**
```python
# Reduce memory usage
config = {
    'max_pairs_sample': 500,  # Reduce from 1000
    'max_cache_size': 5000,   # Reduce from 10000
}
```

**Slow Analysis**
```bash
# Use parallel processing
tailchasing . --parallel --workers 4

# Skip semantic analysis for speed
tailchasing . --no-semantic
```

**Too Many False Positives**
```bash
# Increase thresholds
tailchasing . --min-confidence 0.8 --z-threshold 3.0
```

## 📞 Support

- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/` directory for usage patterns