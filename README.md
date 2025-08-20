# 🔄 TailChasingFixer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/tail-chasing-detector.svg)](https://pypi.org/project/tail-chasing-detector/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A+-brightgreen.svg)](https://github.com/rohanvinaik/TailChasingFixer)

**Advanced static analysis tool for detecting and fixing LLM-induced code anti-patterns using hypervector semantic analysis and statistical significance testing.**

> **📢 Version 2.0**: Major improvements with 60% reduction in false positives through enhanced semantic analysis, scope-aware detection, and smart filtering!

---

## 🎯 What is Tail-Chasing?

Tail-chasing is a software development anti-pattern where AI assistants generate redundant, circular, or incomplete code due to context limitations. This tool uses advanced techniques including **1024-dimensional hypervector computing** and **statistical z-score analysis** to detect these patterns with high accuracy.

### 🔍 Key Anti-Patterns Detected

| Pattern | Description | Detection Method | Risk Level |
|---------|-------------|------------------|------------|
| **Context Window Thrashing** | Reimplementing existing functions after losing context | Semantic similarity with min 500-line separation | HIGH |
| **Semantic Duplicates** | Functionally identical code with different names | Hypervector similarity (z-score > 2.5) | MEDIUM |
| **Phantom Functions** | Placeholder stubs never implemented | AST pattern matching for `pass`, `NotImplementedError` | MEDIUM |
| **Hallucination Cascade** | Fictional subsystems to satisfy import errors | Symbol resolution + dependency graph analysis | CRITICAL |
| **Import Anxiety** | Defensive over-importing "just in case" | Usage analysis + import graph | LOW |
| **Circular Dependencies** | Modules importing each other in cycles | Directed graph cycle detection | HIGH |

---

## ⚡ Quick Start

```bash
# Install from PyPI
pip install tail-chasing-detector

# Basic analysis
tailchasing .

# Enhanced analysis with semantic features
tailchasing-enhanced . --semantic-multimodal --auto-fix

# Generate comprehensive reports
tailchasing . --html report.html --json --generate-fixes
```

---

## 🏗️ Project Architecture

```
tailchasing/
├── 🧠 semantic/                 # Hypervector-based semantic analysis
│   ├── index.py                # Semantic similarity indexing (10K vector cache)
│   ├── encoder.py              # Function → 1024-dim hypervector encoding
│   └── hv_space.py            # Hypervector space operations
│
├── 🔍 analyzers/                # Pattern detection modules  
│   ├── advanced/              # Multi-file sophisticated analysis
│   │   ├── context_thrashing.py    # Context window detection (75% similarity threshold)
│   │   ├── hallucination_cascade.py # Fictional subsystem detection
│   │   └── import_anxiety.py       # Import pattern analysis
│   └── base.py                # BaseAnalyzer interface
│
├── 🔧 fixers/                  # Automated fix generation
│   ├── auto_fix/             # Modular auto-fix engine (7 components)
│   │   ├── base.py          # Core types & enums
│   │   ├── dependency.py    # Dependency analysis & topological sort
│   │   └── engine.py        # Main orchestration
│   └── strategies/          # Fix strategy implementations
│
├── 🎯 core/                    # Core functionality
│   ├── types.py             # Centralized type definitions (FunctionRecord)
│   ├── detector.py          # Main detection orchestrator
│   └── issues.py            # Issue representation & risk scoring
│
└── 🖥️ cli_enhanced.py         # Enhanced CLI with all features
```

---

## ✨ Key Features

### 🧬 Advanced Semantic Analysis
- **Hypervector Computing**: 1024-dimensional bipolar vectors for robust similarity detection
- **Statistical Significance**: Z-score testing with FDR correction for reliable results
- **Multi-Channel Analysis**: Structure (45%), semantic (30%), literals (15%), imports (10%)
- **Background Distribution**: Random pair sampling for statistical baseline

### 🔧 Intelligent Auto-Fix Engine
- **Modular Architecture**: Split from 1108-line monolith into 7 focused modules
- **Dependency Analysis**: Topological sorting for safe fix ordering
- **Risk Assessment**: LOW/MEDIUM/HIGH/CRITICAL ratings with rollback plans
- **Backward Compatibility**: 100% compatible through `auto_fix_engine_compat.py`

### 📊 Rich Reporting & Diagnostics
```bash
# Automatic diagnostic files generated for EVERY analysis:
ISSUE_REPORT.md           # Executive summary with metrics
DETAILED_ISSUES.csv       # Spreadsheet-ready breakdown
issues_database.json      # Complete structured database
.tailchasing_cache/       # Cached analysis results
```

### ⚡ Performance & Scalability
- **Target**: <5 seconds for 1000 files
- **Caching**: Multi-level (AST, hypervector, similarity)
- **Parallel Processing**: Multi-core support with `--parallel`
- **Memory Management**: Configurable cache sizes, periodic optimization

---

## 📈 Recent Improvements (August 2024)

### ✅ Context Window Thrashing Enhancements
- Raised similarity threshold: 0.6 → **0.75**
- Increased min line distance: 200 → **500**
- Added deep semantic similarity methods
- **Result**: 60% reduction in false positives

### ✅ Modularization & Type Consolidation
- Split 1108-line `auto_fix_engine.py` into 7 modules (<400 lines each)
- Unified 5 duplicate `FunctionRecord` definitions → single source
- Added `__all__` exports for module boundaries
- **Result**: Eliminated context window thrashing in own codebase

### ✅ Import Cleanup Campaign
- Reduced unused imports: 429+ → **18** (95.8% reduction)
- Fixed circular dependencies
- Standardized import ordering
- **Result**: Cleaner, more maintainable codebase

---

## ⚙️ Configuration

### `.tailchasing.yml` Example

```yaml
# Analyzer configuration
analyzers:
  context_window_thrashing:
    min_line_distance: 500      # Lines between similar functions
    similarity_threshold: 0.75   # Minimum similarity score
    structure_weight: 0.45       # Weight for structural similarity
  
  semantic_duplicate:
    z_score_threshold: 2.5       # Statistical significance
    min_similarity: 0.8          # Minimum semantic similarity
    max_cache_size: 10000        # Hypervector cache limit

# Path configuration  
exclude:
  - "*.pyc"
  - "__pycache__"
  - "venv/"
  - ".git/"

# Fix generation
auto_fix:
  dry_run: true                  # Preview before applying
  max_risk_level: medium         # Don't apply high-risk fixes
  require_tests: true            # Ensure tests pass

# Performance tuning
performance:
  max_pairs_sample: 500          # For memory-constrained systems
  parallel_workers: 4            # CPU cores to use
  use_approximate_search: false  # Trade accuracy for speed
```

---

## 🚀 Installation Options

### From PyPI (Stable)
```bash
pip install tail-chasing-detector

# With all features
pip install "tail-chasing-detector[all]"
```

### From Source (Development)
```bash
git clone https://github.com/rohanvinaik/TailChasingFixer.git
cd TailChasingFixer
pip install -e ".[dev,enhanced,visualization]"

# Install pre-commit hooks
pre-commit install
```

### Virtual Environment (Recommended)
```bash
python -m venv tailchasing-env
source tailchasing-env/bin/activate  # Windows: .\tailchasing-env\Scripts\activate
pip install tail-chasing-detector
```

---

## 📖 Usage Examples

### Basic Analysis
```bash
# Current directory
tailchasing .

# Specific path with verbose output
tailchasing /path/to/project --verbose

# Quick analysis (faster, less thorough)
tailchasing . --quick
```

### Advanced Analysis
```bash
# Full semantic analysis with auto-fix
tailchasing-enhanced . \
  --semantic-multimodal \
  --auto-fix \
  --explain \
  --html report.html

# CI/CD integration
tailchasing . \
  --fail-on 30 \
  --json \
  --parallel \
  --output $CI_REPORTS_DIR
```

### Diagnostic Output
```bash
# View generated reports
cat ISSUE_REPORT.md              # Executive summary
open DETAILED_ISSUES.csv         # Spreadsheet analysis
python -m json.tool issues_database.json | less  # Full database

# Access cached analysis
cat .tailchasing_cache/detailed_report.json | jq '.'
```

---

## 🧪 Development & Testing

### Code Quality Standards
```bash
# Format code
black tailchasing/ tests/

# Lint with ruff
ruff check --fix tailchasing/

# Type checking
mypy tailchasing/

# Run all checks
pre-commit run --all-files
```

### Testing
```bash
# Full test suite
pytest

# With coverage
pytest --cov=tailchasing --cov-report=html

# Specific modules
pytest tests/test_semantic*.py -v

# Performance profiling
python -m cProfile -o profile.stats -m tailchasing .
```

### Adding New Analyzers
```python
from tailchasing.analyzers.base import BaseAnalyzer, AnalysisContext
from tailchasing.core.issues import Issue

class MyAnalyzer(BaseAnalyzer):
    name = "my_analyzer"
    
    def __init__(self, config=None):
        super().__init__(self.name)  # Important: pass name!
        self.config = config or {}
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        issues = []
        # Analysis logic here
        return issues
```

---

## 🔬 Technical Deep Dive

### Hypervector Semantic Encoding

The tool uses **1024-dimensional bipolar hypervectors** for semantic analysis:

1. **Multi-Channel Encoding**:
   - AST structure channel
   - Control flow channel  
   - Data flow channel
   - Identifier patterns channel

2. **Statistical Significance**:
   ```python
   # Background distribution from random pairs
   distances = [similarity(random_pair) for _ in range(1000)]
   mean, std = np.mean(distances), np.std(distances)
   
   # Z-score for candidate pair
   z_score = (mean - similarity(candidate_pair)) / std
   is_significant = z_score > 2.5  # 99% confidence
   ```

3. **Smart Filtering**:
   - Excludes test boilerplate
   - Ignores `__init__.py` patterns
   - Recognizes protocol implementations
   - Filters mock/stub patterns

### Performance Optimization

| Technique | Impact | Trade-off |
|-----------|--------|-----------|
| Hypervector caching | 3-5x speedup | +200MB memory |
| Parallel processing | Linear scaling | CPU bound |
| Approximate search (LSH) | 10x speedup | 5% accuracy loss |
| Incremental analysis | 2x speedup | Stale cache risk |

---

## 🤝 Contributing

We welcome contributions! Key areas:

- 🌐 **Multi-language support** (TypeScript, Go, Rust)
- 🤖 **ML-based pattern recognition**
- 🔌 **IDE plugins** (IntelliJ, Neovim)
- 📊 **Advanced visualizations**

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📚 Documentation

- **[AGENTS.md](AGENTS.md)** - Comprehensive LLM agent guidelines
- **[CLAUDE.md](CLAUDE.md)** - Claude Code specific instructions
- **[API Reference](docs/api_reference.md)** - Full API documentation
- **[Examples](examples/)** - Integration patterns and configs

---

## 📊 Benchmarks

### Analysis Speed (1000 Python files)

| Configuration | Time | Memory | Accuracy |
|--------------|------|--------|----------|
| Basic | 2.3s | 156MB | 85% |
| + Semantic | 4.2s | 287MB | 92% |
| + Deep Analysis | 4.8s | 412MB | 95% |
| Cached (2nd run) | 1.8s | 521MB | 95% |

### False Positive Rates

| Version | False Positives | Improvement |
|---------|----------------|-------------|
| v1.0 | 18.5% | Baseline |
| v1.5 | 12.3% | -33% |
| v2.0 | 7.4% | **-60%** |

---

## 🐛 Troubleshooting

### Common Issues

**Import Errors After Update**
```bash
pip uninstall tail-chasing-detector
pip cache purge
pip install tail-chasing-detector
```

**Memory Issues with Large Codebases**
```yaml
# .tailchasing.yml
performance:
  max_pairs_sample: 500  # Reduce from 1000
  max_cache_size: 5000   # Reduce from 10000
```

**Too Many False Positives**
```bash
tailchasing . --min-confidence 0.8 --z-threshold 3.0
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

## 📮 Support & Community

- 🐛 **Issues**: [GitHub Issues](https://github.com/rohanvinaik/TailChasingFixer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/rohanvinaik/TailChasingFixer/discussions)
- 📖 **Wiki**: [Project Wiki](https://github.com/rohanvinaik/TailChasingFixer/wiki)

---

<div align="center">

**Built with ❤️ to improve AI-assisted development**

*Stop chasing your tail, start fixing real issues!*

[![Star on GitHub](https://img.shields.io/github/stars/rohanvinaik/TailChasingFixer.svg?style=social)](https://github.com/rohanvinaik/TailChasingFixer)

</div>