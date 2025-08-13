# TailChasing Fixer 🔍

[![Python versions](https://img.shields.io/pypi/pyversions/tail-chasing-detector.svg)](https://pypi.org/project/tail-chasing-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type coverage](https://img.shields.io/badge/type%20coverage-%3E80%25-brightgreen)](https://github.com/rohanvinaik/TailChasingFixer)

**Detect and fix LLM-induced tail-chasing anti-patterns** - circular, redundant, and hallucinatory code patterns that emerge when Large Language Models repeatedly make superficial fixes without addressing root causes.

## 📚 Table of Contents

- [Architecture Overview](#-architecture-overview)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Advanced Features](#-advanced-features)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#license)

## 🏗️ Architecture Overview

TailChasing Fixer uses a modular, layered architecture designed for extensibility and performance:

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Interface                        │
│  (Typer-based CLI with Rich formatting & interactivity) │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Analysis Engine                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Pattern   │  │   Semantic   │  │     Fix      │  │
│  │  Detectors  │  │   Analyzers  │  │  Strategies  │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Core Components                      │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │    AST      │  │    Symbol    │  │   Metrics    │  │
│  │   Parser    │  │    Table     │  │  Collector   │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                  Infrastructure Layer                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Caching   │  │   Parallel   │  │     LLM      │  │
│  │   System    │  │  Processing  │  │  Adapters    │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Key Components

- **Pattern Detectors**: Identify specific anti-patterns (phantoms, duplicates, circular imports)
- **Semantic Analyzers**: Deep semantic analysis using hypervector computing
- **Fix Strategies**: Automated fix generation with impact analysis
- **Convergence Engine**: Ensures fixes converge to stable solutions
- **LLM Integration**: Optional LLM-powered fix generation and explanation

## 🚀 Quick Start

### Basic Analysis

```bash
# Install
pip install tail-chasing-detector

# Analyze current directory
tailchasing .

# Analyze with enhanced detection
tailchasing . --enhanced --semantic

# Generate and apply fixes
tailchasing fix . --interactive
```

### Advanced Usage

```bash
# Full analysis with all features
tailchasing analyze . \
  --enhanced \
  --semantic \
  --git-history \
  --format json \
  --output report.json

# Interactive fix mode with backups
tailchasing fix . \
  --mode interactive \
  --backup \
  --validate

# Generate HTML report with visualizations
tailchasing analyze . \
  --format html \
  --output report.html
```

## 💿 Installation

### From PyPI (Recommended)

```bash
pip install tail-chasing-detector

# With all optional dependencies
pip install tail-chasing-detector[all]

# With specific features
pip install tail-chasing-detector[semantic,llm,viz]
```

### From Source

```bash
git clone https://github.com/rohanvinaik/TailChasingFixer.git
cd TailChasingFixer
pip install -e .

# For development
pip install -e ".[dev]"
```

### Using Poetry

```bash
poetry add tail-chasing-detector

# For development
poetry install --with dev
```

### Using Conda

```bash
conda create -n tailchasing python=3.11
conda activate tailchasing
pip install tail-chasing-detector
```

## ⚙️ Configuration

### Configuration File (.tailchasing.yml)

Create a `.tailchasing.yml` file in your project root:

```yaml
# Path configuration
paths:
  include:
    - src
    - lib
  exclude:
    - tests
    - __pycache__
    - "*.egg-info"

# Analysis settings
analysis:
  enhanced_detection: true
  semantic_analysis: true
  git_history: false
  parallel: true
  cache: true

# Semantic analysis configuration
semantic:
  vector_dim: 16384
  similarity_threshold: 0.85
  min_functions: 30

# Risk thresholds
risk_thresholds:
  warn: 15
  fail: 30

# Fix settings
fixes:
  backup: true
  validate: true
  mode: interactive

# Output settings
output:
  format: text
  show_explanations: true
  severity_threshold: 2

# LLM configuration (optional)
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.3
  max_tokens: 2000
```

### Environment Variables

```bash
# Export API keys for LLM features
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Configure behavior
export TAILCHASING_CONFIG="custom-config.yml"
export TAILCHASING_CACHE_DIR="~/.cache/tailchasing"
export TAILCHASING_LOG_LEVEL="INFO"
```

### Command-Line Options

All configuration options can be overridden via command-line:

```bash
tailchasing analyze . \
  --config custom.yml \
  --enhanced \
  --no-cache \
  --parallel \
  --severity 3
```

## 📖 Usage Examples

### Example 1: Detecting Phantom Functions

```python
# phantom_example.py
class DataProcessor:
    def process(self, data):
        """Process data."""
        pass  # Phantom implementation
    
    def validate(self, data):
        """Validate data."""
        raise NotImplementedError()  # Another phantom
```

Detection:
```bash
$ tailchasing analyze phantom_example.py
🔍 Found 2 phantom functions:
  - DataProcessor.process (line 3): Empty implementation
  - DataProcessor.validate (line 7): NotImplementedError stub
```

### Example 2: Finding Semantic Duplicates

```python
# duplicate_example.py
def calculate_average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0

def compute_mean(values):
    if not values:
        return 0
    return sum(values) / len(values)

def get_avg(data):
    return 0 if not data else sum(data) / len(data)
```

Detection:
```bash
$ tailchasing analyze duplicate_example.py --semantic
🔍 Found semantic duplicate cluster:
  - calculate_average (line 2)
  - compute_mean (line 5)
  - get_avg (line 10)
  Similarity: 94.3%
```

### Example 3: Circular Dependencies

```python
# module_a.py
from module_b import FunctionB

def FunctionA():
    return FunctionB()

# module_b.py
from module_a import FunctionA

def FunctionB():
    return FunctionA()
```

Detection:
```bash
$ tailchasing analyze . 
⚠️  Circular dependency detected:
  module_a.py → module_b.py → module_a.py
  Risk: HIGH (runtime import error likely)
```

## 🎯 Advanced Features

### Semantic Hypervector Analysis

Uses high-dimensional computing for deep semantic understanding:

```bash
# Enable 16384-dimensional semantic analysis
tailchasing analyze . --semantic --vector-dim 16384

# Adjust similarity threshold
tailchasing analyze . --semantic --similarity-threshold 0.9
```

### Multimodal Analysis

Analyzes code across multiple dimensions:
- **Data Flow**: How data moves through functions
- **Control Flow**: Branching and loop patterns
- **Error Handling**: Exception patterns
- **Return Patterns**: What functions return
- **Type Patterns**: Type hints and usage

### Intelligent Auto-Fixing

Generate and apply fixes automatically:

```bash
# Generate fix plan
tailchasing fix . --plan-only --output fixes.json

# Apply fixes interactively
tailchasing fix . --interactive

# Apply all safe fixes automatically
tailchasing fix . --mode automatic --severity 1-2
```

### Convergence Engine

Ensures fixes reach stable solutions:

```python
from tailchasing.engine import ConvergenceEngine

engine = ConvergenceEngine(max_iterations=10)
result = engine.converge(
    initial_issues=issues,
    fix_strategy=strategy,
    validator=validator
)
```

### LLM Integration

Enhance analysis with LLM capabilities:

```bash
# Use GPT-4 for fix generation
tailchasing fix . --llm gpt-4 --explain

# Use Claude for natural language explanations
tailchasing analyze . --llm claude-3 --explain
```

### Benchmark Suite

Evaluate performance across scenarios:

```bash
# Run benchmarks
python benchmarks/run_benchmarks.py single --model gpt-4

# Compare models
python benchmarks/run_benchmarks.py compare \
  --compare-models gpt-4,claude-3,llama-70b

# Track performance over time
python benchmarks/run_benchmarks.py track --plot
```

## 📚 API Reference

### Core Classes

```python
from tailchasing import TailChasingDetector

# Initialize detector
detector = TailChasingDetector(config={
    "enhanced_detection": True,
    "semantic_analysis": True
})

# Analyze codebase
issues = detector.analyze("src/")

# Generate fixes
fixes = detector.generate_fixes(issues)

# Apply fixes
detector.apply_fixes(fixes, validate=True)
```

### Analyzer Interface

```python
from tailchasing.analyzers import SemanticAnalyzer

analyzer = SemanticAnalyzer(vector_dim=16384)
duplicates = analyzer.find_duplicates(
    functions=ast_nodes,
    threshold=0.85
)
```

### Fix Strategies

```python
from tailchasing.fixers import FixStrategy

class CustomStrategy(FixStrategy):
    def generate_fix(self, issue):
        """Generate fix for issue."""
        # Custom fix logic
        return Fix(...)

    def validate_fix(self, fix, context):
        """Validate fix won't cause regressions."""
        return True
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'tailchasing'`

**Solution**:
```bash
# Ensure proper installation
pip install --upgrade tail-chasing-detector

# Or install from source
pip install -e .
```

#### 2. Memory Issues with Large Codebases

**Problem**: High memory usage or OOM errors

**Solution**:
```bash
# Limit parallel processing
tailchasing analyze . --no-parallel

# Use incremental analysis
tailchasing analyze src/ --incremental

# Reduce vector dimensions
tailchasing analyze . --semantic --vector-dim 8192
```

#### 3. Slow Semantic Analysis

**Problem**: Semantic analysis taking too long

**Solution**:
```yaml
# In .tailchasing.yml
semantic:
  min_functions: 50  # Increase threshold
  vector_dim: 8192   # Reduce dimensions
  cache: true        # Enable caching
```

#### 4. LLM Rate Limiting

**Problem**: API rate limit errors

**Solution**:
```bash
# Add delays between LLM calls
tailchasing fix . --llm gpt-4 --delay 1.0

# Use local models
tailchasing fix . --llm local-llama
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Set log level
export TAILCHASING_LOG_LEVEL=DEBUG

# Or via command line
tailchasing analyze . --log-level DEBUG

# Save logs to file
tailchasing analyze . --log-file debug.log
```

### Performance Profiling

```bash
# Profile execution
python -m cProfile -s cumulative tailchasing analyze .

# Memory profiling
python -m memory_profiler tailchasing analyze .
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/TailChasingFixer.git
cd TailChasingFixer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run benchmarks
python benchmarks/run_benchmarks.py single
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
- Semantic analysis powered by hypervector computing research
- Community contributors and early adopters

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/rohanvinaik/TailChasingFixer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rohanvinaik/TailChasingFixer/discussions)
- **Email**: support@tailchasing.dev

---

**Remember**: LLMs are powerful tools, but they can introduce subtle anti-patterns. Use TailChasing Fixer to keep your codebase clean and maintainable! 🚀