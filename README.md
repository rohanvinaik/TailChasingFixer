# Tail-Chasing Detector

[![PyPI version](https://badge.fury.io/py/tail-chasing-detector.svg)](https://badge.fury.io/py/tail-chasing-detector)
[![Python versions](https://img.shields.io/pypi/pyversions/tail-chasing-detector.svg)](https://pypi.org/project/tail-chasing-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/rohanvinaik/TailChasingFixer/actions/workflows/tail-chasing.yml/badge.svg)](https://github.com/rohanvinaik/TailChasingFixer/actions/workflows/tail-chasing.yml)

Detects LLM-assisted *tail-chasing* anti-patterns - a software development anti-pattern where LLMs repeatedly make superficial or circular fixes in response to immediate errors, without addressing underlying causes.

## 🚀 Quick Start

### Command Line

```bash
# Install from PyPI
pip install tail-chasing-detector

# Install with all features
pip install "tail-chasing-detector[all]"

# Run basic analysis
tailchasing .

# With natural language explanations
tailchasing analyze . --explain

# With interactive visualization
tailchasing analyze . --visualize --open-browser

# Combined analysis with all features
tailchasing analyze . --explain --visualize --output report.html
```

### VS Code Extension

1. Open VS Code
2. Search for "Tail-Chasing Detector" in Extensions (Cmd+Shift+X)
3. Click Install
4. The extension will automatically analyze Python files as you edit them

## What is a Tail-Chasing Bug?

A Tail-Chasing Bug occurs when an LLM:
1. **Creates circular dependencies or redundant implementations** (e.g., defining `ComparisonAnalyzer` when `QualityAnalyzer.compare_quality` already exists)
2. **Renames or restructures code solely to match error messages** without resolving core issues
3. **Generates phantom implementations** - inventing classes or methods to satisfy mistaken imports
4. **🆕 Creates hallucination cascades** - entire fictional subsystems to satisfy import errors
5. **🆕 Shows context window thrashing** - reimplements existing functionality after forgetting earlier code
6. **🆕 Exhibits import anxiety** - defensively imports many items "just in case"

This pattern emerges because LLMs operate with limited context windows, causing them to fixate on immediate errors rather than global architectural intent.

## Features

### Core Detection
- **Phantom/placeholder implementation detection**
- **Structural duplicate function detection** 
- **Circular import analysis with runtime-risk weighting**
- **Missing/hallucinated symbol detection**
- **Wrapper/trivial abstraction detection**
- **Temporal chain analysis** of superficial fixes
- **Git history integration** to detect fix patterns

### 🆕 Advanced Features
- **Test-Driven Development Anti-patterns** - Detects mirror tests, brittle assertions, and redundant test coverage
- **Cross-File Semantic Duplication** - Finds duplicates across modules with architectural awareness
- **Cargo Cult Programming Detection** - Identifies boilerplate copied without understanding
- **Interactive Root Cause Visualization** - Traces and visualizes tail-chasing chains
- **Natural Language Explanations** - Human-readable explanations of why patterns are problematic
- **Semantic Hypervector Analysis** - Detects deep semantic duplicates using high-dimensional computing
- **Hallucination Cascade Detection** - Identifies entire fictional subsystems created by LLMs
- **Context Window Thrashing Detection** - Finds reimplementations due to forgotten context
- **Import Anxiety Pattern Detection** - Catches defensive over-importing
- **Enhanced Multi-Modal Semantic Analysis** - Uses data flow, return patterns, and complexity
- **Intelligent Auto-Fix System** - Generates fixes with confidence scores and impact analysis

### Output & Integration
- **Risk scoring system** with configurable thresholds
- **Multiple output formats** (text, JSON, SARIF coming soon)
- **VS Code Extension** for real-time detection
- **CI/CD Integration** ready

## Installation

### From PyPI (Recommended)

```bash
pip install tail-chasing-detector
```

### With Optional Features

```bash
# Visualization support
pip install "tail-chasing-detector[visualization]"

# Machine learning enhancements
pip install "tail-chasing-detector[ml]"

# Performance optimizations
pip install "tail-chasing-detector[performance]"

# All features
pip install "tail-chasing-detector[all]"
```

### From Source

```bash
git clone https://github.com/rohanvinaik/TailChasingFixer.git
cd TailChasingFixer
pip install -e .
```

## Configuration

Create a `.tailchasing.yml` in your project root:

```yaml
paths:
  include:
    - src
  exclude:
    - tests
    - build
    - venv

risk_thresholds:
  warn: 15
  fail: 30

placeholders:
  allow:
    - BasePlugin.initialize

ignore_issue_types:
  - wrapper_abstraction

scoring_weights:
  missing_symbol: 2
  phantom_function: 2
  duplicate_function: 2
  wrapper_abstraction: 1
  semantic_duplicate_function: 3
  prototype_fragmentation: 3

semantic:
  enable: true
  hv_dim: 8192
  min_functions: 30
```

## Issue Types Detected

### Core Issue Types

| Issue Type | Description | Default Weight |
|------------|-------------|----------------|
| `circular_import` | Import cycles that can cause runtime failures | 3 |
| `duplicate_function` | Structurally identical functions with different names | 2 |
| `phantom_function` | Pass-only or NotImplementedError stubs | 2 |
| `missing_symbol` | References to non-existent functions/classes | 2 |
| `wrapper_abstraction` | Trivial wrappers that add no value | 1 |
| `hallucinated_import` | Imports of non-existent modules | 3 |
| `tail_chasing_chain` | Temporal pattern of superficial fixes | 4 |

### Semantic Analysis Issues

| Issue Type | Description | Default Weight |
|------------|-------------|----------------|
| `semantic_duplicate_function` | Functions with different structure but same semantics | 3 |
| `prototype_fragmentation` | Multiple implementations of the same concept | 3 |
| `semantic_stagnant_placeholder` | Stubs that never evolve semantically | 2 |
| `rename_cascade_chain` | Functions renamed without semantic changes | 4 |

### 🆕 Advanced Pattern Issues

| Issue Type | Description | Default Weight |
|------------|-------------|----------------|
| `hallucination_cascade` | Entire fictional subsystems created to satisfy errors | 4 |
| `context_window_thrashing` | Reimplementation due to forgotten context | 3 |
| `import_anxiety` | Defensive over-importing patterns | 1 |
| `enhanced_semantic_duplicate` | Multi-modal semantic duplicate detection | 3 |
| `mirror_test` | Tests that duplicate implementation logic | 2 |
| `brittle_test_assertions` | Tests with overly specific assertions | 2 |
| `incomplete_test_coverage` | Tests missing edge cases and error conditions | 2 |
| `cross_file_duplication` | Semantic duplicates across different files | 3 |
| `cargo_cult_*` | Various patterns of copying without understanding | 1-3 |

## 🆕 Advanced Features

For detailed documentation on advanced features, see [docs/ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md).

### Quick Overview

1. **Hallucination Cascade Detection**: Identifies when LLMs create entire fictional subsystems
2. **Context Window Thrashing**: Detects reimplementations due to lost context
3. **Import Anxiety Patterns**: Finds defensive over-importing
4. **Enhanced Semantic Analysis**: Multi-modal duplicate detection
5. **Intelligent Auto-Fix System**: Generates fixes with confidence scores

### 📊 Interactive Visualizations

Generate interactive HTML reports that visualize tail-chasing chains:

```bash
# Generate and open visualization
tailchasing analyze . --visualize --open-browser
```

The visualization includes:
- Mermaid diagrams showing how errors propagate
- Timeline of tail-chasing events
- Risk scores and severity indicators
- Actionable resolution suggestions

### 💬 Natural Language Explanations

Get human-readable explanations of detected patterns:

```bash
# Get explanations for all issues
tailchasing analyze . --explain

# Explain a specific pattern type
tailchasing explain-pattern phantom_function
```

Explanations include:
- What happened and why it's problematic
- Root cause analysis
- Specific fix recommendations
- Prevention strategies

### Enabling Advanced Features

```yaml
# .tailchasing.yml
enable_advanced_analyzers: true
```

## 🧬 Semantic Hypervector Analysis

The Tail Chasing Detector now includes advanced semantic analysis using hypervector computing (HDC). This feature can detect semantic patterns that traditional AST-based analysis would miss.

### What It Detects

1. **Semantic Duplicates**: Functions that behave the same but have different implementations
2. **Prototype Fragmentation**: Multiple scattered implementations of the same concept
3. **Semantic Stagnation**: Placeholder functions that never evolve meaningfully
4. **Rename Cascades**: Functions that are renamed without changing behavior

### How It Works

The analyzer uses high-dimensional vectors (8192 dimensions by default) to encode the semantic "fingerprint" of each function based on:
- Function and parameter names (tokenized)
- Docstring content
- Called functions
- Control flow patterns
- Literal types used
- Exception handling
- Import dependencies

Functions with similar semantic fingerprints are identified using statistical analysis with false discovery rate control.

### Configuration

Add this to your `.tailchasing.yml`:

```yaml
semantic:
  enable: true
  hv_dim: 8192  # Hypervector dimension
  min_functions: 30  # Minimum functions needed
  z_threshold: 2.5  # Statistical significance threshold
  
  # Adjust channel weights to emphasize different aspects
  channel_weights:
    NAME_TOKENS: 1.0
    CALLS: 1.2  # Emphasize function calls
    DOC_TOKENS: 0.8

# Enable advanced analyzers
enable_advanced_analyzers: true

# Configure advanced features
advanced_thresholds:
  hallucination_min_cluster: 3
  context_window_min_distance: 500
  import_anxiety_unused_ratio: 0.66
```

### Example: Semantic Duplicate Detection

```python
# These would be detected as semantic duplicates despite different implementations:

def calculate_average(numbers):
    """Calculate the mean of a list of numbers."""
    total = sum(numbers)
    count = len(numbers)
    return total / count

def compute_mean(data_list):
    """Compute arithmetic mean of values."""
    accumulator = 0
    for value in data_list:
        accumulator += value
    return accumulator / len(data_list)
```

## Examples

### Detecting Phantom Functions
```python
# Detected as phantom_function
def process_data(data):
    pass

def analyze_results():
    raise NotImplementedError()
```

### Finding Circular Imports
```python
# file_a.py
from file_b import helper

# file_b.py  
from file_a import process  # Circular!
```

### Identifying Missing Symbols
```python
# Hallucinated import - module doesn't exist
from utils.advanced_processor import DataTransformer

# Reference to non-existent function
result = calculate_advanced_metrics(data)  # Never defined!
```

### 🆕 Test Anti-Patterns
```python
# Detected as mirror_test - test duplicates implementation
def test_calculate_average():
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)  # Same logic as implementation!
    count = len(numbers)
    expected = total / count
    assert calculate_average(numbers) == expected

# Detected as brittle_test_assertions
def test_user_creation():
    user = create_user("John", "Doe")
    # Too specific - will break with any formatting change
    assert str(user) == "User(id=12345, name='John Doe', created_at='2024-01-15 10:30:45.123456')"
```

### 🆕 Cargo Cult Patterns
```python
# Detected as cargo_cult_unnecessary_super
class MyClass:  # No parent class!
    def __init__(self):
        super().__init__()  # Unnecessary!
        self.value = 42

# Detected as cargo_cult_trivial_wrapper
def get_user_name(user):
    return user.name  # Just use user.name directly!
```

## Integration with Existing Tools

Tail Chasing Detector can ingest results from other tools:

```bash
# Use with ruff
ruff check --output-format=json > ruff.json
tailchasing . --ingest-json ruff.json

# Combine with pylint
pylint src --output-format=json > pylint.json
tailchasing . --ingest-json pylint.json
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Tail-Chasing Check
  uses: tail-chasing/detector-action@v1
  with:
    fail-on: 30
    config: .tailchasing.yml
```

### Pre-commit Hook

```yaml
repos:
  - repo: https://github.com/rohanvinaik/TailChasingFixer
    rev: v0.1.0
    hooks:
      - id: tail-chasing
        args: ['--fail-on', '20']
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Conceptual inspiration from existing static analyzers and the novel "tail-chasing bug" pattern observed in LLM-assisted development workflows.