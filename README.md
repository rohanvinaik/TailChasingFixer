# TailChasingFixer

[![Python versions](https://img.shields.io/pypi/pyversions/tail-chasing-detector.svg)](https://pypi.org/project/tail-chasing-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Detects LLM-assisted *tail-chasing* anti-patterns - a software development anti-pattern where LLMs repeatedly make superficial or circular fixes in response to immediate errors, without addressing underlying causes.

## ✨ New in v0.1.0: Advanced Features

🧠 **Enhanced Pattern Detection**: Hallucination cascades, context window thrashing, import anxiety  
🔬 **Multimodal Semantic Analysis**: Deep semantic duplicate detection across multiple dimensions  
🔧 **Intelligent Auto-Fixing**: Generate and apply fixes with comprehensive impact analysis  
📊 **Interactive Visualizations**: HTML reports with dependency graphs, heatmaps, and similarity matrices  
🗣️ **Natural Language Explanations**: Detailed explanations of why patterns occur and how to fix them

[📖 See Advanced Features Documentation](docs/ADVANCED_FEATURES.md)

## 🚀 Quick Start

```bash
# Install
pip install tail-chasing-detector

# Basic analysis
tailchasing .

# Enhanced analysis with all new features
tailchasing-enhanced . --enhanced --semantic-multimodal --html report.html

# Auto-fix with detailed explanations
tailchasing-enhanced . --auto-fix --explain --fix-plan fixes.json

# Generate comprehensive HTML report
tailchasing-enhanced . --html full_report.html --enhanced
```

## What is a Tail-Chasing Bug?

A Tail-Chasing Bug occurs when an LLM:
1. **Creates circular dependencies or redundant implementations** (e.g., defining `ComparisonAnalyzer` when `QualityAnalyzer.compare_quality` already exists)
2. **Renames or restructures code solely to match error messages** without resolving core issues
3. **Generates phantom implementations** - inventing classes or methods to satisfy mistaken imports
4. **Creates hallucination cascades** - entire fictional subsystems to satisfy import errors
5. **Shows context window thrashing** - reimplements existing functionality after forgetting earlier code
6. **Exhibits import anxiety** - defensively imports many items "just in case"

This pattern emerges because LLMs operate with limited context windows, causing them to fixate on immediate errors rather than global architectural intent.

## Features

### Core Detection
- **Phantom/placeholder implementation detection**
- **Structural duplicate function detection** 
- **Circular import analysis with runtime-risk weighting**
- **Missing/hallucinated symbol detection**
- **Wrapper/trivial abstraction detection**

### Advanced Semantic Analysis
- **Semantic Hypervector Analysis** - Detects deep semantic duplicates using high-dimensional computing
- **Prototype Fragmentation Detection** - Finds multiple implementations of the same concept
- **Semantic Stagnation Analysis** - Identifies placeholders that never evolve
- **Multi-Modal Semantic Encoding** - Analyzes data flow, return patterns, and complexity

### Advanced Pattern Detection
- **Hallucination Cascade Detection** - Identifies entire fictional subsystems created by LLMs
- **Context Window Thrashing Detection** - Finds reimplementations due to forgotten context
- **Import Anxiety Pattern Detection** - Catches defensive over-importing

### Fix Generation & Suggestions
- **Interactive Fix Scripts** - Generate executable Python scripts to apply fixes
- **Detailed Suggestion Documents** - Markdown files with code examples and explanations
- **In-Terminal Suggestions** - View fix suggestions directly in terminal output
- **Context-Aware Recommendations** - Suggestions based on specific issue patterns

### Output & Integration
- **Risk scoring system** with configurable thresholds
- **Multiple output formats** (text, JSON, HTML)
- **Always shows report paths** in terminal output
- **Fix availability notifications** in standard output

## Installation

### From PyPI

```bash
pip install tail-chasing-detector
```

### From Source

```bash
git clone https://github.com/rohanvinaik/TailChasingFixer.git
cd TailChasingFixer
pip install -e .
```

## Usage

### Basic Analysis

```bash
# Analyze current directory
tailchasing .

# Analyze specific directory
tailchasing /path/to/project
```

### Output Options

```bash
# Generate HTML report
tailchasing . --html

# Generate JSON report
tailchasing . --json

# Save reports to specific directory
tailchasing . --output ./analysis

# Show verbose progress
tailchasing . --verbose
```

### Fix Suggestions

```bash
# Show suggestions in terminal for top issues
tailchasing . --show-suggestions

# Generate interactive fix script and detailed suggestions
tailchasing . --generate-fixes

# Both together
tailchasing . --show-suggestions --generate-fixes
```

### Filtering

```bash
# Exclude specific paths
tailchasing . --exclude tests --exclude build

# Include only specific paths
tailchasing . --include src --include lib

# Disable specific analyzers
tailchasing . --disable semantic_hv --disable hallucination_cascade
```

### CI/CD Integration

```bash
# Fail if risk score exceeds threshold
tailchasing . --fail-on 50

# Generate all reports for CI artifacts
tailchasing . --json --html --generate-fixes --output ./ci-reports
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
  hallucination_cascade: 4
  context_window_thrashing: 3
  import_anxiety: 1

semantic:
  enable: true
  hv_dim: 8192
  min_functions: 30
  z_threshold: 2.5
  
report:
  formats:
    - text
    - html
  output_dir: ./reports
```

## Terminal Output

The tool always shows:
- Analysis summary (issues, risk score, affected modules)
- Paths to all generated reports
- Fix suggestion availability and generation commands

Example output:
```
Tail-Chasing Analysis Complete
========================================
Total Issues: 88
Global Risk Score: 14.43 (OK)
Affected Modules: 42

Top Issues:
  [semantic_duplicate_function] file.py:123 - Semantic duplicate: func1 and func2
  [phantom_function] file.py:456 - Pass-only function: placeholder
  [circular_import] module.py:1 - Circular import cycle of length 3

Generated Reports:
----------------------------------------
TEXT report: ./tailchasing_report.txt
HTML report: ./tailchasing_report.html

Fix Suggestions:
----------------------------------------
Found 75 fixable issues out of 88 total
Run with --generate-fixes to create:
  • Interactive fix script (tailchasing_fixes.py)
  • Detailed suggestions file (tailchasing_suggestions.md)

Example: tailchasing . --generate-fixes
```

## Issue Types Detected

### Core Issues

| Issue Type | Description | Default Weight |
|------------|-------------|----------------|
| `circular_import` | Import cycles that can cause runtime failures | 3 |
| `duplicate_function` | Structurally identical functions | 2 |
| `phantom_function` | Pass-only or NotImplementedError stubs | 2 |
| `missing_symbol` | References to non-existent functions/classes | 2 |
| `wrapper_abstraction` | Trivial wrappers that add no value | 1 |

### Semantic Analysis Issues

| Issue Type | Description | Default Weight |
|------------|-------------|----------------|
| `semantic_duplicate_function` | Functions with same behavior but different code | 3 |
| `prototype_fragmentation` | Multiple implementations of same concept | 3 |
| `semantic_stagnant_placeholder` | Stubs that never evolve | 2 |

**Note**: Semantic analysis includes **smart filtering** to avoid false positives. The system automatically ignores legitimate patterns like `__init__.py` files, test boilerplate, property accessors, and protocol implementations.

### Advanced Pattern Issues

| Issue Type | Description | Default Weight |
|------------|-------------|----------------|
| `hallucination_cascade` | Fictional subsystems created to satisfy errors | 4 |
| `context_window_thrashing` | Reimplementation due to forgotten context | 3 |
| `import_anxiety` | Defensive over-importing patterns | 1 |
| `enhanced_semantic_duplicate` | Multi-modal semantic duplicate | 3 |

## Fix Suggestions

Each detected issue includes actionable fix suggestions:

### In Terminal (with `--show-suggestions`)
```
[semantic_duplicate_function] file.py:123 - Semantic duplicate: func1 and func2
  Suggestions:
    Merge func1 and func2 in file.py
    Review naming conventions - these functions have different names but similar behavior
```

### In Generated Files (with `--generate-fixes`)

**Interactive Fix Script** (`tailchasing_fixes.py`):
- Shows each issue interactively
- Lets you review and apply fixes
- Tracks which fixes were applied

**Detailed Suggestions** (`tailchasing_suggestions.md`):
- Grouped by issue type
- Includes code examples
- Step-by-step instructions
- Command-line snippets

## Semantic Hypervector Analysis

The semantic analyzer detects deep patterns using high-dimensional computing:

### Configuration

```yaml
semantic:
  enable: true
  hv_dim: 8192  # Hypervector dimension
  min_functions: 30  # Minimum functions needed
  z_threshold: 2.5  # Statistical significance threshold
```

### What It Analyzes

Each function is encoded based on:
- Function and parameter names
- Docstring content
- Called functions
- Control flow patterns
- Data flow
- Return patterns
- Exception handling

### Example Detection

These functions would be detected as semantic duplicates:

```python
def calculate_average(numbers):
    """Calculate the mean of a list."""
    total = sum(numbers)
    count = len(numbers)
    return total / count

def compute_mean(data_list):
    """Compute arithmetic mean."""
    accumulator = 0
    for value in data_list:
        accumulator += value
    return accumulator / len(data_list)
```

## Examples

### Phantom Functions
```python
# Detected as phantom_function
def process_data(data):
    pass

def analyze_results():
    raise NotImplementedError()
```

### Circular Imports
```python
# file_a.py
from file_b import helper

# file_b.py  
from file_a import process  # Circular!
```

### Hallucination Cascades
```python
# LLM creates these to satisfy non-existent imports
class DataProcessor:  # Never actually used
    pass

class DataValidator:  # Created to satisfy DataProcessor
    def __init__(self, processor: DataProcessor):
        self.processor = processor

class DataPipeline:  # Created to use both above
    def __init__(self):
        self.processor = DataProcessor()
        self.validator = DataValidator(self.processor)
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Conceptual inspiration from existing static analyzers and the novel "tail-chasing bug" pattern observed in LLM-assisted development workflows.
