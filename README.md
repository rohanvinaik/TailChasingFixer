# Tail Chasing Detector

Detects LLM-assisted *tail-chasing* anti-patterns - a software development anti-pattern where LLMs repeatedly make superficial or circular fixes in response to immediate errors, without addressing underlying causes.

## What is a Tail-Chasing Bug?

A Tail-Chasing Bug occurs when an LLM:
1. **Creates circular dependencies or redundant implementations** (e.g., defining `ComparisonAnalyzer` when `QualityAnalyzer.compare_quality` already exists)
2. **Renames or restructures code solely to match error messages** without resolving core issues
3. **Generates phantom implementations** - inventing classes or methods to satisfy mistaken imports

This pattern emerges because LLMs operate with limited context windows, causing them to fixate on immediate errors rather than global architectural intent.

## Features

- **Phantom/placeholder implementation detection**
- **Structural duplicate function detection** 
- **Circular import analysis with runtime-risk weighting**
- **Missing/hallucinated symbol detection**
- **Wrapper/trivial abstraction detection**
- **Temporal chain analysis** of superficial fixes
- **Git history integration** to detect fix patterns
- **Risk scoring system** with configurable thresholds
- **Multiple output formats** (text, JSON, SARIF coming soon)

## Quick Start

```bash
# Install
pip install -e .

# Run analysis
tailchasing .

# JSON output
tailchasing . --json

# Set failure threshold
tailchasing . --fail-on 30
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
  duplicate_cluster: 2
  wrapper_abstraction: 1
  cycle_participation: 3
  drift_entropy: 1

git:
  enable: true

report:
  formats: ["text", "json"]
  output_dir: ./
```

## Issue Types Detected

| Issue Type | Description | Default Weight |
|------------|-------------|----------------|
| `circular_import` | Import cycles that can cause runtime failures | 3 |
| `duplicate_function` | Structurally identical functions with different names | 2 |
| `phantom_function` | Pass-only or NotImplementedError stubs | 2 |
| `missing_symbol` | References to non-existent functions/classes | 2 |
| `wrapper_abstraction` | Trivial wrappers that add no value | 1 |
| `hallucinated_import` | Imports of non-existent modules | 3 |
| `tail_chasing_chain` | Temporal pattern of superficial fixes | 4 |

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

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Conceptual inspiration from existing static analyzers and the novel "tail-chasing bug" pattern observed in LLM-assisted development workflows.
