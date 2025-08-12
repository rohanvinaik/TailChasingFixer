# TailChasingFixer

[![Python versions](https://img.shields.io/pypi/pyversions/tail-chasing-detector.svg)](https://pypi.org/project/tail-chasing-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive tool that detects and fixes LLM-assisted *tail-chasing* anti-patterns - a software development anti-pattern where LLMs repeatedly make superficial or circular fixes in response to immediate errors, without addressing underlying architectural issues.

## 🎯 What is Tail-Chasing?

Tail-chasing occurs when AI assistants get stuck in unproductive loops while generating code:

1. **Creating circular dependencies** - Modules that import each other in cycles
2. **Generating phantom implementations** - Inventing classes/methods to satisfy mistaken imports
3. **Duplicate implementations** - Creating multiple versions of the same functionality
4. **Context window thrashing** - Reimplementing existing code after forgetting earlier context
5. **Hallucination cascades** - Building entire fictional subsystems to satisfy import errors
6. **Import anxiety** - Defensively importing many items "just in case"

This tool detects these patterns using advanced semantic analysis and provides automated fixes.

## ✨ Key Features

### 🔍 Advanced Detection
- **Semantic Hypervector Analysis** - Deep semantic duplicate detection using 8192-dimensional vectors
- **Smart Filtering** - Reduces false positives by understanding legitimate patterns
- **Multi-Modal Analysis** - Analyzes structure, behavior, data flow, and complexity
- **Cargo Cult Detection** - Identifies copied boilerplate and misused patterns
- **🧬 Chromatin-Inspired Analysis** - Uses polymer physics models from chromatin biology
- **🔬 Loop Extrusion Detection** - Identifies code organization patterns using TAD-like boundaries

### 🔧 Intelligent Auto-Fixing
- **Automated Fix Generation** - Creates safe, tested fixes for detected issues
- **Risk Assessment** - Evaluates fix safety with LOW/MEDIUM/HIGH/CRITICAL ratings
- **Rollback Plans** - Generates undo strategies for every fix
- **Fix Strategies** - Specialized handlers for different issue types

### 📊 Rich Reporting
- **Interactive HTML Reports** - Dependency graphs, heatmaps, similarity matrices
- **Multiple Formats** - Text, JSON, HTML, Markdown suggestions
- **CI/CD Integration** - Fail on threshold, generate artifacts
- **Natural Language Explanations** - Detailed explanations of issues and fixes

## 🚀 Quick Start

```bash
# Install from PyPI
pip install tail-chasing-detector

# Basic analysis
tailchasing .

# Enhanced analysis with semantic detection
tailchasing-enhanced . --enhanced --semantic-multimodal

# Generate comprehensive HTML report
tailchasing . --html report.html

# Auto-fix with explanations
tailchasing-enhanced . --auto-fix --explain

# Show fix suggestions in terminal
tailchasing . --show-suggestions --generate-fixes
```

## 📦 Installation

### From PyPI
```bash
pip install tail-chasing-detector

# With all features
pip install "tail-chasing-detector[all]"
```

### From Source
```bash
git clone https://github.com/rohanvinaik/TailChasingFixer.git
cd TailChasingFixer
pip install -e .

# With development tools
pip install -e ".[dev]"
```

## 🔄 Updating/Reinstalling

### Updating from PyPI
```bash
# Update to latest version
pip install --upgrade tail-chasing-detector

# Update with all features
pip install --upgrade "tail-chasing-detector[all]"
```

### Updating from Source
```bash
# If installed in editable mode
cd /path/to/TailChasingFixer
git pull origin main
pip install -e . --upgrade

# Force reinstall if having issues
pip install -e . --force-reinstall
```

### Uninstalling
```bash
# Uninstall the package
pip uninstall tail-chasing-detector

# Clean up any remaining files (if installed from source)
rm -rf build/ dist/ *.egg-info/
```

### Troubleshooting Installation Issues

#### Dependency Conflicts
If you encounter dependency conflicts with other packages (e.g., semgrep, web3):
```bash
# Use a virtual environment to avoid conflicts
python -m venv tailchasing-env
source tailchasing-env/bin/activate  # On Windows: tailchasing-env\Scripts\activate
pip install tail-chasing-detector

# Or use pipx for isolated installation
pipx install tail-chasing-detector
```

#### Import Errors
```bash
# If you encounter import errors after updating
pip uninstall tail-chasing-detector
pip cache purge  # Clear pip cache
pip install tail-chasing-detector

# For development installation issues
pip install -e . --force-reinstall --no-deps
pip install -e ".[dev]"

# If commands aren't found after installation
python -m tailchasing --help  # Use module syntax
# Or reinstall with:
pip install -e . --force-reinstall
```

#### Known Compatibility Notes
- **boltons**: Not a direct dependency. Conflicts may arise with semgrep. Use virtual environment.
- **pydantic**: Not a direct dependency. Conflicts may arise with web3. Use virtual environment.
- **numpy/scipy**: Pinned to <2.0 for compatibility with older codebases.

## 📖 Usage

### Basic Commands

```bash
# Analyze current directory
tailchasing .

# Analyze specific directory
tailchasing /path/to/project

# Generate reports
tailchasing . --html --json --output ./reports

# Show verbose progress
tailchasing . --verbose
```

### Fix Generation

```bash
# Show suggestions in terminal
tailchasing . --show-suggestions

# Generate fix script and documentation
tailchasing . --generate-fixes

# Auto-apply fixes (use with caution)
tailchasing-enhanced . --auto-fix --dry-run  # Preview first
tailchasing-enhanced . --auto-fix             # Apply fixes
```

### Filtering & Configuration

```bash
# Exclude paths
tailchasing . --exclude tests --exclude build

# Include specific paths only
tailchasing . --include src --include lib

# Disable specific analyzers
tailchasing . --disable semantic_hv --disable cargo_cult

# Use configuration file
tailchasing . --config .tailchasing.yml
```

### CI/CD Integration

```bash
# Fail if risk score exceeds threshold
tailchasing . --fail-on 50

# Generate all artifacts
tailchasing . --json --html --generate-fixes --output ./ci-reports

# GitHub Actions example
tailchasing . --fail-on 30 --json --output $GITHUB_WORKSPACE/reports
```

## ⚙️ Configuration

Create `.tailchasing.yml` in your project root:

```yaml
# Path configuration
paths:
  include:
    - src
    - lib
  exclude:
    - tests
    - build
    - venv
    - .git

# Risk thresholds
risk_thresholds:
  warn: 15
  fail: 30

# Allowed placeholders (won't be flagged)
placeholders:
  allow:
    - BaseClass.abstract_method
    - Interface.template_method

# Ignore specific issue types
ignore_issue_types:
  - wrapper_abstraction  # If you use many wrapper patterns

# Customize scoring weights
scoring_weights:
  circular_import: 3
  missing_symbol: 2
  phantom_function: 2
  duplicate_function: 2
  semantic_duplicate_function: 3
  hallucination_cascade: 4
  context_window_thrashing: 3
  cargo_cult_pattern: 2
  import_anxiety: 1

# Semantic analysis settings
semantic:
  enable: true
  hv_dim: 8192           # Hypervector dimensions
  min_functions: 30      # Minimum functions to analyze
  z_threshold: 2.5       # Statistical significance
  smart_filter: true     # Enable false-positive reduction

# Fix generation settings
fixes:
  auto_fix: false        # Don't auto-apply by default
  require_confirmation: true
  generate_rollback: true
  risk_limit: MEDIUM     # Don't apply HIGH/CRITICAL fixes

# Report settings
report:
  formats:
    - text
    - html
  output_dir: ./reports
  include_suggestions: true
  include_examples: true
```

### Canonical Module Policy

The tool includes a powerful canonical module policy system that identifies and manages canonical vs shadow implementations:

```bash
# Analyze with canonical policy (requires configuration)
tailchasing . --generate-canonical-codemod

# Root cause clustering with canonical awareness
tailchasing . --cluster-root-causes
```

**Example Configuration:**
```yaml
canonical_policy:
  # Define canonical implementation roots (highest priority)
  canonical_roots:
    - "src/core"
    - "mypackage/advanced"
  
  # Define shadow/experimental roots (lower priority)  
  shadow_roots:
    - "experimental"
    - "prototypes"
    - "temp"
  
  # Pattern-based priority adjustments
  priority_patterns:
    ".*experimental.*": -30    # Heavily deprioritize experimental
    ".*test.*": -10            # Deprioritize test files
    ".*/core/.*": 10           # Boost core implementations
  
  auto_suppress_shadows: true  # Auto-suppress shadow issues
  generate_forwarders: true    # Generate import forwarders
```

**Features:**
- **Automatic Shadow Detection**: Identifies duplicate implementations in experimental/shadow paths
- **Priority-Based Canonical Selection**: Uses configurable rules to determine canonical vs shadow
- **Codemod Generation**: Automatically generates scripts to replace shadows with import forwarders
- **Deprecation Warnings**: Adds proper deprecation warnings to maintain backward compatibility
- **Integration with Root Cause Clustering**: Provides enhanced analysis of systemic patterns

## 🎯 Issue Types Detected

### Core Issues
| Issue Type | Description | Risk |
|------------|-------------|------|
| `circular_import` | Import cycles that can cause runtime failures | HIGH |
| `duplicate_function` | Structurally identical functions | MEDIUM |
| `phantom_function` | Pass-only or NotImplementedError stubs | MEDIUM |
| `missing_symbol` | References to non-existent functions/classes | HIGH |
| `unused_code` | Dead code that's never called | LOW |

### Semantic Issues
| Issue Type | Description | Risk |
|------------|-------------|------|
| `semantic_duplicate` | Same behavior, different implementation | MEDIUM |
| `prototype_fragmentation` | Multiple implementations of same concept | HIGH |
| `semantic_stagnation` | Placeholders that never evolve | MEDIUM |

### LLM-Specific Patterns
| Issue Type | Description | Risk |
|------------|-------------|------|
| `hallucination_cascade` | Fictional subsystems created to satisfy errors | CRITICAL |
| `context_window_thrashing` | Reimplementation due to forgotten context | HIGH |
| `import_anxiety` | Defensive over-importing | LOW |
| `cargo_cult_pattern` | Copied boilerplate without understanding | MEDIUM |

### Biology-Inspired Patterns
| Issue Type | Description | Risk |
|------------|-------------|------|
| `chromatin_contact_violation` | Code violating expected polymer distance relationships | MEDIUM |
| `tad_boundary_crossing` | Improper module boundary violations | HIGH |
| `loop_extrusion_anomaly` | Abnormal code organization patterns | MEDIUM |
| `recombination_hotspot` | Area of excessive code churn | HIGH |

### Canonical Policy Issues
| Issue Type | Description | Risk |
|------------|-------------|------|
| `shadow_implementation_detected` | Non-canonical duplicate implementation | MEDIUM |
| `shadow_implementation_suppressed` | Auto-suppressed shadow (informational) | LOW |
| `canonical_policy_codemod` | Generated codemod suggestions available | LOW |

## 📊 Example Output

```
🔍 Tail-Chasing Analysis Complete
══════════════════════════════════════════════════
📊 Summary:
  Total Issues: 47
  Risk Score: 28.5 (⚠️ MEDIUM)
  Affected Files: 23
  
🔴 Critical Issues (3):
  • hallucination_cascade: auth/validator.py:45
    Entire validator subsystem never actually used
    
  • circular_import: models/user.py ↔ models/profile.py
    Runtime failure risk in production

📈 Top Patterns:
  • 12 semantic duplicates (25.5% of issues)
  • 8 phantom functions (17.0% of issues)
  • 5 circular imports (10.6% of issues)

📁 Generated Reports:
  ✓ HTML: ./reports/tailchasing_report.html
  ✓ JSON: ./reports/tailchasing_data.json
  ✓ Fixes: ./reports/suggested_fixes.py

💡 Fix Suggestions Available:
  37 of 47 issues have automated fixes
  Run: tailchasing . --generate-fixes
```

## 🔬 Advanced Features

### Semantic Hypervector Analysis

Detects deep semantic patterns using high-dimensional computing:

```python
# These would be detected as semantic duplicates:
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

def compute_mean(data_list):
    total = 0
    for val in data_list:
        total += val
    return total / len(data_list)
```

### 🧬 Biology-Inspired Polymer Physics Analysis

Applies chromatin biology concepts to code organization:

- **Contact Probability Decay**: Models code relationships using polymer distance metrics
- **TAD Boundary Detection**: Identifies natural module boundaries like topologically associating domains
- **Loop Extrusion Simulation**: Analyzes how code structures form and reorganize
- **Recombination Hotspot Mapping**: Finds areas of high code churn and instability

#### Polymer Physics Configuration

```yaml
# .tailchasing_polymer.yml
alpha: 1.2  # Contact decay exponent (1.0-1.5)
weights:
  tok: 1.0  # Token-level distance weight
  ast: 2.0  # AST-level distance weight  
  mod: 3.0  # Module-level distance weight
  git: 4.0  # Git-level distance weight
tad_patterns:
  - "*.api.*"
  - "*.core.*"
  - "*.models.*"
```

#### Calibration Tool

Optimize parameters based on historical data:

```python
from tailchasing.calibrate import CalibrationTool
from tailchasing.config import PolymerConfig

# Calibrate based on historical thrashing events
tool = CalibrationTool()
result = tool.fit_parameters(
    historical_events,
    codebase_metrics,
    alpha_range=(1.0, 1.5)
)

# Apply optimized parameters
config = PolymerConfig(
    alpha=result.optimal_alpha,
    weights=result.optimal_weights
)
```

### Smart Filtering

Automatically excludes false positives:
- `__init__.py` standard patterns
- Test boilerplate and fixtures
- Property getters/setters
- Protocol/interface implementations
- Django/Flask view patterns

### Cargo Cult Detection

Identifies code copied without understanding:
- Unnecessary `super().__init__()` calls
- Redundant docstrings
- Trivial getters/setters
- Misused design patterns

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Testing guidelines
- Code style requirements
- Pull request process

## 📚 Documentation

- [Advanced Features](docs/ADVANCED_FEATURES.md) - Semantic analysis, auto-fixing
- [API Reference](docs/API.md) - Using as a library
- [Configuration Guide](docs/CONFIG.md) - Detailed configuration options
- [VS Code Extension](vscode-extension/README.md) - IDE integration

## 🐛 Recent Fixes

### v0.2.0 (Latest)
- ✅ Fixed advanced fixers package imports
- ✅ Consolidated cargo cult analyzers with proper parent checking
- ✅ Added missing dataclasses (FixOutcome, StrategyRanking)
- ✅ Removed NotImplementedError from production paths
- ✅ Implemented safe fallbacks for unknown imports

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by patterns observed in LLM-assisted development
- Built with love for the AI-assisted programming community
- Special thanks to all contributors and early adopters

## 📮 Support

- 🐛 [Report Issues](https://github.com/rohanvinaik/TailChasingFixer/issues)
- 💬 [Discussions](https://github.com/rohanvinaik/TailChasingFixer/discussions)
- 📧 Contact: [Create an issue](https://github.com/rohanvinaik/TailChasingFixer/issues/new)

---

**Made with ❤️ to improve AI-assisted development**

*Stop chasing your tail, start fixing real issues!*