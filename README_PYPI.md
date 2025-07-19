# Tail-Chasing Detector

[![PyPI version](https://badge.fury.io/py/tail-chasing-detector.svg)](https://badge.fury.io/py/tail-chasing-detector)
[![Python Versions](https://img.shields.io/pypi/pyversions/tail-chasing-detector.svg)](https://pypi.org/project/tail-chasing-detector/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/rohanvinaik/TailChasingFixer/actions/workflows/tail-chasing.yml/badge.svg)](https://github.com/rohanvinaik/TailChasingFixer/actions/workflows/tail-chasing.yml)

Detects LLM-assisted *tail-chasing* anti-patterns using advanced semantic analysis with hypervector computing.

## Installation

```bash
pip install tail-chasing-detector
```

For additional features:
```bash
pip install tail-chasing-detector[visualization]  # For visualizations
pip install tail-chasing-detector[ml]             # For ML enhancements
pip install tail-chasing-detector[all]            # Everything
```

## Quick Start

```bash
# Analyze current directory
tailchasing .

# Analyze with semantic detection
tailchasing . --semantic

# Generate JSON report
tailchasing . --json > report.json

# Watch for changes
tailchasing watch src/
```

## What is a Tail-Chasing Bug?

A Tail-Chasing Bug occurs when an LLM:
1. **Creates circular dependencies or redundant implementations**
2. **Renames or restructures code solely to match error messages**
3. **Generates phantom implementations** to satisfy mistaken imports

## Features

- 🧠 **Semantic Hypervector Analysis** - Detects deep semantic duplicates beyond structural similarity
- 🔄 **Circular import detection** with runtime risk assessment
- 👻 **Phantom/hallucinated code detection**
- 📊 **Risk scoring** with configurable thresholds
- 🔍 **Git integration** for temporal pattern analysis
- 📈 **Multiple output formats** (text, JSON, HTML)

## Configuration

Create `.tailchasing.yml`:

```yaml
semantic:
  enable: true
  hv_dim: 8192
  
risk_thresholds:
  warn: 15
  fail: 30
```

## Documentation

Full documentation at [https://tail-chasing-detector.readthedocs.io](https://tail-chasing-detector.readthedocs.io)

## License

MIT License - see [LICENSE](LICENSE) for details.