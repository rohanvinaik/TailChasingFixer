# Contributing to TailChasing Fixer

Thank you for your interest in contributing to TailChasing Fixer! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Testing Guidelines](#testing-guidelines)
- [Code Style Rules](#code-style-rules)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read and follow our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback gracefully
- Put the community first

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)
- Pre-commit (for code quality checks)

### First-Time Contributors

Look for issues labeled with:
- `good first issue` - Simple fixes perfect for beginners
- `help wanted` - Issues where we need community help
- `documentation` - Documentation improvements
- `enhancement` - New features or improvements

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/TailChasingFixer.git
cd TailChasingFixer

# Add upstream remote
git remote add upstream https://github.com/rohanvinaik/TailChasingFixer.git
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n tailchasing python=3.11
conda activate tailchasing
```

### 3. Install Dependencies

```bash
# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
tailchasing --version
pytest --version
```

### 4. Configure Git

```bash
# Set up your user information
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Enable helpful Git settings
git config branch.autosetuprebase always
git config pull.rebase true
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tailchasing --cov-report=html

# Run specific test file
pytest tests/test_analyzers/test_semantic_hv.py

# Run specific test
pytest tests/test_analyzers/test_semantic_hv.py::TestSemanticAnalyzer::test_duplicate_detection

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

### Writing Tests

All new features must include tests. Follow these guidelines:

```python
# tests/test_module.py
import pytest
from tailchasing.module import YourClass


class TestYourClass:
    """Test suite for YourClass."""
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        instance = YourClass()
        result = instance.method("input")
        assert result == "expected_output"
    
    def test_edge_case(self):
        """Test edge cases are handled properly."""
        instance = YourClass()
        with pytest.raises(ValueError):
            instance.method(None)
    
    @pytest.mark.parametrize("input,expected", [
        ("test1", "result1"),
        ("test2", "result2"),
        ("test3", "result3"),
    ])
    def test_multiple_inputs(self, input, expected):
        """Test multiple input scenarios."""
        instance = YourClass()
        assert instance.method(input) == expected
```

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
├── fixtures/       # Test data and fixtures
├── benchmarks/     # Performance benchmarks
└── conftest.py     # Shared pytest configuration
```

### Coverage Requirements

- Minimum 80% code coverage for new code
- 100% coverage for critical paths (fix generation, analysis)
- Document any untestable code with comments

## Code Style Rules

### Python Style Guide

We follow PEP 8 with these additions:

```python
# Good: Use type hints
def process_data(data: List[str], threshold: float = 0.8) -> Dict[str, Any]:
    """Process data with given threshold."""
    pass

# Good: Use descriptive variable names
user_count = len(users)
average_score = sum(scores) / len(scores)

# Bad: Single letter variables (except in comprehensions)
n = len(users)  # Don't do this
a = sum(scores) / len(scores)  # Don't do this
```

### Docstring Format

Use Google-style docstrings:

```python
def analyze_patterns(
    self,
    code: str,
    config: Dict[str, Any],
    verbose: bool = False
) -> List[Issue]:
    """Analyze code for tail-chasing patterns.
    
    Performs comprehensive analysis to detect various anti-patterns
    that indicate LLM-induced tail-chasing behavior.
    
    Args:
        code: Source code to analyze
        config: Configuration dictionary with analysis settings
        verbose: If True, provide detailed output
        
    Returns:
        List of detected issues with severity and suggestions
        
    Raises:
        ValueError: If code is empty or invalid
        ConfigError: If configuration is invalid
        
    Examples:
        >>> analyzer = PatternAnalyzer()
        >>> issues = analyzer.analyze_patterns(
        ...     code="def foo(): pass",
        ...     config={"threshold": 0.8}
        ... )
        >>> len(issues)
        1
    """
    pass
```

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from rich.console import Console

# Local application imports
from tailchasing.core import Issue
from tailchasing.analyzers import BaseAnalyzer
from tailchasing.utils import calculate_similarity
```

### Code Quality Checks

Pre-commit runs automatically, but you can run manually:

```bash
# Format code with Black
black tailchasing/ tests/

# Sort imports with isort
isort tailchasing/ tests/

# Lint with Ruff
ruff check tailchasing/

# Type check with MyPy
mypy tailchasing/

# Security check with Bandit
bandit -r tailchasing/
```

## Pull Request Process

### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

Use conventional commit messages:

```bash
# Format: <type>(<scope>): <subject>

git commit -m "feat(analyzer): add semantic duplicate detection"
git commit -m "fix(cli): handle empty input gracefully"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(semantic): add edge case tests"
git commit -m "refactor(core): simplify issue reporting"
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Maintenance tasks

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:

- **Title**: Clear, descriptive title
- **Description**: 
  - What changes were made
  - Why they were needed
  - How they were tested
- **Checklist**:
  - [ ] Tests pass locally
  - [ ] Code follows style guidelines
  - [ ] Documentation updated
  - [ ] Changelog entry added (if applicable)

### 5. Code Review

- Respond to feedback promptly
- Make requested changes
- Re-request review after changes
- Be patient and respectful

## Issue Guidelines

### Reporting Bugs

Use the bug report template and include:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.0]
- TailChasing version: [e.g., 0.2.0]

**Additional context**
Any other relevant information.
```

### Requesting Features

Use the feature request template:

```markdown
**Is your feature request related to a problem?**
Description of the problem.

**Describe the solution**
How you'd like it to work.

**Alternatives considered**
Other solutions you've thought about.

**Additional context**
Any other information or screenshots.
```

## Documentation

### Documentation Structure

```
docs/
├── getting-started/    # Installation and quick start
├── user-guide/        # Detailed usage instructions
├── api-reference/     # API documentation
├── development/       # Developer documentation
└── examples/          # Example code and tutorials
```

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep it up-to-date with code changes

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build HTML documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Community

### Getting Help

- **GitHub Discussions**: Ask questions and share ideas
- **Issue Tracker**: Report bugs and request features
- **Stack Overflow**: Tag questions with `tailchasing-fixer`

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

## Advanced Development

### Running Benchmarks

```bash
# Run performance benchmarks
python benchmarks/run_benchmarks.py single --model mock

# Compare multiple scenarios
python benchmarks/run_benchmarks.py compare --compare-models mock1,mock2

# Track performance over time
python benchmarks/run_benchmarks.py track --plot
```

### Profiling

```bash
# CPU profiling
python -m cProfile -s cumulative -m tailchasing analyze .

# Memory profiling
python -m memory_profiler tailchasing analyze .

# Line profiling
kernprof -l -v tailchasing analyze .
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Or use ipdb for better experience
import ipdb; ipdb.set_trace()
```

## Release Process

Maintainers follow this process:

1. Update version in `tailchasing/version.py`
2. Update CHANGELOG.md
3. Create release PR
4. Merge after approval
5. Tag release: `git tag v0.2.0`
6. Push tag: `git push upstream v0.2.0`
7. GitHub Actions builds and publishes to PyPI

## Questions?

If you have questions not covered here:

1. Check existing issues and discussions
2. Ask in GitHub Discussions
3. Contact maintainers

Thank you for contributing to TailChasing Fixer! 🚀