# Contributing to Tail-Chasing Detector

Thank you for your interest in contributing to the Tail-Chasing Detector! This tool helps identify anti-patterns in LLM-assisted development, and we welcome contributions that help make it better.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in our [issue tracker](https://github.com/rohanvinaik/TailChasingFixer/issues)
2. If not, create a new issue with:
   - A clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment details (Python version, OS, etc.)

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-analyzer`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite: `pytest tests/`
6. Commit your changes (`git commit -am 'Add amazing analyzer'`)
7. Push to your branch (`git push origin feature/amazing-analyzer`)
8. Create a Pull Request

### Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all public functions and classes
- Keep line length under 100 characters

### Adding New Analyzers

To add a new analyzer:

1. Create a new file in `tailchasing/analyzers/`
2. Inherit from `BaseAnalyzer`
3. Implement the `run()` method
4. Add your analyzer to the default list in `plugins.py`
5. Add tests in `tests/test_your_analyzer.py`

Example analyzer:

```python
from typing import Iterable
from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue

class YourAnalyzer(BaseAnalyzer):
    """Detects some specific tail-chasing pattern."""
    
    name = "your_analyzer"
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Run the analysis and yield issues."""
        for file, tree in ctx.ast_index.items():
            # Your analysis logic here
            if problem_found:
                yield Issue(
                    kind="your_issue_type",
                    message="Description of the problem",
                    severity=2,
                    file=file,
                    line=line_number,
                    suggestions=["How to fix it"],
                    confidence=0.8
                )
```

### Testing

- Write unit tests for your analyzer
- Include both positive and negative test cases
- Test edge cases
- Ensure all tests pass before submitting

### Documentation

- Update README.md if adding new features
- Add docstrings to all new code
- Include examples in your analyzer's documentation

## Development Setup

```bash
# Clone the repo
git clone https://github.com/rohanvinaik/TailChasingFixer.git
cd TailChasingFixer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
mypy tailchasing/
```

## Areas for Contribution

### High Priority

1. **Wrapper Detection Analyzer**: Detect trivial wrapper functions that add no value
2. **Drift Entropy Analyzer**: Measure semantic drift in duplicated code
3. **SARIF Output Format**: Add support for GitHub Code Scanning
4. **IDE Integrations**: VS Code, PyCharm extensions

### Medium Priority

1. **Multi-language Support**: Extend beyond Python (JavaScript/TypeScript first)
2. **Fix Suggestions**: Automated fixes for simple issues
3. **Performance Optimization**: Speed up analysis for large codebases
4. **Better Symbol Resolution**: Handle dynamic imports, type annotations

### Nice to Have

1. **Web UI**: Dashboard for visualizing tail-chasing patterns
2. **Historical Analysis**: Track improvement over time
3. **Machine Learning**: Use embeddings for semantic duplicate detection
4. **Integration with Other Tools**: Combine with traditional linters

## Questions?

Feel free to open an issue for any questions about contributing. We're here to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
