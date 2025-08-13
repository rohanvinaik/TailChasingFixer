# Project Structure

## Directory Layout

```
TailChasingFixer/
├── tailchasing/          # Main package source code
│   ├── analyzers/        # Analysis modules for different patterns
│   ├── catalytic/        # Catalytic analysis components
│   ├── chromatin/        # Chromatin-based analysis
│   ├── ci/               # CI/CD integration
│   ├── cli/              # Command-line interface
│   ├── core/             # Core functionality
│   ├── fixers/           # Auto-fix generation
│   ├── guards/           # Guard mode components
│   ├── llm/              # LLM integration
│   ├── semantic/         # Semantic analysis
│   └── utils/            # Utility functions
│
├── tests/                # Test suite
│   ├── integration/      # Integration tests
│   ├── test_semantic/    # Semantic analysis tests
│   └── ...              # Unit tests
│
├── docs/                 # Documentation
│   ├── guides/           # User and developer guides
│   ├── README_COMPREHENSIVE.md
│   └── README_PYPI.md
│
├── config/               # Configuration files
│   ├── examples/         # Example configurations
│   └── README.md
│
├── benchmarks/           # Performance benchmarks
├── demo/                 # Demo files and examples
├── examples/             # Example code patterns
├── scripts/              # Build and utility scripts
├── vscode-extension/     # VS Code extension
├── output/               # Generated output (gitignored)
└── logs/                 # Log files (gitignored)
```

## Key Files in Root

- `README.md` - Main project documentation
- `CLAUDE.md` - Instructions for Claude AI
- `setup.py`, `setup.cfg`, `pyproject.toml` - Python package configuration
- `requirements.txt` - Python dependencies
- `.tailchasing.yml` - Default configuration file
- `LICENSE` - MIT license
- `CONTRIBUTING.md` - Contribution guidelines
- `ROADMAP.md` - Project roadmap

## Configuration Files

- `.gitignore` - Git ignore patterns
- `.pre-commit-config.yaml` - Pre-commit hooks
- `pytest.ini` - Pytest configuration
- `mypy.ini` - Type checking configuration
- `.tcdignore` - TailChasing ignore patterns

## Generated Directories (not in version control)

- `.tailchasing_cache/` - Analysis cache
- `.tailchasing_checkpoints/` - Processing checkpoints
- `.tailchasing_history.db` - History database
- `output/` - Generated reports and fix scripts
- `build/`, `dist/` - Build artifacts
- `*.egg-info/` - Package metadata