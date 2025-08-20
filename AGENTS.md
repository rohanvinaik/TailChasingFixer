# AGENTS

## Development Guidelines
- Use Python 3.11+.
- Install dependencies with `pip install -e .[dev]` and any needed extras.
- Run linting and formatting with `pre-commit run --files <file1> [<file2> ...]`.
- Run the test suite with `pytest` after making changes. Some tests require optional packages such as `hypothesis`, `typer`, `psutil`, and `requests`. Install them as needed or note failures if unavailable.
- Avoid adding TODO/FIXME comments; track issues in GitHub instead.

## Repository Notes
- The `semantic` package relies on `lsh_index.FeatureConfig` and `LSHParams` for incremental analysis.
- Pre-commit configuration may need maintenance; report configuration errors if the hooks fail to run.

