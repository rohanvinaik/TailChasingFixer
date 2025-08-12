"""Main entry point for running tailchasing as a module."""

# Direct import from the cli module (cli.py file)
import importlib.util
import sys
from pathlib import Path

# Get the path to cli.py
cli_path = Path(__file__).parent / "cli.py"

# Import the cli module directly
spec = importlib.util.spec_from_file_location("cli", cli_path)
cli_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli_module)

# Get the main function
main = cli_module.main

if __name__ == "__main__":
    main()
