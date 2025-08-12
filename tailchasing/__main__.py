"""Main entry point for running tailchasing as a module."""

# Import from the cli.py file, not the cli/ directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import main

if __name__ == "__main__":
    main()
