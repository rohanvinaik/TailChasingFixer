"""
CLI utilities for TailChasingFixer.
"""

from .output_manager import OutputManager, VerbosityLevel, OutputFormat
from .profiler import PerformanceProfiler, ComponentTimer, profile_analyzer

# Import main function from the parent cli module
def main():
    """Main CLI entry point - imports from parent cli module."""
    # Import the actual CLI module (cli.py) directly
    from .. import cli
    return cli.main()

__all__ = [
    'OutputManager',
    'VerbosityLevel',
    'OutputFormat',
    'PerformanceProfiler',
    'ComponentTimer',
    'profile_analyzer',
    'main'
]