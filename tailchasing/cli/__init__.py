"""
CLI utilities for TailChasingFixer.
"""

from .output_manager import OutputManager, VerbosityLevel, OutputFormat
from .profiler import PerformanceProfiler, ComponentTimer, profile_analyzer

# Import main function from the cli_main module
def main():
    """Main CLI entry point - imports from cli_main module."""
    # Import directly from the renamed cli_main module
    from .. import cli_main
    return cli_main.main()

__all__ = [
    'OutputManager',
    'VerbosityLevel',
    'OutputFormat',
    'PerformanceProfiler',
    'ComponentTimer',
    'profile_analyzer',
    'main'
]