#!/usr/bin/env python3
"""
Type coverage checker for TailChasing Fixer.

This script ensures that type annotation coverage stays above the required threshold.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class TypeCoverageStats:
    """Statistics for type coverage."""
    total_functions: int = 0
    typed_functions: int = 0
    total_parameters: int = 0
    typed_parameters: int = 0
    total_returns: int = 0
    typed_returns: int = 0
    
    @property
    def function_coverage(self) -> float:
        return (self.typed_functions / self.total_functions * 100) if self.total_functions > 0 else 100.0
    
    @property
    def parameter_coverage(self) -> float:
        return (self.typed_parameters / self.total_parameters * 100) if self.total_parameters > 0 else 100.0
    
    @property
    def return_coverage(self) -> float:
        return (self.typed_returns / self.total_returns * 100) if self.total_returns > 0 else 100.0
    
    @property
    def overall_coverage(self) -> float:
        total_items = self.total_functions + self.total_parameters + self.total_returns
        typed_items = self.typed_functions + self.typed_parameters + self.typed_returns
        return (typed_items / total_items * 100) if total_items > 0 else 100.0


class TypeCoverageAnalyzer:
    """Analyze type annotation coverage in Python files."""
    
    def __init__(self):
        self.stats = TypeCoverageStats()
        self.exclude_patterns = {
            '__init__.py',
            'test_',
            'tests/',
            'scripts/',
            'docs/',
            'examples/',
            'demo/',
            'vscode-extension/'
        }
    
    def should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis."""
        file_str = str(file_path)
        return any(pattern in file_str for pattern in self.exclude_patterns)
    
    def analyze_function(self, node: ast.FunctionDef) -> None:
        """Analyze type annotations in a function definition."""
        # Skip special methods and private functions for less strict coverage
        if node.name.startswith('_'):
            return
        
        self.stats.total_functions += 1
        
        # Check return type annotation
        self.stats.total_returns += 1
        if node.returns is not None:
            self.stats.typed_returns += 1
        
        # Check parameter annotations
        for arg in node.args.args:
            # Skip 'self' and 'cls' parameters
            if arg.arg in ('self', 'cls'):
                continue
                
            self.stats.total_parameters += 1
            if arg.annotation is not None:
                self.stats.typed_parameters += 1
        
        # Count function as typed if it has either return annotation or all params annotated
        param_count = len([arg for arg in node.args.args if arg.arg not in ('self', 'cls')])
        param_typed_count = len([arg for arg in node.args.args if arg.arg not in ('self', 'cls') and arg.annotation is not None])
        
        has_return_annotation = node.returns is not None
        has_all_params_typed = param_count == 0 or param_typed_count == param_count
        
        if has_return_annotation or has_all_params_typed:
            self.stats.typed_functions += 1
    
    def analyze_file(self, file_path: Path) -> None:
        """Analyze type coverage in a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.analyze_function(node)
                elif isinstance(node, ast.AsyncFunctionDef):
                    self.analyze_function(node)  # AsyncFunctionDef is compatible
                    
        except Exception as e:
            print(f"Warning: Failed to analyze {file_path}: {e}", file=sys.stderr)
    
    def analyze_directory(self, root_path: Path) -> None:
        """Analyze type coverage in all Python files in a directory."""
        python_files = list(root_path.rglob("*.py"))
        
        for file_path in python_files:
            if not self.should_exclude(file_path):
                self.analyze_file(file_path)
    
    def print_report(self, min_coverage: float = 80.0) -> bool:
        """Print type coverage report and return True if coverage is sufficient."""
        print("=" * 60)
        print("📊 TYPE COVERAGE REPORT")
        print("=" * 60)
        
        print(f"Functions: {self.stats.typed_functions}/{self.stats.total_functions} "
              f"({self.stats.function_coverage:.1f}%)")
        
        print(f"Parameters: {self.stats.typed_parameters}/{self.stats.total_parameters} "
              f"({self.stats.parameter_coverage:.1f}%)")
        
        print(f"Returns: {self.stats.typed_returns}/{self.stats.total_returns} "
              f"({self.stats.return_coverage:.1f}%)")
        
        print(f"Overall: {self.stats.overall_coverage:.1f}%")
        print("-" * 60)
        
        if self.stats.overall_coverage >= min_coverage:
            print(f"✅ Type coverage {self.stats.overall_coverage:.1f}% meets requirement ({min_coverage}%)")
            return True
        else:
            print(f"❌ Type coverage {self.stats.overall_coverage:.1f}% below requirement ({min_coverage}%)")
            print(f"   Need {min_coverage - self.stats.overall_coverage:.1f}% more coverage")
            return False


def main():
    """Main entry point."""
    # Parse command line arguments
    min_coverage = 80.0
    if len(sys.argv) > 1:
        try:
            min_coverage = float(sys.argv[1])
        except ValueError:
            print(f"Invalid coverage threshold: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)
    
    # Find project root
    root_path = Path(__file__).parent.parent
    
    # Analyze type coverage
    analyzer = TypeCoverageAnalyzer()
    analyzer.analyze_directory(root_path)
    
    # Print report and exit with appropriate code
    if analyzer.print_report(min_coverage):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()