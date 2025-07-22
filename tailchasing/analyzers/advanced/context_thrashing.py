"""
Context Window Thrashing Detection

Detects when LLM forgets earlier context and reimplements functionality.
Pattern: Implement X → 1000 lines later → Implement X again slightly differently
"""

import ast
import difflib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..base import AnalysisContext
from ...core.issues import Issue


@dataclass
class FunctionSignature:
    """Represents a function's semantic signature."""
    name: str
    args: List[str]
    line: int
    file: str
    body_dump: str
    calls: List[str]
    returns: int


class ContextWindowThrashingAnalyzer:
    """Detects reimplementation due to context window limitations."""
    
    name = "context_window_thrashing"
    
    def __init__(self):
        self.min_line_distance = 500
        self.similarity_threshold_min = 0.6
        self.similarity_threshold_max = 0.95
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Analyze files for context window thrashing patterns."""
        issues = []
        
        # Analyze each file separately
        for filepath, tree in ctx.ast_index.items():
            file_issues = self._analyze_file(filepath, tree)
            issues.extend(file_issues)
        
        return issues
    
    def _analyze_file(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Analyze a single file for thrashing patterns."""
        issues = []
        functions = self._extract_functions(filepath, tree)
        
        # Compare all function pairs
        for i, func1 in enumerate(functions):
            for func2 in functions[i+1:]:
                issue = self._compare_functions(func1, func2, filepath)
                if issue:
                    issues.append(issue)
        
        return issues
    
    def _extract_functions(self, filepath: str, tree: ast.AST) -> List[FunctionSignature]:
        """Extract all functions with their signatures."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                sig = FunctionSignature(
                    name=node.name,
                    args=[arg.arg for arg in node.args.args],
                    line=node.lineno,
                    file=filepath,
                    body_dump=ast.dump(node),
                    calls=self._extract_calls(node),
                    returns=self._count_returns(node)
                )
                functions.append(sig)
        
        return functions
    
    def _extract_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract all function calls from a function."""
        calls = []
        
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if isinstance(subnode.func, ast.Name):
                    calls.append(subnode.func.id)
                elif isinstance(subnode.func, ast.Attribute):
                    calls.append(subnode.func.attr)
        
        return calls
    
    def _count_returns(self, node: ast.FunctionDef) -> int:
        """Count return statements in a function."""
        return sum(1 for n in ast.walk(node) if isinstance(n, ast.Return))
    
    def _compare_functions(self,
                          func1: FunctionSignature,
                          func2: FunctionSignature,
                          filepath: str) -> Optional[Issue]:
        """Compare two functions for thrashing patterns."""
        
        # Check line distance
        line_distance = abs(func2.line - func1.line)
        if line_distance < self.min_line_distance:
            return None
        
        # Calculate various similarity metrics
        name_similarity = difflib.SequenceMatcher(
            None, func1.name, func2.name
        ).ratio()
        
        # Argument similarity
        args_similarity = 0.0
        if func1.args or func2.args:
            common_args = set(func1.args) & set(func2.args)
            total_args = set(func1.args) | set(func2.args)
            args_similarity = len(common_args) / len(total_args) if total_args else 0
        
        # Call pattern similarity
        calls_similarity = 0.0
        if func1.calls or func2.calls:
            common_calls = set(func1.calls) & set(func2.calls)
            total_calls = set(func1.calls) | set(func2.calls)
            calls_similarity = len(common_calls) / len(total_calls) if total_calls else 0
        
        # AST structure similarity (simplified)
        ast_similarity = difflib.SequenceMatcher(
            None, func1.body_dump, func2.body_dump
        ).ratio()
        
        # Return pattern similarity
        return_similarity = 1.0 if func1.returns == func2.returns else 0.5
        
        # Weighted overall similarity
        overall_similarity = (
            name_similarity * 0.2 +
            args_similarity * 0.25 +
            calls_similarity * 0.25 +
            ast_similarity * 0.2 +
            return_similarity * 0.1
        )
        
        # Check if within thrashing range
        if self.similarity_threshold_min < overall_similarity < self.similarity_threshold_max:
            return Issue(
                kind="context_window_thrashing",
                message=f"Functions '{func1.name}' and '{func2.name}' are {overall_similarity:.0%} similar "
                       f"but {line_distance} lines apart. Likely context window exceeded.",
                severity=3,
                file=filepath,
                line=func2.line,  # Point to the later occurrence
                symbol=func2.name,
                evidence={
                    'functions': [func1.name, func2.name],
                    'lines': [func1.line, func2.line],
                    'line_distance': line_distance,
                    'similarity_scores': {
                        'overall': overall_similarity,
                        'name': name_similarity,
                        'args': args_similarity,
                        'calls': calls_similarity,
                        'structure': ast_similarity
                    }
                },
                suggestions=[
                    f"Consider merging '{func1.name}' (line {func1.line}) and "
                    f"'{func2.name}' (line {func2.line})",
                    "Extract common functionality to avoid duplication",
                    "Use more descriptive names to avoid confusion"
                ]
            )
        
        return None
