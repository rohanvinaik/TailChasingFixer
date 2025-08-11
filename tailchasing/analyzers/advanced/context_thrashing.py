"""
Context thrashing analyzer for detecting when LLMs exceed context windows.
"""

import ast
import difflib
from typing import List, Dict, Tuple
from .base_advanced import ContextAwareAnalyzer
from ...core.issues import Issue


class ContextThrashingAnalyzer(ContextAwareAnalyzer):
    """Detect when LLM forgets context and reimplements functionality."""
    
    name = "context_thrashing"
    
    def _initialize_specific_config(self):
        """Initialize context thrashing specific configuration."""
        super()._initialize_specific_config()
        self.set_config('min_distance', 500)  # Minimum line distance to consider thrashing
        self.set_threshold('similarity', 0.6)  # Minimum similarity to flag
        self.set_threshold('max_similarity', 0.95)  # Maximum similarity (avoid exact duplicates)
    
    @property
    def min_distance(self):
        """Get minimum distance configuration."""
        return self.get_config('min_distance', 500)
    
    @property
    def similarity_threshold(self):
        """Get similarity threshold."""
        return self.get_threshold('similarity', 0.6)
    
    @property
    def max_similarity(self):
        """Get maximum similarity threshold."""
        return self.get_threshold('max_similarity', 0.95)
    
    def run(self, ctx) -> List[Issue]:
        """Run context thrashing analysis."""
        issues = []
        
        for filepath, tree in ctx.ast_index.items():
            file_issues = self._analyze_file(filepath, tree)
            issues.extend(file_issues)
        
        return issues
    
    def _analyze_file(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Analyze a single file for context thrashing patterns."""
        issues = []
        functions = []
        
        # Collect all functions with their metadata
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'body_dump': ast.dump(node),
                    'node': node
                })
        
        # Compare functions that are far apart
        for i, func1 in enumerate(functions):
            for func2 in functions[i+1:]:
                line_distance = abs(func2['line'] - func1['line'])
                
                if line_distance >= self.min_distance:
                    similarity = self._calculate_similarity(func1, func2)
                    
                    if self.similarity_threshold < similarity < self.max_similarity:
                        issues.append(Issue(
                            kind="context_window_thrashing",
                            message=f"Functions '{func1['name']}' and '{func2['name']}' are {similarity:.0%} similar but {line_distance} lines apart",
                            severity=3,
                            file=filepath,
                            line=func1['line'],
                            symbol=func1['name'],
                            evidence={
                                "function1": func1['name'],
                                "function2": func2['name'],
                                "line1": func1['line'],
                                "line2": func2['line'],
                                "similarity": similarity,
                                "distance": line_distance,
                                "likely_cause": "context_window_exceeded"
                            },
                            suggestions=[
                                f"Consider merging '{func1['name']}' and '{func2['name']}'",
                                "Extract common functionality into a shared helper",
                                "Review if both functions are actually needed",
                                "Break large files into smaller, focused modules"
                            ]
                        ))
        
        return issues
    
    def _calculate_similarity(self, func1: Dict, func2: Dict) -> float:
        """Calculate overall similarity between two functions."""
        # Name similarity
        name_similarity = difflib.SequenceMatcher(
            None, func1['name'], func2['name']
        ).ratio()
        
        # Argument similarity
        args_similarity = 0.0
        if func1['args'] or func2['args']:
            common_args = set(func1['args']) & set(func2['args'])
            total_args = set(func1['args']) | set(func2['args'])
            args_similarity = len(common_args) / len(total_args) if total_args else 0
        
        # AST structure similarity
        ast_similarity = difflib.SequenceMatcher(
            None, func1['body_dump'], func2['body_dump']
        ).ratio()
        
        # Weighted average
        return (name_similarity * 0.3 + args_similarity * 0.3 + ast_similarity * 0.4)
