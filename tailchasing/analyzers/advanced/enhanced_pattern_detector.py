#!/usr/bin/env python3
"""
Enhanced TailChasingFixer components with advanced detection capabilities.
"""

import ast
import re
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import difflib
import networkx as nx
from datetime import datetime
import json

from ...core.issues import Issue


@dataclass
class TailChasingPattern:
    """Enhanced pattern detection with confidence scores."""
    pattern_type: str
    confidence: float
    locations: List[Tuple[str, int]]  # (file, line)
    risk_score: int
    explanation: str
    suggested_fix: str
    related_patterns: List[str]


from .base_advanced import PatternDetectionAnalyzer


class EnhancedPatternDetector(PatternDetectionAnalyzer):
    """Advanced pattern detection beyond simple AST analysis."""
    
    def _initialize_specific_config(self):
        """Initialize enhanced pattern detector specific configuration."""
        super()._initialize_specific_config()
        self.set_threshold('hallucination', 0.7)
        
    def detect_hallucination_cascade(self, 
                                   codebase_ast: Dict[str, ast.AST],
                                   git_history: Optional[Dict] = None) -> List[Issue]:
        """
        Detect when LLM creates entire fictional subsystems.
        
        Pattern: Error in A → Create B → Error in B → Create C → etc.
        """
        issues = []
        
        # Build dependency graph
        dep_graph = nx.DiGraph()
        creation_times = {}
        
        for filepath, tree in codebase_ast.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    dep_graph.add_node(class_name, file=filepath, line=node.lineno)
                    
                    # Find dependencies
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Name):
                            dep_graph.add_edge(class_name, subnode.id)
                    
                    # Get creation time from git
                    if git_history:
                        creation_times[class_name] = git_history.get(
                            f"{filepath}:{class_name}", 
                            datetime.now()
                        )
        
        # Find suspicious clusters
        for component in nx.weakly_connected_components(dep_graph):
            if len(component) > 3:  # Suspicious if many related classes
                # Check if all created recently and together
                times = [creation_times.get(c, datetime.now()) for c in component]
                if times:
                    time_span = (max(times) - min(times)).days
                    
                    if time_span < 2:  # All created within 2 days
                        # Check if isolated from rest of codebase
                        external_refs = sum(
                            1 for n in component 
                            for pred in dep_graph.predecessors(n)
                            if pred not in component
                        )
                        
                        if external_refs < len(component) * 0.2:  # Less than 20% external refs
                            first_node = list(component)[0]
                            node_data = dep_graph.nodes.get(first_node, {})
                            main_file = node_data.get('file', 'unknown')
                            main_line = node_data.get('line', 0)
                            
                            issues.append(Issue(
                                kind="hallucination_cascade",
                                message=f"Detected {len(component)} interdependent classes created together with minimal external references. Likely hallucinated subsystem.",
                                severity=4,
                                file=main_file,
                                line=main_line,
                                evidence={
                                    "components": list(component),
                                    "external_refs": external_refs,
                                    "time_span_days": time_span
                                },
                                suggestions=[
                                    f"Review if {', '.join(list(component)[:3])}{'...' if len(component) > 3 else ''} are actually needed",
                                    "Check if existing functionality could be used instead",
                                    "Consider consolidating related classes"
                                ]
                            ))
        
        return issues
    
    def detect_context_window_thrashing(self, 
                                      file_ast: ast.AST,
                                      filepath: str) -> List[Issue]:
        """
        Detect when LLM forgets context and reimplements functionality.
        
        Pattern: Implement feature → 1000+ lines later → Implement similar feature
        """
        issues = []
        functions = []
        
        # Collect all functions with their line numbers
        for node in ast.walk(file_ast):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'body_dump': ast.dump(node)
                })
        
        # Compare functions that are far apart
        for i, func1 in enumerate(functions):
            for func2 in functions[i+1:]:
                line_distance = abs(func2['line'] - func1['line'])
                
                if line_distance > 500:  # Far apart in file
                    # Check semantic similarity
                    name_similarity = difflib.SequenceMatcher(
                        None, func1['name'], func2['name']
                    ).ratio()
                    
                    # Check if similar arguments
                    if func1['args'] or func2['args']:
                        args_similarity = len(set(func1['args']) & set(func2['args'])) / \
                                        max(len(func1['args']), len(func2['args']), 1)
                    else:
                        args_similarity = 0
                    
                    # Simple AST similarity
                    ast_similarity = difflib.SequenceMatcher(
                        None, func1['body_dump'], func2['body_dump']
                    ).ratio()
                    
                    overall_similarity = (name_similarity + args_similarity + ast_similarity) / 3
                    
                    if 0.6 < overall_similarity < 0.95:  # Similar but not identical
                        issues.append(Issue(
                            kind="context_window_thrashing",
                            message=f"Functions '{func1['name']}' and '{func2['name']}' are {overall_similarity:.0%} similar but {line_distance} lines apart. Likely context window exceeded.",
                            severity=3,
                            file=filepath,
                            line=func1['line'],
                            evidence={
                                "function1": func1['name'],
                                "function2": func2['name'],
                                "line1": func1['line'],
                                "line2": func2['line'],
                                "similarity": overall_similarity,
                                "distance": line_distance
                            },
                            suggestions=[
                                f"Consider merging '{func1['name']}' and '{func2['name']}'",
                                "Extract common functionality into a shared helper",
                                "Review if both functions are actually needed"
                            ]
                        ))
        
        return issues
    
    def detect_import_anxiety(self,
                            imports: Dict[str, Set[str]],
                            usage: Dict[str, Set[str]],
                            filepath: str) -> List[Issue]:
        """
        Detect defensive over-importing pattern.
        
        Pattern: Import errors → Import everything that might be related
        """
        issues = []
        
        for module, imported_items in imports.items():
            used_items = usage.get(module, set())
            unused_items = imported_items - used_items
            
            if len(imported_items) > 5 and len(unused_items) > len(used_items) * 2:
                # Detect import patterns
                import_pattern = self._detect_import_pattern(imported_items)
                
                issues.append(Issue(
                    kind="import_anxiety",
                    message=f"Importing {len(imported_items)} items from {module} but only using {len(used_items)}. Pattern: {import_pattern}",
                    severity=1,
                    file=filepath,
                    line=0,  # Would need line numbers in real implementation
                    evidence={
                        "module": module,
                        "imported_count": len(imported_items),
                        "used_count": len(used_items),
                        "unused_items": list(unused_items),
                        "pattern": import_pattern
                    },
                    suggestions=[
                        f"Remove unused imports: {', '.join(list(unused_items)[:5])}{'...' if len(unused_items) > 5 else ''}",
                        "Import only what you need",
                        "Use 'from module import specific_item' instead of wildcard imports"
                    ]
                ))
        
        return issues
    
    def detect_fix_induced_regression(self,
                                    before_ast: ast.AST,
                                    after_ast: ast.AST,
                                    test_results: Dict[str, bool],
                                    filepath: str) -> List[Issue]:
        """
        Detect when a 'fix' actually breaks other functionality.
        
        Pattern: Fix error in A → Break functionality in B
        """
        issues = []
        
        # Find what changed
        before_funcs = {
            node.name: ast.dump(node) 
            for node in ast.walk(before_ast) 
            if isinstance(node, ast.FunctionDef)
        }
        
        after_funcs = {
            node.name: ast.dump(node)
            for node in ast.walk(after_ast)
            if isinstance(node, ast.FunctionDef)
        }
        
        changed_funcs = {
            name for name in before_funcs
            if name in after_funcs and before_funcs[name] != after_funcs[name]
        }
        
        # Check if changes broke tests
        newly_failing_tests = {
            test: result for test, result in test_results.items()
            if not result  # Failed after change
        }
        
        if changed_funcs and newly_failing_tests:
            issues.append(Issue(
                kind="fix_induced_regression",
                message=f"Changes to {', '.join(changed_funcs)} caused {len(newly_failing_tests)} tests to fail.",
                severity=4,
                file=filepath,
                line=0,
                evidence={
                    "changed_functions": list(changed_funcs),
                    "failing_tests": list(newly_failing_tests.keys())
                },
                suggestions=[
                    "Review changes and ensure they don't break existing functionality",
                    "Add more comprehensive tests",
                    "Consider reverting and taking a different approach"
                ]
            ))
        
        return issues
    
    def _detect_import_pattern(self, items: Set[str]) -> str:
        """Detect patterns in import names."""
        if all('error' in item.lower() or 'exception' in item.lower() for item in items):
            return "error_handling_anxiety"
        elif all(item.istitle() for item in items):
            return "class_import_spree"
        elif len(items) > 10:
            return "import_everything"
        else:
            return "mixed_imports"
    
    def run(self, ctx) -> List[Issue]:
        """Run the enhanced pattern detection analyzer.
        
        Args:
            ctx: Analysis context containing parsed AST and other data
            
        Returns:
            List of detected issues
        """
        all_issues = []
        
        # Run hallucination cascade detection if we have codebase AST
        if hasattr(ctx, 'codebase_ast') and ctx.codebase_ast:
            git_history = getattr(ctx, 'git_history', None)
            all_issues.extend(self.detect_hallucination_cascade(ctx.codebase_ast, git_history))
        
        # Run context window thrashing detection if we have parsed files
        if hasattr(ctx, 'parsed_files') and ctx.parsed_files:
            for filepath, tree in ctx.parsed_files.items():
                all_issues.extend(self.detect_context_window_thrashing(tree, filepath))
        
        # Run import anxiety detection if we have imports analysis
        if hasattr(ctx, 'imports_analysis') and ctx.imports_analysis:
            for filepath, imports_data in ctx.imports_analysis.items():
                all_issues.extend(self.detect_import_anxiety(imports_data, filepath))
        
        return all_issues
