"""
Enhanced semantic analyzer using multimodal approach.
"""

import ast
from typing import List, Tuple
from ..base import Analyzer
from ...core.issues import Issue
from .multimodal_semantic import SemanticDuplicateEnhancer


class EnhancedSemanticAnalyzer(Analyzer):
    """Enhanced semantic analysis with multimodal encoding."""
    
    name = "enhanced_semantic"
    
    def __init__(self):
        self.enhancer = SemanticDuplicateEnhancer()
        self.similarity_threshold = 0.85
    
    def run(self, ctx) -> List[Issue]:
        """Run enhanced semantic analysis."""
        # Extract all functions for analysis
        functions = []
        for func_name, entries in ctx.symbol_table.functions.items():
            for entry in entries:
                functions.append((entry['file'], entry['node']))
        
        if len(functions) < 2:
            return []  # Need at least 2 functions to compare
        
        # Find semantic duplicates using multimodal analysis
        return self.enhancer.find_semantic_duplicates(functions, self.similarity_threshold)
