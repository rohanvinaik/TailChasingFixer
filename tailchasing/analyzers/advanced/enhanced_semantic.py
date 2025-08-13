"""
Enhanced semantic analyzer using multimodal approach.
"""

import ast
from typing import List, Tuple
from .base_advanced import SemanticAwareAnalyzer
from ...core.issues import Issue
from .multimodal_semantic import SemanticDuplicateEnhancer


class EnhancedSemanticAnalyzer(SemanticAwareAnalyzer):
    """Enhanced semantic analysis with multimodal encoding."""
    
    name = "enhanced_semantic"
    
    def _initialize_specific_config(self):
        """Initialize enhanced semantic specific configuration."""
        super()._initialize_specific_config()
        self.set_threshold('similarity', 0.85)
        self.enhancer = SemanticDuplicateEnhancer()
    
    @property
    def similarity_threshold(self):
        """Get similarity threshold."""
        return self.get_threshold('similarity', 0.85)
    
    def run(self, ctx) -> List[Issue]:
        """Run enhanced semantic analysis."""
        # Extract all functions for analysis
        functions = []
        for func_name, entries in ctx.symbol_table.functions.items():
            for entry in entries:
                functions.append((entry['file'], entry['node']))
        
        if len(functions) < 2:
            return []  # Need at least 2 functions to compare
        
        # Apply reasonable limit to prevent timeout on large codebases
        MAX_FUNCTIONS = 500  # Limit to prevent O(n²) explosion
        if len(functions) > MAX_FUNCTIONS:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Large codebase ({len(functions)} functions), limiting enhanced semantic analysis to {MAX_FUNCTIONS} functions")
            # Sort by function size (complexity) and take the largest ones
            # as they're more likely to be duplicated
            functions.sort(key=lambda x: len(ast.dump(x[1])), reverse=True)
            functions = functions[:MAX_FUNCTIONS]
        
        # Set the threshold on the enhancer if needed
        if hasattr(self.enhancer, 'similarity_threshold'):
            self.enhancer.similarity_threshold = self.similarity_threshold
        
        # Find semantic duplicates using multimodal analysis
        return self.enhancer.find_semantic_duplicates(functions)
