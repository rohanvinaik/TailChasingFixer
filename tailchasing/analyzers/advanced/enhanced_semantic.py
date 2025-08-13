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
        """Run enhanced semantic analysis using hierarchical approach."""
        # Extract all functions for analysis
        functions = []
        for func_name, entries in ctx.symbol_table.functions.items():
            for entry in entries:
                functions.append((entry['file'], entry['node']))
        
        if len(functions) < 2:
            return []  # Need at least 2 functions to compare
        
        # Use efficient hierarchical analysis for large codebases
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Running enhanced semantic analysis on {len(functions)} functions using hierarchical approach")
        
        # Set the threshold on the enhancer if needed
        if hasattr(self.enhancer, 'similarity_threshold'):
            self.enhancer.similarity_threshold = self.similarity_threshold
        
        # Find semantic duplicates using optimized multimodal analysis
        return self.enhancer.find_semantic_duplicates(functions)
