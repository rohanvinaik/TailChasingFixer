"""
Enhanced semantic duplicate detection with multi-modal analysis.
"""

import ast
import re
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import hashlib

from ...core.issues import Issue


class SemanticDuplicateEnhancer:
    """Enhanced semantic duplicate detection with multi-modal analysis."""
    
    def __init__(self, vector_dim: int = 8192):
        self.vector_dim = vector_dim
        self.channel_weights = {
            'data_flow': 1.5,
            'return_patterns': 1.3,
            'error_handling': 1.2,
            'loop_patterns': 1.1,
            'name_tokens': 0.8,  # Lower weight - LLMs often use different names
        }
    
    def encode_function_multimodal(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """
        Encode function using multiple semantic channels.
        """
        channels = {}
        
        # Data flow analysis
        channels['data_flow'] = self._analyze_data_flow(func_ast)
        
        # Return pattern analysis
        channels['return_patterns'] = self._analyze_return_patterns(func_ast)
        
        # Error handling patterns
        channels['error_handling'] = self._analyze_error_handling(func_ast)
        
        # Loop structure analysis
        channels['loop_patterns'] = self._analyze_loop_patterns(func_ast)
        
        # Traditional name tokenization
        channels['name_tokens'] = self._tokenize_name(func_ast.name)
        
        # Combine into weighted hypervector
        return self._create_weighted_hypervector(channels)
    
    def find_semantic_duplicates(self, functions: List[Tuple[str, ast.FunctionDef]], 
                               threshold: float = 0.85) -> List[Issue]:
        """Find semantically similar functions across the codebase."""
        issues = []
        
        # Limit the number of functions to analyze to prevent hanging
        MAX_FUNCTIONS = 200
        if len(functions) > MAX_FUNCTIONS:
            # Sample the functions if there are too many
            import random
            functions = random.sample(functions, MAX_FUNCTIONS)
        
        # Encode all functions
        encoded_functions = []
        for filepath, func_ast in functions:
            try:
                vector = self.encode_function_multimodal(func_ast)
                encoded_functions.append((filepath, func_ast, vector))
            except Exception:
                continue  # Skip functions that can't be encoded
        
        # Compare all pairs (with early termination if too many issues)
        MAX_ISSUES = 50
        for i, (file1, func1, vec1) in enumerate(encoded_functions):
            if len(issues) >= MAX_ISSUES:
                break
            for file2, func2, vec2 in encoded_functions[i+1:]:
                # Calculate similarity
                similarity = self._cosine_similarity(vec1, vec2)
                
                if similarity > threshold:
                    # Additional checks to avoid false positives
                    if self._are_likely_duplicates(func1, func2, similarity):
                        issues.append(Issue(
                            kind="semantic_duplicate_multimodal",
                            message=f"High semantic similarity ({similarity:.2%}) between '{func1.name}' and '{func2.name}'",
                            severity=3,
                            file=file1,
                            line=func1.lineno,
                            evidence={
                                "function1": func1.name,
                                "function2": func2.name,
                                "file1": file1,
                                "file2": file2,
                                "similarity": similarity,
                                "channels": self._analyze_channel_contributions(vec1, vec2)
                            },
                            suggestions=[
                                f"Consider consolidating '{func1.name}' and '{func2.name}'",
                                "Extract common functionality into a shared helper",
                                "Review if both implementations are needed"
                            ]
                        ))
        
        return issues
    
    def _analyze_data_flow(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Analyze how data flows through the function."""
        flow_vector = np.zeros(self.vector_dim // 5)
        
        # Track variable assignments and usage
        vars_assigned = set()
        vars_used = set()
        
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        vars_assigned.add(target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                vars_used.add(node.id)
        
        # Encode flow patterns
        flow_patterns = {
            'assign_before_use': len(vars_assigned & vars_used),
            'unused_vars': len(vars_assigned - vars_used),
            'undefined_vars': len(vars_used - vars_assigned),
        }
        
        # Convert to vector
        for i, (pattern, value) in enumerate(flow_patterns.items()):
            if i * 10 + 10 <= len(flow_vector):
                flow_vector[i * 10:(i + 1) * 10] = min(value, 10)  # Cap values
        
        return flow_vector
    
    def _analyze_return_patterns(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Analyze function return patterns."""
        return_vector = np.zeros(self.vector_dim // 5)
        
        returns = [node for node in ast.walk(func_ast) if isinstance(node, ast.Return)]
        
        patterns = {
            'num_returns': min(len(returns), 10),
            'early_returns': min(sum(1 for r in returns if r.lineno < func_ast.lineno + 5), 10),
            'none_returns': min(sum(1 for r in returns if r.value is None), 10),
            'complex_returns': min(sum(1 for r in returns if r.value and 
                                     isinstance(r.value, (ast.Dict, ast.List, ast.Tuple))), 10),
        }
        
        for i, (pattern, value) in enumerate(patterns.items()):
            if i * 10 + 10 <= len(return_vector):
                return_vector[i * 10:(i + 1) * 10] = value
        
        return return_vector
    
    def _analyze_error_handling(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Analyze error handling patterns."""
        error_vector = np.zeros(self.vector_dim // 5)
        
        try_blocks = [node for node in ast.walk(func_ast) if isinstance(node, ast.Try)]
        raises = [node for node in ast.walk(func_ast) if isinstance(node, ast.Raise)]
        
        patterns = {
            'num_try_blocks': min(len(try_blocks), 10),
            'num_raises': min(len(raises), 10),
            'has_finally': min(sum(1 for t in try_blocks if t.finalbody), 10),
            'catches_all': min(sum(1 for t in try_blocks for h in t.handlers if h.type is None), 10),
        }
        
        for i, (pattern, value) in enumerate(patterns.items()):
            if i * 10 + 10 <= len(error_vector):
                error_vector[i * 10:(i + 1) * 10] = value
        
        return error_vector
    
    def _analyze_loop_patterns(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Analyze loop structures."""
        loop_vector = np.zeros(self.vector_dim // 5)
        
        for_loops = [node for node in ast.walk(func_ast) if isinstance(node, ast.For)]
        while_loops = [node for node in ast.walk(func_ast) if isinstance(node, ast.While)]
        comprehensions = [
            node for node in ast.walk(func_ast) 
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp))
        ]
        
        patterns = {
            'num_for_loops': min(len(for_loops), 10),
            'num_while_loops': min(len(while_loops), 10),
            'num_comprehensions': min(len(comprehensions), 10),
            'nested_loops': min(self._count_nested_loops(func_ast), 10),
        }
        
        for i, (pattern, value) in enumerate(patterns.items()):
            if i * 10 + 10 <= len(loop_vector):
                loop_vector[i * 10:(i + 1) * 10] = value
        
        return loop_vector
    
    def _tokenize_name(self, name: str) -> np.ndarray:
        """Tokenize function name into semantic components."""
        name_vector = np.zeros(self.vector_dim // 5)
        
        # Split on common naming patterns
        tokens = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+', name)
        
        # Common programming verbs/nouns
        common_tokens = {
            'get': 0, 'set': 1, 'create': 2, 'delete': 3, 'update': 4,
            'process': 5, 'handle': 6, 'parse': 7, 'validate': 8, 'check': 9,
            'data': 10, 'user': 11, 'file': 12, 'error': 13, 'result': 14,
        }
        
        for token in tokens[:20]:  # Limit to first 20 tokens
            token_lower = token.lower()
            if token_lower in common_tokens:
                name_vector[common_tokens[token_lower]] = 1
            else:
                # Hash unknown tokens to positions
                hash_val = int(hashlib.md5(token_lower.encode()).hexdigest(), 16)
                name_vector[hash_val % (self.vector_dim // 5)] = 1
        
        return name_vector
    
    def _create_weighted_hypervector(self, channels: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine channels into weighted hypervector."""
        combined = np.zeros(self.vector_dim)
        
        start_idx = 0
        for channel_name, channel_vector in channels.items():
            weight = self.channel_weights.get(channel_name, 1.0)
            end_idx = min(start_idx + len(channel_vector), self.vector_dim)
            combined[start_idx:end_idx] = channel_vector[:end_idx-start_idx] * weight
            start_idx = end_idx
            if start_idx >= self.vector_dim:
                break
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        
        return combined
    
    def _count_nested_loops(self, func_ast: ast.FunctionDef) -> int:
        """Count maximum loop nesting depth."""
        max_depth = 0
        
        def count_depth(node, depth=0):
            nonlocal max_depth
            if isinstance(node, (ast.For, ast.While)):
                depth += 1
                max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                count_depth(child, depth)
        
        count_depth(func_ast)
        return max_depth
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _are_likely_duplicates(self, func1: ast.FunctionDef, func2: ast.FunctionDef, 
                              similarity: float) -> bool:
        """Additional checks to filter false positives."""
        # Don't flag very short functions as duplicates
        if len(func1.body) < 3 or len(func2.body) < 3:
            return False
        
        # Don't flag functions with very different names if similarity is borderline
        name_similarity = self._name_similarity(func1.name, func2.name)
        if similarity < 0.9 and name_similarity < 0.3:
            return False
        
        return True
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate name similarity."""
        import difflib
        return difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def _analyze_channel_contributions(self, vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, float]:
        """Analyze which channels contribute most to similarity."""
        channel_size = self.vector_dim // 5
        contributions = {}
        
        for i, channel in enumerate(['data_flow', 'return_patterns', 'error_handling', 
                                   'loop_patterns', 'name_tokens']):
            start_idx = i * channel_size
            end_idx = min(start_idx + channel_size, self.vector_dim)
            
            if end_idx > start_idx:
                channel_vec1 = vec1[start_idx:end_idx]
                channel_vec2 = vec2[start_idx:end_idx]
                contributions[channel] = self._cosine_similarity(channel_vec1, channel_vec2)
        
        return contributions
