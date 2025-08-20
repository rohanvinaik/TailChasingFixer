"""
Enhanced multimodal semantic analyzer with 16384-dimensional hypervector encoding.

This module provides advanced semantic analysis capabilities including:
- Deep data flow pattern analysis
- Control flow complexity measurement  
- Enhanced error handling pattern detection
- Type pattern analysis
- Drift detection across commits
- Weighted channel combination with hypervector operations
"""

import ast
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from functools import lru_cache
import logging

from ...core.issues import Issue
from ...semantic.hv_space import HVSpace


class EnhancedSemanticAnalyzer:
    """Enhanced semantic analyzer with 16384-dimensional hypervector encoding."""
    
    def __init__(self, vector_dim: int = 16384):
        self.vector_dim = vector_dim
        self.logger = logging.getLogger(__name__)
        
        # Initialize hypervector space for enhanced encoding
        self.hv_space = HVSpace(dim=vector_dim, bipolar=True, seed=42)
        
        # Enhanced channel weights with fine-tuned values
        self.channel_weights = {
            'data_flow': 2.0,           # Critical for semantic similarity
            'control_flow': 1.8,        # Control structures are key differentiators
            'return_patterns': 1.6,     # Return behavior is semantically important
            'error_handling': 1.4,      # Error patterns show code maturity
            'loop_structures': 1.3,     # Loop complexity matters
            'type_patterns': 1.2,       # Type usage patterns
            'name_tokens': 0.6,         # Lower weight - names can vary significantly
        }
        
        # Computation cache for expensive operations
        self._pattern_cache: Dict[str, np.ndarray] = {}
        self._ast_cache: Dict[str, Dict[str, Any]] = {}
        
        # Drift detection state
        self.commit_history: List[Dict[str, Any]] = []
        self.semantic_snapshots: Dict[str, np.ndarray] = {}
    
    def encode_function_multimodal(self, func_ast: ast.FunctionDef, cache_key: Optional[str] = None) -> np.ndarray:
        """
        Encode function using multiple semantic channels with aggressive caching.
        """
        # Check cache first
        if cache_key and cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key].copy()
        
        channels = {}
        
        # Enhanced data flow analysis
        channels['data_flow'] = self._analyze_data_flow_enhanced(func_ast)
        
        # Control flow complexity analysis
        channels['control_flow'] = self._analyze_control_flow_complexity(func_ast)
        
        # Return pattern analysis
        channels['return_patterns'] = self._analyze_return_patterns_enhanced(func_ast)
        
        # Error handling patterns
        channels['error_handling'] = self._analyze_error_handling_enhanced(func_ast)
        
        # Loop structure analysis with nesting
        channels['loop_structures'] = self._analyze_loop_structures_enhanced(func_ast)
        
        # Type pattern analysis
        channels['type_patterns'] = self._analyze_type_patterns(func_ast)
        
        # Enhanced name tokenization
        channels['name_tokens'] = self._tokenize_name_enhanced(func_ast.name)
        
        # Combine into weighted hypervector
        result = self._create_weighted_hypervector_enhanced(channels)
        
        # Cache result
        if cache_key:
            self._pattern_cache[cache_key] = result.copy()
        
        return result
    
    def _analyze_data_flow_enhanced(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Enhanced data flow analysis with deeper semantic understanding."""
        channel_dim = self.vector_dim // len(self.channel_weights)
        
        # Use hypervector space for compositional encoding
        data_flow_vectors = []
        
        # Analyze variable lifecycle patterns
        var_lifecycle = self._analyze_variable_lifecycle(func_ast)
        for pattern_name, pattern_data in var_lifecycle.items():
            pattern_hv = self.hv_space.bind(
                self.hv_space.token(f"dataflow_{pattern_name}"),
                self.hv_space.token(str(pattern_data))
            )
            data_flow_vectors.append(pattern_hv)
        
        # Analyze parameter usage patterns
        param_patterns = self._analyze_parameter_usage(func_ast)
        for pattern in param_patterns:
            pattern_hv = self.hv_space.token(f"param_{pattern}")
            data_flow_vectors.append(pattern_hv)
        
        # Analyze data transformation chains
        transform_chains = self._analyze_transformation_chains(func_ast)
        for chain in transform_chains:
            chain_hv = self.hv_space.bundle([
                self.hv_space.token(f"transform_{op}") for op in chain
            ])
            data_flow_vectors.append(chain_hv)
        
        # Bundle all data flow vectors
        if data_flow_vectors:
            bundled = self.hv_space.bundle(data_flow_vectors)
            # Resize to channel dimension
            return self._resize_vector(bundled, channel_dim)
        else:
            return np.zeros(channel_dim, dtype=np.float32)
    
    def _analyze_variable_lifecycle(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze how variables are created, used, and transformed."""
        var_states = defaultdict(lambda: {'assigned': 0, 'used': 0, 'modified': 0})
        
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_states[target.id]['assigned'] += 1
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                var_states[node.target.id]['modified'] += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                var_states[node.id]['used'] += 1
        
        return {
            'total_variables': len(var_states),
            'reassigned_vars': sum(1 for v in var_states.values() if v['assigned'] > 1),
            'unused_vars': sum(1 for v in var_states.values() if v['used'] == 0),
            'write_only_vars': sum(1 for v in var_states.values() if v['assigned'] > 0 and v['used'] == 0),
            'modified_vars': sum(1 for v in var_states.values() if v['modified'] > 0),
        }
    
    def _analyze_parameter_usage(self, func_ast: ast.FunctionDef) -> List[str]:
        """Analyze how function parameters are used."""
        param_names = {arg.arg for arg in func_ast.args.args}
        param_usage = []
        
        # Check if parameters are modified
        for node in ast.walk(func_ast):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                if isinstance(node.target if hasattr(node, 'target') else node.targets[0], ast.Name):
                    name = node.target.id if hasattr(node, 'target') else node.targets[0].id
                    if name in param_names:
                        param_usage.append('param_modified')
        
        # Check if parameters are used in return
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Name) and node.value.id in param_names:
                    param_usage.append('param_returned')
                elif any(isinstance(n, ast.Name) and n.id in param_names 
                        for n in ast.walk(node.value)):
                    param_usage.append('param_in_return')
        
        return param_usage
    
    def _analyze_transformation_chains(self, func_ast: ast.FunctionDef) -> List[List[str]]:
        """Analyze chains of data transformations."""
        chains = []
        
        # Look for method call chains
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                chain = self._extract_method_chain(node)
                if len(chain) > 1:
                    chains.append(chain)
        
        return chains
    
    def _extract_method_chain(self, node: ast.Call) -> List[str]:
        """Extract method call chain from an AST node."""
        chain = []
        current = node
        
        while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            chain.append(current.func.attr)
            current = current.func.value
        
        return list(reversed(chain))
    
    def _analyze_control_flow_complexity(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Analyze control flow complexity patterns."""
        channel_dim = self.vector_dim // len(self.channel_weights)
        
        complexity_vectors = []
        
        # Analyze branching patterns
        branching = self._analyze_branching_patterns(func_ast)
        for pattern, count in branching.items():
            pattern_hv = self.hv_space.bind(
                self.hv_space.token(f"branch_{pattern}"),
                self.hv_space.token(str(min(count, 10)))  # Cap for stability
            )
            complexity_vectors.append(pattern_hv)
        
        # Analyze cyclomatic complexity approximation
        cyclomatic = self._compute_cyclomatic_complexity(func_ast)
        complexity_hv = self.hv_space.bind(
            self.hv_space.token("cyclomatic"),
            self.hv_space.token(str(min(cyclomatic, 20)))
        )
        complexity_vectors.append(complexity_hv)
        
        if complexity_vectors:
            bundled = self.hv_space.bundle(complexity_vectors)
            return self._resize_vector(bundled, channel_dim)
        else:
            return np.zeros(channel_dim, dtype=np.float32)
    
    def _analyze_branching_patterns(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze different types of branching in the function."""
        patterns = {
            'if_statements': 0,
            'nested_if': 0,
            'elif_chains': 0,
            'ternary_ops': 0,
            'match_statements': 0
        }
        
        def count_nested_depth(node, if_depth=0):
            if isinstance(node, ast.If):
                patterns['if_statements'] += 1
                if if_depth > 0:
                    patterns['nested_if'] += 1
                if node.orelse and isinstance(node.orelse[0], ast.If):
                    patterns['elif_chains'] += 1
                
                for child in ast.iter_child_nodes(node):
                    count_nested_depth(child, if_depth + 1)
            elif isinstance(node, ast.IfExp):
                patterns['ternary_ops'] += 1
                for child in ast.iter_child_nodes(node):
                    count_nested_depth(child, if_depth)
            elif isinstance(node, ast.Match):  # Python 3.10+
                patterns['match_statements'] += 1
                for child in ast.iter_child_nodes(node):
                    count_nested_depth(child, if_depth)
            else:
                for child in ast.iter_child_nodes(node):
                    count_nested_depth(child, if_depth)
        
        count_nested_depth(func_ast)
        return patterns
    
    def _compute_cyclomatic_complexity(self, func_ast: ast.FunctionDef) -> int:
        """Compute approximation of cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_ast):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.Match):
                complexity += len(node.cases) if hasattr(node, 'cases') else 1
        
        return complexity
    
    def _analyze_return_patterns_enhanced(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Enhanced return pattern analysis."""
        channel_dim = self.vector_dim // len(self.channel_weights)
        
        return_vectors = []
        returns = [node for node in ast.walk(func_ast) if isinstance(node, ast.Return)]
        
        # Analyze return complexity
        return_complexity = self._categorize_return_complexity(returns)
        for complexity, count in return_complexity.items():
            if count > 0:
                return_hv = self.hv_space.bind(
                    self.hv_space.token(f"return_{complexity}"),
                    self.hv_space.token(str(min(count, 10)))
                )
                return_vectors.append(return_hv)
        
        # Analyze early vs late returns
        if returns and func_ast.body:
            total_lines = len(func_ast.body)
            early_returns = sum(1 for r in returns if r.lineno - func_ast.lineno < total_lines * 0.3)
            if early_returns > 0:
                early_hv = self.hv_space.token(f"early_returns_{min(early_returns, 5)}")
                return_vectors.append(early_hv)
        
        if return_vectors:
            bundled = self.hv_space.bundle(return_vectors)
            return self._resize_vector(bundled, channel_dim)
        else:
            return np.zeros(channel_dim, dtype=np.float32)
    
    def _categorize_return_complexity(self, returns: List[ast.Return]) -> Dict[str, int]:
        """Categorize returns by complexity."""
        categories = {
            'none_return': 0,
            'simple_value': 0,
            'complex_expression': 0,
            'collection_return': 0,
            'conditional_return': 0
        }
        
        for ret in returns:
            if ret.value is None:
                categories['none_return'] += 1
            elif isinstance(ret.value, (ast.Name, ast.Constant)):
                categories['simple_value'] += 1
            elif isinstance(ret.value, (ast.List, ast.Dict, ast.Tuple, ast.Set)):
                categories['collection_return'] += 1
            elif isinstance(ret.value, ast.IfExp):
                categories['conditional_return'] += 1
            else:
                categories['complex_expression'] += 1
        
        return categories
    
    def _analyze_error_handling_enhanced(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Enhanced error handling pattern analysis."""
        channel_dim = self.vector_dim // len(self.channel_weights)
        
        error_vectors = []
        
        # Analyze try-catch patterns
        try_patterns = self._analyze_try_patterns(func_ast)
        for pattern, data in try_patterns.items():
            if data > 0:
                error_hv = self.hv_space.bind(
                    self.hv_space.token(f"error_{pattern}"),
                    self.hv_space.token(str(min(data, 10)))
                )
                error_vectors.append(error_hv)
        
        # Analyze assertion patterns
        assertions = len([node for node in ast.walk(func_ast) if isinstance(node, ast.Assert)])
        if assertions > 0:
            assert_hv = self.hv_space.token(f"assertions_{min(assertions, 5)}")
            error_vectors.append(assert_hv)
        
        if error_vectors:
            bundled = self.hv_space.bundle(error_vectors)
            return self._resize_vector(bundled, channel_dim)
        else:
            return np.zeros(channel_dim, dtype=np.float32)
    
    def _analyze_try_patterns(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze try-except-finally patterns."""
        patterns = {
            'try_blocks': 0,
            'specific_catches': 0,
            'generic_catches': 0,
            'finally_blocks': 0,
            'nested_try': 0,
            'reraise': 0
        }
        
        def analyze_try_depth(node, try_depth=0):
            if isinstance(node, ast.Try):
                patterns['try_blocks'] += 1
                if try_depth > 0:
                    patterns['nested_try'] += 1
                
                if node.finalbody:
                    patterns['finally_blocks'] += 1
                
                for handler in node.handlers:
                    if handler.type is None:
                        patterns['generic_catches'] += 1
                    else:
                        patterns['specific_catches'] += 1
                    
                    # Check for re-raise
                    for stmt in handler.body:
                        if isinstance(stmt, ast.Raise) and stmt.exc is None:
                            patterns['reraise'] += 1
                
                for child in ast.iter_child_nodes(node):
                    analyze_try_depth(child, try_depth + 1)
            else:
                for child in ast.iter_child_nodes(node):
                    analyze_try_depth(child, try_depth)
        
        analyze_try_depth(func_ast)
        return patterns
    
    def _analyze_loop_structures_enhanced(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Enhanced loop structure analysis."""
        channel_dim = self.vector_dim // len(self.channel_weights)
        
        loop_vectors = []
        
        # Analyze loop patterns
        loop_patterns = self._analyze_loop_patterns_detailed(func_ast)
        for pattern, count in loop_patterns.items():
            if count > 0:
                loop_hv = self.hv_space.bind(
                    self.hv_space.token(f"loop_{pattern}"),
                    self.hv_space.token(str(min(count, 10)))
                )
                loop_vectors.append(loop_hv)
        
        if loop_vectors:
            bundled = self.hv_space.bundle(loop_vectors)
            return self._resize_vector(bundled, channel_dim)
        else:
            return np.zeros(channel_dim, dtype=np.float32)
    
    def _analyze_loop_patterns_detailed(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Detailed loop pattern analysis."""
        patterns = {
            'for_loops': 0,
            'while_loops': 0,
            'async_for': 0,
            'nested_loops': 0,
            'comprehensions': 0,
            'break_statements': 0,
            'continue_statements': 0,
            'else_clauses': 0
        }
        
        def analyze_loop_depth(node, loop_depth=0):
            if isinstance(node, (ast.For, ast.AsyncFor)):
                if isinstance(node, ast.AsyncFor):
                    patterns['async_for'] += 1
                else:
                    patterns['for_loops'] += 1
                
                if loop_depth > 0:
                    patterns['nested_loops'] += 1
                
                if node.orelse:
                    patterns['else_clauses'] += 1
                
                for child in ast.iter_child_nodes(node):
                    analyze_loop_depth(child, loop_depth + 1)
            
            elif isinstance(node, ast.While):
                patterns['while_loops'] += 1
                if loop_depth > 0:
                    patterns['nested_loops'] += 1
                
                if node.orelse:
                    patterns['else_clauses'] += 1
                
                for child in ast.iter_child_nodes(node):
                    analyze_loop_depth(child, loop_depth + 1)
            
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                patterns['comprehensions'] += 1
                for child in ast.iter_child_nodes(node):
                    analyze_loop_depth(child, loop_depth)
            
            elif isinstance(node, ast.Break):
                patterns['break_statements'] += 1
            
            elif isinstance(node, ast.Continue):
                patterns['continue_statements'] += 1
            
            else:
                for child in ast.iter_child_nodes(node):
                    analyze_loop_depth(child, loop_depth)
        
        analyze_loop_depth(func_ast)
        return patterns
    
    def _analyze_type_patterns(self, func_ast: ast.FunctionDef) -> np.ndarray:
        """Analyze type usage patterns in the function."""
        channel_dim = self.vector_dim // len(self.channel_weights)
        
        type_vectors = []
        
        # Analyze type annotations
        type_annotations = self._extract_type_annotations(func_ast)
        for annotation in type_annotations:
            type_hv = self.hv_space.token(f"type_{annotation}")
            type_vectors.append(type_hv)
        
        # Analyze type checks (isinstance, type())
        type_checks = self._analyze_type_checks(func_ast)
        for check_type in type_checks:
            check_hv = self.hv_space.token(f"typecheck_{check_type}")
            type_vectors.append(check_hv)
        
        if type_vectors:
            bundled = self.hv_space.bundle(type_vectors)
            return self._resize_vector(bundled, channel_dim)
        else:
            return np.zeros(channel_dim, dtype=np.float32)
    
    def _extract_type_annotations(self, func_ast: ast.FunctionDef) -> List[str]:
        """Extract type annotations from function."""
        annotations = []
        
        # Return type annotation
        if func_ast.returns:
            annotations.append(ast.unparse(func_ast.returns) if hasattr(ast, 'unparse') else 'return_annotated')
        
        # Parameter annotations
        for arg in func_ast.args.args:
            if arg.annotation:
                annotations.append(ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else 'param_annotated')
        
        return annotations
    
    def _analyze_type_checks(self, func_ast: ast.FunctionDef) -> List[str]:
        """Analyze explicit type checking in the function."""
        type_checks = []
        
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == 'isinstance':
                        type_checks.append('isinstance')
                    elif node.func.id == 'type':
                        type_checks.append('type_check')
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'isinstance':
                        type_checks.append('method_isinstance')
        
        return type_checks
    
    def _tokenize_name_enhanced(self, name: str) -> np.ndarray:
        """Enhanced name tokenization using hypervector space."""
        channel_dim = self.vector_dim // len(self.channel_weights)
        
        # Extract semantic tokens from name
        tokens = self._extract_semantic_tokens(name)
        
        if not tokens:
            return np.zeros(channel_dim, dtype=np.float32)
        
        # Create hypervector for each token
        token_vectors = [self.hv_space.token(f"name_{token}") for token in tokens]
        
        # Bundle tokens
        bundled = self.hv_space.bundle(token_vectors)
        
        return self._resize_vector(bundled, channel_dim)
    
    @lru_cache(maxsize=1000)
    def _extract_semantic_tokens(self, name: str) -> Tuple[str, ...]:
        """Extract semantic tokens from function name with caching."""
        # Split camelCase and snake_case
        tokens = []
        
        # Handle snake_case
        if '_' in name:
            parts = name.split('_')
            tokens.extend(part.lower() for part in parts if part)
        else:
            # Handle camelCase
            import re
            parts = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+', name)
            tokens.extend(part.lower() for part in parts)
        
        # Add semantic categories
        semantic_tokens = []
        for token in tokens:
            semantic_tokens.append(token)
            
            # Add semantic category
            if token in {'get', 'fetch', 'retrieve', 'find', 'search', 'query'}:
                semantic_tokens.append('_retrieval_verb')
            elif token in {'set', 'update', 'modify', 'change', 'edit', 'alter'}:
                semantic_tokens.append('_modification_verb')
            elif token in {'create', 'make', 'build', 'generate', 'produce'}:
                semantic_tokens.append('_creation_verb')
            elif token in {'delete', 'remove', 'destroy', 'clear', 'clean'}:
                semantic_tokens.append('_deletion_verb')
            elif token in {'data', 'info', 'information', 'content', 'value'}:
                semantic_tokens.append('_data_noun')
            elif token in {'user', 'client', 'customer', 'person', 'account'}:
                semantic_tokens.append('_entity_noun')
        
        return tuple(semantic_tokens)
    
    def _create_weighted_hypervector_enhanced(self, channels: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine channels into weighted hypervector using hypervector operations."""
        channel_vectors = []
        
        for channel_name, channel_data in channels.items():
            weight = self.channel_weights.get(channel_name, 1.0)
            
            # Create channel role vector
            role_vector = self.hv_space.role(f"CHANNEL_{channel_name.upper()}")
            
            # Bind channel data with role
            bound_vector = self.hv_space.bind(role_vector, channel_data)
            
            # Apply weight by repeated bundling (approximation of scalar multiplication)
            weighted_vectors = [bound_vector] * int(weight * 10)  # Scale weight to integer
            if weighted_vectors:
                weighted_channel = self.hv_space.bundle(weighted_vectors)
                channel_vectors.append(weighted_channel)
        
        # Bundle all weighted channels
        if channel_vectors:
            combined = self.hv_space.bundle(channel_vectors)
            return combined.astype(np.float32)
        else:
            return np.zeros(self.vector_dim, dtype=np.float32)
    
    def _resize_vector(self, vector: np.ndarray, target_size: int) -> np.ndarray:
        """Resize vector to target size using interpolation or truncation."""
        if len(vector) == target_size:
            return vector.astype(np.float32)
        elif len(vector) > target_size:
            # Truncate
            return vector[:target_size].astype(np.float32)
        else:
            # Pad with zeros
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(vector)] = vector
            return padded
    
    def find_semantic_duplicates(self, functions: List[Tuple[str, ast.FunctionDef]], 
                               threshold: float = 0.85) -> List[Issue]:
        """Find semantically similar functions across the codebase."""
        issues = []
        
        # Encode all functions with caching
        encoded_functions = []
        for filepath, func_ast in functions:
            try:
                cache_key = f"{func_ast.name}@{filepath}:{func_ast.lineno}"
                vector = self.encode_function_multimodal(func_ast, cache_key)
                encoded_functions.append((filepath, func_ast, vector))
            except Exception as e:
                self.logger.warning(f"Failed to encode {filepath}:{func_ast.name}: {e}")
                continue
        
        # Use efficient similarity search
        similar_pairs = self._find_similar_pairs_efficient(encoded_functions, threshold)
        
        for (file1, func1, vec1), (file2, func2, vec2), similarity in similar_pairs:
            if self._are_likely_duplicates(func1, func2, similarity):
                issues.append(Issue(
                    kind="semantic_duplicate_enhanced",
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
    
    def _find_similar_pairs_efficient(self, encoded_functions: List[Tuple[str, ast.FunctionDef, np.ndarray]], 
                                    threshold: float) -> List[Tuple[Tuple[str, ast.FunctionDef, np.ndarray], 
                                                                  Tuple[str, ast.FunctionDef, np.ndarray], float]]:
        """Efficient similarity search using optimized distance computation."""
        similar_pairs = []
        
        # Convert to matrix for vectorized computation
        if len(encoded_functions) < 2:
            return similar_pairs
        
        vectors = np.array([vec for _, _, vec in encoded_functions])
        
        # Compute all pairwise similarities efficiently
        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vectors = vectors / norms
        
        # Compute similarity matrix
        similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
        
        # Find pairs above threshold
        n = len(encoded_functions)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    similar_pairs.append((encoded_functions[i], encoded_functions[j], similarity))
        
        return similar_pairs
    
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
        """Analyze which channels contribute most to similarity using approximation."""
        # Since we're using bundled hypervectors, we can't perfectly decompose
        # But we can approximate by comparing with channel prototypes
        contributions = {}
        
        for channel_name in self.channel_weights.keys():
            # Create prototype vector for this channel
            prototype = self.hv_space.role(f"CHANNEL_{channel_name.upper()}")
            
            # Measure similarity to prototype for both vectors
            sim1 = self.hv_space.similarity(vec1, prototype)
            sim2 = self.hv_space.similarity(vec2, prototype)
            
            # Contribution is the product of similarities
            contributions[channel_name] = sim1 * sim2
        
        return contributions
    
    def track_semantic_drift(self, commit_id: str, functions: List[Tuple[str, ast.FunctionDef]]) -> Dict[str, Any]:
        """Track semantic drift over commits to detect confusion vs refactoring."""
        current_snapshot = {}
        
        # Encode all functions in current commit
        for filepath, func_ast in functions:
            func_key = f"{func_ast.name}@{filepath}"
            try:
                vector = self.encode_function_multimodal(func_ast, cache_key=func_key)
                current_snapshot[func_key] = vector
            except Exception as e:
                self.logger.warning(f"Failed to encode {func_key}: {e}")
                continue
        
        drift_analysis = {
            'commit_id': commit_id,
            'timestamp': hash(commit_id),  # Simplified timestamp
            'total_functions': len(current_snapshot),
            'semantic_changes': {},
            'drift_patterns': {},
            'convergence_status': 'unknown'
        }
        
        # Compare with previous snapshot if available
        if self.semantic_snapshots:
            drift_analysis['semantic_changes'] = self._analyze_semantic_changes(
                self.semantic_snapshots, current_snapshot
            )
            drift_analysis['drift_patterns'] = self._detect_drift_patterns(
                self.semantic_snapshots, current_snapshot
            )
            drift_analysis['convergence_status'] = self._assess_convergence(
                self.semantic_snapshots, current_snapshot
            )
        
        # Update snapshots
        self.semantic_snapshots = current_snapshot
        self.commit_history.append(drift_analysis)
        
        # Keep only recent history to prevent memory bloat
        if len(self.commit_history) > 50:
            self.commit_history = self.commit_history[-50:]
        
        return drift_analysis
    
    def _analyze_semantic_changes(self, prev_snapshot: Dict[str, np.ndarray], 
                                 curr_snapshot: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze semantic changes between snapshots."""
        changes = {
            'modified_functions': [],
            'new_functions': [],
            'removed_functions': [],
            'similarity_changes': {}
        }
        
        prev_keys = set(prev_snapshot.keys())
        curr_keys = set(curr_snapshot.keys())
        
        changes['new_functions'] = list(curr_keys - prev_keys)
        changes['removed_functions'] = list(prev_keys - curr_keys)
        
        # Analyze modifications
        common_keys = prev_keys & curr_keys
        for key in common_keys:
            similarity = self.hv_space.similarity(prev_snapshot[key], curr_snapshot[key])
            if similarity < 0.95:  # Threshold for considering a change
                changes['modified_functions'].append(key)
                changes['similarity_changes'][key] = similarity
        
        return changes
    
    def _detect_drift_patterns(self, prev_snapshot: Dict[str, np.ndarray], 
                              curr_snapshot: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect patterns in semantic drift."""
        patterns = {
            'systematic_changes': 0,
            'random_changes': 0,
            'convergent_changes': 0,
            'divergent_changes': 0,
            'average_drift': 0.0
        }
        
        common_keys = set(prev_snapshot.keys()) & set(curr_snapshot.keys())
        if not common_keys:
            return patterns
        
        similarities = []
        for key in common_keys:
            similarity = self.hv_space.similarity(prev_snapshot[key], curr_snapshot[key])
            similarities.append(similarity)
        
        if similarities:
            patterns['average_drift'] = 1.0 - (sum(similarities) / len(similarities))
            
            # Detect systematic vs random changes
            variance = np.var(similarities)
            if variance < 0.01:  # Low variance suggests systematic changes
                patterns['systematic_changes'] = len([s for s in similarities if s < 0.95])
            else:
                patterns['random_changes'] = len([s for s in similarities if s < 0.95])
            
            # Detect convergence vs divergence by analyzing pairwise similarities
            pairwise_similarities = []
            functions = list(common_keys)
            for i in range(len(functions)):
                for j in range(i + 1, len(functions)):
                    sim_prev = self.hv_space.similarity(
                        prev_snapshot[functions[i]], prev_snapshot[functions[j]]
                    )
                    sim_curr = self.hv_space.similarity(
                        curr_snapshot[functions[i]], curr_snapshot[functions[j]]
                    )
                    pairwise_similarities.append(sim_curr - sim_prev)
            
            if pairwise_similarities:
                avg_change = sum(pairwise_similarities) / len(pairwise_similarities)
                if avg_change > 0.01:
                    patterns['convergent_changes'] = len([c for c in pairwise_similarities if c > 0])
                elif avg_change < -0.01:
                    patterns['divergent_changes'] = len([c for c in pairwise_similarities if c < 0])
        
        return patterns
    
    def _assess_convergence(self, prev_snapshot: Dict[str, np.ndarray], 
                           curr_snapshot: Dict[str, np.ndarray]) -> str:
        """Assess whether changes represent convergence, divergence, or refactoring."""
        common_keys = set(prev_snapshot.keys()) & set(curr_snapshot.keys())
        if len(common_keys) < 2:
            return 'insufficient_data'
        
        # Calculate inter-function similarities before and after
        functions = list(common_keys)
        prev_similarities = []
        curr_similarities = []
        
        for i in range(len(functions)):
            for j in range(i + 1, len(functions)):
                prev_sim = self.hv_space.similarity(
                    prev_snapshot[functions[i]], prev_snapshot[functions[j]]
                )
                curr_sim = self.hv_space.similarity(
                    curr_snapshot[functions[i]], curr_snapshot[functions[j]]
                )
                prev_similarities.append(prev_sim)
                curr_similarities.append(curr_sim)
        
        if not prev_similarities:
            return 'insufficient_data'
        
        avg_prev = sum(prev_similarities) / len(prev_similarities)
        avg_curr = sum(curr_similarities) / len(curr_similarities)
        
        # Assess change pattern
        change = avg_curr - avg_prev
        
        if change > 0.05:
            return 'convergence'  # Functions becoming more similar
        elif change < -0.05:
            return 'divergence'   # Functions becoming less similar
        elif abs(change) < 0.02:
            # Check if individual functions changed significantly
            individual_changes = []
            for key in common_keys:
                individual_sim = self.hv_space.similarity(prev_snapshot[key], curr_snapshot[key])
                individual_changes.append(1.0 - individual_sim)
            
            avg_individual_change = sum(individual_changes) / len(individual_changes)
            if avg_individual_change > 0.1:
                return 'refactoring'  # Individual functions changed but relationships preserved
            else:
                return 'stable'
        else:
            return 'mixed_changes'
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of semantic drift over time."""
        if not self.commit_history:
            return {'status': 'no_history'}
        
        recent_commits = self.commit_history[-10:]  # Last 10 commits
        
        summary = {
            'total_commits_tracked': len(self.commit_history),
            'recent_convergence_pattern': [c['convergence_status'] for c in recent_commits],
            'average_drift_rate': 0.0,
            'stability_trend': 'unknown',
            'concerning_patterns': []
        }
        
        # Calculate average drift rate
        drift_rates = [c['drift_patterns'].get('average_drift', 0.0) for c in recent_commits]
        if drift_rates:
            summary['average_drift_rate'] = sum(drift_rates) / len(drift_rates)
        
        # Assess stability trend
        convergence_counts = {
            'convergence': sum(1 for c in recent_commits if c['convergence_status'] == 'convergence'),
            'divergence': sum(1 for c in recent_commits if c['convergence_status'] == 'divergence'),
            'refactoring': sum(1 for c in recent_commits if c['convergence_status'] == 'refactoring'),
            'stable': sum(1 for c in recent_commits if c['convergence_status'] == 'stable')
        }
        
        max_pattern = max(convergence_counts, key=convergence_counts.get)
        summary['stability_trend'] = max_pattern
        
        # Identify concerning patterns
        if convergence_counts['divergence'] > len(recent_commits) * 0.6:
            summary['concerning_patterns'].append('high_divergence_rate')
        
        if summary['average_drift_rate'] > 0.3:
            summary['concerning_patterns'].append('high_drift_rate')
        
        # Check for oscillation (alternating convergence/divergence)
        statuses = [c['convergence_status'] for c in recent_commits]
        oscillations = sum(1 for i in range(1, len(statuses)) 
                          if statuses[i] != statuses[i-1] and 
                          {statuses[i], statuses[i-1]} == {'convergence', 'divergence'})
        
        if oscillations > len(recent_commits) * 0.4:
            summary['concerning_patterns'].append('semantic_oscillation')
        
        return summary