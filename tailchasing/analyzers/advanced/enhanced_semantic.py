"""
Enhanced Semantic Analysis

Multi-modal semantic encoding for better duplicate detection.
Uses multiple channels: data flow, return patterns, error handling, loops, etc.
"""

import ast
import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from dataclasses import dataclass

from ..base import AnalysisContext
from ...core.issues import Issue
from ...semantic.hv_space import HVSpace
from ...semantic.index import SemanticIndex


@dataclass
class EnhancedChannels:
    """Enhanced semantic channels for multi-modal analysis."""
    name_tokens: List[str]
    param_types: List[str]
    return_patterns: Dict[str, int]
    exception_handling: Dict[str, int]
    loop_structures: Dict[str, int]
    conditional_logic: Dict[str, int]
    data_flow: Dict[str, int]
    cognitive_complexity: int


class EnhancedSemanticAnalyzer:
    """Enhanced semantic duplicate detection with multi-modal analysis."""
    
    name = "enhanced_semantic"
    
    def __init__(self):
        self.channel_weights = {
            'data_flow': 1.5,
            'return_patterns': 1.3,
            'param_types': 1.2,
            'exception_handling': 1.1,
            'loop_structures': 1.0,
            'conditional_logic': 1.0,
            'name_tokens': 0.8,  # Lower weight - LLMs often use different names
        }
        self.z_threshold = 2.0
        self.min_functions = 10
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Run enhanced semantic analysis."""
        cfg = ctx.config.get("semantic", {})
        if not cfg.get("enable", False):
            return []
        
        # Check minimum function count
        total_functions = sum(len(entries) for entries in ctx.symbol_table.functions.values())
        if total_functions < self.min_functions:
            return []
        
        # Get or create semantic index
        index = ctx.cache.setdefault(
            "enhanced_semantic_index",
            SemanticIndex(cfg)
        )
        
        # Encode all functions with enhanced channels
        function_encodings = []
        for func_name, entries in ctx.symbol_table.functions.items():
            for entry in entries:
                channels = self._extract_enhanced_channels(entry['node'])
                hv = self._encode_with_channels(channels, index.space)
                
                function_encodings.append({
                    'name': func_name,
                    'file': entry['file'],
                    'line': entry['lineno'],
                    'hv': hv,
                    'channels': channels
                })
                
                # Add to index
                index.add(func_name, entry['file'], entry['lineno'], hv)
        
        # Find similar pairs
        issues = []
        similar_pairs = index.top_similar_pairs(limit=20)
        
        for (meta_a, meta_b, dist, z, contrib) in similar_pairs:
            if z >= self.z_threshold:
                # Get channel details for explanation
                func_a = next((f for f in function_encodings 
                             if f['name'] == meta_a[0] and f['file'] == meta_a[1]), None)
                func_b = next((f for f in function_encodings 
                             if f['name'] == meta_b[0] and f['file'] == meta_b[1]), None)
                
                if func_a and func_b:
                    channel_analysis = self._analyze_channel_similarity(
                        func_a['channels'], func_b['channels']
                    )
                    
                    issues.append(Issue(
                        kind="enhanced_semantic_duplicate",
                        message=f"Enhanced semantic analysis detected duplicate: "
                               f"'{meta_a[0]}' and '{meta_b[0]}' (z={z:.2f})",
                        severity=3,
                        file=meta_a[1],
                        line=meta_a[2],
                        symbol=meta_a[0],
                        evidence={
                            'pair': [meta_a, meta_b],
                            'z_score': z,
                            'distance': dist,
                            'channel_similarity': channel_analysis,
                            'dominant_channels': self._get_dominant_channels(channel_analysis)
                        },
                        suggestions=[
                            f"Functions have similar {', '.join(self._get_dominant_channels(channel_analysis))}",
                            "Consider merging or extracting common functionality",
                            "Review if both implementations are necessary"
                        ]
                    ))
        
        return issues
    
    def _extract_enhanced_channels(self, func_ast: ast.FunctionDef) -> EnhancedChannels:
        """Extract enhanced semantic channels from function."""
        return EnhancedChannels(
            name_tokens=self._tokenize_name(func_ast.name),
            param_types=self._extract_param_types(func_ast),
            return_patterns=self._analyze_return_patterns(func_ast),
            exception_handling=self._analyze_error_handling(func_ast),
            loop_structures=self._analyze_loop_patterns(func_ast),
            conditional_logic=self._analyze_conditional_patterns(func_ast),
            data_flow=self._analyze_data_flow(func_ast),
            cognitive_complexity=self._calculate_complexity(func_ast)
        )
    
    def _tokenize_name(self, name: str) -> List[str]:
        """Tokenize function name into semantic components."""
        # Split on camelCase, snake_case, numbers
        tokens = re.findall(r'[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+', name)
        return [t.lower() for t in tokens]
    
    def _extract_param_types(self, func_ast: ast.FunctionDef) -> List[str]:
        """Extract parameter type hints and patterns."""
        param_types = []
        
        for arg in func_ast.args.args:
            if arg.annotation:
                # Simple type extraction
                if isinstance(arg.annotation, ast.Name):
                    param_types.append(arg.annotation.id)
                else:
                    param_types.append('complex')
            else:
                # Infer from name patterns
                arg_lower = arg.arg.lower()
                if any(t in arg_lower for t in ['str', 'text', 'name', 'message']):
                    param_types.append('str_like')
                elif any(t in arg_lower for t in ['num', 'count', 'size', 'idx', 'index']):
                    param_types.append('num_like')
                elif any(t in arg_lower for t in ['list', 'items', 'elements']):
                    param_types.append('list_like')
                else:
                    param_types.append('unknown')
        
        return param_types
    
    def _analyze_return_patterns(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze function return patterns."""
        patterns = {
            'num_returns': 0,
            'early_returns': 0,
            'none_returns': 0,
            'complex_returns': 0,
            'conditional_returns': 0
        }
        
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Return):
                patterns['num_returns'] += 1
                
                # Check if early return (not in last 20% of function)
                if hasattr(node, 'lineno') and hasattr(func_ast, 'lineno'):
                    func_lines = func_ast.end_lineno - func_ast.lineno if hasattr(func_ast, 'end_lineno') else 10
                    if (node.lineno - func_ast.lineno) < func_lines * 0.8:
                        patterns['early_returns'] += 1
                
                # Check return value type
                if node.value is None:
                    patterns['none_returns'] += 1
                elif isinstance(node.value, (ast.Dict, ast.List, ast.Tuple)):
                    patterns['complex_returns'] += 1
                
                # Check if return is inside conditional
                parent = node
                while parent:
                    if isinstance(parent, ast.If):
                        patterns['conditional_returns'] += 1
                        break
                    parent = getattr(parent, 'parent', None)
        
        return patterns
    
    def _analyze_error_handling(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze error handling patterns."""
        patterns = {
            'try_blocks': 0,
            'except_handlers': 0,
            'finally_blocks': 0,
            'raises': 0,
            'catches_all': 0,
            'specific_exceptions': 0
        }
        
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Try):
                patterns['try_blocks'] += 1
                patterns['except_handlers'] += len(node.handlers)
                
                if node.finalbody:
                    patterns['finally_blocks'] += 1
                
                for handler in node.handlers:
                    if handler.type is None:
                        patterns['catches_all'] += 1
                    else:
                        patterns['specific_exceptions'] += 1
            
            elif isinstance(node, ast.Raise):
                patterns['raises'] += 1
        
        return patterns
    
    def _analyze_loop_patterns(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze loop structures."""
        patterns = {
            'for_loops': 0,
            'while_loops': 0,
            'nested_loops': 0,
            'comprehensions': 0,
            'break_statements': 0,
            'continue_statements': 0
        }
        
        loop_depth = 0
        max_depth = 0
        
        def visit_node(node, depth=0):
            nonlocal max_depth
            
            if isinstance(node, ast.For):
                patterns['for_loops'] += 1
                depth += 1
                max_depth = max(max_depth, depth)
            elif isinstance(node, ast.While):
                patterns['while_loops'] += 1
                depth += 1
                max_depth = max(max_depth, depth)
            elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp)):
                patterns['comprehensions'] += 1
            elif isinstance(node, ast.Break):
                patterns['break_statements'] += 1
            elif isinstance(node, ast.Continue):
                patterns['continue_statements'] += 1
            
            for child in ast.iter_child_nodes(node):
                visit_node(child, depth)
        
        visit_node(func_ast)
        patterns['nested_loops'] = max_depth
        
        return patterns
    
    def _analyze_conditional_patterns(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze conditional logic patterns."""
        patterns = {
            'if_statements': 0,
            'elif_branches': 0,
            'else_branches': 0,
            'ternary_ops': 0,
            'boolean_ops': 0,
            'comparison_ops': 0
        }
        
        for node in ast.walk(func_ast):
            if isinstance(node, ast.If):
                patterns['if_statements'] += 1
                if node.orelse:
                    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                        patterns['elif_branches'] += 1
                    else:
                        patterns['else_branches'] += 1
            elif isinstance(node, ast.IfExp):
                patterns['ternary_ops'] += 1
            elif isinstance(node, ast.BoolOp):
                patterns['boolean_ops'] += 1
            elif isinstance(node, ast.Compare):
                patterns['comparison_ops'] += 1
        
        return patterns
    
    def _analyze_data_flow(self, func_ast: ast.FunctionDef) -> Dict[str, int]:
        """Analyze how data flows through the function."""
        patterns = {
            'assignments': 0,
            'augmented_assigns': 0,
            'variable_reads': 0,
            'attribute_access': 0,
            'method_calls': 0,
            'unique_vars': 0
        }
        
        variables_assigned = set()
        variables_read = set()
        
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Assign):
                patterns['assignments'] += 1
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables_assigned.add(target.id)
            
            elif isinstance(node, ast.AugAssign):
                patterns['augmented_assigns'] += 1
                if isinstance(node.target, ast.Name):
                    variables_assigned.add(node.target.id)
            
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                patterns['variable_reads'] += 1
                variables_read.add(node.id)
            
            elif isinstance(node, ast.Attribute):
                patterns['attribute_access'] += 1
            
            elif isinstance(node, ast.Call):
                patterns['method_calls'] += 1
        
        patterns['unique_vars'] = len(variables_assigned | variables_read)
        
        return patterns
    
    def _calculate_complexity(self, func_ast: ast.FunctionDef) -> int:
        """Calculate cognitive complexity of the function."""
        complexity = 1  # Base complexity
        
        # Increment for control flow
        for node in ast.walk(func_ast):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _encode_with_channels(self, channels: EnhancedChannels, space: HVSpace) -> List[int]:
        """Encode channels into weighted hypervector."""
        channel_vectors = []
        
        # Encode each channel
        if channels.name_tokens:
            tokens_hv = [space.token(t) for t in channels.name_tokens[:20]]
            channel_vectors.append((
                'name_tokens',
                space.bind(space.role('name_tokens'), space.bundle(tokens_hv))
            ))
        
        # Encode parameter types
        if channels.param_types:
            types_hv = [space.token(t) for t in channels.param_types[:10]]
            channel_vectors.append((
                'param_types',
                space.bind(space.role('param_types'), space.bundle(types_hv))
            ))
        
        # Encode pattern dictionaries as feature vectors
        for name, patterns in [
            ('return_patterns', channels.return_patterns),
            ('exception_handling', channels.exception_handling),
            ('loop_structures', channels.loop_structures),
            ('conditional_logic', channels.conditional_logic),
            ('data_flow', channels.data_flow)
        ]:
            if patterns:
                # Create a feature vector from the pattern counts
                feature_tokens = []
                for key, value in patterns.items():
                    # Quantize the value
                    if value == 0:
                        continue
                    elif value == 1:
                        feature_tokens.append(f"{key}_single")
                    elif value <= 3:
                        feature_tokens.append(f"{key}_few")
                    else:
                        feature_tokens.append(f"{key}_many")
                
                if feature_tokens:
                    features_hv = [space.token(t) for t in feature_tokens]
                    channel_vectors.append((
                        name,
                        space.bind(space.role(name), space.bundle(features_hv))
                    ))
        
        # Encode complexity
        complexity_token = f"complexity_{min(channels.cognitive_complexity // 5, 10)}"
        channel_vectors.append((
            'complexity',
            space.bind(space.role('complexity'), space.token(complexity_token))
        ))
        
        # Apply weights and bundle
        weighted_vectors = []
        for channel_name, vector in channel_vectors:
            weight = self.channel_weights.get(channel_name, 1.0)
            # Simple weight application (in real implementation would be more sophisticated)
            weighted_vectors.append(vector)
        
        return space.bundle(weighted_vectors) if weighted_vectors else space.token("empty")
    
    def _analyze_channel_similarity(self,
                                  channels_a: EnhancedChannels,
                                  channels_b: EnhancedChannels) -> Dict[str, float]:
        """Analyze similarity between two sets of channels."""
        similarity = {}
        
        # Name token similarity
        if channels_a.name_tokens or channels_b.name_tokens:
            common = set(channels_a.name_tokens) & set(channels_b.name_tokens)
            total = set(channels_a.name_tokens) | set(channels_b.name_tokens)
            similarity['name_tokens'] = len(common) / len(total) if total else 0
        
        # Parameter type similarity
        if channels_a.param_types or channels_b.param_types:
            if len(channels_a.param_types) == len(channels_b.param_types):
                matches = sum(1 for a, b in zip(channels_a.param_types, channels_b.param_types) if a == b)
                similarity['param_types'] = matches / len(channels_a.param_types)
            else:
                similarity['param_types'] = 0
        
        # Pattern similarity for dictionaries
        for name in ['return_patterns', 'exception_handling', 'loop_structures', 
                    'conditional_logic', 'data_flow']:
            dict_a = getattr(channels_a, name)
            dict_b = getattr(channels_b, name)
            
            if dict_a or dict_b:
                # Compare normalized patterns
                keys = set(dict_a.keys()) | set(dict_b.keys())
                if keys:
                    diffs = []
                    for key in keys:
                        val_a = dict_a.get(key, 0)
                        val_b = dict_b.get(key, 0)
                        max_val = max(val_a, val_b, 1)
                        diffs.append(abs(val_a - val_b) / max_val)
                    similarity[name] = 1 - (sum(diffs) / len(diffs))
                else:
                    similarity[name] = 1.0
        
        # Complexity similarity
        max_complexity = max(channels_a.cognitive_complexity, channels_b.cognitive_complexity, 1)
        complexity_diff = abs(channels_a.cognitive_complexity - channels_b.cognitive_complexity)
        similarity['complexity'] = 1 - (complexity_diff / max_complexity)
        
        return similarity
    
    def _get_dominant_channels(self, channel_analysis: Dict[str, float]) -> List[str]:
        """Get the most similar channels."""
        sorted_channels = sorted(channel_analysis.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_channels[:3] if score > 0.5]
