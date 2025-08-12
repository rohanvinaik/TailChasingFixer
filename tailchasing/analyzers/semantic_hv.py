"""
Semantic hypervector analyzer for tail-chasing detection.

This analyzer integrates hypervector encoding, similarity analysis,
prototype detection, and drift analysis to identify semantic patterns
that indicate tail-chasing behavior.
"""

from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
import ast
import hashlib
from difflib import SequenceMatcher
import numpy as np

from ..core.issues import Issue
from ..analyzers.base import Analyzer, AnalysisContext
from ..semantic.encoder import encode_function
from ..semantic.index import SemanticIndex
from ..semantic.similarity import SimilarityAnalyzer
from ..semantic.prototypes import PrototypeDetector
from ..semantic.drift import SemanticDriftAnalyzer
from ..semantic.smart_filter import SemanticDuplicateFilter


class SemanticHVAnalyzer(Analyzer):
    """
    Detects semantic tail-chasing patterns using hypervector computing.
    
    Identifies:
    - Semantic duplicate functions (beyond structural similarity)
    - Prototype fragmentation (multiple implementations of same concept)
    - Semantic stagnation (placeholders that don't evolve)
    - Rename cascades (same semantics, different names over time)
    """
    
    name = "semantic_hv"
    
    def __init__(self):
        self.index: Optional[SemanticIndex] = None
        self.similarity_analyzer: Optional[SimilarityAnalyzer] = None
        self.prototype_detector: Optional[PrototypeDetector] = None
        self.drift_analyzer: Optional[SemanticDriftAnalyzer] = None
        self.smart_filter: Optional[SemanticDuplicateFilter] = None
        self._feature_cache: Dict[str, Dict] = {}
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Run semantic hypervector analysis."""
        # Get configuration
        config = ctx.config.get('semantic', {})
        if not config.get('enable', False):
            return []
        
        # Check minimum function threshold
        min_functions = config.get('min_functions', 40)
        total_functions = sum(len(entries) for entries in ctx.symbol_table.functions.values())
        
        if total_functions < min_functions:
            return []
        
        # Initialize components
        self._initialize_components(ctx, config)
        
        # Encode functions
        self._encode_functions(ctx)
        
        # Run analyses
        issues = []
        
        # 1. Find semantic duplicates
        issues.extend(self._find_semantic_duplicates())
        
        # 2. Detect prototype fragmentation
        issues.extend(self._detect_prototype_fragmentation())
        
        # 3. Analyze semantic drift (if git integration available)
        if ctx.config.get('git_integration', {}).get('enabled', False):
            issues.extend(self._analyze_semantic_drift(ctx))
        
        # 4. Check for semantic stagnation
        issues.extend(self._detect_semantic_stagnation())
        
        return issues
    
    def _initialize_components(self, ctx: AnalysisContext, config: Dict) -> None:
        """Initialize semantic analysis components."""
        # Get cache directory
        cache_dir = None
        if config.get('incremental_cache'):
            cache_dir = Path(config['incremental_cache'])
        
        # Merge resource limits into config
        resource_limits = ctx.config.get('resource_limits', {})
        config['resource_limits'] = resource_limits
        
        # Initialize index
        if 'semantic_index' in ctx.cache:
            self.index = ctx.cache['semantic_index']
        else:
            self.index = SemanticIndex(config, cache_dir)
            ctx.cache['semantic_index'] = self.index
        
        # Initialize analyzers
        self.similarity_analyzer = SimilarityAnalyzer(config)
        self.prototype_detector = PrototypeDetector(self.index.space, config)
        self.drift_analyzer = SemanticDriftAnalyzer(self.index.space, config)
        self.smart_filter = SemanticDuplicateFilter()
    
    def _encode_functions(self, ctx: AnalysisContext) -> None:
        """Encode all functions as hypervectors."""
        config = ctx.config.get('semantic', {})
        
        # Store AST index for smart filtering
        self._ast_index = ctx.ast_index
        
        for func_name, entries in ctx.symbol_table.functions.items():
            for entry in entries:
                file = entry['file']
                line = entry['lineno']
                node = entry['node']
                
                # Generate unique ID
                func_id = f"{func_name}@{file}:{line}"
                
                # Skip if already encoded (incremental mode)
                if func_id in self.index.id_to_index:
                    continue
                
                try:
                    # Encode function
                    hv = encode_function(node, file, self.index.space, config)
                    
                    # Extract features for later analysis
                    from ..semantic.encoder import FunctionFeatureExtractor
                    extractor = FunctionFeatureExtractor()
                    features = extractor.extract(node)
                    self._feature_cache[func_id] = features
                    
                    # Add to index
                    self.index.add(func_name, file, line, hv, {
                        'features': features,
                        'args': entry.get('args', [])
                    })
                    
                except Exception as e:
                    # Log encoding error but continue
                    pass
    
    def _find_semantic_duplicates(self) -> List[Issue]:
        """Find semantically similar functions."""
        issues = []
        
        # Get all similar pairs with a reasonable limit to prevent hanging
        pairs = self.index.find_all_similar_pairs(limit=100)
        
        # Apply FDR correction
        significant_pairs = self.similarity_analyzer.filter_significant_pairs(pairs)
        
        # Apply smart filtering to remove false positives (like __init__.py patterns)
        if self.smart_filter and significant_pairs:
            # Convert to format expected by smart filter
            filter_pairs = []
            for id1, id2, distance, z_score, analysis in significant_pairs:
                func1 = self._parse_func_id(id1)
                func2 = self._parse_func_id(id2)
                filter_pairs.append((
                    (func1['name'], func1['file'], func1['line']),
                    (func2['name'], func2['file'], func2['line']),
                    distance, z_score, {}
                ))
            
            # Filter out legitimate patterns
            filtered_pairs = self.smart_filter.filter_semantic_duplicates(
                filter_pairs, 
                # We need the AST index - get it from context
                getattr(self, '_ast_index', {})
            )
            
            # Convert back to original format
            filtered_indices = set()
            for i, (func1_info, func2_info, _, _, _) in enumerate(filter_pairs):
                for filtered_func1, filtered_func2, _, _, _ in filtered_pairs:
                    if (func1_info == filtered_func1 and func2_info == filtered_func2):
                        filtered_indices.add(i)
                        break
            
            significant_pairs = [pair for i, pair in enumerate(significant_pairs) if i in filtered_indices]
        
        # Create issues for significant pairs
        for id1, id2, distance, z_score, analysis in significant_pairs:
            # Extract function info
            func1 = self._parse_func_id(id1)
            func2 = self._parse_func_id(id2)
            
            # Skip if structural duplicates (already caught by other analyzer)
            if self._are_structural_duplicates(func1, func2):
                continue
            
            # Compute channel contributions
            features1 = self._feature_cache.get(id1, {})
            features2 = self._feature_cache.get(id2, {})
            
            channel_contrib = self.similarity_analyzer.analyze_channel_contributions(
                self.index.entries[self.index.id_to_index[id1]][1],
                self.index.entries[self.index.id_to_index[id2]][1],
                self.index.space,
                features1,
                features2
            )
            
            issue = Issue(
                kind="semantic_duplicate_function",
                message=f"Semantic duplicate: {func1['name']} and {func2['name']} "
                       f"(z-score: {z_score:.2f}, distance: {distance:.3f})",
                severity=self._compute_duplicate_severity(z_score, analysis),
                file=func1['file'],
                line=func1['line'],
                symbol=func1['name'],
                evidence={
                    'pair': [func1, func2],
                    'z_score': z_score,
                    'distance': distance,
                    'p_value': analysis.get('p_value', 0),
                    'channel_contributions': channel_contrib,
                    'same_file': analysis.get('files_same', False),
                    'name_similar': analysis.get('names_similar', False)
                },
                suggestions=self._get_duplicate_suggestions(func1, func2, analysis)
            )
            
            issues.append(issue)
        
        return issues
    
    def _detect_prototype_fragmentation(self) -> List[Issue]:
        """Detect fragmented prototype patterns."""
        issues = []
        
        # Get function clusters
        clusters = self.similarity_analyzer.cluster_similar_functions(
            self.index.entries,
            self.index.space
        )
        
        # Find prototypes
        prototypes = self.prototype_detector.find_prototypes(clusters)
        
        # Create issues for fragmented prototypes
        for proto in prototypes:
            if proto['risk_score'] < 0.5:
                continue
            
            issue = Issue(
                kind="prototype_fragmentation",
                message=f"Fragmented implementation pattern: {proto['pattern']} "
                       f"({proto['size']} variants)",
                severity=self._compute_fragmentation_severity(proto),
                evidence={
                    'prototype_id': proto['id'],
                    'pattern': proto['pattern'],
                    'size': proto['size'],
                    'cohesion': proto['cohesion'],
                    'risk_score': proto['risk_score'],
                    'functions': proto['functions'][:10]  # Limit for readability
                },
                suggestions=self._get_fragmentation_suggestions(proto)
            )
            
            issues.append(issue)
        
        return issues
    
    def _analyze_semantic_drift(self, ctx: AnalysisContext) -> List[Issue]:
        """Analyze semantic drift patterns over time."""
        issues = []
        
        # This would integrate with git history
        # For now, return empty list
        # TODO: Implement git integration
        
        return issues
    
    def _detect_semantic_stagnation(self) -> List[Issue]:
        """Detect semantically stagnant placeholders."""
        issues = []
        
        # Look for functions with low channel diversity
        for entry_id, hv, metadata in self.index.entries:
            if hv is None or metadata.get('removed', False):
                continue
            
            features = metadata.get('features', {})
            if not features:
                continue
            
            # Check for placeholder indicators
            func_info = self._parse_func_id(entry_id)
            
            # Count active channels
            active_channels = sum(1 for v in features.values() if v)
            
            # Check for stub patterns
            is_stub = (
                active_channels <= 2 and
                ('DOC_TOKENS' not in features or not features['DOC_TOKENS']) and
                ('CALLS' not in features or len(features.get('CALLS', [])) <= 1)
            )
            
            if is_stub:
                # Check against common stub patterns
                stub_patterns = ['pass', 'notimplemented', 'todo', 'fixme']
                name_lower = func_info['name'].lower()
                
                if any(pat in name_lower for pat in stub_patterns):
                    severity = 3
                else:
                    severity = 2
                
                issue = Issue(
                    kind="semantic_stagnant_placeholder",
                    message=f"Semantically empty placeholder: {func_info['name']}",
                    severity=severity,
                    file=func_info['file'],
                    line=func_info['line'],
                    symbol=func_info['name'],
                    evidence={
                        'active_channels': active_channels,
                        'features': features
                    },
                    suggestions=[
                        "Implement the function logic or remove if not needed",
                        "If this is intentional, add a docstring explaining why"
                    ]
                )
                
                issues.append(issue)
        
        return issues
    
    def _parse_func_id(self, func_id: str) -> Dict[str, any]:
        """Parse function ID into components."""
        name, location = func_id.split('@')
        file, line = location.split(':')
        return {
            'name': name,
            'file': file,
            'line': int(line)
        }
    
    def _are_structural_duplicates(self, f1: dict, f2: dict, *, 
                                   seq_threshold: float = 0.9, skel_threshold: float = 0.9) -> bool:
        """Check if functions are structural duplicates."""
        n1 = f1.get("ast_node")
        n2 = f2.get("ast_node")
        if not isinstance(n1, ast.AST) or not isinstance(n2, ast.AST):
            return False

        def normalize(node: ast.AST) -> ast.AST:
            class N(ast.NodeTransformer):
                def visit_Name(self, n: ast.Name):
                    return ast.copy_location(ast.Name(id="ID", ctx=n.ctx), n)
                def visit_Attribute(self, n: ast.Attribute):
                    self.generic_visit(n)
                    return ast.copy_location(ast.Attribute(value=n.value, attr="ATTR", ctx=n.ctx), n)
                def visit_Constant(self, n: ast.Constant):
                    tag = "NUM" if isinstance(n.value, (int, float, complex)) else \
                          "STR" if isinstance(n.value, str) else \
                          "BOOL" if isinstance(n.value, bool) else "CONST"
                    return ast.copy_location(ast.Name(id=tag, ctx=ast.Load()), n)
                def visit_FunctionDef(self, n: ast.FunctionDef):
                    # strip decorators and docstring
                    body = n.body
                    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                        body = body[1:]
                    n = ast.FunctionDef(name="FUNC", args=n.args, body=body, decorator_list=[], returns=None, type_comment=None)
                    return self.generic_visit(n)
                visit_AsyncFunctionDef = visit_FunctionDef
            return N().visit(ast.fix_missing_locations(ast.parse(ast.unparse(node))))

        def seq(node: ast.AST) -> list[str]:
            out: list[str] = []
            for sub in ast.walk(node):
                out.append(type(sub).__name__)
            return out

        def bigrams(tokens): 
            return set(zip(tokens, tokens[1:]))

        sk1 = ast.dump(normalize(n1), annotate_fields=False, include_attributes=False)
        sk2 = ast.dump(normalize(n2), annotate_fields=False, include_attributes=False)
        seq1, seq2 = seq(ast.parse(sk1)), seq(ast.parse(sk2))
        bg1, bg2 = bigrams(seq1), bigrams(seq2)
        jacc = len(bg1 & bg2) / max(1, len(bg1 | bg2))
        ratio = SequenceMatcher(None, sk1, sk2).ratio()
        return (jacc >= seq_threshold) and (ratio >= skel_threshold)
    
    def _compute_duplicate_severity(self, z_score: float, analysis: Dict) -> int:
        """Compute severity for semantic duplicate."""
        base_severity = 2
        
        # Higher z-score = more significant
        if z_score > 4.0:
            base_severity += 2
        elif z_score > 3.0:
            base_severity += 1
        
        # Same file duplicates are worse
        if analysis.get('files_same', False):
            base_severity += 1
        
        # Different names but same semantics is suspicious
        if not analysis.get('names_similar', False):
            base_severity += 1
        
        return min(base_severity, 5)
    
    def _compute_fragmentation_severity(self, proto: Dict) -> int:
        """Compute severity for prototype fragmentation."""
        base_severity = 2
        
        # More fragments = worse
        if proto['size'] >= 10:
            base_severity += 2
        elif proto['size'] >= 5:
            base_severity += 1
        
        # High risk patterns
        if proto['pattern'] in ['reimplementation', 'wrapper_proliferation']:
            base_severity += 1
        
        # Low cohesion = more scattered
        if proto['cohesion'] < 0.7:
            base_severity += 1
        
        return min(base_severity, 5)
    
    def _get_duplicate_suggestions(self, func1: Dict, func2: Dict, 
                                  analysis: Dict) -> List[str]:
        """Get suggestions for resolving semantic duplicates."""
        suggestions = []
        
        if analysis.get('files_same', False):
            suggestions.append(
                f"Merge {func1['name']} and {func2['name']} in {func1['file']}"
            )
        else:
            suggestions.append(
                "Extract common functionality to a shared module"
            )
            suggestions.append(
                f"Consider if {func1['name']} and {func2['name']} serve the same purpose"
            )
        
        if not analysis.get('names_similar', False):
            suggestions.append(
                "Review naming conventions - these functions have different names but similar behavior"
            )
        
        return suggestions
    
    def _get_fragmentation_suggestions(self, proto: Dict) -> List[str]:
        """Get suggestions from prototype detector."""
        consolidation = self.prototype_detector.suggest_consolidation(proto)
        
        suggestions = []
        for priority in ['immediate', 'refactor', 'investigate']:
            suggestions.extend(consolidation.get(priority, []))
        
        return suggestions[:5]  # Limit to 5 suggestions