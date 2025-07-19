"""
Prototype detection and canonical pattern identification.

Identifies clusters of semantically similar functions that may represent
fragmented implementations of the same concept.
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict

from .hv_space import HVSpace


class PrototypeDetector:
    """
    Detects canonical prototypes and their variants.
    
    Helps identify when multiple functions are semantic variations
    of the same underlying pattern (tail-chasing fragmentation).
    """
    
    def __init__(self, space: HVSpace, config: Dict):
        self.space = space
        self.config = config
        self.prototypes: Dict[str, Dict] = {}
        self.prototype_threshold = config.get('prototype_threshold', 0.8)
    
    def find_prototypes(
        self,
        clusters: List[Dict],
        min_size: int = 3
    ) -> List[Dict]:
        """
        Identify prototype patterns from clusters.
        
        Args:
            clusters: List of cluster dictionaries from similarity analysis
            min_size: Minimum cluster size to consider as prototype
        
        Returns:
            List of prototype dictionaries
        """
        prototypes = []
        
        for cluster in clusters:
            if cluster['size'] < min_size:
                continue
            
            # Analyze cluster for prototype characteristics
            analysis = self._analyze_cluster(cluster)
            
            if analysis['is_prototype_candidate']:
                prototype = {
                    'id': f"proto_{len(prototypes)}",
                    'cluster_id': cluster['id'],
                    'size': cluster['size'],
                    'functions': cluster['functions'],
                    'centroid': cluster['prototype'],
                    'cohesion': cluster['cohesion'],
                    'pattern': analysis['pattern'],
                    'risk_score': analysis['fragmentation_risk']
                }
                prototypes.append(prototype)
                self.prototypes[prototype['id']] = prototype
        
        return prototypes
    
    def _analyze_cluster(self, cluster: Dict) -> Dict:
        """
        Analyze cluster to determine if it represents a prototype pattern.
        
        Returns analysis including pattern type and fragmentation risk.
        """
        functions = cluster['functions']
        
        # Extract patterns from function names
        name_patterns = self._extract_name_patterns(functions)
        
        # Check for common tail-chasing patterns
        pattern_scores = {
            'reimplementation': self._score_reimplementation_pattern(functions),
            'wrapper_proliferation': self._score_wrapper_pattern(functions),
            'util_fragmentation': self._score_util_fragmentation(functions),
            'interface_variants': self._score_interface_variants(functions)
        }
        
        # Determine dominant pattern
        dominant_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        
        # Calculate fragmentation risk
        risk_score = self._calculate_fragmentation_risk(
            cluster['size'],
            cluster['cohesion'],
            pattern_scores
        )
        
        return {
            'is_prototype_candidate': risk_score > 0.5,
            'pattern': dominant_pattern[0],
            'pattern_scores': pattern_scores,
            'fragmentation_risk': risk_score,
            'name_patterns': name_patterns
        }
    
    def _extract_name_patterns(self, function_ids: List[str]) -> Dict[str, int]:
        """Extract common naming patterns from function IDs."""
        from ..semantic.encoder import split_identifier
        
        # Extract function names
        names = [fid.split('@')[0] for fid in function_ids]
        
        # Count token frequencies
        token_freq = defaultdict(int)
        for name in names:
            tokens = split_identifier(name)
            for token in tokens:
                token_freq[token] += 1
        
        # Find tokens that appear in multiple functions
        patterns = {}
        for token, freq in token_freq.items():
            if freq >= len(names) * 0.4:  # 40% threshold
                patterns[token] = freq
        
        return patterns
    
    def _score_reimplementation_pattern(self, function_ids: List[str]) -> float:
        """Score likelihood of reimplementation pattern."""
        names = [fid.split('@')[0] for fid in function_ids]
        files = [fid.split('@')[1].split(':')[0] for fid in function_ids]
        
        # High score if similar names across different files
        unique_files = len(set(files))
        unique_names = len(set(names))
        
        if unique_files > 1 and unique_names < len(names) * 0.5:
            return 0.8
        
        # Check for version suffixes (e.g., _v2, _new, _alt)
        version_patterns = ['_v', '_new', '_alt', '_old', '_temp', '2', '3']
        version_count = sum(
            1 for name in names
            if any(pat in name.lower() for pat in version_patterns)
        )
        
        if version_count > len(names) * 0.3:
            return 0.7
        
        return 0.2
    
    def _score_wrapper_pattern(self, function_ids: List[str]) -> float:
        """Score likelihood of wrapper proliferation."""
        names = [fid.split('@')[0] for fid in function_ids]
        
        wrapper_indicators = [
            'wrap', 'wrapper', 'proxy', 'delegate', 'forward',
            'call', 'invoke', 'dispatch'
        ]
        
        wrapper_count = sum(
            1 for name in names
            if any(ind in name.lower() for ind in wrapper_indicators)
        )
        
        return wrapper_count / len(names)
    
    def _score_util_fragmentation(self, function_ids: List[str]) -> float:
        """Score likelihood of utility function fragmentation."""
        names = [fid.split('@')[0] for fid in function_ids]
        
        util_indicators = [
            'util', 'helper', 'common', 'shared', 'core',
            'base', 'lib', 'tools'
        ]
        
        util_count = sum(
            1 for name in names
            if any(ind in name.lower() for ind in util_indicators)
        )
        
        # Also check for generic operation names
        operation_indicators = [
            'get', 'set', 'check', 'validate', 'process',
            'handle', 'convert', 'parse', 'format'
        ]
        
        op_count = sum(
            1 for name in names
            if any(ind in name.lower() for ind in operation_indicators)
        )
        
        return max(util_count / len(names), op_count / len(names) * 0.7)
    
    def _score_interface_variants(self, function_ids: List[str]) -> float:
        """Score likelihood of interface variant pattern."""
        names = [fid.split('@')[0] for fid in function_ids]
        
        # Check for async/sync variants
        async_count = sum(1 for n in names if 'async' in n.lower())
        sync_count = sum(1 for n in names if 'sync' in n.lower())
        
        if async_count > 0 and sync_count > 0:
            return 0.6
        
        # Check for different parameter variants
        variant_indicators = ['with_', 'without_', 'from_', 'to_']
        variant_count = sum(
            1 for name in names
            if any(name.lower().startswith(ind) or f'_{ind}' in name.lower() 
                  for ind in variant_indicators)
        )
        
        return variant_count / len(names) * 0.8
    
    def _calculate_fragmentation_risk(
        self,
        cluster_size: int,
        cohesion: float,
        pattern_scores: Dict[str, float]
    ) -> float:
        """
        Calculate overall fragmentation risk score.
        
        Higher score indicates higher risk of tail-chasing fragmentation.
        """
        # Size factor: larger clusters indicate more fragmentation
        size_factor = min(cluster_size / 10, 1.0)
        
        # Cohesion factor: lower cohesion might indicate forced similarity
        cohesion_factor = 1.0 - cohesion
        
        # Pattern factor: maximum pattern score
        pattern_factor = max(pattern_scores.values())
        
        # Weighted combination
        risk = (
            0.3 * size_factor +
            0.2 * cohesion_factor +
            0.5 * pattern_factor
        )
        
        return min(risk, 1.0)
    
    def suggest_consolidation(self, prototype: Dict) -> Dict[str, List[str]]:
        """
        Suggest consolidation strategy for a prototype cluster.
        
        Returns suggestions for reducing fragmentation.
        """
        pattern = prototype['pattern']
        suggestions = {
            'immediate': [],
            'refactor': [],
            'investigate': []
        }
        
        if pattern == 'reimplementation':
            suggestions['immediate'].append(
                "Identify the most complete implementation and remove others"
            )
            suggestions['refactor'].append(
                "Extract common functionality to a shared module"
            )
            
        elif pattern == 'wrapper_proliferation':
            suggestions['immediate'].append(
                "Replace thin wrappers with direct calls where possible"
            )
            suggestions['refactor'].append(
                "Create a single configurable wrapper if abstraction is needed"
            )
            
        elif pattern == 'util_fragmentation':
            suggestions['refactor'].append(
                "Consolidate utility functions into a single utils module"
            )
            suggestions['investigate'].append(
                "Check if standard library or external packages provide this functionality"
            )
            
        elif pattern == 'interface_variants':
            suggestions['refactor'].append(
                "Use optional parameters instead of multiple function variants"
            )
            suggestions['refactor'].append(
                "Consider using function overloading or a builder pattern"
            )
        
        # Add generic suggestions
        if prototype['size'] > 5:
            suggestions['investigate'].append(
                f"High fragmentation ({prototype['size']} variants) suggests architectural issues"
            )
        
        return suggestions