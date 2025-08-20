"""
Temporal drift analysis for semantic patterns.

Tracks how function semantics evolve over time/commits to detect
tail-chasing patterns like rename cascades and semantic stagnation.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import numpy as np

from .hv_space import HVSpace


class SemanticDriftAnalyzer:
    """
    Analyzes semantic drift patterns over time.
    
    Detects tail-chasing patterns like:
    - Rename cascades (same semantics, different names)
    - Semantic stagnation (placeholders that don't evolve)
    - Ping-pong refactoring (alternating between implementations)
    """
    
    def __init__(self, space: HVSpace, config: Dict):
        self.space = space
        self.config = config
        
        # Function histories: func_id -> [(timestamp, hv, metadata)]
        self.histories: Dict[str, List[Tuple[datetime, np.ndarray, Dict]]] = {}
        
        # Drift chains: track sequences of related changes
        self.drift_chains: List[Dict] = []
        
        # Configuration
        self.window_size = config.get('drift_window', 10)
        self.stagnation_threshold = config.get('stagnation_threshold', 0.95)
        self.cascade_threshold = config.get('cascade_threshold', 0.85)
    
    def add_snapshot(
        self,
        func_id: str,
        timestamp: datetime,
        hv: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add a function snapshot to history."""
        if func_id not in self.histories:
            self.histories[func_id] = []
        
        self.histories[func_id].append((timestamp, hv, metadata or {}))
        
        # Keep only recent history
        if len(self.histories[func_id]) > self.window_size:
            self.histories[func_id] = self.histories[func_id][-self.window_size:]
    
    def analyze_function_drift(self, func_id: str) -> Dict:
        """
        Analyze drift pattern for a specific function.
        
        Returns analysis including drift type and severity.
        """
        if func_id not in self.histories or len(self.histories[func_id]) < 2:
            return {
                'drift_type': 'insufficient_history',
                'severity': 0,
                'details': {}
            }
        
        history = self.histories[func_id]
        
        # Compute semantic distances between consecutive versions
        distances = []
        for i in range(1, len(history)):
            _, hv_prev, _ = history[i-1]
            _, hv_curr, _ = history[i]
            dist = self.space.distance(hv_prev, hv_curr)
            distances.append(dist)
        
        # Analyze patterns
        analysis = {
            'stagnation_score': self._compute_stagnation_score(distances),
            'volatility_score': self._compute_volatility_score(distances),
            'trend': self._compute_drift_trend(history),
            'ping_pong_score': self._detect_ping_pong(history)
        }
        
        # Determine dominant pattern
        if analysis['stagnation_score'] > 0.8:
            drift_type = 'semantic_stagnation'
            severity = analysis['stagnation_score']
        elif analysis['ping_pong_score'] > 0.7:
            drift_type = 'ping_pong_refactoring'
            severity = analysis['ping_pong_score']
        elif analysis['volatility_score'] > 0.6:
            drift_type = 'high_volatility'
            severity = analysis['volatility_score']
        else:
            drift_type = 'normal_evolution'
            severity = 0.2
        
        return {
            'drift_type': drift_type,
            'severity': severity,
            'details': analysis
        }
    
    def _compute_stagnation_score(self, distances: List[float]) -> float:
        """
        Compute stagnation score from distance sequence.
        
        High score indicates function isn't evolving semantically.
        """
        if not distances:
            return 0.0
        
        # Count near-zero distances
        stagnant_count = sum(1 for d in distances if d < 0.1)
        
        # Weight recent stagnation more heavily
        weighted_stagnation = 0.0
        for i, dist in enumerate(distances):
            weight = (i + 1) / len(distances)  # More recent = higher weight
            if dist < 0.1:
                weighted_stagnation += weight
        
        base_score = stagnant_count / len(distances)
        weighted_score = weighted_stagnation / (len(distances) * 0.5)
        
        return min((base_score + weighted_score) / 2, 1.0)
    
    def _compute_volatility_score(self, distances: List[float]) -> float:
        """
        Compute volatility score from distance sequence.
        
        High score indicates frequent semantic changes.
        """
        if len(distances) < 2:
            return 0.0
        
        # Compute variance of distances
        mean_dist = sum(distances) / len(distances)
        variance = sum((d - mean_dist) ** 2 for d in distances) / len(distances)
        
        # Normalize by expected variance
        expected_variance = 0.25  # Baseline for normal evolution
        
        return min(variance / expected_variance, 1.0)
    
    def _compute_drift_trend(self, history: List[Tuple]) -> str:
        """
        Compute overall drift trend.
        
        Returns: 'expanding', 'contracting', 'stable', or 'oscillating'
        """
        if len(history) < 3:
            return 'stable'
        
        # Compare first, middle, and last hypervectors
        first_hv = history[0][1]
        mid_hv = history[len(history) // 2][1]
        last_hv = history[-1][1]
        
        # Compute distances
        first_to_mid = self.space.distance(first_hv, mid_hv)
        mid_to_last = self.space.distance(mid_hv, last_hv)
        first_to_last = self.space.distance(first_hv, last_hv)
        
        # Determine trend
        if first_to_last > (first_to_mid + mid_to_last) * 0.8:
            return 'expanding'
        elif first_to_last < 0.2:
            return 'contracting'
        elif abs(first_to_mid - mid_to_last) < 0.1:
            return 'stable'
        else:
            return 'oscillating'
    
    def _detect_ping_pong(self, history: List[Tuple]) -> float:
        """
        Detect ping-pong pattern (alternating between similar states).
        
        Returns score indicating likelihood of ping-pong refactoring.
        """
        if len(history) < 4:
            return 0.0
        
        # Check for alternating patterns
        alternating_score = 0.0
        
        for i in range(2, len(history)):
            # Compare current with 2 versions ago
            _, hv_curr, _ = history[i]
            _, hv_prev2, _ = history[i-2]
            
            similarity = self.space.similarity(hv_curr, hv_prev2)
            
            if similarity > 0.9:  # Very similar to 2 versions ago
                alternating_score += 1
        
        return alternating_score / (len(history) - 2)
    
    def detect_rename_cascades(self, time_window: Optional[int] = None) -> List[Dict]:
        """
        Detect rename cascade patterns across functions.
        
        Returns list of cascade chains where functions with similar
        semantics appear to be renamed versions of each other.
        """
        if time_window is None:
            time_window = self.window_size
        
        cascades = []
        
        # Group functions by recent timestamps
        recent_functions = defaultdict(list)
        
        for func_id, history in self.histories.items():
            if history:
                latest_time, latest_hv, latest_meta = history[-1]
                recent_functions[func_id].append((latest_time, latest_hv, latest_meta))
        
        # Find potential rename pairs
        func_ids = list(recent_functions.keys())
        
        for i in range(len(func_ids)):
            for j in range(i + 1, len(func_ids)):
                func1_id = func_ids[i]
                func2_id = func_ids[j]
                
                # Skip if same base name
                if func1_id.split('@')[0] == func2_id.split('@')[0]:
                    continue
                
                # Get latest hypervectors
                _, hv1, meta1 = recent_functions[func1_id][-1]
                _, hv2, meta2 = recent_functions[func2_id][-1]
                
                # Check semantic similarity
                similarity = self.space.similarity(hv1, hv2)
                
                if similarity > self.cascade_threshold:
                    # Potential rename cascade
                    cascade = {
                        'type': 'rename_cascade',
                        'functions': [func1_id, func2_id],
                        'similarity': similarity,
                        'evidence': self._gather_cascade_evidence(func1_id, func2_id)
                    }
                    cascades.append(cascade)
        
        # Merge cascades into chains
        merged_cascades = self._merge_cascades(cascades)
        
        return merged_cascades
    
    def _gather_cascade_evidence(self, func1_id: str, func2_id: str) -> Dict:
        """
        Gather evidence for rename cascade detection.
        """
        evidence = {
            'name_similarity': self._compute_name_similarity(func1_id, func2_id),
            'file_relationship': self._analyze_file_relationship(func1_id, func2_id),
            'temporal_overlap': self._check_temporal_overlap(func1_id, func2_id)
        }
        
        # Strong evidence if one appears as the other disappears
        if evidence['temporal_overlap'] < 0.2 and evidence['name_similarity'] > 0.3:
            evidence['confidence'] = 'high'
        elif evidence['file_relationship'] == 'same_module':
            evidence['confidence'] = 'medium'
        else:
            evidence['confidence'] = 'low'
        
        return evidence
    
    def _compute_name_similarity(self, func1_id: str, func2_id: str) -> float:
        """Compute name similarity between function IDs."""
        from ..semantic.encoder import split_identifier
        
        name1 = func1_id.split('@')[0]
        name2 = func2_id.split('@')[0]
        
        tokens1 = set(split_identifier(name1))
        tokens2 = set(split_identifier(name2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def _analyze_file_relationship(self, func1_id: str, func2_id: str) -> str:
        """Analyze file relationship between functions."""
        file1 = func1_id.split('@')[1].split(':')[0]
        file2 = func2_id.split('@')[1].split(':')[0]
        
        if file1 == file2:
            return 'same_file'
        
        # Check if in same module
        parts1 = file1.split('/')
        parts2 = file2.split('/')
        
        if len(parts1) > 1 and len(parts2) > 1:
            if parts1[:-1] == parts2[:-1]:
                return 'same_module'
        
        return 'different_modules'
    
    def _check_temporal_overlap(self, func1_id: str, func2_id: str) -> float:
        """
        Check temporal overlap between function histories.
        
        Returns overlap ratio (0 = no overlap, 1 = complete overlap).
        """
        if func1_id not in self.histories or func2_id not in self.histories:
            return 0.0
        
        hist1 = self.histories[func1_id]
        hist2 = self.histories[func2_id]
        
        if not hist1 or not hist2:
            return 0.0
        
        # Get time ranges
        start1, end1 = hist1[0][0], hist1[-1][0]
        start2, end2 = hist2[0][0], hist2[-1][0]
        
        # Compute overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0
        
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        total_duration = (max(end1, end2) - min(start1, start2)).total_seconds()
        
        return overlap_duration / total_duration if total_duration > 0 else 0.0
    
    def _merge_cascades(self, cascades: List[Dict]) -> List[Dict]:
        """
        Merge individual cascades into chains.
        """
        if not cascades:
            return []
        
        # Build graph of related functions
        graph = defaultdict(set)
        
        for cascade in cascades:
            funcs = cascade['functions']
            graph[funcs[0]].add(funcs[1])
            graph[funcs[1]].add(funcs[0])
        
        # Find connected components
        visited = set()
        chains = []
        
        for func_id in graph:
            if func_id not in visited:
                # DFS to find component
                chain = []
                stack = [func_id]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        chain.append(current)
                        stack.extend(graph[current] - visited)
                
                if len(chain) > 1:
                    chains.append({
                        'type': 'rename_cascade_chain',
                        'functions': chain,
                        'length': len(chain),
                        'risk': self._assess_cascade_risk(chain)
                    })
        
        return chains
    
    def _assess_cascade_risk(self, chain: List[str]) -> str:
        """
        Assess risk level of a rename cascade chain.
        """
        if len(chain) >= 5:
            return 'critical'
        elif len(chain) >= 3:
            return 'high'
        else:
            return 'medium'
    
    def get_drift_summary(self) -> Dict:
        """
        Get summary of all drift patterns detected.
        """
        summary = {
            'total_functions_tracked': len(self.histories),
            'patterns': defaultdict(int),
            'high_risk_functions': []
        }
        
        # Analyze each function
        for func_id in self.histories:
            analysis = self.analyze_function_drift(func_id)
            
            summary['patterns'][analysis['drift_type']] += 1
            
            if analysis['severity'] > 0.7:
                summary['high_risk_functions'].append({
                    'function': func_id,
                    'type': analysis['drift_type'],
                    'severity': analysis['severity']
                })
        
        # Add cascade analysis
        cascades = self.detect_rename_cascades()
        summary['rename_cascades'] = len(cascades)
        summary['cascade_chains'] = cascades
        
        return dict(summary)