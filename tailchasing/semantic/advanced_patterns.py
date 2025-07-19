"""
Advanced tail-chasing pattern detectors using hypervectors.

Detects complex patterns like:
- Evolutionary dead ends (functions that stop evolving after initial implementation)
- Conceptual drift (gradual semantic changes that diverge from original intent)
- Implementation ping-pong (alternating between approaches)
- Cross-module duplication patterns
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

from .hv_space import HVSpace
from .drift import SemanticDriftAnalyzer


class EvolutionaryAnalyzer:
    """Analyzes evolutionary patterns in code semantics."""
    
    def __init__(self, space: HVSpace, config: Dict):
        self.space = space
        self.config = config
        self.evolution_threshold = config.get('evolution_threshold', 0.1)
        self.stagnation_window = config.get('stagnation_window_days', 30)
    
    def detect_evolutionary_dead_ends(
        self, 
        histories: Dict[str, List[Tuple[datetime, np.ndarray, Dict]]]
    ) -> List[Dict]:
        """Find functions that stopped evolving after initial burst."""
        dead_ends = []
        
        for func_id, history in histories.items():
            if len(history) < 3:
                continue
            
            # Analyze evolution rate over time
            evolution_rate = self._compute_evolution_rate(history)
            
            # Check for initial burst followed by stagnation
            if self._is_dead_end_pattern(evolution_rate):
                dead_ends.append({
                    'function': func_id,
                    'pattern': 'evolutionary_dead_end',
                    'initial_changes': evolution_rate['initial_burst'],
                    'stagnation_period': evolution_rate['stagnation_days'],
                    'risk': self._assess_dead_end_risk(evolution_rate)
                })
        
        return dead_ends
    
    def _compute_evolution_rate(self, history: List[Tuple]) -> Dict:
        """Compute semantic evolution rate over time."""
        if len(history) < 2:
            return {'rate': 0, 'initial_burst': 0, 'stagnation_days': 0}
        
        # Group changes by time window
        changes_per_window = defaultdict(list)
        window_size = timedelta(days=7)  # Weekly windows
        
        for i in range(1, len(history)):
            time_prev, hv_prev, _ = history[i-1]
            time_curr, hv_curr, _ = history[i]
            
            window = time_curr.replace(microsecond=0, second=0, minute=0, hour=0)
            distance = self.space.distance(hv_prev, hv_curr)
            changes_per_window[window].append(distance)
        
        # Analyze pattern
        windows = sorted(changes_per_window.keys())
        if not windows:
            return {'rate': 0, 'initial_burst': 0, 'stagnation_days': 0}
        
        # Initial burst (first 2 weeks)
        initial_burst = sum(
            sum(changes_per_window[w]) for w in windows[:2]
        ) if len(windows) >= 2 else 0
        
        # Recent stagnation
        recent_windows = windows[-4:] if len(windows) >= 4 else windows
        recent_activity = sum(
            sum(changes_per_window[w]) for w in recent_windows
        )
        
        stagnation_days = (datetime.now() - history[-1][0]).days
        
        return {
            'rate': len(changes_per_window),
            'initial_burst': initial_burst,
            'recent_activity': recent_activity,
            'stagnation_days': stagnation_days
        }
    
    def _is_dead_end_pattern(self, evolution_rate: Dict) -> bool:
        """Check if evolution matches dead-end pattern."""
        return (
            evolution_rate['initial_burst'] > 0.5 and
            evolution_rate['recent_activity'] < 0.1 and
            evolution_rate['stagnation_days'] > self.stagnation_window
        )
    
    def _assess_dead_end_risk(self, evolution_rate: Dict) -> str:
        """Assess risk level of evolutionary dead end."""
        if evolution_rate['stagnation_days'] > 90:
            return 'high'
        elif evolution_rate['stagnation_days'] > 45:
            return 'medium'
        else:
            return 'low'


class ConceptualDriftDetector:
    """Detects conceptual drift in function semantics."""
    
    def __init__(self, space: HVSpace, config: Dict):
        self.space = space
        self.config = config
        self.drift_threshold = config.get('conceptual_drift_threshold', 0.6)
    
    def detect_conceptual_drift(
        self,
        histories: Dict[str, List[Tuple[datetime, np.ndarray, Dict]]]
    ) -> List[Dict]:
        """Find functions that have drifted from original concept."""
        drifts = []
        
        for func_id, history in histories.items():
            if len(history) < 5:  # Need enough history
                continue
            
            # Compare current to original
            original_hv = history[0][1]
            current_hv = history[-1][1]
            
            total_drift = self.space.distance(original_hv, current_hv)
            
            if total_drift > self.drift_threshold:
                # Analyze drift trajectory
                trajectory = self._analyze_drift_trajectory(history)
                
                drifts.append({
                    'function': func_id,
                    'pattern': 'conceptual_drift',
                    'total_drift': total_drift,
                    'trajectory': trajectory,
                    'stages': self._identify_drift_stages(history),
                    'recommendation': self._get_drift_recommendation(trajectory)
                })
        
        return drifts
    
    def _analyze_drift_trajectory(self, history: List[Tuple]) -> str:
        """Analyze the trajectory of semantic drift."""
        if len(history) < 3:
            return 'insufficient_data'
        
        # Calculate progressive distances from original
        original_hv = history[0][1]
        distances = []
        
        for _, hv, _ in history[1:]:
            dist = self.space.distance(original_hv, hv)
            distances.append(dist)
        
        # Analyze trend
        if all(distances[i] <= distances[i+1] for i in range(len(distances)-1)):
            return 'monotonic_drift'
        elif distances[-1] < distances[len(distances)//2]:
            return 'returning_drift'
        else:
            return 'chaotic_drift'
    
    def _identify_drift_stages(self, history: List[Tuple]) -> List[Dict]:
        """Identify distinct stages in the drift."""
        stages = []
        threshold = 0.3  # Significant change threshold
        
        current_stage_start = 0
        
        for i in range(1, len(history)):
            _, hv_prev, _ = history[i-1]
            time_curr, hv_curr, _ = history[i]
            
            distance = self.space.distance(hv_prev, hv_curr)
            
            if distance > threshold:
                # New stage detected
                stages.append({
                    'start_index': current_stage_start,
                    'end_index': i-1,
                    'duration': (history[i-1][0] - history[current_stage_start][0]).days,
                    'change_magnitude': distance
                })
                current_stage_start = i
        
        # Add final stage
        if current_stage_start < len(history) - 1:
            stages.append({
                'start_index': current_stage_start,
                'end_index': len(history) - 1,
                'duration': (history[-1][0] - history[current_stage_start][0]).days
            })
        
        return stages
    
    def _get_drift_recommendation(self, trajectory: str) -> str:
        """Get recommendation based on drift trajectory."""
        recommendations = {
            'monotonic_drift': "Function has continuously drifted from original purpose. Consider splitting into separate functions.",
            'returning_drift': "Function explored alternatives but returned closer to original. Document the journey and rationale.",
            'chaotic_drift': "Function shows unstable semantic evolution. Needs architectural review and clear purpose definition."
        }
        return recommendations.get(trajectory, "Review function evolution and clarify intent.")


class CrossModuleDuplicationDetector:
    """Detects duplication patterns across modules."""
    
    def __init__(self, space: HVSpace, config: Dict):
        self.space = space
        self.config = config
        self.module_threshold = config.get('cross_module_threshold', 0.85)
    
    def detect_cross_module_patterns(
        self,
        index_entries: List[Tuple[str, np.ndarray, Dict]]
    ) -> List[Dict]:
        """Find semantic duplication patterns across modules."""
        # Group by module
        module_functions = defaultdict(list)
        
        for entry_id, hv, metadata in index_entries:
            if hv is None:
                continue
            
            file_path = entry_id.split('@')[1].split(':')[0]
            module = self._get_module_from_path(file_path)
            module_functions[module].append((entry_id, hv, metadata))
        
        # Find cross-module duplicates
        patterns = []
        modules = list(module_functions.keys())
        
        for i in range(len(modules)):
            for j in range(i+1, len(modules)):
                module_a = modules[i]
                module_b = modules[j]
                
                duplicates = self._find_module_duplicates(
                    module_functions[module_a],
                    module_functions[module_b]
                )
                
                if duplicates:
                    patterns.append({
                        'pattern': 'cross_module_duplication',
                        'modules': [module_a, module_b],
                        'duplicate_pairs': duplicates,
                        'severity': self._assess_duplication_severity(duplicates)
                    })
        
        return patterns
    
    def _get_module_from_path(self, file_path: str) -> str:
        """Extract module name from file path."""
        parts = file_path.split('/')
        if len(parts) > 1:
            return parts[-2]  # Parent directory as module
        return 'root'
    
    def _find_module_duplicates(
        self,
        functions_a: List[Tuple],
        functions_b: List[Tuple]
    ) -> List[Dict]:
        """Find semantic duplicates between two modules."""
        duplicates = []
        
        for id_a, hv_a, meta_a in functions_a:
            for id_b, hv_b, meta_b in functions_b:
                similarity = self.space.similarity(hv_a, hv_b)
                
                if similarity > self.module_threshold:
                    duplicates.append({
                        'function_a': id_a,
                        'function_b': id_b,
                        'similarity': similarity,
                        'should_consolidate': self._should_consolidate(meta_a, meta_b)
                    })
        
        return duplicates
    
    def _should_consolidate(self, meta_a: Dict, meta_b: Dict) -> bool:
        """Determine if functions should be consolidated."""
        # Check if they have similar interfaces
        args_a = set(meta_a.get('args', []))
        args_b = set(meta_b.get('args', []))
        
        if args_a == args_b:
            return True
        
        # Check feature overlap
        features_a = meta_a.get('features', {})
        features_b = meta_b.get('features', {})
        
        common_channels = set(features_a.keys()) & set(features_b.keys())
        if len(common_channels) > 3:
            return True
        
        return False
    
    def _assess_duplication_severity(self, duplicates: List[Dict]) -> str:
        """Assess severity of cross-module duplication."""
        if len(duplicates) > 10:
            return 'critical'
        elif len(duplicates) > 5:
            return 'high'
        elif len(duplicates) > 2:
            return 'medium'
        else:
            return 'low'


class ImplementationPingPongDetector:
    """Detects ping-pong patterns between implementation approaches."""
    
    def __init__(self, space: HVSpace, config: Dict):
        self.space = space
        self.config = config
        self.similarity_threshold = config.get('pingpong_threshold', 0.9)
    
    def detect_implementation_pingpong(
        self,
        histories: Dict[str, List[Tuple[datetime, np.ndarray, Dict]]]
    ) -> List[Dict]:
        """Find functions that alternate between implementations."""
        pingpongs = []
        
        for func_id, history in histories.items():
            if len(history) < 4:
                continue
            
            # Look for alternating patterns
            pattern = self._find_alternating_pattern(history)
            
            if pattern['is_pingpong']:
                pingpongs.append({
                    'function': func_id,
                    'pattern': 'implementation_pingpong',
                    'alternations': pattern['alternations'],
                    'approaches': pattern['approaches'],
                    'waste_score': pattern['waste_score'],
                    'recommendation': 'Choose one approach and document why'
                })
        
        return pingpongs
    
    def _find_alternating_pattern(self, history: List[Tuple]) -> Dict:
        """Find alternating implementation patterns."""
        # Cluster implementations
        implementations = []
        for _, hv, _ in history:
            implementations.append(hv)
        
        # Simple clustering based on similarity
        clusters = self._cluster_implementations(implementations)
        
        # Check for alternating pattern
        if len(clusters) < 2:
            return {'is_pingpong': False}
        
        # Track cluster assignments over time
        assignments = []
        for impl in implementations:
            for cluster_id, cluster_hvs in enumerate(clusters):
                if any(self.space.similarity(impl, chv) > self.similarity_threshold 
                      for chv in cluster_hvs):
                    assignments.append(cluster_id)
                    break
        
        # Count alternations
        alternations = 0
        for i in range(1, len(assignments)):
            if assignments[i] != assignments[i-1]:
                alternations += 1
        
        is_pingpong = alternations >= 3 and len(clusters) == 2
        
        return {
            'is_pingpong': is_pingpong,
            'alternations': alternations,
            'approaches': len(clusters),
            'waste_score': alternations / len(history) if is_pingpong else 0
        }
    
    def _cluster_implementations(self, implementations: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Simple clustering of implementations."""
        if not implementations:
            return []
        
        clusters = [[implementations[0]]]
        
        for impl in implementations[1:]:
            assigned = False
            
            for cluster in clusters:
                if any(self.space.similarity(impl, c) > self.similarity_threshold for c in cluster):
                    cluster.append(impl)
                    assigned = True
                    break
            
            if not assigned:
                clusters.append([impl])
        
        return clusters
