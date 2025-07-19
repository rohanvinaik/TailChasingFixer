"""
Statistical similarity analysis for hypervectors.

Provides significance testing, false discovery rate control,
and channel contribution analysis.
"""

import math
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from scipy import stats


def benjamini_hochberg_fdr(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Benjamini-Hochberg FDR correction to p-values.
    
    Args:
        p_values: List of p-values
        alpha: False discovery rate threshold
    
    Returns:
        List of booleans indicating significant results
    """
    n = len(p_values)
    if n == 0:
        return []
    
    # Sort p-values with indices
    sorted_pvals = sorted(enumerate(p_values), key=lambda x: x[1])
    
    # Apply BH procedure
    significant = [False] * n
    
    for rank, (orig_idx, pval) in enumerate(sorted_pvals, 1):
        threshold = (rank / n) * alpha
        if pval <= threshold:
            # All p-values up to this rank are significant
            for j in range(rank):
                significant[sorted_pvals[j][0]] = True
        else:
            break
    
    return significant


def z_to_p_value(z_score: float, two_tailed: bool = True) -> float:
    """
    Convert z-score to p-value.
    
    Args:
        z_score: Standard score
        two_tailed: Use two-tailed test
    
    Returns:
        p-value
    """
    if two_tailed:
        return 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        return 1 - stats.norm.cdf(z_score)


class SimilarityAnalyzer:
    """
    Analyzes semantic similarity with statistical rigor.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.fdr_alpha = config.get('fdr_q', 0.05)
        self.z_threshold = config.get('z_threshold', 2.5)
    
    def filter_significant_pairs(
        self, 
        pairs: List[Tuple[str, str, float, float, Dict]]
    ) -> List[Tuple[str, str, float, float, Dict]]:
        """
        Filter pairs using FDR control.
        
        Args:
            pairs: List of (id1, id2, distance, z_score, analysis)
        
        Returns:
            Filtered list of significant pairs
        """
        if not pairs:
            return []
        
        # Convert z-scores to p-values
        p_values = [z_to_p_value(z) for _, _, _, z, _ in pairs]
        
        # Apply FDR correction
        significant = benjamini_hochberg_fdr(p_values, self.fdr_alpha)
        
        # Filter pairs
        filtered = []
        for i, (is_sig, pair) in enumerate(zip(significant, pairs)):
            if is_sig:
                # Add p-value to analysis
                id1, id2, dist, z, analysis = pair
                analysis['p_value'] = p_values[i]
                analysis['fdr_significant'] = True
                filtered.append((id1, id2, dist, z, analysis))
        
        return filtered
    
    def cluster_similar_functions(
        self,
        entries: List[Tuple[str, np.ndarray, Dict]],
        space,
        min_cluster_size: int = 2
    ) -> List[Dict]:
        """
        Cluster functions by semantic similarity.
        
        Uses simple greedy clustering based on z-score threshold.
        
        Returns:
            List of cluster dictionaries
        """
        # Filter valid entries
        valid_entries = [(id, hv, meta) for id, hv, meta in entries
                        if hv is not None and not meta.get("removed", False)]
        
        if len(valid_entries) < min_cluster_size:
            return []
        
        # Build similarity graph
        edges = []
        for i in range(len(valid_entries)):
            for j in range(i + 1, len(valid_entries)):
                dist = space.distance(valid_entries[i][1], valid_entries[j][1])
                # Assume background stats are available from index
                z_score = self._compute_z_score(dist, space)
                
                if z_score >= self.z_threshold:
                    edges.append((i, j, dist, z_score))
        
        # Greedy clustering
        clusters = []
        assigned = set()
        
        # Sort edges by z-score (highest first)
        edges.sort(key=lambda x: -x[3])
        
        for i, j, dist, z in edges:
            if i in assigned and j in assigned:
                continue
            
            # Find or create cluster
            cluster_idx = None
            
            # Check if either node is already in a cluster
            for idx, cluster in enumerate(clusters):
                if i in cluster['members'] or j in cluster['members']:
                    cluster_idx = idx
                    break
            
            if cluster_idx is None:
                # Create new cluster
                cluster = {
                    'members': set(),
                    'prototype': None,
                    'avg_z_score': 0.0,
                    'function_ids': []
                }
                clusters.append(cluster)
                cluster_idx = len(clusters) - 1
            else:
                cluster = clusters[cluster_idx]
            
            # Add members
            if i not in assigned:
                cluster['members'].add(i)
                assigned.add(i)
            if j not in assigned:
                cluster['members'].add(j)
                assigned.add(j)
        
        # Post-process clusters
        final_clusters = []
        for cluster in clusters:
            if len(cluster['members']) >= min_cluster_size:
                # Get function IDs
                func_ids = [valid_entries[i][0] for i in cluster['members']]
                
                # Compute prototype (centroid)
                hvs = [valid_entries[i][1] for i in cluster['members']]
                prototype = space.bundle(hvs)
                
                # Compute average intra-cluster z-score
                z_scores = []
                members = list(cluster['members'])
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        dist = space.distance(
                            valid_entries[members[i]][1],
                            valid_entries[members[j]][1]
                        )
                        z = self._compute_z_score(dist, space)
                        z_scores.append(z)
                
                avg_z = sum(z_scores) / len(z_scores) if z_scores else 0.0
                
                final_clusters.append({
                    'id': len(final_clusters),
                    'size': len(func_ids),
                    'functions': func_ids,
                    'prototype': prototype,
                    'avg_z_score': avg_z,
                    'cohesion': self._compute_cohesion(hvs, space)
                })
        
        return final_clusters
    
    def _compute_z_score(self, distance: float, space) -> float:
        """Compute z-score (would get stats from index in real implementation)."""
        # Placeholder - in real implementation, get from index
        mean, std = 0.5, 0.05
        return (mean - distance) / std
    
    def _compute_cohesion(self, hvs: List[np.ndarray], space) -> float:
        """
        Compute cluster cohesion metric.
        
        Returns average similarity within cluster.
        """
        if len(hvs) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                sim = space.similarity(hvs[i], hvs[j])
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def analyze_channel_contributions(
        self,
        hv1: np.ndarray,
        hv2: np.ndarray,
        space,
        features1: Dict[str, List[str]],
        features2: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Estimate channel contributions to similarity.
        
        This is approximate since we can't perfectly decompose bound vectors,
        but we can estimate by comparing channel token overlap.
        """
        contributions = {}
        
        all_channels = set(features1.keys()) | set(features2.keys())
        
        for channel in all_channels:
            tokens1 = set(features1.get(channel, []))
            tokens2 = set(features2.get(channel, []))
            
            if not tokens1 and not tokens2:
                contributions[channel] = 0.0
                continue
            
            # Jaccard similarity of tokens
            intersection = tokens1 & tokens2
            union = tokens1 | tokens2
            
            if union:
                jaccard = len(intersection) / len(union)
            else:
                jaccard = 0.0
            
            # Weight by channel presence
            presence = (len(tokens1) + len(tokens2)) / 2.0
            contributions[channel] = jaccard * min(presence / 10, 1.0)
        
        # Normalize
        total = sum(contributions.values())
        if total > 0:
            for channel in contributions:
                contributions[channel] /= total
        
        return contributions