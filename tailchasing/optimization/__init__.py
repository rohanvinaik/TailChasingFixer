"""
Optimization module for tail-chasing detection.

This module provides advanced optimization techniques that break traditional
analysis barriers, including cluster-based analysis that avoids dependency sorting.
"""

from .cluster_engine import ClusterBasedAnalyzer, CodeCluster, InfluentialPattern

__all__ = [
    'ClusterBasedAnalyzer',
    'CodeCluster', 
    'InfluentialPattern'
]