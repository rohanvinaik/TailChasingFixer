"""
Chromatin-inspired performance visualization and reporting.

This module provides tools for visualizing performance bottlenecks using
Hi-C style contact matrices, TAD boundary detection, and polymer physics
concepts from chromatin organization.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class TAD:
    """
    Topologically Associating Domain - represents a functional module boundary.
    
    TADs are regions that interact more frequently within themselves than with
    other regions, analogous to code modules with high internal cohesion.
    """
    
    start: int
    end: int
    name: str
    activity_level: float = 0.5  # 0.0 (inactive) to 1.0 (highly active)
    
    def size(self) -> int:
        """Calculate TAD size."""
        return self.end - self.start
    
    def contains(self, position: int) -> bool:
        """Check if position is within TAD boundaries."""
        return self.start <= position <= self.end


@dataclass
class ThrashCluster:
    """
    Represents a cluster of thrashing positions in the performance landscape.
    
    Similar to chromatin loops that bring distant regions together, thrash
    clusters represent code regions that frequently interact during performance
    bottlenecks.
    """
    
    positions: List[int]
    risk_score: float  # 0.0 to 1.0
    frequency: int
    avg_latency: float
    
    def center(self) -> float:
        """Calculate cluster center position."""
        if not self.positions:
            return 0.0
        return sum(self.positions) / len(self.positions)


class HiCHeatmapGenerator:
    """
    Generates Hi-C style heatmaps for performance contact matrices.
    
    Hi-C (High-throughput Chromatin Conformation Capture) is a technique
    for studying 3D genome organization. We adapt this to visualize
    performance interaction patterns.
    """
    
    def __init__(self):
        self.console = Console()
        # Unicode blocks for different intensity levels
        self.blocks = [' ', '░', '▒', '▓', '█']
        
    def generate_contact_heatmap(
        self,
        contact_matrix: np.ndarray,
        tads: Optional[List[TAD]] = None,
        title: str = "Performance Contact Matrix"
    ) -> str:
        """
        Generate ASCII/Unicode heatmap from contact matrix.
        
        Args:
            contact_matrix: Square matrix of contact frequencies
            tads: Optional list of TAD boundaries to overlay
            title: Title for the heatmap
            
        Returns:
            String representation of the heatmap
        """
        if contact_matrix.size == 0:
            return "Empty contact matrix"
            
        # Normalize matrix to 0-1 range
        normalized = self._normalize_matrix(contact_matrix)
        
        # Convert to ASCII blocks
        ascii_matrix = self._matrix_to_ascii(normalized)
        
        # Build output
        lines = [title, "=" * len(title)]
        
        # Add matrix with TAD boundaries if provided
        if tads:
            lines.append(self._add_tad_boundaries(ascii_matrix, tads))
        else:
            for row in ascii_matrix:
                lines.append(''.join(row))
                
        # Add scale legend
        lines.append("")
        lines.append("Scale: " + ' '.join([f"{b}={i*0.25:.2f}" for i, b in enumerate(self.blocks)]))
        
        return '\n'.join(lines)
    
    def highlight_thrash_clusters(
        self,
        matrix: np.ndarray,
        risk_scores: Dict[Tuple[int, int], float]
    ) -> str:
        """
        Highlight high-risk thrashing clusters in the matrix.
        
        Args:
            matrix: Contact matrix
            risk_scores: Dictionary of (i, j) -> risk score
            
        Returns:
            Highlighted matrix visualization
        """
        if matrix.size == 0:
            return "Empty matrix"
            
        normalized = self._normalize_matrix(matrix)
        ascii_matrix = self._matrix_to_ascii(normalized)
        
        # Overlay risk scores with colored markers
        for (i, j), risk in risk_scores.items():
            if 0 <= i < len(ascii_matrix) and 0 <= j < len(ascii_matrix[0]):
                if risk > 0.8:
                    ascii_matrix[i][j] = '●'  # High risk
                elif risk > 0.5:
                    ascii_matrix[i][j] = '◐'  # Medium risk
                elif risk > 0.2:
                    ascii_matrix[i][j] = '○'  # Low risk
                    
        lines = ["Thrash Risk Clusters", "=" * 20]
        for row in ascii_matrix:
            lines.append(''.join(row))
            
        lines.append("")
        lines.append("Risk indicators: ○=Low ◐=Medium ●=High")
        
        return '\n'.join(lines)
    
    def show_tad_boundaries(
        self,
        matrix: np.ndarray,
        tad_map: Dict[str, TAD]
    ) -> str:
        """
        Visualize TAD boundaries on the contact matrix.
        
        Args:
            matrix: Contact matrix
            tad_map: Dictionary of TAD name -> TAD object
            
        Returns:
            Matrix with TAD boundaries highlighted
        """
        if matrix.size == 0:
            return "Empty matrix"
            
        normalized = self._normalize_matrix(matrix)
        ascii_matrix = self._matrix_to_ascii(normalized)
        
        # Create boundary overlay
        lines = ["TAD Boundary Visualization", "=" * 25]
        
        # Add TAD labels on top
        tad_labels = []
        for name, tad in tad_map.items():
            tad_labels.append(f"{name[:10]:^10}")
        lines.append("TADs: " + " | ".join(tad_labels))
        lines.append("-" * 25)
        
        # Add matrix
        for row in ascii_matrix:
            lines.append(''.join(row))
            
        # Add TAD activity levels
        lines.append("")
        lines.append("Activity levels:")
        for name, tad in tad_map.items():
            activity_bar = '█' * int(tad.activity_level * 10)
            lines.append(f"  {name}: {activity_bar} ({tad.activity_level:.2f})")
            
        return '\n'.join(lines)
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix values to 0-1 range."""
        if matrix.size == 0:
            return matrix
            
        min_val = matrix.min()
        max_val = matrix.max()
        
        if max_val == min_val:
            return np.zeros_like(matrix)
            
        return (matrix - min_val) / (max_val - min_val)
    
    def _matrix_to_ascii(self, normalized: np.ndarray) -> List[List[str]]:
        """Convert normalized matrix to ASCII block characters."""
        ascii_matrix = []
        
        for row in normalized:
            ascii_row = []
            for val in row:
                # Map value to block character
                idx = min(int(val * (len(self.blocks) - 1)), len(self.blocks) - 1)
                ascii_row.append(self.blocks[idx])
            ascii_matrix.append(ascii_row)
            
        return ascii_matrix
    
    def _add_tad_boundaries(self, ascii_matrix: List[List[str]], tads: List[TAD]) -> str:
        """Add TAD boundary markers to the matrix."""
        # This would add visual markers for TAD boundaries
        # For simplicity, returning the basic matrix
        lines = []
        for row in ascii_matrix:
            lines.append(''.join(row))
        return '\n'.join(lines)


class PolymerMetricsReport:
    """
    Generate polymer physics-based metrics for performance analysis.
    
    Uses concepts from polymer physics and chromatin organization to
    quantify performance characteristics.
    """
    
    def __init__(self, alpha: float = 1.2):
        """
        Initialize with polymer physics parameters.
        
        Args:
            alpha: Contact probability decay exponent (typically 1.0-1.5)
        """
        self.alpha = alpha
        self.console = Console()
    
    def calculate_polymer_distances(
        self,
        tads: List[TAD],
        interactions: List[Tuple[int, int, float]]
    ) -> Dict[str, Any]:
        """
        Calculate polymer-based distance metrics.
        
        Args:
            tads: List of TAD regions
            interactions: List of (pos1, pos2, strength) interactions
            
        Returns:
            Dictionary of distance metrics
        """
        metrics = {
            "intra_tad_distances": {},
            "inter_tad_distances": {},
            "global_metrics": {}
        }
        
        # Categorize interactions by TAD membership
        for tad in tads:
            intra_interactions = []
            
            for pos1, pos2, strength in interactions:
                if tad.contains(pos1) and tad.contains(pos2):
                    distance = abs(pos2 - pos1)
                    intra_interactions.append((distance, strength))
            
            if intra_interactions:
                distances = [d for d, _ in intra_interactions]
                strengths = [s for _, s in intra_interactions]
                
                metrics["intra_tad_distances"][tad.name] = {
                    "mean": np.mean(distances),
                    "std": np.std(distances),
                    "median": np.median(distances),
                    "mean_strength": np.mean(strengths),
                    "count": len(intra_interactions)
                }
        
        # Calculate inter-TAD distances
        for i, tad1 in enumerate(tads):
            for j, tad2 in enumerate(tads[i+1:], i+1):
                inter_interactions = []
                
                for pos1, pos2, strength in interactions:
                    if (tad1.contains(pos1) and tad2.contains(pos2)) or \
                       (tad2.contains(pos1) and tad1.contains(pos2)):
                        distance = abs(pos2 - pos1)
                        inter_interactions.append((distance, strength))
                
                if inter_interactions:
                    distances = [d for d, _ in inter_interactions]
                    strengths = [s for _, s in inter_interactions]
                    
                    pair_name = f"{tad1.name}-{tad2.name}"
                    metrics["inter_tad_distances"][pair_name] = {
                        "mean": np.mean(distances),
                        "std": np.std(distances),
                        "median": np.median(distances),
                        "mean_strength": np.mean(strengths),
                        "count": len(inter_interactions)
                    }
        
        # Global metrics
        if interactions:
            all_distances = [abs(p2 - p1) for p1, p2, _ in interactions]
            all_strengths = [s for _, _, s in interactions]
            
            metrics["global_metrics"] = {
                "mean_distance": np.mean(all_distances),
                "std_distance": np.std(all_distances),
                "mean_strength": np.mean(all_strengths),
                "total_interactions": len(interactions)
            }
        
        return metrics
    
    def calculate_contact_probabilities(
        self,
        interactions: List[Tuple[int, int, float]],
        max_distance: int = 1000
    ) -> Dict[str, Any]:
        """
        Calculate contact probability decay with distance.
        
        Uses power-law decay: P(d) = d^(-alpha)
        
        Args:
            interactions: List of interactions
            max_distance: Maximum distance to consider
            
        Returns:
            Contact probability distribution
        """
        # Bin interactions by distance
        distance_bins = np.logspace(0, np.log10(max_distance), 20)
        bin_counts = np.zeros(len(distance_bins) - 1)
        bin_strengths = np.zeros(len(distance_bins) - 1)
        
        for pos1, pos2, strength in interactions:
            distance = abs(pos2 - pos1)
            if distance > 0:
                bin_idx = np.searchsorted(distance_bins, distance) - 1
                if 0 <= bin_idx < len(bin_counts):
                    bin_counts[bin_idx] += 1
                    bin_strengths[bin_idx] += strength
        
        # Calculate probabilities
        probabilities = []
        for i in range(len(bin_counts)):
            if bin_counts[i] > 0:
                prob = bin_strengths[i] / bin_counts[i]
            else:
                prob = 0
            probabilities.append(prob)
        
        # Fit power law
        valid_indices = [i for i, c in enumerate(bin_counts) if c > 0]
        if valid_indices:
            distances = [(distance_bins[i] + distance_bins[i+1])/2 for i in valid_indices]
            probs = [probabilities[i] for i in valid_indices]
            
            # Calculate decay rate (simplified)
            if len(distances) > 1:
                log_distances = np.log(distances)
                log_probs = np.log([p + 1e-10 for p in probs])  # Avoid log(0)
                decay_rate = -np.polyfit(log_distances, log_probs, 1)[0]
            else:
                decay_rate = self.alpha
        else:
            decay_rate = self.alpha
        
        return {
            "distance_bins": distance_bins.tolist(),
            "probabilities": probabilities,
            "decay_rate": decay_rate,
            "statistics": {
                "mean_contact_distance": np.mean([abs(p2-p1) for p1, p2, _ in interactions]) if interactions else 0,
                "contact_decay_rate": decay_rate,
                "short_range_fraction": sum(1 for p1, p2, _ in interactions if abs(p2-p1) < 50) / len(interactions) if interactions else 0,
                "long_range_fraction": sum(1 for p1, p2, _ in interactions if abs(p2-p1) > 200) / len(interactions) if interactions else 0
            }
        }
    
    def predict_thrash_reduction(
        self,
        fix_strategies: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict thrash reduction for different fix strategies.
        
        Args:
            fix_strategies: List of proposed fixes with impact scores
            
        Returns:
            Predictions for each strategy
        """
        predictions = {}
        
        for strategy in fix_strategies:
            name = strategy.get("name", "unknown")
            impact = strategy.get("impact_score", 0.5)
            complexity = strategy.get("complexity", 0.5)
            confidence = strategy.get("confidence", 0.5)
            
            # Polymer physics-inspired prediction model
            # Higher impact, lower complexity = better reduction
            reduction = impact * (1 - complexity * 0.5) * confidence
            
            # Risk assessment
            implementation_risk = complexity * (1 - confidence)
            
            # ROI calculation
            roi = (reduction / (complexity + 0.1)) * confidence
            
            predictions[name] = {
                "estimated_reduction": reduction,
                "implementation_risk": implementation_risk,
                "roi_score": roi,
                "recommended_priority": self._get_priority(roi, implementation_risk)
            }
        
        return predictions
    
    def _get_priority(self, roi: float, risk: float) -> str:
        """Determine priority based on ROI and risk."""
        if roi > 2.0 and risk < 0.3:
            return "High"
        elif roi > 1.0 or risk < 0.5:
            return "Medium"
        else:
            return "Low"
    
    def visualize_replication_timing(
        self,
        timeline_data: List[Dict[str, Any]]
    ) -> str:
        """
        Visualize replication timing patterns.
        
        In chromatin biology, replication timing correlates with activity.
        We use this concept to visualize execution timeline patterns.
        
        Args:
            timeline_data: List of timeline events
            
        Returns:
            ASCII visualization of timing
        """
        if not timeline_data:
            return "No timeline data available"
        
        lines = ["Replication Timing Schedule", "=" * 27]
        
        # Sort by timestamp
        sorted_timeline = sorted(timeline_data, key=lambda x: x.get("timestamp", 0))
        
        # Create timeline visualization
        max_time = max(event.get("timestamp", 0) + event.get("duration", 0) 
                      for event in sorted_timeline)
        
        for event in sorted_timeline:
            name = event.get("name", "unknown")[:20]
            start = event.get("timestamp", 0)
            duration = event.get("duration", 0)
            impact = event.get("impact", 0.5)
            status = event.get("status", "unknown")
            
            # Create visual bar
            start_pos = int((start / max_time) * 50) if max_time > 0 else 0
            duration_len = max(1, int((duration / max_time) * 50)) if max_time > 0 else 1
            
            bar = ' ' * start_pos + '█' * duration_len
            
            # Color coding based on impact
            if impact > 0.7:
                indicator = "🔴"  # High impact
            elif impact > 0.4:
                indicator = "🟡"  # Medium impact
            else:
                indicator = "🟢"  # Low impact
            
            lines.append(f"{name:20} {indicator} {bar}")
            lines.append(f"{'':20}     {start:.1f}ms - {start+duration:.1f}ms ({status})")
            lines.append("")
        
        # Add scale
        lines.append("Time scale: " + ''.join([str(i%10) if i%5==0 else '-' for i in range(51)]))
        lines.append(f"            0ms" + " " * 43 + f"{max_time:.1f}ms")
        
        return '\n'.join(lines)
    
    def generate_comprehensive_report(
        self,
        tads: List[TAD],
        interactions: List[Tuple[int, int, float]],
        fix_strategies: List[Dict[str, Any]],
        timeline_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive polymer physics report.
        
        Args:
            tads: TAD regions
            interactions: Interaction data
            fix_strategies: Proposed fixes
            timeline_data: Execution timeline
            
        Returns:
            Comprehensive report dictionary
        """
        report = {}
        
        # Calculate all metrics
        report["polymer_distances"] = self.calculate_polymer_distances(tads, interactions)
        report["contact_probabilities"] = self.calculate_contact_probabilities(interactions)
        report["thrash_predictions"] = self.predict_thrash_reduction(fix_strategies)
        report["timeline_analysis"] = {
            "visualization": self.visualize_replication_timing(timeline_data),
            "events": timeline_data
        }
        
        # Summary metrics
        polymer_metrics = report["polymer_distances"]
        contact_stats = report["contact_probabilities"]["statistics"]
        predictions = report["thrash_predictions"]
        
        # Calculate overall health score
        health_score = self._calculate_health_score(polymer_metrics, predictions)
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(predictions)
        
        report["summary_metrics"] = {
            "overall_health_score": health_score,
            "optimization_potential": optimization_potential,
            "mean_contact_distance": contact_stats.get("mean_contact_distance", 0),
            "contact_decay_rate": contact_stats.get("contact_decay_rate", self.alpha),
            "total_tads": len(tads),
            "total_interactions": len(interactions),
            "proposed_fixes": len(fix_strategies),
            "stability_index": self._calculate_stability_index(polymer_metrics)
        }
        
        return report
    
    def _calculate_health_score(
        self,
        polymer_metrics: Dict[str, Any],
        predictions: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate overall system health score (0-1)."""
        # Base score from polymer metrics
        global_metrics = polymer_metrics.get("global_metrics", {})
        
        if not global_metrics:
            base_score = 0.5
        else:
            # Lower mean distance = better
            mean_dist = global_metrics.get("mean_distance", 100)
            distance_score = 1.0 / (1 + mean_dist / 100)
            
            # Higher mean strength = better
            mean_strength = global_metrics.get("mean_strength", 0.5)
            
            base_score = (distance_score + mean_strength) / 2
        
        # Adjust based on improvement potential
        if predictions:
            max_reduction = max(p.get("estimated_reduction", 0) for p in predictions.values())
            # If high reduction possible, current health is lower
            health_score = base_score * (1 - max_reduction * 0.3)
        else:
            health_score = base_score
        
        return max(0.0, min(1.0, health_score))
    
    def _calculate_optimization_potential(
        self,
        predictions: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate overall optimization potential (0-1)."""
        if not predictions:
            return 0.0
        
        # Average of all ROI scores
        roi_scores = [p.get("roi_score", 0) for p in predictions.values()]
        avg_roi = np.mean(roi_scores)
        
        # Normalize to 0-1 range
        return min(1.0, avg_roi / 3.0)  # ROI of 3.0 maps to 1.0
    
    def _calculate_stability_index(
        self,
        polymer_metrics: Dict[str, Any]
    ) -> float:
        """Calculate system stability index based on variance metrics."""
        intra_tad = polymer_metrics.get("intra_tad_distances", {})
        
        if not intra_tad:
            return 0.5
        
        # Lower variance = more stable
        variances = []
        for tad_metrics in intra_tad.values():
            std = tad_metrics.get("std", 0)
            mean = tad_metrics.get("mean", 1)
            if mean > 0:
                cv = std / mean  # Coefficient of variation
                variances.append(cv)
        
        if variances:
            avg_cv = np.mean(variances)
            # Lower CV = higher stability
            stability = 1.0 / (1 + avg_cv)
        else:
            stability = 0.5
        
        return max(0.0, min(1.0, stability))


def integrate_chromatin_analysis(
    existing_report: Dict[str, Any],
    contact_matrix: np.ndarray,
    tads: List[TAD],
    interactions: List[Tuple[int, int, float]],
    fix_strategies: List[Dict[str, Any]],
    timeline_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Integrate chromatin analysis into existing performance reports.
    
    Args:
        existing_report: Current performance report
        contact_matrix: Interaction matrix
        tads: TAD regions
        interactions: Interaction data
        fix_strategies: Proposed fixes
        timeline_data: Timeline events
        
    Returns:
        Enhanced report with chromatin analysis
    """
    # Create generators
    heatmap_gen = HiCHeatmapGenerator()
    metrics_reporter = PolymerMetricsReport()
    
    # Generate chromatin analysis
    chromatin_analysis = {
        "contact_matrix_summary": {
            "dimensions": contact_matrix.shape,
            "total_contacts": int(np.sum(contact_matrix)),
            "max_contact_strength": float(np.max(contact_matrix)),
            "mean_contact_strength": float(np.mean(contact_matrix))
        },
        "tad_analysis": {
            "total_tads": len(tads),
            "average_tad_size": np.mean([tad.size() for tad in tads]),
            "tad_details": [
                {
                    "name": tad.name,
                    "size": tad.size(),
                    "activity": tad.activity_level
                }
                for tad in tads
            ]
        },
        "polymer_metrics": metrics_reporter.calculate_polymer_distances(tads, interactions),
        "contact_probabilities": metrics_reporter.calculate_contact_probabilities(interactions),
        "thrash_predictions": metrics_reporter.predict_thrash_reduction(fix_strategies),
        "visualization_data": {
            "heatmap": heatmap_gen.generate_contact_heatmap(contact_matrix, tads),
            "timeline": metrics_reporter.visualize_replication_timing(timeline_data)
        }
    }
    
    # Calculate risk analysis
    risk_positions = []
    for i in range(contact_matrix.shape[0]):
        for j in range(contact_matrix.shape[1]):
            if contact_matrix[i, j] > np.median(contact_matrix) * 1.5:
                risk_positions.append((i, j))
    
    chromatin_analysis["risk_analysis"] = {
        "high_risk_positions": len(risk_positions),
        "risk_distribution": {
            "low": sum(1 for v in contact_matrix.flatten() if v < np.percentile(contact_matrix, 33)),
            "medium": sum(1 for v in contact_matrix.flatten() if np.percentile(contact_matrix, 33) <= v < np.percentile(contact_matrix, 67)),
            "high": sum(1 for v in contact_matrix.flatten() if v >= np.percentile(contact_matrix, 67))
        },
        "average_risk_score": float(np.mean(contact_matrix))
    }
    
    # Merge with existing report
    enhanced_report = existing_report.copy()
    enhanced_report["chromatin_analysis"] = chromatin_analysis
    
    return enhanced_report


def generate_comparative_matrices(
    before_matrix: np.ndarray,
    after_matrix: np.ndarray,
    tads: List[TAD],
    strategy_name: str
) -> Dict[str, Any]:
    """
    Generate comparative analysis between before/after matrices.
    
    Args:
        before_matrix: Original contact matrix
        after_matrix: Optimized contact matrix
        tads: TAD regions
        strategy_name: Name of optimization strategy
        
    Returns:
        Comparative analysis report
    """
    generator = HiCHeatmapGenerator()
    
    # Calculate difference matrix
    diff_matrix = before_matrix - after_matrix
    
    # Calculate metrics
    total_before = np.sum(before_matrix)
    total_after = np.sum(after_matrix)
    reduction = total_before - total_after
    reduction_pct = (reduction / total_before * 100) if total_before > 0 else 0
    
    # TAD-specific improvements
    tad_improvements = {}
    for tad in tads:
        # Calculate contacts within TAD region
        tad_range = range(
            int(tad.start * before_matrix.shape[0] / 100),
            min(int(tad.end * before_matrix.shape[0] / 100), before_matrix.shape[0])
        )
        
        if tad_range:
            before_contacts = sum(
                before_matrix[i, j]
                for i in tad_range
                for j in tad_range
            )
            after_contacts = sum(
                after_matrix[i, j]
                for i in tad_range
                for j in tad_range
            )
            
            tad_improvements[tad.name] = {
                "contacts_before": int(before_contacts),
                "contacts_after": int(after_contacts),
                "reduction": int(before_contacts - after_contacts),
                "reduction_percentage": ((before_contacts - after_contacts) / before_contacts * 100) if before_contacts > 0 else 0
            }
    
    return {
        "strategy_name": strategy_name,
        "metrics": {
            "total_contacts_before": int(total_before),
            "total_contacts_after": int(total_after),
            "absolute_reduction": int(reduction),
            "reduction_percentage": reduction_pct,
            "improvement_score": min(1.0, reduction_pct / 30)  # 30% reduction = perfect score
        },
        "visualizations": {
            "before_heatmap": generator.generate_contact_heatmap(before_matrix, tads, "Before Optimization"),
            "after_heatmap": generator.generate_contact_heatmap(after_matrix, tads, "After Optimization"),
            "difference_heatmap": generator.generate_contact_heatmap(np.abs(diff_matrix), tads, "Change Map")
        },
        "tad_specific_improvements": tad_improvements
    }