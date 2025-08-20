"""
Enhanced Metrics - Advanced metrics for measuring code quality improvements.

This module provides sophisticated metrics for estimating fix impact,
measuring module coupling, and tracking pattern evolution over time.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


@dataclass
class FixImpactMetrics:
    """Metrics estimating the impact of applying fixes."""
    
    estimated_lines_removed: int
    estimated_lines_added: int
    net_line_reduction: int
    files_affected: int
    modules_affected: int
    complexity_reduction: float
    maintainability_increase: float
    test_coverage_impact: float
    refactoring_effort_hours: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'


@dataclass
class ModuleCouplingMetrics:
    """Metrics measuring coupling between modules."""
    
    module_name: str
    afferent_coupling: int  # Number of modules depending on this module
    efferent_coupling: int  # Number of modules this module depends on
    instability: float  # I = Ce / (Ca + Ce)
    abstractness: float  # Ratio of abstract types to total types
    distance_from_main_sequence: float  # |A + I - 1|
    coupling_score: float  # Overall coupling metric
    highly_coupled_with: List[Tuple[str, int]]  # (module, coupling_strength)


@dataclass 
class PatternEvolutionMetrics:
    """Metrics tracking how patterns evolve over time."""
    
    pattern_id: str
    pattern_type: str
    first_occurrence: datetime
    last_occurrence: datetime
    growth_rate: float  # Occurrences per day
    spread_velocity: float  # New files affected per day
    mutation_count: int  # Number of variations
    stability_score: float  # How consistent the pattern is
    predicted_occurrences: int  # Predicted future occurrences
    lifecycle_stage: str  # 'emerging', 'growing', 'stable', 'declining'


class EnhancedMetricsCalculator:
    """
    Calculates advanced metrics for code quality analysis.
    
    Provides:
    1. Fix impact estimation
    2. Module coupling analysis
    3. Pattern evolution tracking
    4. Code reduction forecasting
    5. Refactoring effort estimation
    """
    
    def __init__(self):
        self.module_graph = nx.DiGraph()
        self.pattern_history: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        self.module_metrics: Dict[str, ModuleCouplingMetrics] = {}
        
    def calculate_fix_impact(self, issues: List, context) -> FixImpactMetrics:
        """
        Calculate the estimated impact of fixing detected issues.
        
        Args:
            issues: List of detected issues
            context: Analysis context with code information
            
        Returns:
            FixImpactMetrics with estimated impacts
        """
        lines_removed = 0
        lines_added = 0
        files_affected = set()
        modules_affected = set()
        complexity_reduction = 0.0
        
        for issue in issues:
            # Estimate based on issue type
            if issue.type == 'duplicate_function':
                # Duplicates can be mostly removed
                lines_removed += self._estimate_function_lines(issue) * (issue.details.get('count', 2) - 1)
                lines_added += 5  # Import statement
                complexity_reduction += 0.1
                
            elif issue.type == 'circular_import':
                # Circular imports require refactoring
                lines_removed += 10
                lines_added += 20  # New abstraction layer
                complexity_reduction += 0.2
                
            elif issue.type == 'phantom_function':
                # Phantom functions need implementation
                lines_removed += 2  # Remove stub
                lines_added += 15  # Actual implementation
                
            elif issue.type == 'missing_symbol':
                # Missing symbols need implementation
                lines_added += 10
                
            elif issue.type == 'semantic_duplicate_function':
                # Semantic duplicates harder to merge
                dup_count = issue.details.get('duplicate_count', 2)
                lines_removed += self._estimate_function_lines(issue) * (dup_count - 1) * 0.7
                lines_added += 10  # Unified implementation
                complexity_reduction += 0.15
                
            elif 'base_class_extraction' in issue.type:
                # Base class extraction
                lines_removed += issue.details.get('estimated_line_reduction', 50)
                lines_added += 30  # Base class implementation
                complexity_reduction += 0.3
            
            # Track affected files and modules
            if issue.file_path and issue.file_path != '<global>':
                files_affected.add(issue.file_path)
                module = self._get_module_from_path(issue.file_path)
                if module:
                    modules_affected.add(module)
        
        # Calculate net reduction
        net_reduction = lines_removed - lines_added
        
        # Estimate refactoring effort
        effort_hours = self._estimate_refactoring_effort(
            lines_removed + lines_added,
            len(files_affected),
            complexity_reduction
        )
        
        # Calculate maintainability increase (simplified model)
        maintainability_increase = (
            0.3 * (net_reduction / max(1000, net_reduction)) +  # Line reduction factor
            0.3 * complexity_reduction +  # Complexity factor
            0.2 * (1.0 - len(modules_affected) / max(10, len(modules_affected))) +  # Module impact
            0.2 * (len(issues) / max(100, len(issues)))  # Issue resolution factor
        )
        
        # Estimate test coverage impact
        test_coverage_impact = -0.1 if net_reduction > 100 else 0.05
        
        # Determine risk level
        if effort_hours > 40 or len(modules_affected) > 10:
            risk_level = 'HIGH'
        elif effort_hours > 16 or len(modules_affected) > 5:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return FixImpactMetrics(
            estimated_lines_removed=lines_removed,
            estimated_lines_added=lines_added,
            net_line_reduction=net_reduction,
            files_affected=len(files_affected),
            modules_affected=len(modules_affected),
            complexity_reduction=min(1.0, complexity_reduction),
            maintainability_increase=min(1.0, maintainability_increase),
            test_coverage_impact=test_coverage_impact,
            refactoring_effort_hours=effort_hours,
            risk_level=risk_level
        )
    
    def calculate_module_coupling(self, context) -> Dict[str, ModuleCouplingMetrics]:
        """
        Calculate coupling metrics for all modules.
        
        Args:
            context: Analysis context with import information
            
        Returns:
            Dictionary of module coupling metrics
        """
        # Build module dependency graph
        self._build_module_graph(context)
        
        # Calculate metrics for each module
        for module in self.module_graph.nodes():
            # Afferent coupling (incoming edges)
            ca = self.module_graph.in_degree(module)
            
            # Efferent coupling (outgoing edges)
            ce = self.module_graph.out_degree(module)
            
            # Instability metric
            instability = ce / max(1, ca + ce)
            
            # Abstractness (simplified - based on module name patterns)
            abstractness = self._estimate_abstractness(module, context)
            
            # Distance from main sequence
            distance = abs(abstractness + instability - 1)
            
            # Overall coupling score (lower is better)
            coupling_score = (
                0.3 * (ce / max(1, ce)) +  # Normalized efferent
                0.2 * (ca / max(1, ca)) +  # Normalized afferent
                0.3 * instability +
                0.2 * distance
            )
            
            # Find highly coupled modules
            coupled_modules = []
            for neighbor in self.module_graph.neighbors(module):
                weight = self.module_graph[module][neighbor].get('weight', 1)
                coupled_modules.append((neighbor, weight))
            coupled_modules.sort(key=lambda x: x[1], reverse=True)
            
            self.module_metrics[module] = ModuleCouplingMetrics(
                module_name=module,
                afferent_coupling=ca,
                efferent_coupling=ce,
                instability=instability,
                abstractness=abstractness,
                distance_from_main_sequence=distance,
                coupling_score=coupling_score,
                highly_coupled_with=coupled_modules[:5]
            )
        
        return self.module_metrics
    
    def track_pattern_evolution(self, patterns: List, timestamp: Optional[datetime] = None) -> List[PatternEvolutionMetrics]:
        """
        Track how patterns evolve over time.
        
        Args:
            patterns: List of detected patterns
            timestamp: Timestamp of detection (default: now)
            
        Returns:
            List of pattern evolution metrics
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        evolution_metrics = []
        
        for pattern in patterns:
            pattern_id = self._get_pattern_id(pattern)
            
            # Record occurrence
            self.pattern_history[pattern_id].append((timestamp, pattern.get('count', 1)))
            
            # Calculate metrics if enough history
            history = self.pattern_history[pattern_id]
            if len(history) >= 2:
                metrics = self._calculate_evolution_metrics(pattern_id, pattern.get('type', 'unknown'), history)
                evolution_metrics.append(metrics)
        
        return evolution_metrics
    
    def _estimate_function_lines(self, issue) -> int:
        """Estimate lines of code for a function."""
        # Simple heuristic based on issue details
        if 'line_count' in issue.details:
            return issue.details['line_count']
        elif 'complexity' in issue.details:
            return issue.details['complexity'] * 5  # Rough estimate
        else:
            return 10  # Default estimate
    
    def _get_module_from_path(self, file_path: str) -> Optional[str]:
        """Extract module name from file path."""
        if file_path.startswith('<'):
            return None
        
        path = Path(file_path)
        parts = []
        
        for part in path.parts[:-1]:
            if part not in ('.', '..', '', '__pycache__'):
                parts.append(part)
        
        if path.stem != '__init__':
            parts.append(path.stem)
        
        return '.'.join(parts) if parts else None
    
    def _estimate_refactoring_effort(self, total_lines: int, file_count: int, complexity: float) -> float:
        """Estimate refactoring effort in hours."""
        # Base effort: 100 lines per hour
        base_hours = total_lines / 100
        
        # File overhead: 0.5 hours per file
        file_hours = file_count * 0.5
        
        # Complexity multiplier
        complexity_multiplier = 1 + complexity
        
        return (base_hours + file_hours) * complexity_multiplier
    
    def _build_module_graph(self, context):
        """Build directed graph of module dependencies."""
        self.module_graph.clear()
        
        # Add edges from import information
        if hasattr(context, 'import_graph'):
            for from_module, to_modules in context.import_graph.items():
                from_name = self._get_module_from_path(from_module)
                if from_name:
                    for to_module in to_modules:
                        to_name = self._get_module_from_path(to_module)
                        if to_name and from_name != to_name:
                            if self.module_graph.has_edge(from_name, to_name):
                                # Increase weight for multiple imports
                                self.module_graph[from_name][to_name]['weight'] += 1
                            else:
                                self.module_graph.add_edge(from_name, to_name, weight=1)
    
    def _estimate_abstractness(self, module: str, context) -> float:
        """Estimate abstractness of a module."""
        # Simple heuristic based on module name and content
        abstract_indicators = ['base', 'abstract', 'interface', 'protocol', 'abc']
        concrete_indicators = ['impl', 'concrete', 'handler', 'service', 'controller']
        
        module_lower = module.lower()
        
        # Check for abstract indicators
        abstract_score = sum(1 for ind in abstract_indicators if ind in module_lower)
        concrete_score = sum(1 for ind in concrete_indicators if ind in module_lower)
        
        if abstract_score > concrete_score:
            return 0.8
        elif concrete_score > abstract_score:
            return 0.2
        else:
            return 0.5  # Neutral
    
    def _get_pattern_id(self, pattern) -> str:
        """Generate unique ID for a pattern."""
        if hasattr(pattern, 'pattern_hash'):
            return pattern.pattern_hash
        elif isinstance(pattern, dict) and 'hash' in pattern:
            return pattern['hash']
        else:
            # Generate from pattern characteristics
            import hashlib
            pattern_str = str(pattern)
            return hashlib.md5(pattern_str.encode()).hexdigest()[:8]
    
    def _calculate_evolution_metrics(self, pattern_id: str, pattern_type: str, history: List[Tuple[datetime, int]]) -> PatternEvolutionMetrics:
        """Calculate evolution metrics from pattern history."""
        first_occurrence = history[0][0]
        last_occurrence = history[-1][0]
        
        # Calculate growth rate
        time_span = (last_occurrence - first_occurrence).days or 1
        total_occurrences = sum(count for _, count in history)
        growth_rate = total_occurrences / time_span
        
        # Calculate spread velocity (simplified)
        unique_counts = len(set(count for _, count in history))
        spread_velocity = unique_counts / time_span
        
        # Count mutations (different occurrence counts)
        mutation_count = len(set(count for _, count in history)) - 1
        
        # Calculate stability (consistency of pattern)
        if len(history) > 1:
            counts = [count for _, count in history]
            stability_score = 1.0 - (np.std(counts) / (np.mean(counts) + 1))
        else:
            stability_score = 0.5
        
        # Predict future occurrences (simple linear projection)
        if len(history) >= 3:
            recent_growth = (history[-1][1] - history[-3][1]) / 2
            predicted_occurrences = max(0, int(history[-1][1] + recent_growth * 7))
        else:
            predicted_occurrences = history[-1][1]
        
        # Determine lifecycle stage
        if len(history) < 3:
            lifecycle_stage = 'emerging'
        elif growth_rate > 0.5:
            lifecycle_stage = 'growing'
        elif growth_rate < -0.1:
            lifecycle_stage = 'declining'
        else:
            lifecycle_stage = 'stable'
        
        return PatternEvolutionMetrics(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            first_occurrence=first_occurrence,
            last_occurrence=last_occurrence,
            growth_rate=growth_rate,
            spread_velocity=spread_velocity,
            mutation_count=mutation_count,
            stability_score=stability_score,
            predicted_occurrences=predicted_occurrences,
            lifecycle_stage=lifecycle_stage
        )


class MetricsDashboard:
    """
    Dashboard for displaying enhanced metrics.
    
    Provides formatted output and visualizations of metrics.
    """
    
    def __init__(self, calculator: EnhancedMetricsCalculator):
        self.calculator = calculator
    
    def generate_impact_report(self, fix_impact: FixImpactMetrics) -> str:
        """Generate formatted fix impact report."""
        report = f"""
Fix Impact Analysis
===================
Estimated Line Changes:
  • Lines Removed: {fix_impact.estimated_lines_removed:,}
  • Lines Added: {fix_impact.estimated_lines_added:,}
  • Net Reduction: {fix_impact.net_line_reduction:,}

Scope:
  • Files Affected: {fix_impact.files_affected}
  • Modules Affected: {fix_impact.modules_affected}

Quality Improvements:
  • Complexity Reduction: {fix_impact.complexity_reduction:.1%}
  • Maintainability Increase: {fix_impact.maintainability_increase:.1%}
  • Test Coverage Impact: {fix_impact.test_coverage_impact:+.1%}

Effort Estimate:
  • Refactoring Hours: {fix_impact.refactoring_effort_hours:.1f}
  • Risk Level: {fix_impact.risk_level}
"""
        return report
    
    def generate_coupling_report(self, coupling_metrics: Dict[str, ModuleCouplingMetrics]) -> str:
        """Generate formatted coupling report."""
        # Sort by coupling score
        sorted_modules = sorted(coupling_metrics.values(), key=lambda x: x.coupling_score, reverse=True)
        
        report = """
Module Coupling Analysis
========================
Top 10 Most Coupled Modules:
"""
        for metric in sorted_modules[:10]:
            report += f"""
{metric.module_name}:
  • Afferent Coupling: {metric.afferent_coupling}
  • Efferent Coupling: {metric.efferent_coupling}
  • Instability: {metric.instability:.2f}
  • Distance from Main Sequence: {metric.distance_from_main_sequence:.2f}
  • Coupling Score: {metric.coupling_score:.2f}
"""
            if metric.highly_coupled_with:
                report += "  • Highly Coupled With:\n"
                for module, weight in metric.highly_coupled_with[:3]:
                    report += f"    - {module} (weight: {weight})\n"
        
        return report
    
    def generate_evolution_report(self, evolution_metrics: List[PatternEvolutionMetrics]) -> str:
        """Generate formatted pattern evolution report."""
        if not evolution_metrics:
            return "No pattern evolution data available."
        
        report = """
Pattern Evolution Analysis
==========================
"""
        # Group by lifecycle stage
        by_stage = defaultdict(list)
        for metric in evolution_metrics:
            by_stage[metric.lifecycle_stage].append(metric)
        
        for stage in ['emerging', 'growing', 'stable', 'declining']:
            if stage in by_stage:
                report += f"\n{stage.upper()} Patterns ({len(by_stage[stage])}):\n"
                for metric in by_stage[stage][:5]:
                    report += f"""
  {metric.pattern_type} ({metric.pattern_id[:8]}):
    • Growth Rate: {metric.growth_rate:.2f} occurrences/day
    • Spread Velocity: {metric.spread_velocity:.2f} files/day
    • Stability: {metric.stability_score:.2f}
    • Predicted Next Week: {metric.predicted_occurrences}
"""
        
        return report