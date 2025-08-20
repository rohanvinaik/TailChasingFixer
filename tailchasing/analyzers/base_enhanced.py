"""
Enhanced analyzer base classes to eliminate semantic duplication.

This module consolidates the duplicate __init__ methods and utility functions
found across 5 analyzers, implementing a proper inheritance hierarchy with
the Strategy pattern for different analysis approaches.
"""

import ast
import logging
from abc import abstractmethod
from ..core.types import (
    AnalysisStrategy,
    AnalysisContext, 
    AnalysisConfig
)
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import difflib

from ..core.issues import Issue
from .base import Analyzer, AnalysisContext


class AnalysisStrategy(Enum):
    """Different analysis strategies available."""
    PATTERN_DETECTION = "pattern_detection"
    SEMANTIC_ANALYSIS = "semantic_analysis"
    STRUCTURAL_ANALYSIS = "structural_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"


@dataclass
class AnalysisConfig:
    """Configuration for analyzers with common settings."""
    # Similarity thresholds
    similarity_threshold: float = 0.6
    high_similarity_threshold: float = 0.8
    statistical_significance_threshold: float = 0.05
    
    # Pattern detection thresholds
    hallucination_threshold: float = 0.7
    context_thrashing_threshold: float = 0.8
    import_anxiety_threshold: float = 3.0
    
    # Distance and clustering thresholds
    min_line_distance: int = 200
    context_window_distance: int = 500
    min_cluster_size: int = 3
    max_cluster_size: int = 20
    
    # Confidence and risk thresholds
    confidence_threshold: float = 0.7
    density_threshold: float = 0.3
    isolation_threshold: float = 0.6
    
    # Import analysis thresholds
    min_imports_for_anxiety: int = 5
    unused_ratio_threshold: float = 0.5
    anxiety_score_threshold: float = 0.6
    error_handling_threshold: float = 0.4
    
    # Temporal analysis
    temporal_threshold_hours: int = 48
    
    # Custom configuration
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        """Create config from dictionary with fallback to defaults."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.custom_config[key] = value
        return config


class UtilityMixin:
    """
    Mixin providing common utility functions used across analyzers.
    
    Consolidates duplicate implementations of name similarity, duplicate detection,
    and confidence calculation methods.
    """
    
    @staticmethod
    def calculate_name_similarity(name1: str, name2: str) -> float:
        """
        Calculate similarity between two names.
        
        Consolidates the duplicate _name_similarity implementations.
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            Similarity score between 0 and 1
        """
        if name1 == name2:
            return 1.0
        
        # Normalize names for comparison
        norm1 = name1.lower()
        norm2 = name2.lower()
        
        # Direct sequence matching
        direct_similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        # Check for common patterns (get_X vs fetch_X, etc.)
        pattern_similarity = 0.0
        
        # Extract base names by removing common prefixes/suffixes
        prefixes = ['get_', 'set_', 'fetch_', 'load_', 'save_', 'create_', 'make_', 'build_']
        suffixes = ['_func', '_method', '_handler', '_processor', '_manager']
        
        base1 = norm1
        base2 = norm2
        
        for prefix in prefixes:
            if base1.startswith(prefix):
                base1 = base1[len(prefix):]
            if base2.startswith(prefix):
                base2 = base2[len(prefix):]
        
        for suffix in suffixes:
            if base1.endswith(suffix):
                base1 = base1[:-len(suffix)]
            if base2.endswith(suffix):
                base2 = base2[:-len(suffix)]
        
        if base1 == base2:
            pattern_similarity = 0.8
        elif base1 and base2:
            pattern_similarity = difflib.SequenceMatcher(None, base1, base2).ratio() * 0.6
        
        return max(direct_similarity, pattern_similarity)
    
    @staticmethod
    def are_likely_duplicates(func1: ast.FunctionDef, func2: ast.FunctionDef, 
                            similarity: float, min_lines: int = 3) -> bool:
        """
        Determine if two functions are likely duplicates.
        
        Consolidates the duplicate _are_likely_duplicates implementations.
        
        Args:
            func1: First function AST
            func2: Second function AST  
            similarity: Calculated similarity score
            min_lines: Minimum lines to consider for duplication
            
        Returns:
            True if functions are likely duplicates
        """
        # Don't flag very short functions as duplicates
        if len(func1.body) < min_lines or len(func2.body) < min_lines:
            return False
        
        # Don't flag test functions against each other
        if (func1.name.startswith('test_') and func2.name.startswith('test_')):
            return False
        
        # Don't flag __init__ methods unless similarity is very high
        if func1.name == '__init__' and func2.name == '__init__':
            return similarity > 0.95
        
        # Don't flag functions with very different argument counts
        arg_count1 = len(func1.args.args) if func1.args else 0
        arg_count2 = len(func2.args.args) if func2.args else 0
        
        if abs(arg_count1 - arg_count2) > 2:
            return False
        
        # Check for common boilerplate patterns
        boilerplate_indicators = [
            'pass', 'NotImplementedError', 'raise NotImplementedError',
            'return None', '...'
        ]
        
        func1_code = ast.unparse(func1) if hasattr(ast, 'unparse') else ast.dump(func1)
        func2_code = ast.unparse(func2) if hasattr(ast, 'unparse') else ast.dump(func2)
        
        func1_is_boilerplate = any(indicator in func1_code for indicator in boilerplate_indicators)
        func2_is_boilerplate = any(indicator in func2_code for indicator in boilerplate_indicators)
        
        # Don't flag boilerplate functions unless they're identical
        if func1_is_boilerplate or func2_is_boilerplate:
            return similarity > 0.98
        
        return similarity > 0.85
    
    @staticmethod
    def calculate_confidence(factors: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate confidence score from multiple factors.
        
        Args:
            factors: Dictionary of factor name to value
            weights: Optional weights for factors
            
        Returns:
            Weighted confidence score between 0 and 1
        """
        if not factors:
            return 0.0
        
        default_weights = {
            'similarity': 0.4,
            'pattern_match': 0.3,
            'structural_similarity': 0.2,
            'name_similarity': 0.1
        }
        
        if weights:
            default_weights.update(weights)
        
        total_score = 0.0
        total_weight = 0.0
        
        for factor, value in factors.items():
            weight = default_weights.get(factor, 0.1)
            total_score += value * weight
            total_weight += weight
        
        return min(total_score / max(total_weight, 0.1), 1.0)


# Using AnalysisStrategy protocol from consolidated types
# (AnalysisStrategyProtocol is now AnalysisStrategy)


class ConfigurableAnalyzer(Analyzer, UtilityMixin):
    """
    Base class for configurable analyzers.
    
    Consolidates the duplicate __init__ method pattern found across
    5 different analyzer classes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, 
                 strategy: Optional[AnalysisStrategy] = None):
        """
        Initialize configurable analyzer.
        
        This consolidates the identical __init__ patterns from:
        - ContextThrashingAnalyzer.__init__
        - EnhancedPatternDetector.__init__
        - EnhancedSemanticAnalyzer.__init__
        - HallucinationCascadeAnalyzer.__init__
        - ImportAnxietyAnalyzer.__init__
        
        Args:
            config: Configuration dictionary
            strategy: Analysis strategy to use
        """
        super().__init__()
        
        # Store raw config and create typed config
        self.raw_config = config or {}
        self.config = AnalysisConfig.from_dict(self.raw_config)
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Analysis strategy
        self.strategy = strategy or AnalysisStrategy.PATTERN_DETECTION
        self._strategy_impl: Optional[AnalysisStrategyProtocol] = None
        
        # Common attributes that were duplicated across analyzers
        self.patterns: List[Any] = []
        self.thresholds: Dict[str, float] = {}
        self.cache: Dict[str, Any] = {}
        
        # Initialize specific configuration
        self._initialize_specific_config()
    
    def _initialize_specific_config(self) -> None:
        """
        Initialize analyzer-specific configuration.
        
        Override this method in subclasses to set up specific thresholds
        and parameters from the config.
        """
        # Set up common thresholds that were duplicated
        self.thresholds.update({
            'similarity': self.config.similarity_threshold,
            'high_similarity': self.config.high_similarity_threshold,
            'confidence': self.config.confidence_threshold,
            'statistical_significance': self.config.statistical_significance_threshold
        })
    
    def set_strategy(self, strategy: Union[AnalysisStrategy, AnalysisStrategyProtocol]) -> None:
        """
        Set the analysis strategy.
        
        Args:
            strategy: Strategy enum or strategy implementation
        """
        if isinstance(strategy, AnalysisStrategy):
            self.strategy = strategy
            self._strategy_impl = None
        else:
            self._strategy_impl = strategy
    
    def get_threshold(self, name: str, default: float = 0.5) -> float:
        """Get a threshold value by name."""
        return self.thresholds.get(name, default)
    
    def set_threshold(self, name: str, value: float) -> None:
        """Set a threshold value."""
        self.thresholds[name] = value
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run the analysis using the configured strategy.
        
        Args:
            ctx: Analysis context
            
        Returns:
            List of issues found
        """
        if self._strategy_impl:
            return self._strategy_impl.analyze(ctx, self.config)
        else:
            return self._run_default_analysis(ctx)
    
    @abstractmethod
    def _run_default_analysis(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run the default analysis implementation.
        
        Subclasses must implement this method.
        """
        pass


class AdvancedAnalyzer(ConfigurableAnalyzer):
    """
    Advanced analyzer with pattern detection capabilities.
    
    Provides additional functionality for complex pattern analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 strategy: Optional[AnalysisStrategy] = None):
        """Initialize advanced analyzer."""
        super().__init__(config, strategy)
        
        # Additional advanced capabilities
        self.pattern_cache: Dict[str, List[Any]] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Check for external dependencies
        self.git_available = self._check_git_availability()
    
    def _check_git_availability(self) -> bool:
        """Check if git is available for temporal analysis."""
        try:
            import subprocess
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
    
    def _initialize_specific_config(self) -> None:
        """Initialize advanced analyzer specific configuration."""
        super()._initialize_specific_config()
        
        # Add advanced-specific thresholds
        self.thresholds.update({
            'hallucination': self.config.hallucination_threshold,
            'context_thrashing': self.config.context_thrashing_threshold,
            'import_anxiety': self.config.import_anxiety_threshold,
            'isolation': self.config.isolation_threshold,
            'density': self.config.density_threshold
        })
    
    def detect_patterns(self, ctx: AnalysisContext, pattern_types: Optional[List[str]] = None) -> Dict[str, List[Any]]:
        """
        Detect patterns in the codebase.
        
        Args:
            ctx: Analysis context
            pattern_types: Specific pattern types to detect
            
        Returns:
            Dictionary of pattern type to detected patterns
        """
        patterns = {}
        
        # Use cache if available
        cache_key = f"patterns_{hash(str(sorted(pattern_types or [])))}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        # Detect different pattern types
        available_patterns = {
            'duplicates': self._detect_duplicate_patterns,
            'cascades': self._detect_cascade_patterns,
            'anxiety': self._detect_anxiety_patterns,
            'thrashing': self._detect_thrashing_patterns
        }
        
        for pattern_type, detector in available_patterns.items():
            if pattern_types is None or pattern_type in pattern_types:
                try:
                    patterns[pattern_type] = detector(ctx)
                except Exception as e:
                    self.logger.warning(f"Failed to detect {pattern_type} patterns: {e}")
                    patterns[pattern_type] = []
        
        # Cache results
        self.pattern_cache[cache_key] = patterns
        return patterns
    
    def _detect_duplicate_patterns(self, ctx: AnalysisContext) -> List[Any]:
        """Detect duplicate patterns in the codebase."""
        # Implementation would go here
        return []
    
    def _detect_cascade_patterns(self, ctx: AnalysisContext) -> List[Any]:
        """Detect cascade patterns in the codebase."""
        # Implementation would go here
        return []
    
    def _detect_anxiety_patterns(self, ctx: AnalysisContext) -> List[Any]:
        """Detect import anxiety patterns in the codebase."""
        # Implementation would go here
        return []
    
    def _detect_thrashing_patterns(self, ctx: AnalysisContext) -> List[Any]:
        """Detect context thrashing patterns in the codebase."""
        # Implementation would go here
        return []
    
    def _run_default_analysis(self, ctx: AnalysisContext) -> List[Issue]:
        """Default analysis implementation for advanced analyzers."""
        issues = []
        
        # Detect patterns based on strategy
        if self.strategy == AnalysisStrategy.PATTERN_DETECTION:
            patterns = self.detect_patterns(ctx)
            issues.extend(self._patterns_to_issues(patterns))
        
        return issues
    
    def _patterns_to_issues(self, patterns: Dict[str, List[Any]]) -> List[Issue]:
        """Convert detected patterns to issues."""
        issues = []
        
        for pattern_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                # Convert pattern to issue - this would be specific to each pattern type
                issue = Issue(
                    kind=f"advanced_{pattern_type}",
                    message=f"Detected {pattern_type} pattern",
                    severity=2,
                    file="<multiple>",
                    line=0,
                    evidence={'pattern': pattern}
                )
                issues.append(issue)
        
        return issues


# Strategy implementations for common analysis types
class PatternDetectionStrategy:
    """Strategy for pattern-based analysis."""
    
    def analyze(self, context: AnalysisContext, config: AnalysisConfig) -> List[Issue]:
        """Analyze using pattern detection."""
        # Implementation would go here
        return []
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "pattern_detection"


class SemanticAnalysisStrategy:
    """Strategy for semantic analysis."""
    
    def analyze(self, context: AnalysisContext, config: AnalysisConfig) -> List[Issue]:
        """Analyze using semantic methods."""
        # Implementation would go here
        return []
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "semantic_analysis"


# Factory function for creating analyzers
def create_analyzer(analyzer_type: str, config: Optional[Dict[str, Any]] = None) -> ConfigurableAnalyzer:
    """
    Factory function for creating analyzers with the new hierarchy.
    
    Args:
        analyzer_type: Type of analyzer to create
        config: Configuration for the analyzer
        
    Returns:
        Configured analyzer instance
    """
    analyzers = {
        'configurable': ConfigurableAnalyzer,
        'advanced': AdvancedAnalyzer
    }
    
    analyzer_class = analyzers.get(analyzer_type, ConfigurableAnalyzer)
    return analyzer_class(config)