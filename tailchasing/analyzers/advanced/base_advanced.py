"""Base class for advanced analyzers with common initialization patterns.

This module provides a base class that eliminates duplicate initialization
code across advanced analyzer classes.
"""

from typing import Dict, Any, List
from abc import ABC, abstractmethod
import logging

from ..base import Analyzer
from ...core.issues import Issue


class BaseAdvancedAnalyzer(ABC):
    """Base class for advanced analyzers with shared initialization.
    
    This class provides common initialization patterns for advanced analyzers
    that need to track patterns, thresholds, and configuration values.
    
    Attributes:
        name: Analyzer name (to be set by subclasses)
        patterns: List of detected patterns
        thresholds: Dictionary of threshold values
        config_values: Dictionary of configuration values
        logger: Logger instance for the analyzer
    """
    
    name: str = "base_advanced"
    
    def __init__(self, **kwargs):
        """Initialize the advanced analyzer with common attributes.
        
        Args:
            **kwargs: Additional configuration values to be stored
        """
        # Common attributes for all advanced analyzers
        self.patterns: List[Any] = []
        self.thresholds: Dict[str, float] = {}
        self.config_values: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store any additional configuration passed in
        self.config_values.update(kwargs)
        
        # Allow subclasses to set their specific configurations
        self._initialize_specific_config()
    
    def _initialize_specific_config(self):
        """Initialize analyzer-specific configuration.
        
        This method should be overridden by subclasses to set their
        specific thresholds and configuration values.
        """
        pass
    
    def set_threshold(self, name: str, value: float):
        """Set a threshold value.
        
        Args:
            name: Name of the threshold
            value: Threshold value
        """
        self.thresholds[name] = value
    
    def get_threshold(self, name: str, default: float = 0.5) -> float:
        """Get a threshold value.
        
        Args:
            name: Name of the threshold
            default: Default value if threshold not set
            
        Returns:
            Threshold value
        """
        return self.thresholds.get(name, default)
    
    def set_config(self, name: str, value: Any):
        """Set a configuration value.
        
        Args:
            name: Name of the configuration
            value: Configuration value
        """
        self.config_values[name] = value
    
    def get_config(self, name: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            name: Name of the configuration
            default: Default value if configuration not set
            
        Returns:
            Configuration value
        """
        return self.config_values.get(name, default)
    
    @abstractmethod
    def run(self, ctx) -> List[Issue]:
        """Run the analyzer.
        
        This method must be implemented by subclasses.
        
        Args:
            ctx: Analysis context
            
        Returns:
            List of detected issues
        """
        pass


class ContextAwareAnalyzer(BaseAdvancedAnalyzer, Analyzer):
    """Base class for analyzers that track context-related patterns.
    
    This class is specifically for analyzers that need to track
    distance, similarity, and other context-related metrics.
    """
    
    def _initialize_specific_config(self):
        """Initialize context-aware specific configuration."""
        # Common thresholds for context-aware analyzers
        self.set_threshold('similarity', 0.7)
        self.set_threshold('min_distance', 100)
        self.set_config('track_patterns', True)


class SemanticAwareAnalyzer(BaseAdvancedAnalyzer, Analyzer):
    """Base class for analyzers that perform semantic analysis.
    
    This class is specifically for analyzers that need to track
    semantic patterns, vector dimensions, and similarity metrics.
    """
    
    def _initialize_specific_config(self):
        """Initialize semantic-aware specific configuration."""
        # Common configuration for semantic analyzers
        self.set_threshold('similarity', 0.85)
        self.set_config('vector_dim', 8192)
        self.set_config('use_cache', True)


class PatternDetectionAnalyzer(BaseAdvancedAnalyzer):
    """Base class for analyzers that detect code patterns.
    
    This class is specifically for analyzers that detect various
    code patterns and anti-patterns.
    """
    
    def _initialize_specific_config(self):
        """Initialize pattern detection specific configuration."""
        # Common configuration for pattern detectors
        self.set_threshold('confidence', 0.7)
        self.set_config('max_patterns', 100)
        self.set_config('merge_similar', True)