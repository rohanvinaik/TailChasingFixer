"""
TailChasing Polymer Physics Module: Chromatin-inspired performance analysis.

This module provides tools for analyzing performance bottlenecks using concepts
from chromatin organization and polymer physics, including Hi-C contact matrices,
TAD boundary detection, and polymer distance metrics.
"""

__version__ = "0.1.0"

from .calibrate import CalibrationTool, CodebaseMetrics, ThrashEvent
from .config import PolymerConfig, get_config, get_config_manager, save_config
from .reporting import (
    HiCHeatmapGenerator,
    PolymerMetricsReport,
    TAD,
    ThrashCluster,
    generate_comparative_matrices,
    integrate_chromatin_analysis,
)

__all__ = [
    # Configuration
    "PolymerConfig",
    "get_config",
    "get_config_manager",
    "save_config",
    # Calibration
    "CalibrationTool",
    "ThrashEvent",
    "CodebaseMetrics",
    # Reporting
    "HiCHeatmapGenerator",
    "PolymerMetricsReport",
    "TAD",
    "ThrashCluster",
    "generate_comparative_matrices",
    "integrate_chromatin_analysis",
]