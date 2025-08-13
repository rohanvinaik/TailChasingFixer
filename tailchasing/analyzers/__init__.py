"""Analyzers for detecting tail-chasing patterns."""

from .base import Analyzer, AnalysisContext, BaseAnalyzer
from .import_graph import ImportGraphAnalyzer
from .duplicates import DuplicateFunctionAnalyzer
from .placeholders import PlaceholderAnalyzer
from .missing_symbols import MissingSymbolAnalyzer
from .git_chains import GitChainAnalyzer

# Improved analyzers with better accuracy
from .scope_aware_duplicates import ScopeAwareDuplicateAnalyzer
from .llm_detector_improved import ImprovedLLMDetector
from .runtime_aware_symbols import RuntimeAwareSymbolAnalyzer
from .mock_aware_placeholders import MockAwarePlaceholderAnalyzer

__all__ = [
    "Analyzer",
    "AnalysisContext",
    "BaseAnalyzer",
    "ImportGraphAnalyzer", 
    "DuplicateFunctionAnalyzer",
    "PlaceholderAnalyzer",
    "MissingSymbolAnalyzer",
    "GitChainAnalyzer",
    # Improved analyzers
    "ScopeAwareDuplicateAnalyzer",
    "ImprovedLLMDetector",
    "RuntimeAwareSymbolAnalyzer",
    "MockAwarePlaceholderAnalyzer",
]
