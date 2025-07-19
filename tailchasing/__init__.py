"""Tail-Chasing Detector - Detect LLM-assisted anti-patterns in Python code."""

__version__ = "0.1.0"
__author__ = "Rohan Vinaik"
__email__ = "rohanpvinaik@gmail.com"

from .core.issues import Issue
from .core.scoring import RiskScorer
from .analyzers.base import Analyzer

__all__ = ["Issue", "RiskScorer", "Analyzer", "__version__"]