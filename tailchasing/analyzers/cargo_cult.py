"""
Cargo cult detector - alias to the enhanced implementation.

This module is maintained for backward compatibility.
The actual implementation is in cargo_cult_detector.py.
"""

# Import the enhanced detector with proper parent __init__ checking
from .cargo_cult_detector import CargoCultDetector, EnhancedCargoCultDetector

# Alias for backward compatibility
CargoCultDetector = EnhancedCargoCultDetector

__all__ = ['CargoCultDetector', 'EnhancedCargoCultDetector']