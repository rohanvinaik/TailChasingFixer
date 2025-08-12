"""Plugin system for loading analyzers."""

from typing import List, Dict, Any
import importlib
import os
import pkgutil

from .analyzers.base import Analyzer
from .analyzers.import_graph import ImportGraphAnalyzer
from .analyzers.duplicates import DuplicateFunctionAnalyzer
from .analyzers.placeholders import PlaceholderAnalyzer
from .analyzers.missing_symbols import MissingSymbolAnalyzer
from .analyzers.git_chains import GitChainAnalyzer
from .analyzers.semantic_hv import SemanticHVAnalyzer

# Import new analyzers
try:
    from .analyzers.tdd_antipatterns import TDDAntipatternAnalyzer
    from .analyzers.cross_file_duplication import CrossFileDuplicationAnalyzer
    from .analyzers.cargo_cult import CargoCultDetector
    from .analyzers.root_cause_tracer import RootCauseTracer
    NEW_ANALYZERS_AVAILABLE = True
except ImportError:
    NEW_ANALYZERS_AVAILABLE = False

# Import advanced analyzers
try:
    from .analyzers.advanced.context_thrashing import ContextThrashingAnalyzer
    from .analyzers.advanced.hallucination_cascade import HallucinationCascadeAnalyzer
    from .analyzers.advanced.import_anxiety import ImportAnxietyAnalyzer
    from .analyzers.advanced.enhanced_semantic import EnhancedSemanticAnalyzer
    ADVANCED_ANALYZERS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYZERS_AVAILABLE = False


# Default analyzers that are always available
DEFAULT_ANALYZERS = [
    ImportGraphAnalyzer(),
    DuplicateFunctionAnalyzer(),
    PlaceholderAnalyzer(),
    MissingSymbolAnalyzer(),
    GitChainAnalyzer(),
    SemanticHVAnalyzer(),
]

# Add new analyzers if available
if NEW_ANALYZERS_AVAILABLE:
    DEFAULT_ANALYZERS.extend([
        TDDAntipatternAnalyzer(),
        CrossFileDuplicationAnalyzer(),
        CargoCultDetector(),
    ])

# Advanced analyzers (optional)
ADVANCED_ANALYZERS = []
if ADVANCED_ANALYZERS_AVAILABLE:
    ADVANCED_ANALYZERS = [
        HallucinationCascadeAnalyzer(),
        ContextThrashingAnalyzer(),
        ImportAnxietyAnalyzer(),
        EnhancedSemanticAnalyzer(),
    ]


def load_analyzers(config: Dict[str, Any]) -> List[Analyzer]:
    """Load analyzers based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of analyzer instances
    """
    # Start with default analyzers
    analyzers = DEFAULT_ANALYZERS.copy()
    
    # Add advanced analyzers if enabled
    if config.get("enable_advanced_analyzers", False) and ADVANCED_ANALYZERS_AVAILABLE:
        analyzers.extend(ADVANCED_ANALYZERS)
    
    # Check if any analyzers are disabled in config
    disabled = config.get("disabled_analyzers", [])
    analyzers = [a for a in analyzers if a.name not in disabled]
    
    # Future enhancement: support loading custom analyzers
    # This is safely disabled by default
    if config.get("enable_experimental_plugins", False):
        # When enabled, support loading from:
        # - Entry points (for installed packages)
        # - Plugin directories
        # - Config-specified modules
        pass  # Not implemented yet - requires explicit opt-in
    
    return analyzers


def discover_analyzers(plugin_dir: str = None) -> Dict[str, type]:
    """Discover available analyzer plugins.
    
    Args:
        plugin_dir: Optional directory to search for plugins
        
    Returns:
        Dictionary mapping analyzer names to classes
    """
    discovered = {}
    
    # Discover built-in analyzers
    from . import analyzers
    
    for importer, modname, ispkg in pkgutil.iter_modules(analyzers.__path__):
        if modname in ["base", "__init__"]:
            continue
            
        try:
            module = importlib.import_module(f".analyzers.{modname}", package="tailchasing")
            
            # Look for analyzer classes
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, "name") and 
                    hasattr(attr, "run")):
                    discovered[attr.name] = attr
                    
        except ImportError:
            pass
            
    # External plugin discovery (safely disabled by default)
    if plugin_dir and os.path.exists(plugin_dir):
        # Only process if explicitly provided and exists
        # This prevents any issues with missing plugin directories
        try:
            # Future enhancement: scan plugin_dir for analyzer modules
            pass  # Not implemented - requires explicit plugin directory
        except Exception:
            # Silently ignore plugin loading errors
            pass
    
    return discovered
