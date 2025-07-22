"""Plugin system for loading analyzers."""

from typing import List, Dict, Any
import importlib
import pkgutil

from .analyzers.base import Analyzer
from .analyzers.import_graph import ImportGraphAnalyzer
from .analyzers.duplicates import DuplicateFunctionAnalyzer
from .analyzers.placeholders import PlaceholderAnalyzer
from .analyzers.missing_symbols import MissingSymbolAnalyzer
from .analyzers.git_chains import GitChainAnalyzer
from .analyzers.semantic_hv import SemanticHVAnalyzer

# Import advanced analyzers
try:
    from .analyzers.advanced import (
        HallucinationCascadeAnalyzer,
        ContextWindowThrashingAnalyzer,
        ImportAnxietyAnalyzer,
        EnhancedSemanticAnalyzer
    )
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

# Advanced analyzers (optional)
ADVANCED_ANALYZERS = []
if ADVANCED_ANALYZERS_AVAILABLE:
    ADVANCED_ANALYZERS = [
        HallucinationCascadeAnalyzer(),
        ContextWindowThrashingAnalyzer(),
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
    
    # TODO: In the future, support loading custom analyzers from:
    # - Entry points (for installed packages)
    # - Plugin directories
    # - Config-specified modules
    
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
            
    # TODO: Discover external plugins from plugin_dir
    
    return discovered
