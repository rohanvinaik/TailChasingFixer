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

# Import fast duplicate analyzer
try:
    from .analyzers.fast_duplicates import FastDuplicateAnalyzer
    FAST_DUPLICATES_AVAILABLE = True
except ImportError:
    FAST_DUPLICATES_AVAILABLE = False

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

# Import canonical policy analyzer
try:
    from .analyzers.canonical_policy import CanonicalPolicyAnalyzer
    CANONICAL_POLICY_AVAILABLE = True
except ImportError:
    CANONICAL_POLICY_AVAILABLE = False

# Import circular import resolver
try:
    from .analyzers.circular_import_resolver import CircularImportResolver
    CIRCULAR_IMPORT_RESOLVER_AVAILABLE = True
except ImportError:
    CIRCULAR_IMPORT_RESOLVER_AVAILABLE = False

# Import phantom triage analyzer
try:
    from .analyzers.phantom_triage import PhantomTriageAnalyzer
    PHANTOM_TRIAGE_AVAILABLE = True
except ImportError:
    PHANTOM_TRIAGE_AVAILABLE = False

# Import context thrashing analyzer
try:
    from .analyzers.context_thrashing import ContextThrashingAnalyzer
    CONTEXT_THRASHING_AVAILABLE = True
except ImportError:
    CONTEXT_THRASHING_AVAILABLE = False

# Import chromatin contact analyzer
try:
    from .analyzers.chromatin_contact import ChromatinContactAnalyzer
    CHROMATIN_CONTACT_AVAILABLE = True
except ImportError:
    CHROMATIN_CONTACT_AVAILABLE = False

# Import enhanced placeholder analyzer
try:
    from .analyzers.enhanced_placeholders import EnhancedPlaceholderAnalyzer
    ENHANCED_PLACEHOLDERS_AVAILABLE = True
except ImportError:
    ENHANCED_PLACEHOLDERS_AVAILABLE = False

# Import enhanced missing symbols analyzer
try:
    from .analyzers.enhanced_missing_symbols import EnhancedMissingSymbolAnalyzer
    ENHANCED_MISSING_SYMBOLS_AVAILABLE = True
except ImportError:
    ENHANCED_MISSING_SYMBOLS_AVAILABLE = False


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
    
    # Replace regular duplicate analyzer with fast version if configured
    duplicates_config = config.get("duplicates", {})
    if duplicates_config.get("use_fast_detection", True) and FAST_DUPLICATES_AVAILABLE:
        # Remove regular duplicate analyzer
        analyzers = [a for a in analyzers if a.name != "duplicates"]
        # Add fast duplicate analyzer
        analyzers.insert(1, FastDuplicateAnalyzer())  # Insert after import_graph
    
    # Add advanced analyzers if enabled
    if config.get("enable_advanced_analyzers", False) and ADVANCED_ANALYZERS_AVAILABLE:
        analyzers.extend(ADVANCED_ANALYZERS)
    
    # Add canonical policy analyzer if configured
    canonical_config = config.get("canonical_policy", {})
    if (canonical_config.get("canonical_roots") or canonical_config.get("shadow_roots")) and CANONICAL_POLICY_AVAILABLE:
        analyzers.append(CanonicalPolicyAnalyzer(config))
    
    # Add circular import resolver if enabled
    circular_config = config.get("circular_import_resolver", {})
    if circular_config.get("enabled", True) and CIRCULAR_IMPORT_RESOLVER_AVAILABLE:
        analyzers.append(CircularImportResolver(config))
    
    # Add phantom triage analyzer if enabled
    phantom_config = config.get("placeholders", {})
    if phantom_config.get("triage_enabled", True) and PHANTOM_TRIAGE_AVAILABLE:
        analyzers.append(PhantomTriageAnalyzer(config))
    
    # Add context thrashing analyzer if enabled
    context_config = config.get("context_thrashing", {})
    if context_config.get("enabled", True) and CONTEXT_THRASHING_AVAILABLE:
        analyzers.append(ContextThrashingAnalyzer(config))
    
    # Add chromatin contact analyzer if enabled
    chromatin_config = config.get("chromatin_contact", {})
    if chromatin_config.get("enabled", True) and CHROMATIN_CONTACT_AVAILABLE:
        analyzers.append(ChromatinContactAnalyzer(config))
    
    # Add enhanced placeholder analyzer if enabled
    enhanced_placeholders_config = config.get("enhanced_placeholders", {})
    if enhanced_placeholders_config.get("enabled", True) and ENHANCED_PLACEHOLDERS_AVAILABLE:
        # Replace regular placeholder analyzer with enhanced version
        analyzers = [a for a in analyzers if a.name != "placeholders"]
        analyzers.append(EnhancedPlaceholderAnalyzer())
    
    # Add enhanced missing symbols analyzer if enabled
    enhanced_missing_config = config.get("enhanced_missing_symbols", {})
    if enhanced_missing_config.get("enabled", False) and ENHANCED_MISSING_SYMBOLS_AVAILABLE:
        # Replace regular missing symbols analyzer with enhanced version
        analyzers = [a for a in analyzers if a.name != "missing_symbols"]
        analyzers.append(EnhancedMissingSymbolAnalyzer())
    
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
