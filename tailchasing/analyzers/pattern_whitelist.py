"""
Pattern whitelist system for marking known legitimate patterns.

This module allows users to configure patterns that should not be flagged
as issues, even if they appear to be duplicates or other problems.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re


@dataclass
class WhitelistPattern:
    """A pattern that should be whitelisted from detection."""
    pattern_type: str  # 'method', 'class', 'module', 'file'
    name_pattern: str  # Can be exact match or regex
    context: Optional[str] = None  # Optional context like 'in_test', 'in_mock'
    reason: str = ""  # Why this is whitelisted
    is_regex: bool = False
    
    def matches(self, name: str, file_path: str = "", class_name: str = "") -> bool:
        """Check if this pattern matches the given context."""
        # Check name match
        if self.is_regex:
            if not re.match(self.name_pattern, name):
                return False
        else:
            if name != self.name_pattern:
                return False
        
        # Check context if specified
        if self.context:
            if self.context == 'in_test' and 'test' not in file_path.lower():
                return False
            if self.context == 'in_mock' and 'mock' not in class_name.lower():
                return False
        
        return True


class PatternWhitelist:
    """Manages whitelisted patterns across the codebase."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.patterns: List[WhitelistPattern] = []
        
        # Add default patterns that are almost always legitimate
        self._add_default_patterns()
        
        # Add user-configured patterns
        if config:
            self._load_from_config(config)
    
    def _add_default_patterns(self):
        """Add default patterns that are commonly legitimate."""
        defaults = [
            # Serialization methods that appear in many classes
            WhitelistPattern(
                pattern_type='method',
                name_pattern='dict_for_update',
                reason='Common serialization pattern for API models'
            ),
            WhitelistPattern(
                pattern_type='method',
                name_pattern='to_dict',
                reason='Standard serialization method'
            ),
            WhitelistPattern(
                pattern_type='method',
                name_pattern='from_dict',
                reason='Standard deserialization method'
            ),
            WhitelistPattern(
                pattern_type='method',
                name_pattern='as_dict',
                reason='Alternative serialization method'
            ),
            
            # Common interface methods
            WhitelistPattern(
                pattern_type='method',
                name_pattern=r'^(get|set|has|is)_\w+',
                is_regex=True,
                reason='Property accessor pattern'
            ),
            
            # Test patterns
            WhitelistPattern(
                pattern_type='method',
                name_pattern='setUp',
                context='in_test',
                reason='Standard test setup method'
            ),
            WhitelistPattern(
                pattern_type='method',
                name_pattern='tearDown',
                context='in_test',
                reason='Standard test teardown method'
            ),
            
            # Mock patterns
            WhitelistPattern(
                pattern_type='class',
                name_pattern=r'^(Mock|Fake|Stub|Dummy)\w+',
                is_regex=True,
                reason='Test double naming convention'
            ),
            
            # Dataclass methods
            WhitelistPattern(
                pattern_type='method',
                name_pattern='__post_init__',
                reason='Dataclass initialization hook'
            ),
        ]
        
        self.patterns.extend(defaults)
    
    def _load_from_config(self, config: Dict[str, Any]):
        """Load patterns from configuration."""
        if 'whitelist_patterns' in config:
            for pattern_config in config['whitelist_patterns']:
                pattern = WhitelistPattern(
                    pattern_type=pattern_config.get('type', 'method'),
                    name_pattern=pattern_config['pattern'],
                    context=pattern_config.get('context'),
                    reason=pattern_config.get('reason', 'User configured'),
                    is_regex=pattern_config.get('regex', False)
                )
                self.patterns.append(pattern)
    
    def is_whitelisted(self, name: str, pattern_type: str = 'method',
                      file_path: str = "", class_name: str = "") -> bool:
        """Check if a pattern is whitelisted."""
        for pattern in self.patterns:
            if pattern.pattern_type != pattern_type:
                continue
            
            if pattern.matches(name, file_path, class_name):
                return True
        
        return False
    
    def get_reason(self, name: str, pattern_type: str = 'method',
                   file_path: str = "", class_name: str = "") -> Optional[str]:
        """Get the reason why a pattern is whitelisted."""
        for pattern in self.patterns:
            if pattern.pattern_type != pattern_type:
                continue
            
            if pattern.matches(name, file_path, class_name):
                return pattern.reason
        
        return None


# Global whitelist instance that can be shared
_global_whitelist: Optional[PatternWhitelist] = None


def get_whitelist(config: Optional[Dict[str, Any]] = None) -> PatternWhitelist:
    """Get or create the global whitelist instance."""
    global _global_whitelist
    
    if _global_whitelist is None or config is not None:
        _global_whitelist = PatternWhitelist(config)
    
    return _global_whitelist


def is_whitelisted(name: str, pattern_type: str = 'method',
                  file_path: str = "", class_name: str = "",
                  config: Optional[Dict[str, Any]] = None) -> bool:
    """Quick check if a pattern is whitelisted."""
    whitelist = get_whitelist(config)
    return whitelist.is_whitelisted(name, pattern_type, file_path, class_name)