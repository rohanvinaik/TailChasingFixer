"""
Import resolution and circular dependency breaking strategies.

Handles import-related issues including missing imports, import anxiety,
and circular dependencies. Extracted from fix_strategies.py to reduce
context window thrashing between import-related functions.
"""

from typing import List, Dict, Any, Optional

from .base import BaseFixStrategy, Action, Patch, SimplePatch, RiskLevel
from ...core.issues import Issue


class ImportResolutionStrategy(BaseFixStrategy):
    """
    Strategy for handling missing imports and import-related issues.
    
    Handles:
    - Missing symbols and imports
    - Import anxiety (too many unused imports)
    - Unused imports cleanup
    - Import organization
    """
    
    def __init__(self):
        super().__init__("ImportResolution")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle missing imports, import errors, and related issues."""
        return issue.kind in [
            "missing_symbol",
            "missing_import",
            "import_anxiety", 
            "unused_import",
            "import_error"
        ]
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix for import issues."""
        if not self.can_handle(issue):
            return None
        
        # For simple import fixes, return a SimplePatch for compatibility
        if issue.kind in ["missing_symbol", "missing_import"]:
            return self._create_simple_import_fix(issue)
        
        # For complex cases, use full Patch with actions
        actions = self._generate_fix_actions(issue, context)
        
        if not actions:
            return None
        
        return Patch(
            actions=actions,
            description=f"Fix {issue.kind}: {issue.symbol or 'import issue'}",
            confidence=0.9,  # Import fixes are usually reliable
            risk_level=RiskLevel.LOW,  # Import fixes are generally low risk
            estimated_time=self._estimate_time(actions),
            dependencies=[],  # Import fixes have no dependencies
            rollback_plan=[self.create_backup_action(action) for action in actions],
            validation_tests=self._generate_validation_tests(issue),
            side_effects=["Import statement changes"]
        )
    
    def _create_simple_import_fix(self, issue: Issue) -> SimplePatch:
        """Create a simple import fix for basic missing symbol issues."""
        symbol = issue.symbol or (issue.evidence.get('used_symbols', ['unknown'])[0] if issue.evidence else 'unknown')
        
        # Determine the import statement
        import_statement = self._determine_import_statement(symbol)
        
        return SimplePatch(
            file_path=issue.file,
            content=import_statement,
            line_number=1,
            description=f"Add missing import for {symbol}"
        )
    
    def _determine_import_statement(self, symbol: str) -> str:
        """Determine the appropriate import statement for a symbol."""
        # Common import mappings
        common_imports = {
            'json': 'import json',
            'os': 'import os',
            'sys': 'import sys',
            're': 'import re',
            'datetime': 'from datetime import datetime',
            'Path': 'from pathlib import Path',
            'Optional': 'from typing import Optional',
            'List': 'from typing import List',
            'Dict': 'from typing import Dict',
            'Any': 'from typing import Any',
            'Union': 'from typing import Union',
            'Tuple': 'from typing import Tuple',
            'Set': 'from typing import Set',
            'Callable': 'from typing import Callable',
            'Iterator': 'from typing import Iterator',
            'Iterable': 'from typing import Iterable',
            'Protocol': 'from typing import Protocol',
            'TypeVar': 'from typing import TypeVar',
            'Generic': 'from typing import Generic',
            'ABC': 'from abc import ABC',
            'abstractmethod': 'from abc import abstractmethod',
            'dataclass': 'from dataclasses import dataclass',
            'field': 'from dataclasses import field',
            'Enum': 'from enum import Enum',
            'IntEnum': 'from enum import IntEnum'
        }
        
        if symbol in common_imports:
            return common_imports[symbol]
        
        # Try to infer from symbol name patterns
        if symbol.startswith('np') or 'numpy' in symbol.lower():
            return 'import numpy as np'
        elif symbol.startswith('pd') or 'pandas' in symbol.lower():
            return 'import pandas as pd'
        elif symbol.startswith('plt') or 'matplotlib' in symbol.lower():
            return 'import matplotlib.pyplot as plt'
        elif 'torch' in symbol.lower():
            return 'import torch'
        elif 'tensorflow' in symbol.lower() or symbol.startswith('tf'):
            return 'import tensorflow as tf'
        
        # Generic import suggestion
        return f"# TODO: Add import for {symbol}\\n# from module import {symbol}"
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to fix import issues."""
        actions = []
        
        if issue.kind in ["missing_symbol", "missing_import"]:
            actions.extend(self._fix_missing_symbol(issue, context))
        elif issue.kind == "import_anxiety":
            actions.extend(self._fix_import_anxiety(issue, context))
        elif issue.kind == "unused_import":
            actions.extend(self._fix_unused_imports(issue, context))
        
        return actions
    
    def _fix_missing_symbol(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Fix missing symbol by adding appropriate import."""
        actions = []
        
        symbol = issue.symbol or "unknown_symbol"
        import_statement = self._determine_import_statement(symbol)
        
        # Read current file content for backup
        file_content = ""
        try:
            with open(issue.file, 'r') as f:
                file_content = f.read()
        except Exception:
            return actions
        
        # Add import at appropriate location
        new_content = self._insert_import_statement(file_content, import_statement)
        
        actions.append(Action(
            type="modify_file",
            target=issue.file,
            content=new_content,
            backup_content=file_content,
            metadata={"added_import": import_statement, "symbol": symbol}
        ))
        
        return actions


class CircularDependencyBreaker(BaseFixStrategy):
    """
    Strategy for breaking circular import dependencies.
    
    Uses multiple strategies:
    1. Move imports to local function scope
    2. Create interface modules for complex cycles
    3. Loop extrusion algorithms (if available)
    """
    
    def __init__(self, chromatin_analyzer=None):
        super().__init__("CircularDependencyBreaker")
        self.chromatin_analyzer = chromatin_analyzer
        self._loop_extrusion_breaker = None
        
        # Initialize loop extrusion if chromatin analyzer available
        if chromatin_analyzer:
            try:
                from ..loop_extrusion import LoopExtrusionBreaker
                self._loop_extrusion_breaker = LoopExtrusionBreaker(chromatin_analyzer)
                self.logger.info("Loop extrusion capabilities enabled")
            except ImportError as e:
                self.logger.warning(f"Loop extrusion not available: {e}")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle circular import issues."""
        return issue.kind in [
            "circular_import",
            "circular_dependency",
            "import_cycle"
        ]
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix to break circular dependencies."""
        actions = self._generate_fix_actions(issue, context)
        
        if not actions:
            return None
        
        # Circular dependency fixes are higher risk
        risk_level = RiskLevel.HIGH
        confidence = 0.7  # More conservative confidence
        
        return Patch(
            actions=actions,
            description=f"Break circular dependency: {issue.symbol or 'import cycle'}",
            confidence=confidence,
            risk_level=risk_level,
            estimated_time=self._estimate_time(actions),
            dependencies=[],  # No dependencies for circular fixes
            rollback_plan=[self.create_backup_action(action) for action in actions],
            validation_tests=self._extend_validation_for_cycle(self._generate_validation_tests(issue), issue),
            side_effects=["Import structure changes", "Possible API changes"]
        )
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to break circular dependencies."""
        actions = []
        
        if not issue.evidence or 'cycle' not in issue.evidence:
            return actions
        
        cycle = issue.evidence['cycle']
        if len(cycle) < 2:
            return actions
        
        # Strategy 1: Loop extrusion (if available and beneficial)
        if self._should_use_loop_extrusion(cycle, issue):
            loop_actions = self._create_loop_extrusion_actions(cycle, issue, context)
            if loop_actions:
                actions.extend(loop_actions)
                self.logger.info(f"Generated {len(loop_actions)} loop extrusion actions")
                return actions  # Use loop extrusion as primary strategy
        
        # Strategy 2: Move imports to function level (fallback)
        actions.extend(self._create_local_import_actions(cycle, issue))
        
        # Strategy 3: Create interface module (if cycle is complex)
        if len(cycle) > 2:
            actions.extend(self._create_interface_module_actions(cycle, issue))
        
        return actions
    
    def _should_use_loop_extrusion(self, cycle: List[str], issue: Issue) -> bool:
        """Determine if loop extrusion should be used for this cycle."""
        if not self._loop_extrusion_breaker:
            return False
        
        # Use for complex cycles with multiple modules
        if len(cycle) >= 3:
            return True
        
        # Use if there's evidence of high binding strength
        if issue.evidence and issue.evidence.get('binding_strength', 0) > 0.7:
            return True
        
        return False
    
    def _create_local_import_actions(self, cycle: List[str], issue: Issue) -> List[Action]:
        """Create actions to move imports to local scope."""
        actions = []
        
        if len(cycle) < 2:
            return actions
        
        file_a = cycle[0]
        file_b = cycle[1]
        
        try:
            with open(file_a, 'r') as f:
                content_a = f.read()
            
            # Find imports of file_b in file_a and move them to functions
            new_content = self._move_imports_to_functions(content_a, file_b)
            
            if new_content != content_a:
                actions.append(Action(
                    type="modify_file",
                    target=file_a,
                    content=new_content,
                    backup_content=content_a,
                    metadata={
                        "moved_imports_from": file_b,
                        "strategy": "local_imports"
                    }
                ))
        
        except Exception as e:
            self.logger.warning(f"Could not move imports in {file_a}: {e}")
        
        return actions