"""
Import resolution strategy for fixing import-related issues.

This module handles missing symbols, import anxiety, unused imports,
and other import-related problems.
"""

import ast
from typing import List, Dict, Any, Optional

from .base import BaseFixStrategy, Action, RiskLevel, Patch
from ...core.issues import Issue


class ImportResolutionStrategy(BaseFixStrategy):
    """Strategy for handling missing imports and import-related issues."""
    
    def __init__(self):
        super().__init__("ImportResolution")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle missing imports, import errors, and related issues."""
        return issue.kind in [
            "missing_symbol",
            "import_anxiety", 
            "unused_import",
            "import_error"
        ]
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix for import issues."""
        if not self.can_handle(issue):
            return None
        
        actions = self._generate_fix_actions(issue, context)
        if not actions:
            return None
        
        # Create rollback plan
        rollback_plan = [self.create_backup_action(action) for action in actions]
        
        # Determine confidence based on issue type
        confidence = self._calculate_confidence(issue, context)
        
        # Create patch
        return Patch(
            actions=actions,
            description=self._get_fix_description(issue),
            confidence=confidence,
            risk_level=self._get_base_risk_level(),
            estimated_time=0.1,  # Import fixes are quick
            rollback_plan=rollback_plan,
            validation_tests=[f"python -m py_compile {issue.file}"],
            side_effects=self._get_side_effects(issue)
        )
    
    def _get_base_risk_level(self) -> RiskLevel:
        return RiskLevel.LOW  # Import fixes are generally low risk
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to fix import issues."""
        actions = []
        
        if issue.kind == "missing_symbol":
            actions.extend(self._fix_missing_symbol(issue, context))
        elif issue.kind == "import_anxiety":
            actions.extend(self._fix_import_anxiety(issue, context))
        elif issue.kind == "unused_import":
            actions.extend(self._fix_unused_imports(issue, context))
        elif issue.kind == "import_error":
            actions.extend(self._fix_import_error(issue, context))
        
        return actions
    
    def _fix_missing_symbol(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Fix missing symbol by adding appropriate import."""
        actions = []
        
        symbol = issue.symbol or "unknown_symbol"
        
        # Try to determine correct import
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
            'Tuple': 'from typing import Tuple',
            'Set': 'from typing import Set',
            'Union': 'from typing import Union',
            'Type': 'from typing import Type',
            'Callable': 'from typing import Callable',
            'Iterator': 'from typing import Iterator',
            'Generator': 'from typing import Generator',
            'defaultdict': 'from collections import defaultdict',
            'Counter': 'from collections import Counter',
            'dataclass': 'from dataclasses import dataclass',
            'field': 'from dataclasses import field',
            'abstractmethod': 'from abc import abstractmethod',
            'ABC': 'from abc import ABC',
            'Enum': 'from enum import Enum',
            'logging': 'import logging',
            'asyncio': 'import asyncio',
            'requests': 'import requests',
            'numpy': 'import numpy as np',
            'pandas': 'import pandas as pd',
        }
        
        if symbol in common_imports:
            import_statement = common_imports[symbol]
        else:
            # Try to infer from context
            import_statement = self._infer_import(symbol, context)
        
        # Read current file content for backup
        file_content = ""
        try:
            with open(issue.file, 'r') as f:
                file_content = f.read()
        except Exception:
            pass
        
        # Add import at top of file after existing imports
        new_content = self._add_import_to_content(file_content, import_statement)
        
        actions.append(Action(
            type="modify_file",
            target=issue.file,
            content=new_content,
            backup_content=file_content,
            metadata={"added_import": import_statement, "symbol": symbol}
        ))
        
        return actions
    
    def _fix_import_anxiety(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Fix import anxiety by organizing and cleaning imports."""
        actions = []
        
        if not issue.file:
            return actions
        
        try:
            with open(issue.file, 'r') as f:
                content = f.read()
            
            # Parse AST to analyze imports
            tree = ast.parse(content)
            
            # Collect and organize imports
            stdlib_imports = []
            third_party_imports = []
            local_imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_line = f"import {alias.name}"
                        if alias.asname:
                            import_line += f" as {alias.asname}"
                        
                        if self._is_stdlib(alias.name):
                            stdlib_imports.append(import_line)
                        elif self._is_local(alias.name):
                            local_imports.append(import_line)
                        else:
                            third_party_imports.append(import_line)
                            
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module = node.module
                        names = []
                        for alias in node.names:
                            if alias.name == '*':
                                names.append('*')
                            else:
                                name = alias.name
                                if alias.asname:
                                    name += f" as {alias.asname}"
                                names.append(name)
                        
                        if len(names) == 1:
                            import_line = f"from {module} import {names[0]}"
                        else:
                            import_line = f"from {module} import {', '.join(names)}"
                        
                        if self._is_stdlib(module):
                            stdlib_imports.append(import_line)
                        elif self._is_local(module):
                            local_imports.append(import_line)
                        else:
                            third_party_imports.append(import_line)
            
            # Remove duplicates and sort each group
            stdlib_imports = sorted(list(set(stdlib_imports)))
            third_party_imports = sorted(list(set(third_party_imports)))
            local_imports = sorted(list(set(local_imports)))
            
            # Create organized import section with proper spacing
            organized_imports = []
            if stdlib_imports:
                organized_imports.extend(stdlib_imports)
            if third_party_imports:
                if organized_imports:
                    organized_imports.append("")  # Blank line
                organized_imports.extend(third_party_imports)
            if local_imports:
                if organized_imports:
                    organized_imports.append("")  # Blank line
                organized_imports.extend(local_imports)
            
            organized_imports_str = "\n".join(organized_imports)
            
            # Remove old imports and add organized ones
            new_content = self._replace_imports_in_content(content, organized_imports_str)
            
            actions.append(Action(
                type="modify_file", 
                target=issue.file,
                content=new_content,
                backup_content=content,
                metadata={
                    "organized_imports": len(stdlib_imports) + len(third_party_imports) + len(local_imports),
                    "stdlib": len(stdlib_imports),
                    "third_party": len(third_party_imports),
                    "local": len(local_imports)
                }
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not analyze imports in {issue.file}: {e}")
        
        return actions
    
    def _fix_unused_imports(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Remove unused imports."""
        actions = []
        
        if not issue.file or not issue.evidence:
            return actions
        
        unused_items = issue.evidence.get('unused_items', [])
        if not unused_items:
            return actions
        
        try:
            with open(issue.file, 'r') as f:
                lines = f.readlines()
            
            # Remove lines containing unused imports
            new_lines = []
            removed_count = 0
            
            for line in lines:
                should_remove = False
                for unused in unused_items:
                    # More precise matching
                    if (f"import {unused}" in line and f"import {unused}." not in line) or \
                       (f"from {unused} import" in line) or \
                       (f"import {unused} " in line) or \
                       (line.strip() == f"import {unused}"):
                        should_remove = True
                        removed_count += 1
                        break
                
                if not should_remove:
                    new_lines.append(line)
            
            new_content = ''.join(new_lines)
            
            actions.append(Action(
                type="modify_file",
                target=issue.file, 
                content=new_content,
                backup_content=''.join(lines),
                metadata={"removed_imports": unused_items, "count": removed_count}
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not remove unused imports from {issue.file}: {e}")
        
        return actions
    
    def _fix_import_error(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Fix import errors."""
        # This would handle cases like circular imports, module not found, etc.
        # For now, return empty list - would need more sophisticated handling
        return []
    
    def _replace_imports_in_content(self, content: str, new_imports: str) -> str:
        """Replace imports section in file content."""
        lines = content.split('\n')
        
        # Find the extent of import statements
        first_import = -1
        last_import = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and not stripped.startswith('#'):
                if first_import == -1:
                    first_import = i
                last_import = i
        
        if first_import != -1:
            # Replace import section
            new_lines = lines[:first_import] + new_imports.split('\n') + lines[last_import+1:]
            return '\n'.join(new_lines)
        else:
            # Add imports at the beginning (after shebang/encoding if present)
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('#!') or line.startswith('# -*- coding'):
                    insert_pos = i + 1
                else:
                    break
            
            if insert_pos > 0:
                new_lines = lines[:insert_pos] + [''] + new_imports.split('\n') + [''] + lines[insert_pos:]
            else:
                new_lines = new_imports.split('\n') + [''] + lines
            
            return '\n'.join(new_lines)
    
    def _add_import_to_content(self, content: str, import_statement: str) -> str:
        """Add an import statement to the content."""
        lines = content.split('\n')
        
        # Find where to insert the import
        last_import = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and not stripped.startswith('#'):
                last_import = i
        
        if last_import != -1:
            # Add after last import
            lines.insert(last_import + 1, import_statement)
        else:
            # Add at the beginning (after shebang/encoding if present)
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.startswith('#!') or line.startswith('# -*- coding'):
                    insert_pos = i + 1
                else:
                    break
            
            if insert_pos > 0:
                lines.insert(insert_pos, '')
                lines.insert(insert_pos + 1, import_statement)
            else:
                lines.insert(0, import_statement)
                lines.insert(1, '')
        
        return '\n'.join(lines)
    
    def _infer_import(self, symbol: str, context: Optional[Dict[str, Any]]) -> str:
        """Try to infer the correct import for a symbol."""
        # Check if it looks like a class (capitalized)
        if symbol[0].isupper():
            # Might be from typing or a class
            if symbol.endswith('Error') or symbol.endswith('Exception'):
                # Generate a reasonable fallback for exceptions
                return f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    class {symbol}(Exception): pass"
            elif symbol in {'List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union', 'Any', 'Type', 'Callable'}:
                # Common typing imports
                return f"from typing import {symbol}"
            else:
                # Generate a type stub for unknown classes
                return f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from typing import Any\n    {symbol}: Any"
        else:
            # Likely a function or variable - generate a safe stub
            return f"from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    def {symbol}(*args, **kwargs): ..."
    
    def _is_stdlib(self, module: str) -> bool:
        """Check if a module is from the standard library."""
        stdlib_modules = {
            'os', 'sys', 're', 'json', 'math', 'random', 'datetime', 'time',
            'collections', 'itertools', 'functools', 'typing', 'pathlib',
            'urllib', 'http', 'email', 'html', 'xml', 'csv', 'io', 'sqlite3',
            'hashlib', 'hmac', 'secrets', 'uuid', 'copy', 'pickle', 'shelve',
            'tempfile', 'glob', 'shutil', 'zipfile', 'tarfile', 'gzip',
            'logging', 'warnings', 'traceback', 'inspect', 'ast', 'abc',
            'enum', 'dataclasses', 'contextlib', 'asyncio', 'threading',
            'multiprocessing', 'subprocess', 'queue', 'socket', 'ssl',
            'platform', 'locale', 'gettext', 'argparse', 'configparser'
        }
        
        # Check the base module name
        base_module = module.split('.')[0]
        return base_module in stdlib_modules
    
    def _is_local(self, module: str) -> bool:
        """Check if a module is a local import."""
        return module.startswith('.') or module.startswith('..')
    
    def _calculate_confidence(self, issue: Issue, context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence for the fix."""
        if issue.kind == "missing_symbol":
            symbol = issue.symbol or ""
            # High confidence for common symbols
            if symbol in ['json', 'os', 'sys', 're', 'Path', 'List', 'Dict', 'Optional']:
                return 0.95
            else:
                return 0.7
        elif issue.kind == "unused_import":
            return 0.9  # High confidence in removing unused imports
        elif issue.kind == "import_anxiety":
            return 0.8  # Good confidence in organizing imports
        else:
            return 0.6
    
    def _get_fix_description(self, issue: Issue) -> str:
        """Get a description of the fix."""
        if issue.kind == "missing_symbol":
            return f"Add import for missing symbol '{issue.symbol}'"
        elif issue.kind == "unused_import":
            return "Remove unused imports"
        elif issue.kind == "import_anxiety":
            return "Organize and clean up imports"
        elif issue.kind == "import_error":
            return "Fix import error"
        else:
            return "Fix import issue"
    
    def _get_side_effects(self, issue: Issue) -> List[str]:
        """Get potential side effects of the fix."""
        if issue.kind == "missing_symbol":
            return ["May require installing missing package"]
        elif issue.kind == "unused_import":
            return ["May break code if import was actually used indirectly"]
        elif issue.kind == "import_anxiety":
            return ["May change import order which could affect initialization"]
        else:
            return []
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Import issues should be fixed before other issues."""
        return []  # No dependencies - imports are usually foundational