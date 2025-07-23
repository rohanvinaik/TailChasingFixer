"""
Enhanced fix suggestions generator for TailChasingFixer.

This module provides detailed, actionable code suggestions for each type of issue.
"""

import ast
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import textwrap


class FixSuggestionGenerator:
    """Generates detailed fix suggestions with code examples."""
    
    def __init__(self):
        self.suggestion_templates = {
            'semantic_duplicate_function': self._suggest_semantic_duplicate_fix,
            'duplicate_function': self._suggest_duplicate_fix,
            'phantom_function': self._suggest_phantom_fix,
            'missing_symbol': self._suggest_missing_symbol_fix,
            'circular_import': self._suggest_circular_import_fix,
            'prototype_fragmentation': self._suggest_prototype_fix,
            'hallucination_cascade': self._suggest_hallucination_fix,
            'import_anxiety': self._suggest_import_anxiety_fix,
            'wrapper_abstraction': self._suggest_wrapper_fix,
            'context_window_thrashing': self._suggest_context_thrashing_fix,
        }
    
    def generate_suggestions(self, issue: 'Issue', context: Optional[Dict] = None) -> List[str]:
        """Generate detailed suggestions for an issue."""
        generator = self.suggestion_templates.get(issue.kind)
        if generator:
            return generator(issue, context or {})
        return self._default_suggestions(issue)
    
    def _suggest_semantic_duplicate_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for semantic duplicates."""
        evidence = issue.evidence
        func1, func2 = evidence['pair']
        same_file = evidence.get('same_file', False)
        
        suggestions = []
        
        # 1. Immediate action with code example
        if same_file:
            suggestions.append(
                f"IMMEDIATE: Merge duplicate functions in {func1['file']}:\n"
                f"```python\n"
                f"# Remove {func2['name']} and update all calls to use {func1['name']}\n"
                f"# Or create a single function that both can use:\n"
                f"def _shared_implementation(*args, **kwargs):\n"
                f"    # Move common logic here\n"
                f"    pass\n\n"
                f"def {func1['name']}(*args, **kwargs):\n"
                f"    return _shared_implementation(*args, **kwargs)\n"
                f"```"
            )
        else:
            suggestions.append(
                f"IMMEDIATE: Extract shared functionality:\n"
                f"```python\n"
                f"# In a shared module (e.g., {self._suggest_shared_module_name(func1, func2)}):\n"
                f"def shared_{self._extract_common_name(func1['name'], func2['name'])}(*args, **kwargs):\n"
                f"    # Move common implementation here\n"
                f"    pass\n\n"
                f"# Then in {func1['file']}:\n"
                f"from {self._suggest_shared_module_name(func1, func2)} import shared_...\n"
                f"```"
            )
        
        # 2. Analysis suggestion
        suggestions.append(
            f"ANALYZE: These functions have {evidence['z_score']:.1f} semantic similarity:\n"
            f"- {func1['name']} at {func1['file']}:{func1['line']}\n"
            f"- {func2['name']} at {func2['file']}:{func2['line']}\n"
            f"Review if they truly serve different purposes."
        )
        
        # 3. Refactoring approach
        if evidence.get('channel_contributions'):
            dominant = self._get_dominant_channel(evidence['channel_contributions'])
            suggestions.append(
                f"REFACTOR: High similarity in {dominant} suggests these functions "
                f"share the same core algorithm. Consider creating a base implementation "
                f"with parameters for variations."
            )
        
        # 4. Prevention
        suggestions.append(
            "PREVENT: Before implementing new functions, search for existing ones with:\n"
            "- Similar names (even with different prefixes)\n"
            "- Similar parameter patterns\n"
            "- Similar return types"
        )
        
        return suggestions
    
    def _suggest_duplicate_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for structural duplicates."""
        suggestions = []
        
        suggestions.append(
            "IMMEDIATE: These functions are structurally identical. Options:\n"
            "1. Delete all but one and update imports\n"
            "2. If they need different names, use aliasing:\n"
            "   ```python\n"
            "   # Keep one implementation\n"
            "   def original_function(...):\n"
            "       ...\n"
            "   \n"
            "   # Create aliases for other names\n"
            "   alternative_name = original_function\n"
            "   ```"
        )
        
        suggestions.append(
            "CHECK: Run this command to find all usages:\n"
            "```bash\n"
            "grep -r 'function_name' --include='*.py' .\n"
            "```"
        )
        
        return suggestions
    
    def _suggest_phantom_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for phantom/placeholder functions."""
        placeholder_type = issue.evidence.get('placeholder_type', 'unknown')
        
        suggestions = []
        
        if placeholder_type == 'pass-only_function':
            suggestions.append(
                f"IMPLEMENT: Function '{issue.symbol}' needs implementation:\n"
                f"```python\n"
                f"def {issue.symbol}(self, *args, **kwargs):\n"
                f"    # TODO: Implement based on function name and context\n"
                f"    # Consider:\n"
                f"    # 1. What should this function return?\n"
                f"    # 2. What side effects should it have?\n"
                f"    # 3. What validation is needed?\n"
                f"    raise NotImplementedError(\n"
                f"        'Implementation needed for {issue.symbol}'\n"
                f"    )\n"
                f"```"
            )
        
        elif placeholder_type == 'notimplementederror_stub':
            suggestions.append(
                f"IMPLEMENT: Replace NotImplementedError with actual logic:\n"
                f"1. Understand the interface requirements\n"
                f"2. Look at similar functions for patterns\n"
                f"3. Implement incrementally with tests"
            )
        
        suggestions.append(
            "ALTERNATIVES:\n"
            "- If this is an interface, use ABC:\n"
            "  ```python\n"
            "  from abc import ABC, abstractmethod\n"
            "  \n"
            "  class MyInterface(ABC):\n"
            "      @abstractmethod\n"
            "      def my_method(self): pass\n"
            "  ```\n"
            "- If truly optional, document why:\n"
            "  ```python\n"
            "  def optional_hook(self):\n"
            "      \"\"\"Override in subclasses if needed.\"\"\"\n"
            "      pass\n"
            "  ```"
        )
        
        return suggestions
    
    def _suggest_missing_symbol_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for missing symbols."""
        symbol = issue.evidence.get('symbol', issue.symbol)
        
        suggestions = [
            f"LOCATE: Symbol '{symbol}' is imported but not defined. Check:\n"
            f"1. Is it defined in another module? Update import path\n"
            f"2. Was it renamed? Update import statement\n"
            f"3. Was it removed? Remove import or restore symbol\n"
            f"4. Is it from an external package? Install the package",
            
            f"SEARCH: Find where '{symbol}' might be defined:\n"
            f"```bash\n"
            f"# Search in current project\n"
            f"grep -r 'def {symbol}' --include='*.py' .\n"
            f"grep -r 'class {symbol}' --include='*.py' .\n"
            f"\n"
            f"# Check if it's from an installed package\n"
            f"python -c \"import {symbol.split('.')[0]}\"\n"
            f"```",
            
            "FIX OPTIONS:\n"
            "- Remove the import if unused\n"
            "- Implement the missing symbol\n"
            "- Fix the import path\n"
            "- Install missing dependency"
        ]
        
        return suggestions
    
    def _suggest_circular_import_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for circular imports."""
        cycle = issue.evidence.get('cycle', [])
        
        suggestions = [
            "IMMEDIATE FIX - Move imports inside functions:\n"
            "```python\n"
            "# Instead of top-level import\n"
            "# from module_b import something\n"
            "\n"
            "def my_function():\n"
            "    from module_b import something  # Import when needed\n"
            "    return something()\n"
            "```",
            
            "BETTER FIX - Restructure code:\n"
            "1. Create an interface module for shared types\n"
            "2. Move common dependencies to a separate module\n"
            "3. Use dependency injection instead of direct imports",
            
            f"ANALYZE: Circular dependency chain:\n{' -> '.join(cycle)}\n"
            "Consider which module should truly own each piece of functionality."
        ]
        
        if len(cycle) > 3:
            suggestions.append(
                "REFACTOR: Long circular chain suggests architectural issues.\n"
                "Consider introducing a service layer or facade pattern."
            )
        
        return suggestions
    
    def _suggest_prototype_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for prototype fragmentation."""
        pattern = issue.evidence.get('pattern', 'unknown')
        functions = issue.evidence.get('functions', [])
        
        suggestions = [
            f"CONSOLIDATE: {len(functions)} similar implementations detected.\n"
            f"Pattern: {pattern}\n\n"
            f"1. Identify the most complete implementation\n"
            f"2. Extract common behavior to a base class or shared function\n"
            f"3. Replace duplicates with calls to shared implementation",
            
            "EXAMPLE CONSOLIDATION:\n"
            "```python\n"
            "# Base implementation\n"
            "class BaseProcessor:\n"
            "    def process(self, data, **options):\n"
            "        # Common logic here\n"
            "        result = self._core_algorithm(data)\n"
            "        return self._format_output(result, **options)\n"
            "\n"
            "# Specific variants\n"
            "class VariantA(BaseProcessor):\n"
            "    def _format_output(self, result, **options):\n"
            "        # Variant-specific formatting\n"
            "        pass\n"
            "```"
        ]
        
        if len(functions) > 5:
            suggestions.append(
                "HIGH PRIORITY: With this many variants, consider:\n"
                "- Strategy pattern for different behaviors\n"
                "- Configuration-driven approach\n"
                "- Plugin architecture"
            )
        
        return suggestions
    
    def _suggest_hallucination_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for hallucination cascades."""
        entities = issue.evidence.get('entities', [])
        
        suggestions = [
            f"CRITICAL: Detected fictional subsystem with {len(entities)} interdependent classes.\n"
            f"Entities: {', '.join(entities[:5])}{'...' if len(entities) > 5 else ''}\n\n"
            f"This often happens when AI assistants create classes to satisfy errors.",
            
            "IMMEDIATE ACTIONS:\n"
            "1. Map what each class actually does\n"
            "2. Check if existing functionality can replace them\n"
            "3. Document legitimate use cases before removing",
            
            "VALIDATION QUESTIONS:\n"
            "- Do these classes have any external users?\n"
            "- Do they implement real business logic?\n"
            "- Could existing stdlib/framework features replace them?",
            
            "CLEANUP APPROACH:\n"
            "```python\n"
            "# 1. Find all usages\n"
            "# grep -r 'ClassName' --include='*.py' .\n"
            "\n"
            "# 2. For each class, check if it can be replaced with:\n"
            "# - Standard library (e.g., dataclasses, enums)\n"
            "# - Existing project code\n"
            "# - Simple functions instead of classes\n"
            "```"
        ]
        
        return suggestions
    
    def _suggest_import_anxiety_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for import anxiety."""
        suggestions = [
            "CLEAN IMPORTS:\n"
            "```python\n"
            "# Use tools to automatically remove unused imports:\n"
            "# pip install autoflake\n"
            "# autoflake --remove-unused-variables --in-place file.py\n"
            "```",
            
            "IMPORT BEST PRACTICES:\n"
            "1. Import only what you use\n"
            "2. Prefer explicit imports over wildcards\n"
            "3. Group imports: stdlib, third-party, local\n"
            "4. Use TYPE_CHECKING for type-only imports",
            
            "PREVENT: Configure your IDE/linter to warn about unused imports:\n"
            "```yaml\n"
            "# .flake8 or setup.cfg\n"
            "[flake8]\n"
            "select = F401  # unused imports\n"
            "```"
        ]
        
        return suggestions
    
    def _suggest_wrapper_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for wrapper abstractions."""
        suggestions = [
            "EVALUATE: Is this wrapper adding value?\n"
            "- Does it simplify the interface?\n"
            "- Does it add validation/transformation?\n"
            "- Does it provide better naming?\n"
            "If NO to all: remove the wrapper",
            
            "REFACTOR OPTIONS:\n"
            "1. Direct delegation (if wrapper adds no value):\n"
            "   ```python\n"
            "   # Instead of wrapper, use assignment\n"
            "   process_data = original_function\n"
            "   ```\n"
            "2. Partial application (if only fixing some args):\n"
            "   ```python\n"
            "   from functools import partial\n"
            "   process_data = partial(original_function, format='json')\n"
            "   ```"
        ]
        
        return suggestions
    
    def _suggest_context_thrashing_fix(self, issue: 'Issue', context: Dict) -> List[str]:
        """Suggest fixes for context window thrashing."""
        suggestions = [
            "MERGE DUPLICATES: Functions separated by many lines often duplicate logic.\n"
            "1. Use IDE's 'Find Similar Code' feature\n"
            "2. Extract common patterns to module top\n"
            "3. Add cross-references in docstrings",
            
            "PREVENT CONTEXT LOSS:\n"
            "- Keep related functions together\n"
            "- Use descriptive names that indicate relationships\n"
            "- Add 'See Also' sections in docstrings\n"
            "- Create a module-level docstring describing all functions",
            
            "ORGANIZE CODE:\n"
            "```python\n"
            "# Group related functionality\n"
            "class DataProcessor:\n"
            "    \"\"\"Groups all data processing functions.\"\"\"\n"
            "    \n"
            "    @staticmethod\n"
            "    def process_type_a(data):\n"
            "        \"\"\"Process type A data.\"\"\"\n"
            "        pass\n"
            "    \n"
            "    @staticmethod\n"
            "    def process_type_b(data):\n"
            "        \"\"\"Process type B data (similar to type A).\"\"\"\n"
            "        pass\n"
            "```"
        ]
        
        return suggestions
    
    def _default_suggestions(self, issue: 'Issue') -> List[str]:
        """Default suggestions for unhandled issue types."""
        return [
            f"Review {issue.kind} issue at {issue.file}:{issue.line}",
            "Consider refactoring to improve code quality",
            "Add tests to prevent regression"
        ]
    
    def _suggest_shared_module_name(self, func1: Dict, func2: Dict) -> str:
        """Suggest a name for a shared module."""
        # Extract common path components
        path1 = Path(func1['file'])
        path2 = Path(func2['file'])
        
        common = []
        for p1, p2 in zip(path1.parts, path2.parts):
            if p1 == p2:
                common.append(p1)
            else:
                break
        
        if common:
            return '.'.join(common) + '.shared'
        return 'shared.common'
    
    def _extract_common_name(self, name1: str, name2: str) -> str:
        """Extract common parts from two function names."""
        # Simple approach - find common prefix/suffix
        # Could be enhanced with better string matching
        for i in range(min(len(name1), len(name2))):
            if name1[i] != name2[i]:
                if i > 3:  # At least some common prefix
                    return name1[:i].rstrip('_')
                break
        
        # Try common suffix
        for i in range(1, min(len(name1), len(name2))):
            if name1[-i] != name2[-i]:
                if i > 3:
                    return name1[-i:].lstrip('_')
                break
        
        return 'common'
    
    def _get_dominant_channel(self, contributions: Dict[str, float]) -> str:
        """Get the most significant channel from contributions."""
        if not contributions:
            return 'implementation'
        
        return max(contributions.items(), key=lambda x: x[1])[0]


class InteractiveFixGenerator:
    """Generate interactive fix scripts."""
    
    def generate_fix_script(self, issues: List['Issue'], output_path: Path) -> str:
        """Generate an executable Python script to fix issues."""
        script = '''#!/usr/bin/env python3
"""
Auto-generated fix script for tail-chasing issues.
Review each fix before applying.
"""

import os
import sys
from pathlib import Path

class TailChasingFixer:
    def __init__(self):
        self.fixes_applied = 0
        self.fixes_skipped = 0
    
    def fix_all(self):
        """Apply all fixes interactively."""
        fixes = [
'''
        
        # Add each fix
        for i, issue in enumerate(issues):
            if issue.kind in ['semantic_duplicate_function', 'duplicate_function']:
                script += self._generate_duplicate_fix_code(issue, i)
            elif issue.kind == 'phantom_function':
                script += self._generate_phantom_fix_code(issue, i)
        
        script += '''
        ]
        
        for i, fix in enumerate(fixes):
            print(f"\\n[Fix {i+1}/{len(fixes)}]")
            print(f"Issue: {fix['issue']}")
            print(f"File: {fix['file']}")
            print(f"Suggestion: {fix['suggestion']}")
            
            response = input("\\nApply this fix? [y/N/q]: ").lower()
            if response == 'q':
                break
            elif response == 'y':
                fix['apply']()
                self.fixes_applied += 1
            else:
                self.fixes_skipped += 1
        
        print(f"\\nCompleted: {self.fixes_applied} fixes applied, {self.fixes_skipped} skipped")

if __name__ == "__main__":
    fixer = TailChasingFixer()
    fixer.fix_all()
'''
        
        return script
    
    def _generate_duplicate_fix_code(self, issue: 'Issue', index: int) -> str:
        """Generate code to fix duplicate functions."""
        evidence = issue.evidence
        func1, func2 = evidence.get('pair', ({}, {}))
        
        return f'''
            {{
                'issue': '{issue.kind}',
                'file': '{issue.file}',
                'suggestion': 'Merge {func1.get("name", "function1")} and {func2.get("name", "function2")}',
                'apply': lambda: self._merge_functions(
                    '{func1.get("file", "")}', 
                    '{func1.get("name", "")}', 
                    '{func2.get("file", "")}', 
                    '{func2.get("name", "")}'
                )
            }},
'''
    
    def _generate_phantom_fix_code(self, issue: 'Issue', index: int) -> str:
        """Generate code to fix phantom functions."""
        return f'''
            {{
                'issue': '{issue.kind}',
                'file': '{issue.file}',
                'suggestion': 'Implement or remove {issue.symbol}',
                'apply': lambda: self._fix_phantom(
                    '{issue.file}', 
                    {issue.line}, 
                    '{issue.symbol}'
                )
            }},
'''


# Export the main class
__all__ = ['FixSuggestionGenerator', 'InteractiveFixGenerator']
