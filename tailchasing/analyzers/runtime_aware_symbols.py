"""
Runtime-aware symbol detection that understands Python execution contexts.

This analyzer properly handles:
- __file__ in scripts vs modules
- __name__ == "__main__" patterns
- Runtime-available symbols
- Mock and test contexts
"""

import ast
import os
from typing import Set, Dict, List, Any, Optional
from pathlib import Path

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


class RuntimeAwareSymbolAnalyzer(BaseAnalyzer):
    """Detects truly undefined symbols while understanding runtime contexts."""
    
    name = "runtime_aware_symbols"
    
    def __init__(self):
        super().__init__()
        # Python runtime symbols that are always available
        self.runtime_symbols = {
            '__file__',      # Available when script is run
            '__name__',      # Always available
            '__package__',   # Available in packages
            '__doc__',       # Module docstring
            '__cached__',    # Bytecode cache
            '__spec__',      # Module spec
            '__loader__',    # Module loader
            '__builtins__',  # Builtin namespace
            '__import__',    # Import function
            '__debug__',     # Debug flag
            '__annotations__',  # Type annotations
        }
        
        # Symbols available in specific contexts
        self.context_symbols = {
            'test': {
                'pytest', 'unittest', 'mock', 'patch', 'fixture',
                'setup', 'teardown', 'skipIf', 'skipUnless',
                'TestCase', 'TestSuite', 'TestLoader'
            },
            'setup': {
                'setup', 'find_packages', 'Extension', 'setup_requires',
                'install_requires', 'extras_require', 'python_requires'
            },
            'main': {
                'sys', 'os', 'argparse', 'logging', 'pathlib'
            }
        }
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Find undefined symbols with runtime awareness."""
        issues = []
        
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
            
            # Determine file context
            file_context = self._determine_file_context(file, tree)
            
            # Check for undefined symbols
            visitor = RuntimeAwareSymbolVisitor(file, ctx, self, file_context)
            visitor.visit(tree)
            issues.extend(visitor.issues)
        
        return issues
    
    def _determine_file_context(self, file: str, tree: ast.AST) -> Dict[str, Any]:
        """Determine the runtime context of a file."""
        context = {
            'is_script': False,
            'is_test': False,
            'is_setup': False,
            'is_module': True,
            'has_main_guard': False,
            'is_mock': False,
            'available_symbols': set(self.runtime_symbols)
        }
        
        file_path = Path(file)
        file_name = file_path.name.lower()
        
        # Check if it's a test file
        if 'test' in file_name or 'test' in str(file_path):
            context['is_test'] = True
            context['available_symbols'].update(self.context_symbols['test'])
        
        # Check if it's a setup file
        if file_name == 'setup.py' or 'setup' in file_name:
            context['is_setup'] = True
            context['is_script'] = True  # setup.py is typically run as script
            context['available_symbols'].update(self.context_symbols['setup'])
        
        # Check if it's a mock file
        if 'mock' in file_name:
            context['is_mock'] = True
        
        # Check for main guard (indicates script usage)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if self._is_main_guard(node):
                    context['has_main_guard'] = True
                    context['is_script'] = True
                    context['available_symbols'].update(self.context_symbols['main'])
        
        # Scripts and files with main guard have __file__ available
        if context['is_script'] or context['has_main_guard']:
            context['available_symbols'].add('__file__')
        
        # Check if file is directly executable
        if file_path.suffix == '.py':
            # Check for shebang
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#!') and 'python' in first_line:
                        context['is_script'] = True
                        context['available_symbols'].add('__file__')
            except:
                pass
        
        return context
    
    def _is_main_guard(self, node: ast.If) -> bool:
        """Check if an if statement is a main guard."""
        if isinstance(node.test, ast.Compare):
            if isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__':
                if len(node.test.comparators) == 1:
                    comp = node.test.comparators[0]
                    if isinstance(comp, ast.Constant) and comp.value == '__main__':
                        return True
        return False


class RuntimeAwareSymbolVisitor(ast.NodeVisitor):
    """Visitor that checks for undefined symbols with runtime awareness."""
    
    def __init__(self, file: str, ctx: AnalysisContext, analyzer: RuntimeAwareSymbolAnalyzer, 
                 file_context: Dict[str, Any]):
        self.file = file
        self.ctx = ctx
        self.analyzer = analyzer
        self.file_context = file_context
        self.issues: List[Issue] = []
        
        # Track defined symbols
        self.defined_symbols: Set[str] = set(file_context['available_symbols'])
        self.imported_symbols: Set[str] = set()
        self.local_symbols: List[Set[str]] = [set()]  # Stack of local scopes
        
    def visit_Import(self, node: ast.Import):
        """Track imported modules."""
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.imported_symbols.add(name)
            self.defined_symbols.add(name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track imported symbols."""
        for alias in node.names:
            if alias.name == '*':
                # Star import - we can't track everything
                pass
            else:
                name = alias.asname or alias.name
                self.imported_symbols.add(name)
                self.defined_symbols.add(name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function definitions and parameters."""
        self.defined_symbols.add(node.name)
        
        # Create new scope with parameters
        new_scope = set()
        for arg in node.args.args:
            new_scope.add(arg.arg)
        if node.args.vararg:
            new_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            new_scope.add(node.args.kwarg.arg)
        
        self.local_symbols.append(new_scope)
        self.generic_visit(node)
        self.local_symbols.pop()
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class definitions."""
        self.defined_symbols.add(node.name)
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name):
        """Check for undefined names."""
        if isinstance(node.ctx, ast.Load):
            name = node.id
            
            # Check if defined
            if not self._is_defined(name):
                # Special handling for __file__
                if name == '__file__':
                    # Only report if not in script context
                    if not self.file_context['is_script'] and not self.file_context['has_main_guard']:
                        issue = Issue(
                            kind="undefined_symbol_runtime",
                            message=f"__file__ used in module context (only available when run as script)",
                            severity=2,
                            file=self.file,
                            line=safe_get_lineno(node),
                            symbol=name,
                            evidence={
                                "context": "module",
                                "has_main_guard": self.file_context['has_main_guard']
                            },
                            suggestions=[
                                "Add if __name__ == '__main__': guard",
                                "Use importlib.resources for resource access",
                                "Use Path(__file__).parent only in script context"
                            ],
                            confidence=0.8
                        )
                        self.issues.append(issue)
                
                # Check for other undefined symbols
                elif name not in self.analyzer.runtime_symbols:
                    # Skip if it's a mock/test pattern
                    if self.file_context['is_test'] or self.file_context['is_mock']:
                        if name.startswith('Mock') or name.startswith('mock_'):
                            return  # Expected in test context
                    
                    issue = Issue(
                        kind="undefined_symbol",
                        message=f"Undefined symbol: {name}",
                        severity=3,
                        file=self.file,
                        line=safe_get_lineno(node),
                        symbol=name,
                        evidence={
                            "file_context": self.file_context['is_test'] or self.file_context['is_mock']
                        },
                        suggestions=self._get_suggestions(name),
                        confidence=0.9
                    )
                    self.issues.append(issue)
        
        elif isinstance(node.ctx, ast.Store):
            # Add to current scope
            self.local_symbols[-1].add(node.id)
        
        self.generic_visit(node)
    
    def _is_defined(self, name: str) -> bool:
        """Check if a name is defined in any scope."""
        # Check local scopes
        for scope in self.local_symbols:
            if name in scope:
                return True
        
        # Check defined symbols
        if name in self.defined_symbols:
            return True
        
        # Check imported
        if name in self.imported_symbols:
            return True
        
        # Check builtins
        import builtins
        if hasattr(builtins, name):
            return True
        
        return False
    
    def _get_suggestions(self, name: str) -> List[str]:
        """Get suggestions for undefined symbol."""
        suggestions = []
        
        # Common import suggestions
        import_map = {
            'Path': "from pathlib import Path",
            'List': "from typing import List",
            'Dict': "from typing import Dict",
            'Optional': "from typing import Optional",
            'Union': "from typing import Union",
            'Any': "from typing import Any",
            'Tuple': "from typing import Tuple",
            'Set': "from typing import Set",
        }
        
        if name in import_map:
            suggestions.append(f"Add: {import_map[name]}")
        
        # Check for typos in defined symbols
        from difflib import get_close_matches
        all_defined = self.defined_symbols | self.imported_symbols
        close_matches = get_close_matches(name, all_defined, n=2, cutoff=0.8)
        for match in close_matches:
            suggestions.append(f"Did you mean '{match}'?")
        
        return suggestions if suggestions else ["Check import statements", "Define the symbol"]