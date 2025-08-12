"""
Missing Symbol Resolver - Suggests where to implement missing symbols.

This analyzer not only detects missing symbols but also suggests the most
appropriate location to implement them based on usage patterns and module structure.
"""

import ast
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from ..core.issues import Issue
from .base import BaseAnalyzer


@dataclass
class MissingSymbol:
    """Represents a missing symbol and its usage context."""
    
    symbol_name: str
    symbol_type: str  # 'class', 'function', 'variable', 'module', 'exception'
    usage_locations: List[Tuple[str, int]]  # (file_path, line_number)
    import_attempts: Set[str]  # Modules from which import was attempted
    usage_context: List[str]  # How it's used (called, instantiated, raised, etc.)
    suggested_location: Optional[str] = None
    suggested_implementation: Optional[str] = None


@dataclass
class ImplementationSuggestion:
    """Suggestion for implementing a missing symbol."""
    
    symbol_name: str
    suggested_file: str
    suggested_module: str
    implementation_template: str
    reasoning: str
    related_symbols: List[str]


class MissingSymbolResolver(BaseAnalyzer):
    """
    Resolves missing symbols by suggesting implementation locations and templates.
    
    Features:
    1. Detects missing symbols and their usage patterns
    2. Suggests appropriate implementation locations
    3. Generates implementation templates
    4. Groups related missing symbols
    5. Identifies common missing patterns (exceptions, utilities, etc.)
    """
    
    def __init__(self):
        super().__init__()
        self.missing_symbols: Dict[str, MissingSymbol] = {}
        self.symbol_usage: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
        self.module_symbols: Dict[str, Set[str]] = defaultdict(set)
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze(self, context) -> List[Issue]:
        """Analyze and resolve missing symbols."""
        issues = []
        
        # First, build complete symbol table
        self._build_symbol_inventory(context)
        
        # Find all symbol usages
        self._find_symbol_usages(context)
        
        # Identify missing symbols
        missing = self._identify_missing_symbols(context)
        
        # Group related missing symbols
        symbol_groups = self._group_related_symbols(missing)
        
        # Generate implementation suggestions
        suggestions = self._generate_implementation_suggestions(missing, symbol_groups)
        
        # Create issues with solutions
        for symbol_name, symbol in missing.items():
            suggestion = suggestions.get(symbol_name)
            issues.append(self._create_missing_symbol_issue(symbol, suggestion))
        
        # Add summary issue if many missing symbols
        if len(missing) > 10:
            issues.append(self._create_summary_issue(missing, symbol_groups))
        
        return issues
    
    def _build_symbol_inventory(self, context):
        """Build inventory of all defined symbols."""
        for file_path, ast_tree in context.ast_index.items():
            module_name = self._path_to_module(file_path)
            
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.ClassDef):
                    self.module_symbols[module_name].add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    self.module_symbols[module_name].add(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_name = alias.asname if alias.asname else alias.name
                        self.module_symbols[module_name].add(imported_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imported_name = alias.asname if alias.asname else alias.name
                            self.module_symbols[module_name].add(imported_name)
    
    def _find_symbol_usages(self, context):
        """Find all symbol usages in the codebase."""
        for file_path, ast_tree in context.ast_index.items():
            for node in ast.walk(ast_tree):
                # Function/method calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        self.symbol_usage[node.func.id].append(
                            (file_path, node.lineno, 'call')
                        )
                    elif isinstance(node.func, ast.Attribute):
                        self.symbol_usage[node.func.attr].append(
                            (file_path, node.lineno, 'method_call')
                        )
                
                # Class instantiation (captured by Call above)
                
                # Exception raising
                elif isinstance(node, ast.Raise):
                    if node.exc:
                        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                            self.symbol_usage[node.exc.func.id].append(
                                (file_path, node.lineno, 'exception')
                            )
                        elif isinstance(node.exc, ast.Name):
                            self.symbol_usage[node.exc.id].append(
                                (file_path, node.lineno, 'exception')
                            )
                
                # Type annotations
                elif isinstance(node, ast.AnnAssign):
                    if node.annotation:
                        if isinstance(node.annotation, ast.Name):
                            self.symbol_usage[node.annotation.id].append(
                                (file_path, node.lineno, 'type_annotation')
                            )
                
                # Import attempts
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.import_graph[file_path].add(node.module)
                        for alias in node.names:
                            name = alias.name
                            self.symbol_usage[name].append(
                                (file_path, node.lineno, f'import_from:{node.module}')
                            )
    
    def _identify_missing_symbols(self, context) -> Dict[str, MissingSymbol]:
        """Identify symbols that are used but not defined."""
        missing = {}
        
        # Get all defined symbols
        all_defined = set()
        for symbols in self.module_symbols.values():
            all_defined.update(symbols)
        
        # Add Python builtins
        import builtins
        all_defined.update(dir(builtins))
        
        # Check each used symbol
        for symbol_name, usages in self.symbol_usage.items():
            if symbol_name not in all_defined:
                # Determine symbol type from usage
                symbol_type = self._infer_symbol_type(usages)
                
                # Extract import attempts
                import_attempts = set()
                usage_locations = []
                usage_contexts = []
                
                for file_path, line_no, context in usages:
                    if context.startswith('import_from:'):
                        import_attempts.add(context.split(':')[1])
                    else:
                        usage_locations.append((file_path, line_no))
                        usage_contexts.append(context)
                
                if usage_locations:  # Only if actually used, not just import attempted
                    missing[symbol_name] = MissingSymbol(
                        symbol_name=symbol_name,
                        symbol_type=symbol_type,
                        usage_locations=usage_locations,
                        import_attempts=import_attempts,
                        usage_context=usage_contexts
                    )
        
        return missing
    
    def _infer_symbol_type(self, usages: List[Tuple[str, int, str]]) -> str:
        """Infer the type of a symbol from its usage."""
        contexts = [usage[2] for usage in usages]
        
        if any('exception' in c for c in contexts):
            return 'exception'
        elif any('call' in c for c in contexts):
            if any(c[0].isupper() for c in contexts if isinstance(c, str)):
                return 'class'  # Capitalized calls are likely classes
            return 'function'
        elif any('type_annotation' in c for c in contexts):
            return 'class'
        else:
            return 'variable'
    
    def _group_related_symbols(self, missing: Dict[str, MissingSymbol]) -> Dict[str, List[str]]:
        """Group related missing symbols."""
        groups = defaultdict(list)
        
        # Group by common patterns
        for symbol_name, symbol in missing.items():
            # Exception group
            if symbol.symbol_type == 'exception' or 'Error' in symbol_name or 'Exception' in symbol_name:
                groups['exceptions'].append(symbol_name)
            
            # Validation group
            elif 'validate' in symbol_name.lower() or 'check' in symbol_name.lower():
                groups['validation'].append(symbol_name)
            
            # Handler group
            elif 'handler' in symbol_name.lower() or 'process' in symbol_name.lower():
                groups['handlers'].append(symbol_name)
            
            # Model group
            elif 'model' in symbol_name.lower() or symbol_name.endswith('Model'):
                groups['models'].append(symbol_name)
            
            # Utility group
            elif 'util' in symbol_name.lower() or 'helper' in symbol_name.lower():
                groups['utilities'].append(symbol_name)
            
            # Group by import attempts
            for module in symbol.import_attempts:
                groups[f"module:{module}"].append(symbol_name)
        
        return dict(groups)
    
    def _generate_implementation_suggestions(
        self, 
        missing: Dict[str, MissingSymbol],
        groups: Dict[str, List[str]]
    ) -> Dict[str, ImplementationSuggestion]:
        """Generate implementation suggestions for missing symbols."""
        suggestions = {}
        
        for symbol_name, symbol in missing.items():
            if symbol.symbol_type == 'exception':
                suggestion = self._suggest_exception_implementation(symbol, groups.get('exceptions', []))
            elif symbol.symbol_type == 'class':
                suggestion = self._suggest_class_implementation(symbol)
            elif symbol.symbol_type == 'function':
                suggestion = self._suggest_function_implementation(symbol)
            else:
                suggestion = self._suggest_variable_implementation(symbol)
            
            if suggestion:
                suggestions[symbol_name] = suggestion
        
        return suggestions
    
    def _suggest_exception_implementation(self, symbol: MissingSymbol, related: List[str]) -> ImplementationSuggestion:
        """Suggest implementation for missing exception."""
        # Determine module based on import attempts or usage location
        if symbol.import_attempts:
            module = list(symbol.import_attempts)[0]
            suggested_file = module.replace('.', '/') + '/exceptions.py'
        else:
            # Use most common usage location
            common_path = Path(symbol.usage_locations[0][0]).parent
            suggested_file = str(common_path / 'exceptions.py')
            module = str(common_path).replace('/', '.')
        
        # Generate template
        base_exception = 'Exception'
        if 'Validation' in symbol.symbol_name:
            base_exception = 'ValueError'
        elif 'NotFound' in symbol.symbol_name:
            base_exception = 'LookupError'
        
        template = f'''class {symbol.symbol_name}({base_exception}):
    """Custom exception for {symbol.symbol_name.replace('Error', '').lower()} errors."""
    pass'''
        
        return ImplementationSuggestion(
            symbol_name=symbol.symbol_name,
            suggested_file=suggested_file,
            suggested_module=module,
            implementation_template=template,
            reasoning=f"Exception class used in {len(symbol.usage_locations)} locations",
            related_symbols=related
        )
    
    def _suggest_class_implementation(self, symbol: MissingSymbol) -> ImplementationSuggestion:
        """Suggest implementation for missing class."""
        # Determine module
        if symbol.import_attempts:
            module = list(symbol.import_attempts)[0]
            suggested_file = module.replace('.', '/') + '.py'
        else:
            common_path = Path(symbol.usage_locations[0][0]).parent
            suggested_file = str(common_path / 'models.py')
            module = str(common_path).replace('/', '.') + '.models'
        
        # Generate template based on usage
        if 'Model' in symbol.symbol_name:
            template = f'''class {symbol.symbol_name}:
    """Data model for {symbol.symbol_name.replace('Model', '').lower()}."""
    
    def __init__(self):
        """Initialize {symbol.symbol_name}."""
        pass'''
        else:
            template = f'''class {symbol.symbol_name}:
    """{symbol.symbol_name} implementation."""
    
    def __init__(self):
        """Initialize {symbol.symbol_name}."""
        pass'''
        
        return ImplementationSuggestion(
            symbol_name=symbol.symbol_name,
            suggested_file=suggested_file,
            suggested_module=module,
            implementation_template=template,
            reasoning=f"Class instantiated in {len(symbol.usage_locations)} locations",
            related_symbols=[]
        )
    
    def _suggest_function_implementation(self, symbol: MissingSymbol) -> ImplementationSuggestion:
        """Suggest implementation for missing function."""
        # Determine module
        if symbol.import_attempts:
            module = list(symbol.import_attempts)[0]
            parts = module.split('.')
            if 'util' in parts[-1] or 'helper' in parts[-1]:
                suggested_file = module.replace('.', '/') + '.py'
            else:
                suggested_file = module.replace('.', '/') + '/utils.py'
        else:
            common_path = Path(symbol.usage_locations[0][0]).parent
            suggested_file = str(common_path / 'utils.py')
            module = str(common_path).replace('/', '.') + '.utils'
        
        # Generate template
        template = f'''def {symbol.symbol_name}(*args, **kwargs):
    """{symbol.symbol_name} function.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Result of {symbol.symbol_name} operation
    """
    # TODO: Implement {symbol.symbol_name}
    raise NotImplementedError("{symbol.symbol_name} not yet implemented")'''
        
        return ImplementationSuggestion(
            symbol_name=symbol.symbol_name,
            suggested_file=suggested_file,
            suggested_module=module,
            implementation_template=template,
            reasoning=f"Function called in {len(symbol.usage_locations)} locations",
            related_symbols=[]
        )
    
    def _suggest_variable_implementation(self, symbol: MissingSymbol) -> ImplementationSuggestion:
        """Suggest implementation for missing variable/constant."""
        # Usually constants or config values
        if symbol.import_attempts:
            module = list(symbol.import_attempts)[0]
            suggested_file = module.replace('.', '/') + '/constants.py'
        else:
            common_path = Path(symbol.usage_locations[0][0]).parent
            suggested_file = str(common_path / 'constants.py')
            module = str(common_path).replace('/', '.') + '.constants'
        
        # Generate template
        if symbol.symbol_name.isupper():
            template = f'{symbol.symbol_name} = None  # TODO: Set appropriate value'
        else:
            template = f'{symbol.symbol_name} = None  # TODO: Initialize properly'
        
        return ImplementationSuggestion(
            symbol_name=symbol.symbol_name,
            suggested_file=suggested_file,
            suggested_module=module,
            implementation_template=template,
            reasoning=f"Variable/constant used in {len(symbol.usage_locations)} locations",
            related_symbols=[]
        )
    
    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        path = Path(file_path)
        parts = []
        
        for part in path.parts[:-1]:
            if part not in ('.', '..', ''):
                parts.append(part)
        
        if path.stem != '__init__':
            parts.append(path.stem)
            
        return '.'.join(parts)
    
    def _create_missing_symbol_issue(self, symbol: MissingSymbol, suggestion: Optional[ImplementationSuggestion]) -> Issue:
        """Create issue for missing symbol with implementation suggestion."""
        details = {
            "symbol_type": symbol.symbol_type,
            "usage_count": len(symbol.usage_locations),
            "usage_files": list(set(loc[0] for loc in symbol.usage_locations)),
            "import_attempts": list(symbol.import_attempts),
            "usage_contexts": list(set(symbol.usage_context))
        }
        
        if suggestion:
            details.update({
                "suggested_location": suggestion.suggested_file,
                "suggested_module": suggestion.suggested_module,
                "implementation_template": suggestion.implementation_template,
                "reasoning": suggestion.reasoning
            })
        
        return Issue(
            type="missing_symbol_with_suggestion",
            severity="HIGH",
            message=f"Missing {symbol.symbol_type} '{symbol.symbol_name}' used in {len(symbol.usage_locations)} places",
            file_path=symbol.usage_locations[0][0] if symbol.usage_locations else "<unknown>",
            line_number=symbol.usage_locations[0][1] if symbol.usage_locations else 0,
            details=details
        )
    
    def _create_summary_issue(self, missing: Dict[str, MissingSymbol], groups: Dict[str, List[str]]) -> Issue:
        """Create summary issue for missing symbols."""
        # Count by type
        type_counts = defaultdict(int)
        for symbol in missing.values():
            type_counts[symbol.symbol_type] += 1
        
        return Issue(
            type="missing_symbols_summary",
            severity="HIGH",
            message=f"Found {len(missing)} missing symbols across codebase",
            file_path="<global>",
            line_number=0,
            details={
                "total_missing": len(missing),
                "by_type": dict(type_counts),
                "grouped_symbols": {k: v for k, v in groups.items() if len(v) > 1},
                "recommendation": "Implement missing symbols in suggested locations",
                "priority_order": [
                    "1. Exceptions (runtime failures)",
                    "2. Classes (type errors)",
                    "3. Functions (functionality gaps)",
                    "4. Variables/constants (configuration)"
                ]
            }
        )