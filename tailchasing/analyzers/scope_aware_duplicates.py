"""
Scope-aware duplicate detection that understands class hierarchies and namespaces.

This analyzer properly handles:
- Methods with same names in different classes
- Property methods and dataclass patterns
- Interface implementations across classes
- Inheritance hierarchies
"""

import ast
from typing import Dict, List, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno
from .pattern_whitelist import is_whitelisted


@dataclass
class FunctionSignature:
    """Complete function signature including scope context."""
    name: str
    class_name: Optional[str]
    module: str
    parameters: List[str]
    decorators: List[str]
    is_property: bool
    is_test: bool
    is_mock: bool
    is_abstract: bool
    body_hash: str
    line: int
    
    @property
    def qualified_name(self) -> str:
        """Get fully qualified name including class scope."""
        if self.class_name:
            return f"{self.module}::{self.class_name}.{self.name}"
        return f"{self.module}::{self.name}"
    
    @property
    def is_special_method(self) -> bool:
        """Check if this is a special Python method."""
        return self.name.startswith('__') and self.name.endswith('__')
    
    @property
    def is_interface_method(self) -> bool:
        """Check if this is likely an interface method."""
        # Common interface method names that are expected to be duplicated
        interface_methods = {
            'get', 'set', 'update', 'delete', 'create', 'read', 'write',
            'connect', 'disconnect', 'start', 'stop', 'run', 'execute',
            'process', 'handle', 'validate', 'serialize', 'deserialize',
            'encode', 'decode', 'parse', 'format', 'render', 'compile',
            'save', 'load', 'open', 'close', 'init', 'cleanup', 'reset',
            'to_dict', 'from_dict', 'to_json', 'from_json', 'dict_for_update',
            'transform', 'convert', 'export', 'import', 'fetch', 'push',
            'pull', 'sync', 'refresh', 'reload', 'clear', 'flush'
        }
        return self.name in interface_methods or self.name.startswith('to_') or self.name.startswith('from_')


class ScopeAwareDuplicateAnalyzer(BaseAnalyzer):
    """Detects true code duplicates while respecting scope and design patterns."""
    
    name = "scope_aware_duplicates"
    
    def __init__(self):
        super().__init__()
        self.class_hierarchies: Dict[str, Set[str]] = {}  # Track inheritance
        self.interface_methods: Set[str] = set()  # Track interface patterns
        
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Find duplicates with proper scope awareness."""
        issues = []
        
        # First pass: collect all function signatures with scope
        signatures = self._collect_signatures(ctx)
        
        # Second pass: build class hierarchies
        self._build_class_hierarchies(ctx)
        
        # Third pass: identify true duplicates
        duplicates = self._find_true_duplicates(signatures)
        
        # Generate issues for true duplicates only
        for dup_group in duplicates:
            if self._is_legitimate_duplicate_group(dup_group):
                continue
                
            primary = dup_group[0]
            for secondary in dup_group[1:]:
                issue = self._create_duplicate_issue(primary, secondary)
                issues.append(issue)
        
        return issues
    
    def _collect_signatures(self, ctx: AnalysisContext) -> List[FunctionSignature]:
        """Collect all function signatures with full scope context."""
        signatures = []
        
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            visitor = ScopeAwareVisitor(file, ctx)
            visitor.visit(tree)
            signatures.extend(visitor.signatures)
        
        return signatures
    
    def _build_class_hierarchies(self, ctx: AnalysisContext):
        """Build inheritance hierarchies to understand class relationships."""
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    bases = []
                    
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(base.attr)
                    
                    self.class_hierarchies[class_name] = set(bases)
    
    def _find_true_duplicates(self, signatures: List[FunctionSignature]) -> List[List[FunctionSignature]]:
        """Find true duplicates considering scope and context."""
        # Group by body hash first
        by_hash = defaultdict(list)
        for sig in signatures:
            if not self._should_check_for_duplicates(sig):
                continue
            by_hash[sig.body_hash].append(sig)
        
        # Filter to only groups with actual duplicates
        duplicate_groups = []
        for hash_value, sigs in by_hash.items():
            if len(sigs) > 1:
                # Further filter by checking if they're truly duplicates
                true_dups = self._filter_true_duplicates(sigs)
                if len(true_dups) > 1:
                    duplicate_groups.append(true_dups)
        
        return duplicate_groups
    
    def _should_check_for_duplicates(self, sig: FunctionSignature) -> bool:
        """Determine if a function should be checked for duplicates."""
        # Check whitelist first
        if is_whitelisted(sig.name, 'method', sig.module, sig.class_name or ""):
            return False
        
        # Skip special methods
        if sig.is_special_method:
            return False
        
        # Skip abstract methods
        if sig.is_abstract:
            return False
        
        # Skip property getters/setters (they're often similar by design)
        if sig.is_property:
            return False
        
        # Skip test methods (often have similar setup/teardown)
        if sig.is_test and sig.name in ['setUp', 'tearDown', 'setUpClass', 'tearDownClass']:
            return False
        
        # Skip mock methods
        if sig.is_mock:
            return False
        
        # Skip if method name strongly suggests delegation/utility usage
        utility_indicators = ['dict_for_update', 'to_dict', 'from_dict', 'as_dict']
        if sig.name in utility_indicators and sig.class_name:
            return False  # These are expected to be similar across classes
        
        return True
    
    def _filter_true_duplicates(self, sigs: List[FunctionSignature]) -> List[FunctionSignature]:
        """Filter to only true duplicates, not legitimate patterns."""
        # If all are in different classes and have interface-like names, not duplicates
        if all(sig.class_name for sig in sigs):
            classes = set(sig.class_name for sig in sigs)
            if len(classes) == len(sigs) and all(sig.is_interface_method for sig in sigs):
                return []  # Different classes implementing same interface
        
        # Check if these are delegating to a shared utility
        if self._are_delegating_to_utility(sigs):
            return []  # Legitimate delegation pattern
        
        # If they're in related classes (inheritance), might be overrides
        if self._are_related_by_inheritance(sigs):
            return []  # Legitimate overrides
        
        # Check for dataclass patterns
        if self._is_dataclass_pattern(sigs):
            return []  # Legitimate dataclass methods
        
        # Check for property pattern
        if self._is_property_pattern(sigs):
            return []  # Legitimate property methods
        
        # Check for serialization pattern
        if self._is_serialization_pattern(sigs):
            return []  # Legitimate serialization methods
        
        return sigs
    
    def _are_related_by_inheritance(self, sigs: List[FunctionSignature]) -> bool:
        """Check if signatures are related by class inheritance."""
        classes = [sig.class_name for sig in sigs if sig.class_name]
        if len(classes) < 2:
            return False
        
        # Check if any class inherits from another
        for cls1 in classes:
            for cls2 in classes:
                if cls1 != cls2:
                    if cls2 in self.class_hierarchies.get(cls1, set()):
                        return True
                    if cls1 in self.class_hierarchies.get(cls2, set()):
                        return True
        
        return False
    
    def _are_delegating_to_utility(self, sigs: List[FunctionSignature]) -> bool:
        """Check if functions are delegating to a shared utility function."""
        # Common delegation patterns
        delegation_patterns = [
            'dict_for_update', 'to_dict', 'from_dict', 'serialize', 'deserialize',
            'validate', 'clean', 'save', 'load', 'export', 'import',
            'convert', 'transform', 'process', 'handle', 'dispatch'
        ]
        
        # Check if function names suggest delegation
        for pattern in delegation_patterns:
            if all(pattern in sig.name.lower() for sig in sigs):
                # If they're in different classes, likely delegating
                if len(set(sig.class_name for sig in sigs if sig.class_name)) > 1:
                    return True
        
        # Check if the function body is very small (likely just calling another function)
        # This would need actual AST analysis of the body, for now use name heuristics
        if all(sig.name in delegation_patterns for sig in sigs):
            return True
            
        return False
    
    def _is_property_pattern(self, sigs: List[FunctionSignature]) -> bool:
        """Check if functions follow property getter/setter patterns."""
        # Already handled by is_property flag, but check for manual property patterns
        property_prefixes = ['get_', 'set_', 'is_', 'has_', 'can_']
        
        for prefix in property_prefixes:
            if all(sig.name.startswith(prefix) for sig in sigs):
                # Different classes with same property pattern is OK
                if len(set(sig.class_name for sig in sigs if sig.class_name)) > 1:
                    return True
        
        return False
    
    def _is_serialization_pattern(self, sigs: List[FunctionSignature]) -> bool:
        """Check if functions follow serialization/deserialization patterns."""
        serialization_methods = {
            'to_json', 'from_json', 'to_dict', 'from_dict', 'to_xml', 'from_xml',
            'to_yaml', 'from_yaml', 'to_string', 'from_string', 'serialize', 'deserialize',
            'dump', 'dumps', 'load', 'loads', 'dict_for_update', 'as_dict',
            '__getstate__', '__setstate__', '__reduce__', '__reduce_ex__'
        }
        
        # Check if all signatures are serialization methods
        if all(sig.name in serialization_methods for sig in sigs):
            # Different classes implementing serialization is expected
            if len(set(sig.class_name for sig in sigs if sig.class_name)) > 1:
                return True
        
        return False
    
    def _is_dataclass_pattern(self, sigs: List[FunctionSignature]) -> bool:
        """Check if this is a legitimate dataclass pattern."""
        # Common dataclass method names
        dataclass_methods = {
            '__init__', '__repr__', '__eq__', '__hash__',
            '__post_init__', 'replace', 'asdict', 'astuple'
        }
        
        # Check if all signatures are dataclass-like methods
        if all(sig.name in dataclass_methods for sig in sigs):
            # Check if they have @dataclass decorator or similar
            for sig in sigs:
                if 'dataclass' in sig.decorators or 'dataclasses' in sig.decorators:
                    return True
        
        return False
    
    def _is_legitimate_duplicate_group(self, dup_group: List[FunctionSignature]) -> bool:
        """Check if a duplicate group is legitimate (not an issue)."""
        # All in test files - test patterns often repeat
        if all('test' in sig.module.lower() for sig in dup_group):
            # Common test patterns are OK
            if dup_group[0].name in ['test_basic', 'test_default', 'test_empty']:
                return True
        
        # All are mock implementations
        if all(sig.is_mock or 'mock' in sig.class_name.lower() if sig.class_name else False 
               for sig in dup_group):
            return True
        
        # Protocol implementations
        if all('protocol' in sig.module.lower() or 'interface' in sig.module.lower() 
               for sig in dup_group):
            return True
        
        return False
    
    def _create_duplicate_issue(self, primary: FunctionSignature, secondary: FunctionSignature) -> Issue:
        """Create an issue for a true duplicate."""
        return Issue(
            kind="scope_aware_duplicate",
            message=f"Duplicate implementation of {secondary.qualified_name} (duplicates {primary.qualified_name})",
            severity=2,  # Lower severity since we're more confident
            file=secondary.module,
            line=secondary.line,
            symbol=secondary.qualified_name,
            evidence={
                "primary_location": primary.qualified_name,
                "primary_line": primary.line,
                "is_different_class": primary.class_name != secondary.class_name,
                "is_same_module": primary.module == secondary.module
            },
            suggestions=[
                "Consider extracting to a shared utility function",
                "Use inheritance or composition to share code",
                "Create a base class with the common implementation"
            ] if primary.class_name != secondary.class_name else [
                "Remove the duplicate function",
                "Consolidate into a single implementation"
            ],
            confidence=0.9  # High confidence since we've filtered false positives
        )


class ScopeAwareVisitor(ast.NodeVisitor):
    """Visitor that maintains scope context while traversing AST."""
    
    def __init__(self, file: str, ctx: AnalysisContext):
        self.file = file
        self.ctx = ctx
        self.signatures: List[FunctionSignature] = []
        self.current_class: Optional[str] = None
        self.class_decorators: List[str] = []
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition maintaining scope."""
        old_class = self.current_class
        old_decorators = self.class_decorators
        
        self.current_class = node.name
        self.class_decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        self.generic_visit(node)
        
        self.current_class = old_class
        self.class_decorators = old_decorators
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition and create signature."""
        self._process_function(node)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition and create signature."""
        self._process_function(node, is_async=True)
        self.generic_visit(node)
    
    def _process_function(self, node: ast.FunctionDef, is_async: bool = False):
        """Process a function and create its signature."""
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        sig = FunctionSignature(
            name=node.name,
            class_name=self.current_class,
            module=self.file,
            parameters=self._get_parameters(node),
            decorators=decorators,
            is_property='property' in decorators,
            is_test=node.name.startswith('test_') or self.current_class and 'Test' in self.current_class,
            is_mock=node.name.startswith('mock_') or self.current_class and 'Mock' in self.current_class,
            is_abstract='abstractmethod' in decorators,
            body_hash=self._hash_function_body(node),
            line=safe_get_lineno(node)
        )
        
        self.signatures.append(sig)
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return "unknown"
    
    def _get_parameters(self, node: ast.FunctionDef) -> List[str]:
        """Extract parameter names from function."""
        params = []
        for arg in node.args.args:
            # Skip 'self' and 'cls' parameters
            if arg.arg not in ['self', 'cls']:
                params.append(arg.arg)
        return params
    
    def _hash_function_body(self, node: ast.FunctionDef) -> str:
        """Create a hash of the function body for comparison."""
        # Remove docstrings for comparison
        body = node.body.copy()
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # Skip docstring
        
        # Convert to string representation for hashing
        import hashlib
        body_str = ast.dump(ast.Module(body=body, type_ignores=[]))
        return hashlib.md5(body_str.encode()).hexdigest()