"""
Enhanced cargo cult detector with intelligent parent __init__ checking.

This module detects unnecessary super().__init__() calls and other
cargo-cult programming patterns where code is copied without understanding.
"""

import ast
from typing import Dict, Optional, Set, List, Union
from pathlib import Path

from ..core.issues import Issue
from .base import Analyzer, AnalysisContext


class CargoCultDetector(Analyzer):
    """Detect cargo cult programming patterns including unnecessary super() calls."""
    
    name = "cargo_cult"
    
    def __init__(self):
        """Initialize the detector with caching."""
        self._has_init_cache: Dict[str, bool] = {}
        self._module_index: Dict[str, ast.ClassDef] = {}
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Run cargo cult detection analysis."""
        issues = []
        
        # Build module index for all classes
        self._build_module_index(ctx)
        
        # Check each file
        for filepath, tree in ctx.ast_index.items():
            file_issues = self._analyze_file(filepath, tree, ctx)
            issues.extend(file_issues)
        
        return issues
    
    def _build_module_index(self, ctx: AnalysisContext) -> None:
        """Build an index of all classes in the codebase."""
        self._module_index.clear()
        
        for filepath, tree in ctx.ast_index.items():
            # Extract module name from filepath
            module_parts = Path(filepath).stem.split('/')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Store with simple name and qualified name
                    self._module_index[node.name] = node
                    
                    # Also store with module prefix if we can determine it
                    if module_parts:
                        qualified_name = f"{'.'.join(module_parts)}.{node.name}"
                        self._module_index[qualified_name] = node
    
    def _analyze_file(self, filepath: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Analyze a single file for cargo cult patterns."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for unnecessary super().__init__() calls
                init_issue = self._check_unnecessary_super_init(node, filepath, ctx)
                if init_issue:
                    issues.append(init_issue)
        
        return issues
    
    def _check_unnecessary_super_init(self, cls: ast.ClassDef, filepath: str, 
                                     ctx: AnalysisContext) -> Optional[Issue]:
        """Check if a class has unnecessary super().__init__() call."""
        # Find __init__ method
        init_method = None
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                init_method = item
                break
        
        if not init_method:
            return None
        
        # Check if it calls super().__init__()
        has_super_init = False
        super_line = 0
        
        for stmt in ast.walk(init_method):
            if isinstance(stmt, ast.Call):
                # Check for super().__init__() pattern
                if (isinstance(stmt.func, ast.Attribute) and 
                    stmt.func.attr == "__init__" and
                    isinstance(stmt.func.value, ast.Call) and
                    isinstance(stmt.func.value.func, ast.Name) and
                    stmt.func.value.func.id == "super"):
                    has_super_init = True
                    super_line = stmt.lineno
                    break
        
        if not has_super_init:
            return None
        
        # Check if parent actually has a meaningful __init__
        if not self._parent_has_init(cls, self._module_index):
            return Issue(
                kind="unnecessary_super_init",
                message=f"Class {cls.name} calls super().__init__() but no parent class has a meaningful __init__",
                severity=2,
                file=filepath,
                line=super_line,
                symbol=f"{cls.name}.__init__",
                evidence={
                    "pattern": "cargo_cult",
                    "reason": "no_parent_init",
                    "class_name": cls.name,
                    "bases": [self._get_base_name(base) for base in cls.bases]
                },
                suggestions=[
                    "Remove the unnecessary super().__init__() call",
                    "Only call super().__init__() when parent class has initialization logic",
                    "Check if parent classes actually need initialization"
                ]
            )
        
        return None
    
    def _parent_has_init(self, cls: ast.ClassDef, module_index: Dict[str, ast.ClassDef]) -> bool:
        """
        Determine if any ancestor class has a meaningful __init__ method.
        
        This method performs a depth-first search through the inheritance hierarchy
        to check if any parent class defines a non-trivial __init__ method.
        
        Args:
            cls: The class to check
            module_index: Index of all classes in the codebase
            
        Returns:
            True if any ancestor has a meaningful __init__, False otherwise
        """
        # Use cached results when available
        cache = getattr(self, "_has_init_cache", None)
        if cache is None:
            cache = self._has_init_cache = {}
        
        def has_nontrivial_init(c: ast.ClassDef) -> bool:
            """Check if a class has a non-trivial __init__ method."""
            for node in c.body:
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    # Filter out docstrings and comments
                    body = []
                    for stmt in node.body:
                        # Skip docstrings (Expr nodes with Constant values)
                        if isinstance(stmt, ast.Expr):
                            if isinstance(stmt.value, ast.Constant):
                                continue  # Skip docstring
                            elif isinstance(stmt.value, ast.Str):  # Python 3.7 compatibility
                                continue  # Skip docstring
                        body.append(stmt)
                    
                    # Empty body or only docstring - trivial
                    if not body:
                        return False
                    
                    # Single pass statement - trivial
                    if len(body) == 1 and isinstance(body[0], ast.Pass):
                        return False
                    
                    # Single raise statement - meaningful (requires override)
                    if len(body) == 1 and isinstance(body[0], ast.Raise):
                        return True
                    
                    # Anything else is non-trivial
                    return True
            
            # No __init__ found
            return False
        
        def resolve_base_name(base: ast.expr) -> Optional[str]:
            """Resolve a base class expression to a string name."""
            if isinstance(base, ast.Name):
                return base.id
            
            if isinstance(base, ast.Attribute):
                # Handle module.Class or package.module.Class
                parts = []
                current = base
                
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                    return ".".join(reversed(parts))
            
            # Could be ast.Subscript for generics like List[T]
            if isinstance(base, ast.Subscript):
                return resolve_base_name(base.value)
            
            return None
        
        def dfs_has_init(class_name: str, visited: Set[str]) -> bool:
            """Depth-first search to check if class or ancestors have __init__."""
            # Check cache first
            if class_name in cache:
                return cache[class_name]
            
            # Avoid infinite recursion in circular inheritance
            if class_name in visited:
                return False
            visited.add(class_name)
            
            # Get the class node
            node = module_index.get(class_name)
            
            # If we found the class and it has non-trivial __init__, we're done
            if node and has_nontrivial_init(node):
                cache[class_name] = True
                return True
            
            # If class not found in our index, be conservative
            if not node:
                # Could be external class (e.g., from standard library)
                # Be conservative and assume it might have __init__
                if class_name not in ("object", "type", "typing.Protocol", "Protocol", "ABC"):
                    cache[class_name] = True
                    return True
                else:
                    cache[class_name] = False
                    return False
            
            # Recursively check bases
            bases = []
            for base in node.bases:
                base_name = resolve_base_name(base)
                if base_name:
                    bases.append(base_name)
            
            # Check each base
            for base_name in bases:
                # Skip known trivial bases
                if base_name in ("object", "type", "typing.Protocol", "Protocol", "ABC"):
                    continue
                
                if dfs_has_init(base_name, visited):
                    cache[class_name] = True
                    return True
            
            # No meaningful __init__ found in hierarchy
            cache[class_name] = False
            return False
        
        # Check each base class of the given class
        for base in cls.bases:
            base_name = resolve_base_name(base)
            
            # Skip object base class
            if not base_name or base_name == "object":
                continue
            
            # If any base has meaningful __init__, return True
            if dfs_has_init(base_name, set()):
                return True
        
        # No bases have meaningful __init__
        return False
    
    def _get_base_name(self, base: ast.expr) -> str:
        """Get string representation of a base class for reporting."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            parts = []
            current = base
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
                return ".".join(reversed(parts))
        elif isinstance(base, ast.Subscript):
            # Handle generics like List[T]
            return self._get_base_name(base.value) + "[...]"
        return "unknown"


class EnhancedCargoCultDetector(CargoCultDetector):
    """Extended version with additional cargo cult pattern detection."""
    
    def _analyze_file(self, filepath: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Analyze file for multiple cargo cult patterns."""
        issues = super()._analyze_file(filepath, tree, ctx)
        
        # Add more cargo cult patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for unnecessary object inheritance (Python 3)
                issues.extend(self._check_unnecessary_object_inheritance(node, filepath))
                
                # Check for trivial getters/setters
                issues.extend(self._check_trivial_accessors(node, filepath))
                
                # Check for single-method classes that could be functions
                issues.extend(self._check_unnecessary_classes(node, filepath))
        
        return issues
    
    def _check_unnecessary_object_inheritance(self, cls: ast.ClassDef, filepath: str) -> List[Issue]:
        """Check for explicit 'object' inheritance in Python 3."""
        issues = []
        
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id == "object":
                issues.append(Issue(
                    kind="unnecessary_object_inheritance",
                    message=f"Class {cls.name} explicitly inherits from 'object' (not needed in Python 3)",
                    severity=1,
                    file=filepath,
                    line=cls.lineno,
                    symbol=cls.name,
                    evidence={
                        "pattern": "cargo_cult",
                        "reason": "python2_legacy"
                    },
                    suggestions=[
                        "Remove explicit 'object' inheritance",
                        "In Python 3, all classes implicitly inherit from object"
                    ]
                ))
        
        return issues
    
    def _check_trivial_accessors(self, cls: ast.ClassDef, filepath: str) -> List[Issue]:
        """Check for trivial getters and setters that add no value."""
        issues = []
        properties = {}  # Track get/set methods by property name
        
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                # Check for get_xxx or set_xxx pattern
                if node.name.startswith("get_"):
                    prop_name = node.name[4:]
                    if prop_name not in properties:
                        properties[prop_name] = {}
                    properties[prop_name]["getter"] = node
                    
                elif node.name.startswith("set_"):
                    prop_name = node.name[4:]
                    if prop_name not in properties:
                        properties[prop_name] = {}
                    properties[prop_name]["setter"] = node
        
        # Check if getters/setters are trivial
        for prop_name, methods in properties.items():
            if "getter" in methods and self._is_trivial_getter(methods["getter"]):
                if "setter" in methods and self._is_trivial_setter(methods["setter"]):
                    issues.append(Issue(
                        kind="trivial_accessors",
                        message=f"Trivial getter/setter for '{prop_name}' in {cls.name}",
                        severity=2,
                        file=filepath,
                        line=methods["getter"].lineno,
                        symbol=f"{cls.name}.{prop_name}",
                        evidence={
                            "pattern": "cargo_cult",
                            "reason": "java_style",
                            "getter": methods["getter"].name,
                            "setter": methods.get("setter", {}).name if "setter" in methods else None
                        },
                        suggestions=[
                            "Use direct attribute access or @property decorator",
                            "Only add getters/setters when they provide validation or transformation",
                            f"Consider: @property def {prop_name}(self): return self._{prop_name}"
                        ]
                    ))
        
        return issues
    
    def _check_unnecessary_classes(self, cls: ast.ClassDef, filepath: str) -> List[Issue]:
        """Check for classes that could be simple functions."""
        issues = []
        
        # Get non-special methods
        methods = []
        has_init = False
        
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == "__init__":
                    has_init = True
                elif not node.name.startswith("_"):
                    methods.append(node)
        
        # Single public method that doesn't use self -> could be a function
        if len(methods) == 1 and not self._method_uses_self(methods[0]):
            issues.append(Issue(
                kind="unnecessary_class",
                message=f"Class {cls.name} has single method '{methods[0].name}' that doesn't use self",
                severity=2,
                file=filepath,
                line=cls.lineno,
                symbol=cls.name,
                evidence={
                    "pattern": "cargo_cult",
                    "reason": "overengineering",
                    "method": methods[0].name
                },
                suggestions=[
                    f"Convert to a simple function: def {methods[0].name}(...)",
                    "Classes should encapsulate state and behavior",
                    "Use functions for stateless operations"
                ]
            ))
        
        return issues
    
    def _is_trivial_getter(self, method: ast.FunctionDef) -> bool:
        """Check if a method is a trivial getter."""
        if len(method.body) != 1:
            return False
        
        stmt = method.body[0]
        # Check for: return self.xxx
        if isinstance(stmt, ast.Return) and stmt.value:
            if isinstance(stmt.value, ast.Attribute):
                if isinstance(stmt.value.value, ast.Name) and stmt.value.value.id == "self":
                    return True
        
        return False
    
    def _is_trivial_setter(self, method: ast.FunctionDef) -> bool:
        """Check if a method is a trivial setter."""
        if len(method.body) != 1:
            return False
        
        stmt = method.body[0]
        # Check for: self.xxx = value
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id == "self":
                        # Check if assigning parameter value
                        if isinstance(stmt.value, ast.Name):
                            # Should be one of the parameters
                            param_names = [arg.arg for arg in method.args.args[1:]]  # Skip self
                            if stmt.value.id in param_names:
                                return True
        
        return False
    
    def _method_uses_self(self, method: ast.FunctionDef) -> bool:
        """Check if a method actually uses self."""
        for node in ast.walk(method):
            if isinstance(node, ast.Name) and node.id == "self":
                # Skip the parameter definition
                if node not in method.args.args:
                    return True
        return False