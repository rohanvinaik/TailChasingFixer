"""
Enhanced cargo cult detector with intelligent parent __init__ checking.

This module detects unnecessary super().__init__() calls and other
cargo-cult programming patterns where code is copied without understanding.
"""

import ast
import re
from typing import Dict, Optional, Set, List, Union
from pathlib import Path
from collections import defaultdict

from ..core.issues import Issue
from ..core.utils import safe_get_lineno
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
                    super_line = safe_get_lineno(stmt)
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
    
    def __init__(self):
        """Initialize with boilerplate patterns."""
        super().__init__()
        self.boilerplate_patterns = {
            'unnecessary_super': {'pattern': 'super().__init__() with no parent __init__', 'severity': 1},
            'redundant_docstring': {'pattern': 'Docstring that just repeats function name', 'severity': 1},
            'pointless_inheritance': {'pattern': 'Inherits from object or unnecessary base class', 'severity': 1},
            'unused_imports': {'pattern': 'Common imports that are never used', 'severity': 2},
            'copy_paste_comments': {'pattern': 'Comments that do not match the code', 'severity': 2},
            'unnecessary_abstractions': {'pattern': 'Over-engineered simple functionality', 'severity': 3},
            'misused_patterns': {'pattern': 'Design patterns applied incorrectly', 'severity': 3}
        }
    
    def _analyze_file(self, filepath: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Analyze file for multiple cargo cult patterns."""
        issues = super()._analyze_file(filepath, tree, ctx)
        
        # Add comprehensive cargo cult pattern checks
        issues.extend(self._check_redundant_docstrings(filepath, tree))
        issues.extend(self._check_copy_paste_comments(filepath, tree))
        issues.extend(self._check_unnecessary_abstractions(filepath, tree))
        issues.extend(self._check_misused_patterns(filepath, tree))
        issues.extend(self._check_boilerplate_overuse(filepath, tree))
        
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
                    line=safe_get_lineno(cls),
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
                        line=safe_get_lineno(methods["getter"]),
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
                line=safe_get_lineno(cls),
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
    
    def _check_redundant_docstrings(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Check for docstrings that add no value."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    name = node.name
                    doc_lower = docstring.lower().strip()
                    
                    # Pattern 1: Docstring is just the name
                    if doc_lower.replace('_', ' ') == name.replace('_', ' ').lower():
                        issues.append(self._create_cargo_cult_issue(
                            "redundant_docstring",
                            f"Docstring just repeats function name '{name}'",
                            filepath, safe_get_lineno(node), name,
                            "Write meaningful docstrings that explain purpose and behavior"
                        ))
                    
                    # Pattern 2: Generic placeholder docstrings
                    generic_patterns = [
                        r'^(this|the) (function|method|class)',
                        r'^(does|performs) .+ing\.$',
                        r'^todo:?\s*$',
                        r'^placeholder',
                        r'^function to ',
                        r'^method (that|to) '
                    ]
                    
                    if any(re.match(pattern, doc_lower) for pattern in generic_patterns):
                        issues.append(self._create_cargo_cult_issue(
                            "generic_docstring",
                            f"Generic placeholder docstring in '{name}'",
                            filepath, safe_get_lineno(node), name,
                            "Replace with specific documentation or remove if obvious"
                        ))
                    
                    # Pattern 3: Docstring doesn't match function
                    if isinstance(node, ast.FunctionDef):
                        doc_params = re.findall(r':param\s+(\w+):', docstring)
                        actual_params = [arg.arg for arg in node.args.args]
                        phantom_params = set(doc_params) - set(actual_params)
                        if phantom_params:
                            issues.append(self._create_cargo_cult_issue(
                                "mismatched_docstring",
                                f"Docstring references non-existent parameters: {phantom_params}",
                                filepath, safe_get_lineno(node), name,
                                "Update docstring to match actual function signature"
                            ))
        
        return issues
    
    def _check_copy_paste_comments(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Check for comments that don't match the code."""
        issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            return issues
        
        comment_patterns = [
            (r'#\s*TODO:\s*implement\s+(\w+)', 'todo_already_implemented'),
            (r'#\s*FIXME:\s*(\w+)', 'fixme_not_applicable'),
            (r'#\s*(\w+)\s+returns\s+(\w+)', 'comment_wrong_return'),
            (r'#\s*[Hh]andle[s]?\s+(\w+)', 'comment_wrong_handling')
        ]
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if line_stripped.startswith('#'):
                for pattern, issue_type in comment_patterns:
                    match = re.search(pattern, line_stripped)
                    if match:
                        if issue_type == 'todo_already_implemented':
                            func_name = match.group(1)
                            if self._function_exists_nearby(lines, i, func_name):
                                issues.append(Issue(
                                    kind="outdated_todo_comment",
                                    message=f"TODO comment for already implemented '{func_name}'",
                                    severity=1,
                                    file=filepath,
                                    line=i + 1,
                                    evidence={'pattern': 'cargo_cult', 'comment': line_stripped},
                                    suggestions=["Remove outdated TODO comment"]
                                ))
        
        return issues
    
    def _check_unnecessary_abstractions(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Check for over-engineered simple functionality."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Pattern: Factory for single concrete class
                if 'factory' in node.name.lower():
                    methods = [m for m in node.body if isinstance(m, ast.FunctionDef)]
                    create_methods = [m for m in methods if 'create' in m.name.lower()]
                    
                    if create_methods:
                        created_types = set()
                        for method in create_methods:
                            for subnode in ast.walk(method):
                                if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                                    created_types.add(subnode.func.id)
                        
                        if len(created_types) == 1:
                            issues.append(self._create_cargo_cult_issue(
                                "unnecessary_factory",
                                f"Factory class {node.name} only creates one type",
                                filepath, safe_get_lineno(node), node.name,
                                "Consider direct instantiation instead of factory pattern"
                            ))
        
        # Check for getter/setter overuse
        getters_setters = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('get_') or node.name.startswith('set_'):
                    prop_name = node.name[4:]
                    getters_setters[prop_name].append(node)
        
        for prop_name, methods in getters_setters.items():
            if len(methods) >= 2:
                all_trivial = all(self._is_trivial_accessor(m) for m in methods)
                if all_trivial:
                    issues.append(Issue(
                        kind="unnecessary_getters_setters",
                        message=f"Trivial getters/setters for '{prop_name}' add no value",
                        severity=2,
                        file=filepath,
                        line=safe_get_lineno(methods[0]),
                        evidence={'property': prop_name, 'pattern': 'cargo_cult'},
                        suggestions=[
                            "Use direct attribute access or @property decorator",
                            "Only add getters/setters when they provide validation or transformation"
                        ]
                    ))
        
        return issues
    
    def _check_misused_patterns(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Check for incorrectly applied design patterns."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Pattern: Singleton with __new__ but allows multiple instances
                has_new = any(
                    isinstance(m, ast.FunctionDef) and m.name == '__new__'
                    for m in node.body
                )
                
                if has_new and 'singleton' in node.name.lower():
                    new_method = next(
                        m for m in node.body 
                        if isinstance(m, ast.FunctionDef) and m.name == '__new__'
                    )
                    
                    has_instance_check = any(
                        isinstance(n, ast.If) for n in ast.walk(new_method)
                    )
                    
                    if not has_instance_check:
                        issues.append(self._create_cargo_cult_issue(
                            "broken_singleton",
                            f"Singleton pattern in {node.name} doesn't prevent multiple instances",
                            filepath, safe_get_lineno(node), node.name,
                            "Implement proper instance caching in __new__ or use a decorator"
                        ))
                
                # Pattern: Observer without proper notification mechanism
                if 'observer' in node.name.lower():
                    methods = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
                    has_update = any('update' in m or 'notify' in m for m in methods)
                    has_attach = any('attach' in m or 'subscribe' in m or 'register' in m for m in methods)
                    
                    if has_attach and not has_update:
                        issues.append(self._create_cargo_cult_issue(
                            "incomplete_observer",
                            f"Observer pattern in {node.name} missing notification mechanism",
                            filepath, safe_get_lineno(node), node.name,
                            "Add update/notify method to complete observer pattern"
                        ))
        
        return issues
    
    def _check_boilerplate_overuse(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Check for excessive boilerplate code."""
        issues = []
        
        boilerplate_count = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if self._is_boilerplate_function(node):
                    boilerplate_count += 1
        
        if total_functions > 5 and boilerplate_count / total_functions > 0.5:
            issues.append(Issue(
                kind="excessive_boilerplate",
                message=f"File contains {boilerplate_count}/{total_functions} boilerplate functions",
                severity=2,
                file=filepath,
                line=1,
                evidence={
                    'boilerplate_ratio': boilerplate_count / total_functions,
                    'pattern': 'cargo_cult'
                },
                suggestions=[
                    "Remove unnecessary boilerplate code",
                    "Focus on implementing actual functionality",
                    "Avoid copying patterns without understanding their purpose"
                ]
            ))
        
        return issues
    
    def _is_trivial_accessor(self, method: ast.FunctionDef) -> bool:
        """Check if method is a trivial getter/setter."""
        return self._is_trivial_getter(method) or self._is_trivial_setter(method)
    
    def _is_boilerplate_function(self, func: ast.FunctionDef) -> bool:
        """Check if function is likely boilerplate."""
        # Empty function
        if len(func.body) == 1 and isinstance(func.body[0], ast.Pass):
            return True
        
        # Only raises NotImplementedError
        if len(func.body) == 1 and isinstance(func.body[0], ast.Raise):
            return True
        
        # Only has docstring
        if len(func.body) == 1 and isinstance(func.body[0], ast.Expr):
            if isinstance(func.body[0].value, (ast.Constant, ast.Str)):
                return True
        
        return False
    
    def _function_exists_nearby(self, lines: List[str], comment_line: int, func_name: str) -> bool:
        """Check if function exists near the comment."""
        start = max(0, comment_line - 5)
        end = min(len(lines), comment_line + 10)
        
        for i in range(start, end):
            if f'def {func_name}' in lines[i]:
                return True
        return False
    
    def _create_cargo_cult_issue(self, kind: str, message: str, filepath: str, 
                                line: int, symbol: str, suggestion: str) -> Issue:
        """Create a cargo cult pattern issue."""
        return Issue(
            kind=f"cargo_cult_{kind}",
            message=message,
            severity=self.boilerplate_patterns.get(kind, {}).get('severity', 2),
            file=filepath,
            line=line,
            symbol=symbol,
            evidence={'pattern': 'cargo_cult', 'type': kind},
            suggestions=[suggestion, "Understand the purpose before copying patterns"]
        )