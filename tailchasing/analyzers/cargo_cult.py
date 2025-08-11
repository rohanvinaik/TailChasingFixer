"""
Detect 'cargo cult' programming patterns where LLMs copy boilerplate
or patterns without understanding their purpose.
"""

import ast
import re
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from ..core.issues import Issue
from .base import AnalysisContext


class CargoCultDetector:
    """Detect cargo cult programming patterns in LLM-generated code."""
    
    name = "cargo_cult_patterns"
    
    def __init__(self):
        # Common cargo cult patterns
        self.boilerplate_patterns = {
            'unnecessary_super': {
                'pattern': 'super().__init__() with no parent __init__',
                'severity': 1
            },
            'redundant_docstring': {
                'pattern': 'Docstring that just repeats function name',
                'severity': 1
            },
            'pointless_inheritance': {
                'pattern': 'Inherits from object or unnecessary base class',
                'severity': 1
            },
            'unused_imports': {
                'pattern': 'Common imports that are never used',
                'severity': 2
            },
            'copy_paste_comments': {
                'pattern': 'Comments that do not match the code',
                'severity': 2
            },
            'unnecessary_abstractions': {
                'pattern': 'Over-engineered simple functionality',
                'severity': 3
            },
            'misused_patterns': {
                'pattern': 'Design patterns applied incorrectly',
                'severity': 3
            }
        }
        
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Detect cargo cult patterns across the codebase."""
        issues = []
        
        for file, tree in ctx.ast_index.items():
            # Check various cargo cult patterns
            issues.extend(self._check_unnecessary_super(file, tree))
            issues.extend(self._check_redundant_docstrings(file, tree))
            issues.extend(self._check_pointless_inheritance(file, tree))
            issues.extend(self._check_copy_paste_comments(file, tree, ctx))
            issues.extend(self._check_unnecessary_abstractions(file, tree))
            issues.extend(self._check_misused_patterns(file, tree))
            issues.extend(self._check_boilerplate_overuse(file, tree, ctx))
        
        return issues
    
    def _check_unnecessary_super(self, file: str, tree: ast.AST) -> List[Issue]:
        """Check for unnecessary super() calls."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if class has explicit base classes
                has_real_base = any(
                    base for base in node.bases 
                    if not (isinstance(base, ast.Name) and base.id == 'object')
                )
                
                # Look for __init__ method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # Check for super().__init__() call
                        for stmt in item.body:
                            if self._is_super_init_call(stmt):
                                # Check if parent has __init__
                                if not has_real_base or not self._parent_has_init(node, ctx):
                                    issues.append(Issue(
                                        kind="unnecessary_super_call",
                                        message=f"Unnecessary super().__init__() in {node.name} - parent has no __init__ or class has no real base",
                                        severity=1,
                                        file=file,
                                        line=stmt.lineno,
                                        symbol=f"{node.name}.__init__",
                                        evidence={
                                            'pattern': 'cargo_cult',
                                            'reason': 'copied_boilerplate'
                                        },
                                        suggestions=[
                                            "Remove unnecessary super().__init__() call",
                                            "Only call super() when parent class has an __init__ to call"
                                        ]
                                    ))
        
        return issues
    
    def _check_redundant_docstrings(self, file: str, tree: ast.AST) -> List[Issue]:
        """Check for docstrings that add no value."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    # Check for redundant patterns
                    name = node.name
                    doc_lower = docstring.lower().strip()
                    
                    # Pattern 1: Docstring is just the name
                    if doc_lower.replace('_', ' ') == name.replace('_', ' ').lower():
                        issues.append(self._create_cargo_cult_issue(
                            "redundant_docstring",
                            f"Docstring just repeats function name '{name}'",
                            file, node.lineno, name,
                            "Write meaningful docstrings that explain purpose and behavior"
                        ))
                    
                    # Pattern 2: Generic placeholder docstrings
                    generic_patterns = [
                        r'^(this|the) (function|method|class)',
                        r'^(does|performs) .+ing\.$',  # "Does something."
                        r'^todo:?\s*$',
                        r'^placeholder',
                        r'^function to ',
                        r'^method (that|to) '
                    ]
                    
                    if any(re.match(pattern, doc_lower) for pattern in generic_patterns):
                        issues.append(self._create_cargo_cult_issue(
                            "generic_docstring",
                            f"Generic placeholder docstring in '{name}'",
                            file, node.lineno, name,
                            "Replace with specific documentation or remove if obvious"
                        ))
                    
                    # Pattern 3: Docstring doesn't match function
                    if isinstance(node, ast.FunctionDef):
                        # Check if docstring mentions parameters that don't exist
                        doc_params = re.findall(r':param\s+(\w+):', docstring)
                        actual_params = [arg.arg for arg in node.args.args]
                        
                        phantom_params = set(doc_params) - set(actual_params)
                        if phantom_params:
                            issues.append(self._create_cargo_cult_issue(
                                "mismatched_docstring",
                                f"Docstring references non-existent parameters: {phantom_params}",
                                file, node.lineno, name,
                                "Update docstring to match actual function signature"
                            ))
        
        return issues
    
    def _check_pointless_inheritance(self, file: str, tree: ast.AST) -> List[Issue]:
        """Check for unnecessary class inheritance."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for explicit object inheritance (Python 3 doesn't need it)
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'object':
                        issues.append(self._create_cargo_cult_issue(
                            "unnecessary_object_inheritance",
                            f"Class {node.name} explicitly inherits from 'object'",
                            file, node.lineno, node.name,
                            "Remove explicit 'object' inheritance (not needed in Python 3)"
                        ))
                
                # Check for single-method classes that could be functions
                methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
                non_init_methods = [m for m in methods if m.name != '__init__']
                
                if len(methods) == 1 and len(non_init_methods) == 1:
                    method = non_init_methods[0]
                    # Check if method uses self
                    uses_self = any(
                        isinstance(node, ast.Name) and node.id == 'self'
                        for node in ast.walk(method)
                    )
                    
                    if not uses_self:
                        issues.append(self._create_cargo_cult_issue(
                            "unnecessary_class",
                            f"Class {node.name} with single method '{method.name}' that doesn't use self",
                            file, node.lineno, node.name,
                            "Consider converting to a simple function instead of a class"
                        ))
        
        return issues
    
    def _check_copy_paste_comments(self, file: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Check for comments that don't match the code."""
        issues = []
        
        # This requires source code analysis
        try:
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except:
            return issues
        
        # Common patterns of mismatched comments
        comment_patterns = [
            (r'#\s*TODO:\s*implement\s+(\w+)', 'todo_already_implemented'),
            (r'#\s*FIXME:\s*(\w+)', 'fixme_not_applicable'),
            (r'#\s*(\w+)\s+returns\s+(\w+)', 'comment_wrong_return'),
            (r'#\s*[Hh]andle[s]?\s+(\w+)', 'comment_wrong_handling')
        ]
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                for pattern, issue_type in comment_patterns:
                    match = re.search(pattern, line)
                    if match:
                        # Verify if comment matches reality
                        if issue_type == 'todo_already_implemented':
                            func_name = match.group(1)
                            # Check if function is actually implemented
                            if self._function_exists_nearby(lines, i, func_name):
                                issues.append(Issue(
                                    kind="outdated_todo_comment",
                                    message=f"TODO comment for already implemented '{func_name}'",
                                    severity=1,
                                    file=file,
                                    line=i + 1,
                                    evidence={'pattern': 'cargo_cult', 'comment': line},
                                    suggestions=["Remove outdated TODO comment"]
                                ))
        
        return issues
    
    def _check_unnecessary_abstractions(self, file: str, tree: ast.AST) -> List[Issue]:
        """Check for over-engineered simple functionality."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Pattern: Factory for single concrete class
                if 'factory' in node.name.lower():
                    methods = [m for m in node.body if isinstance(m, ast.FunctionDef)]
                    create_methods = [m for m in methods if 'create' in m.name.lower()]
                    
                    if create_methods:
                        # Check if factory only creates one type
                        created_types = set()
                        for method in create_methods:
                            for subnode in ast.walk(method):
                                if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                                    created_types.add(subnode.func.id)
                        
                        if len(created_types) == 1:
                            issues.append(self._create_cargo_cult_issue(
                                "unnecessary_factory",
                                f"Factory class {node.name} only creates one type",
                                file, node.lineno, node.name,
                                "Consider direct instantiation instead of factory pattern"
                            ))
                
                # Pattern: Abstract base class with single implementation
                if any(decorator.id == 'abstractmethod' if isinstance(decorator, ast.Name) else False
                       for method in node.body if isinstance(method, ast.FunctionDef)
                       for decorator in method.decorator_list):
                    # This is an abstract class - check if it has multiple implementations
                    # (This would require cross-file analysis)
                    pass
        
        # Check for getter/setter overuse
        getters_setters = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('get_') or node.name.startswith('set_'):
                    prop_name = node.name[4:]
                    getters_setters[prop_name].append(node)
        
        for prop_name, methods in getters_setters.items():
            if len(methods) >= 2:  # Has both getter and setter
                # Check if they just get/set without logic
                all_trivial = all(self._is_trivial_accessor(m) for m in methods)
                if all_trivial:
                    issues.append(Issue(
                        kind="unnecessary_getters_setters",
                        message=f"Trivial getters/setters for '{prop_name}' add no value",
                        severity=2,
                        file=file,
                        line=methods[0].lineno,
                        evidence={'property': prop_name, 'pattern': 'cargo_cult'},
                        suggestions=[
                            "Use direct attribute access or @property decorator",
                            "Only add getters/setters when they provide validation or transformation"
                        ]
                    ))
        
        return issues
    
    def _check_misused_patterns(self, file: str, tree: ast.AST) -> List[Issue]:
        """Check for incorrectly applied design patterns."""
        issues = []
        
        # Pattern: Singleton with __new__ but allows multiple instances
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_new = any(
                    isinstance(m, ast.FunctionDef) and m.name == '__new__'
                    for m in node.body
                )
                
                if has_new and 'singleton' in node.name.lower():
                    # Check if __new__ actually implements singleton
                    new_method = next(
                        m for m in node.body 
                        if isinstance(m, ast.FunctionDef) and m.name == '__new__'
                    )
                    
                    # Simple check: does it have instance caching?
                    has_instance_check = any(
                        isinstance(n, ast.If) for n in ast.walk(new_method)
                    )
                    
                    if not has_instance_check:
                        issues.append(self._create_cargo_cult_issue(
                            "broken_singleton",
                            f"Singleton pattern in {node.name} doesn't prevent multiple instances",
                            file, node.lineno, node.name,
                            "Implement proper instance caching in __new__ or use a decorator"
                        ))
        
        # Pattern: Observer without proper notification mechanism
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and 'observer' in node.name.lower():
                methods = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
                
                # Check for update/notify methods
                has_update = any('update' in m or 'notify' in m for m in methods)
                has_attach = any('attach' in m or 'subscribe' in m or 'register' in m for m in methods)
                
                if has_attach and not has_update:
                    issues.append(self._create_cargo_cult_issue(
                        "incomplete_observer",
                        f"Observer pattern in {node.name} missing notification mechanism",
                        file, node.lineno, node.name,
                        "Add update/notify method to complete observer pattern"
                    ))
        
        return issues
    
    def _check_boilerplate_overuse(self, file: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Check for excessive boilerplate code."""
        issues = []
        
        # Count boilerplate patterns
        boilerplate_count = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                
                # Check for boilerplate patterns
                if self._is_boilerplate_function(node):
                    boilerplate_count += 1
        
        if total_functions > 5 and boilerplate_count / total_functions > 0.5:
            issues.append(Issue(
                kind="excessive_boilerplate",
                message=f"File contains {boilerplate_count}/{total_functions} boilerplate functions",
                severity=2,
                file=file,
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
    
    def _is_super_init_call(self, stmt: ast.AST) -> bool:
        """Check if statement is a super().__init__() call."""
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == '__init__':
                if isinstance(call.func.value, ast.Call):
                    if isinstance(call.func.value.func, ast.Name) and call.func.value.func.id == 'super':
                        return True
        return False
    
    def _parent_has_init(self, class_node: ast.ClassDef, ctx: AnalysisContext) -> bool:
        """Check if parent class has __init__ method."""
        # This is simplified - would need full inheritance chain analysis
        return True  # Conservative assumption
    
    def _create_cargo_cult_issue(self, kind: str, message: str, file: str, 
                                line: int, symbol: str, suggestion: str) -> Issue:
        """Create a cargo cult pattern issue."""
        return Issue(
            kind=f"cargo_cult_{kind}",
            message=message,
            severity=self.boilerplate_patterns.get(kind, {}).get('severity', 2),
            file=file,
            line=line,
            symbol=symbol,
            evidence={'pattern': 'cargo_cult', 'type': kind},
            suggestions=[suggestion, "Understand the purpose before copying patterns"]
        )
    
    def _function_exists_nearby(self, lines: List[str], comment_line: int, func_name: str) -> bool:
        """Check if function exists near the comment."""
        # Look within 10 lines
        start = max(0, comment_line - 5)
        end = min(len(lines), comment_line + 10)
        
        for i in range(start, end):
            if f'def {func_name}' in lines[i]:
                return True
        return False
    
    def _is_trivial_accessor(self, method: ast.FunctionDef) -> bool:
        """Check if method is a trivial getter/setter."""
        if len(method.body) != 1:
            return False
        
        stmt = method.body[0]
        
        # Trivial getter: return self.x
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Attribute):
            if isinstance(stmt.value.value, ast.Name) and stmt.value.value.id == 'self':
                return True
        
        # Trivial setter: self.x = value
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Attribute):
                if isinstance(stmt.targets[0].value, ast.Name) and stmt.targets[0].value.id == 'self':
                    return True
        
        return False
    
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
            if isinstance(func.body[0].value, ast.Constant):
                return True
        
        return False
