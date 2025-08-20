"""
Base Class Extractor - Automatically suggests base classes for duplicate code patterns.

This analyzer identifies common patterns across classes and functions that could
be extracted into base classes or shared utilities.
"""

import ast
import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ..core.issues import Issue
from .base import BaseAnalyzer


@dataclass
class CommonPattern:
    """Represents a common pattern found across multiple classes/functions."""
    
    pattern_type: str  # 'method', 'attribute', 'init_pattern', 'validation'
    pattern_hash: str
    occurrences: List[Tuple[str, str, int]]  # (file, class/func name, line)
    pattern_ast: Optional[ast.AST] = None
    pattern_code: str = ""
    shared_attributes: Set[str] = field(default_factory=set)
    shared_methods: Set[str] = field(default_factory=set)


@dataclass
class BaseClassSuggestion:
    """Suggestion for a base class extraction."""
    
    suggested_name: str
    target_classes: List[str]
    common_methods: List[str]
    common_attributes: List[str]
    estimated_line_reduction: int
    file_paths: Set[str]


class BaseClassExtractor(BaseAnalyzer):
    """
    Analyzes classes and functions to suggest base class extractions.
    
    Detects:
    1. Common initialization patterns
    2. Duplicate method implementations
    3. Shared attribute sets
    4. Common validation logic
    5. Repeated error handling patterns
    6. Similar data transformation methods
    """
    
    def __init__(self):
        super().__init__()
        self.class_patterns: Dict[str, CommonPattern] = {}
        self.method_signatures: Dict[str, List[Tuple[str, str, ast.FunctionDef]]] = defaultdict(list)
        self.class_attributes: Dict[str, Set[str]] = defaultdict(set)
        self.class_methods: Dict[str, Set[str]] = defaultdict(set)
        self.init_patterns: Dict[str, List[Tuple[str, str, ast.FunctionDef]]] = defaultdict(list)
        
    def analyze(self, context) -> List[Issue]:
        """Analyze codebase for base class extraction opportunities."""
        issues = []
        
        # First pass: collect all class information
        self._collect_class_information(context)
        
        # Analyze common patterns
        method_patterns = self._analyze_method_patterns()
        init_patterns = self._analyze_init_patterns()
        attribute_patterns = self._analyze_attribute_patterns()
        
        # Generate base class suggestions
        suggestions = self._generate_base_class_suggestions(
            method_patterns, init_patterns, attribute_patterns
        )
        
        # Create issues from suggestions
        for suggestion in suggestions:
            if suggestion.estimated_line_reduction > 20:  # Significant reduction
                issues.append(self._create_base_class_issue(suggestion))
        
        # Find validation pattern duplicates
        validation_patterns = self._find_validation_patterns(context)
        if validation_patterns:
            issues.append(self._create_validation_base_issue(validation_patterns))
        
        # Find common error handling patterns
        error_patterns = self._find_error_handling_patterns(context)
        if error_patterns:
            issues.append(self._create_error_handling_base_issue(error_patterns))
        
        return issues
    
    def _collect_class_information(self, context):
        """Collect information about all classes in the codebase."""
        for file_path, ast_tree in context.ast_index.items():
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.ClassDef):
                    class_key = f"{file_path}:{node.name}"
                    
                    # Collect methods
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            self.class_methods[class_key].add(item.name)
                            
                            # Store method with signature
                            sig_hash = self._hash_function_signature(item)
                            self.method_signatures[sig_hash].append(
                                (file_path, node.name, item)
                            )
                            
                            # Special handling for __init__
                            if item.name == '__init__':
                                init_hash = self._hash_init_pattern(item)
                                self.init_patterns[init_hash].append(
                                    (file_path, node.name, item)
                                )
                        
                        # Collect attributes (from __init__ and class level)
                        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            self.class_attributes[class_key].add(item.target.id)
                    
                    # Also collect attributes from __init__
                    init_method = self._find_init_method(node)
                    if init_method:
                        attrs = self._extract_init_attributes(init_method)
                        self.class_attributes[class_key].update(attrs)
    
    def _hash_function_signature(self, func: ast.FunctionDef) -> str:
        """Create hash of function signature and structure."""
        # Include function structure but not exact variable names
        structure_parts = []
        
        # Add parameter count and types if annotated
        param_count = len(func.args.args)
        structure_parts.append(f"params:{param_count}")
        
        # Add return type if annotated
        if func.returns:
            structure_parts.append(f"returns:{ast.dump(func.returns)}")
        
        # Add body structure (types of statements)
        for stmt in func.body:
            if not isinstance(stmt, (ast.Expr, ast.Pass)):  # Skip docstrings
                structure_parts.append(type(stmt).__name__)
        
        structure_str = "|".join(structure_parts)
        return hashlib.md5(structure_str.encode()).hexdigest()[:8]
    
    def _hash_init_pattern(self, init_func: ast.FunctionDef) -> str:
        """Create hash of __init__ pattern."""
        pattern_parts = []
        
        # Extract self assignments
        for stmt in init_func.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            # Record attribute being set
                            pattern_parts.append(f"self.{target.attr}")
        
        pattern_str = "|".join(sorted(pattern_parts))
        return hashlib.md5(pattern_str.encode()).hexdigest()[:8]
    
    def _find_init_method(self, class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find __init__ method in class."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                return item
        return None
    
    def _extract_init_attributes(self, init_func: ast.FunctionDef) -> Set[str]:
        """Extract attributes set in __init__."""
        attributes = set()
        
        for stmt in init_func.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == 'self':
                            attributes.add(target.attr)
        
        return attributes
    
    def _analyze_method_patterns(self) -> List[CommonPattern]:
        """Analyze common method patterns across classes."""
        patterns = []
        
        for sig_hash, occurrences in self.method_signatures.items():
            if len(occurrences) >= 3:  # At least 3 occurrences
                # Get the actual method names
                method_names = {occ[2].name for occ in occurrences}
                
                # Skip if all are special methods
                if all(name.startswith('__') and name.endswith('__') for name in method_names):
                    continue
                
                pattern = CommonPattern(
                    pattern_type='method',
                    pattern_hash=sig_hash,
                    occurrences=[(occ[0], occ[1], occ[2].lineno) for occ in occurrences],
                    shared_methods=method_names
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_init_patterns(self) -> List[CommonPattern]:
        """Analyze common initialization patterns."""
        patterns = []
        
        for init_hash, occurrences in self.init_patterns.items():
            if len(occurrences) >= 3:
                pattern = CommonPattern(
                    pattern_type='init_pattern',
                    pattern_hash=init_hash,
                    occurrences=[(occ[0], occ[1], occ[2].lineno) for occ in occurrences]
                )
                
                # Extract common attributes from these inits
                for _, _, init_func in occurrences:
                    attrs = self._extract_init_attributes(init_func)
                    if not pattern.shared_attributes:
                        pattern.shared_attributes = attrs
                    else:
                        pattern.shared_attributes &= attrs  # Intersection
                
                if pattern.shared_attributes:
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_attribute_patterns(self) -> List[CommonPattern]:
        """Analyze common attribute sets across classes."""
        patterns = []
        
        # Group classes by attribute sets
        attr_groups = defaultdict(list)
        for class_key, attributes in self.class_attributes.items():
            if len(attributes) >= 3:  # Meaningful attribute set
                attr_key = "|".join(sorted(attributes))
                attr_groups[attr_key].append(class_key)
        
        # Find common attribute sets
        for attr_key, classes in attr_groups.items():
            if len(classes) >= 3:
                attributes = set(attr_key.split("|"))
                pattern = CommonPattern(
                    pattern_type='attribute',
                    pattern_hash=hashlib.md5(attr_key.encode()).hexdigest()[:8],
                    occurrences=[(c.split(":")[0], c.split(":")[1], 0) for c in classes],
                    shared_attributes=attributes
                )
                patterns.append(pattern)
        
        return patterns
    
    def _generate_base_class_suggestions(
        self, 
        method_patterns: List[CommonPattern],
        init_patterns: List[CommonPattern],
        attribute_patterns: List[CommonPattern]
    ) -> List[BaseClassSuggestion]:
        """Generate concrete base class suggestions."""
        suggestions = []
        
        # Combine patterns to find classes that share multiple patterns
        class_pattern_map = defaultdict(set)
        
        for pattern in method_patterns + init_patterns + attribute_patterns:
            for file_path, class_name, _ in pattern.occurrences:
                class_key = f"{file_path}:{class_name}"
                class_pattern_map[class_key].add(pattern.pattern_hash)
        
        # Find groups of classes with similar patterns
        pattern_groups = defaultdict(list)
        for class_key, pattern_hashes in class_pattern_map.items():
            if len(pattern_hashes) >= 2:  # Share at least 2 patterns
                pattern_key = "|".join(sorted(pattern_hashes))
                pattern_groups[pattern_key].append(class_key)
        
        # Create suggestions for significant groups
        for pattern_key, classes in pattern_groups.items():
            if len(classes) >= 3:
                # Collect common elements
                common_methods = set()
                common_attributes = set()
                
                for class_key in classes:
                    common_methods.update(self.class_methods.get(class_key, set()))
                    common_attributes.update(self.class_attributes.get(class_key, set()))
                
                # Find actual common elements (intersection)
                if len(classes) > 1:
                    for class_key in classes[1:]:
                        common_methods &= self.class_methods.get(class_key, set())
                        common_attributes &= self.class_attributes.get(class_key, set())
                
                if common_methods or common_attributes:
                    suggestion = BaseClassSuggestion(
                        suggested_name=self._suggest_base_class_name(classes),
                        target_classes=[c.split(":")[1] for c in classes],
                        common_methods=list(common_methods),
                        common_attributes=list(common_attributes),
                        estimated_line_reduction=len(common_methods) * 5 * len(classes),
                        file_paths={c.split(":")[0] for c in classes}
                    )
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _suggest_base_class_name(self, classes: List[str]) -> str:
        """Suggest a name for the base class."""
        # Extract class names
        class_names = [c.split(":")[1] for c in classes]
        
        # Find common prefix/suffix
        if all('Handler' in name for name in class_names):
            return "BaseHandler"
        elif all('Service' in name for name in class_names):
            return "BaseService"
        elif all('Model' in name for name in class_names):
            return "BaseModel"
        elif all('View' in name for name in class_names):
            return "BaseView"
        elif all('Controller' in name for name in class_names):
            return "BaseController"
        else:
            # Generic base class name
            return "BaseClass"
    
    def _find_validation_patterns(self, context) -> List[Tuple[str, List[str]]]:
        """Find common validation patterns."""
        validation_patterns = defaultdict(list)
        
        for file_path, ast_tree in context.ast_index.items():
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.FunctionDef):
                    if 'validate' in node.name.lower() or 'check' in node.name.lower():
                        # Hash the validation logic
                        pattern_hash = self._hash_function_signature(node)
                        validation_patterns[pattern_hash].append(f"{file_path}:{node.name}")
        
        # Return patterns with multiple occurrences
        return [(pattern, occurrences) for pattern, occurrences in validation_patterns.items() 
                if len(occurrences) >= 3]
    
    def _find_error_handling_patterns(self, context) -> List[Tuple[str, List[str]]]:
        """Find common error handling patterns."""
        error_patterns = defaultdict(list)
        
        for file_path, ast_tree in context.ast_index.items():
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Try):
                    # Hash the try-except structure
                    pattern_parts = []
                    for handler in node.handlers:
                        if handler.type:
                            if isinstance(handler.type, ast.Name):
                                pattern_parts.append(handler.type.id)
                    
                    if pattern_parts:
                        pattern_hash = hashlib.md5("|".join(pattern_parts).encode()).hexdigest()[:8]
                        error_patterns[pattern_hash].append(file_path)
        
        return [(pattern, occurrences) for pattern, occurrences in error_patterns.items()
                if len(occurrences) >= 5]
    
    def _create_base_class_issue(self, suggestion: BaseClassSuggestion) -> Issue:
        """Create issue for base class extraction suggestion."""
        return Issue(
            type="base_class_extraction_opportunity",
            severity="MEDIUM",
            message=f"Extract base class for {len(suggestion.target_classes)} similar classes",
            file_path="<global>",
            line_number=0,
            details={
                "suggested_name": suggestion.suggested_name,
                "target_classes": suggestion.target_classes,
                "common_methods": suggestion.common_methods,
                "common_attributes": suggestion.common_attributes,
                "estimated_line_reduction": suggestion.estimated_line_reduction,
                "affected_files": list(suggestion.file_paths),
                "recommendation": f"Create {suggestion.suggested_name} with shared functionality",
                "benefits": [
                    f"Reduce ~{suggestion.estimated_line_reduction} lines of duplicate code",
                    "Improve maintainability",
                    "Enforce consistent behavior",
                    "Simplify testing"
                ]
            }
        )
    
    def _create_validation_base_issue(self, patterns: List[Tuple[str, List[str]]]) -> Issue:
        """Create issue for validation pattern extraction."""
        total_occurrences = sum(len(occs) for _, occs in patterns)
        
        return Issue(
            type="validation_pattern_duplication",
            severity="MEDIUM",
            message=f"Found {len(patterns)} duplicate validation patterns",
            file_path="<global>",
            line_number=0,
            details={
                "pattern_count": len(patterns),
                "total_duplicates": total_occurrences,
                "examples": [occs[:3] for _, occs in patterns[:3]],
                "recommendation": "Extract common validation logic to base validator class",
                "suggested_structure": {
                    "BaseValidator": ["validate()", "check_required()", "check_format()"],
                    "ValidationMixin": ["Common validation methods"],
                }
            }
        )
    
    def _create_error_handling_base_issue(self, patterns: List[Tuple[str, List[str]]]) -> Issue:
        """Create issue for error handling pattern extraction."""
        return Issue(
            type="error_handling_pattern_duplication",
            severity="LOW",
            message=f"Found {len(patterns)} duplicate error handling patterns",
            file_path="<global>",
            line_number=0,
            details={
                "pattern_count": len(patterns),
                "recommendation": "Consider extracting common error handling to base class or decorator",
                "suggested_approaches": [
                    "Create ErrorHandlerMixin for common patterns",
                    "Use decorators for consistent error handling",
                    "Extract to context managers for resource handling"
                ]
            }
        )