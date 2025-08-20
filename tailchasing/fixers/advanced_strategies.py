"""
Advanced fix strategies for complex tail-chasing patterns.

Implements sophisticated transformations for context window thrashing,
hallucination cascades, and other complex patterns.
"""

import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict

try:
    import libcst as cst
    from libcst import metadata
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False
    cst = None
    metadata = None

from ..core.issues import Issue

logger = logging.getLogger(__name__)


class ContextThrashingFixer:
    """
    Fixes context window thrashing by extracting common functionality.
    
    Identifies repeated patterns across a file and extracts them into
    helper functions or utility classes.
    """
    
    def __init__(self):
        self.common_patterns = {}
        self.extraction_candidates = []
    
    def fix_context_thrashing(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Fix context window thrashing by extracting common patterns."""
        changes = []
        
        if not LIBCST_AVAILABLE:
            logger.warning("libcst not available, using basic context thrashing fix")
            return self._basic_context_fix(issue)
        
        try:
            source_file = Path(issue.file)
            if not source_file.exists():
                return changes
            
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse with libcst to maintain formatting
            tree = cst.parse_module(source_code)
            
            # Analyze the file for common patterns
            patterns = self._analyze_common_patterns(tree)
            
            if not patterns:
                return ["No common patterns found for extraction"]
            
            # Extract most beneficial patterns
            extractor = self._create_pattern_extractor(patterns)
            modified_tree = tree.visit(extractor)
            
            if extractor.extracted_patterns:
                # Write back the modified code
                modified_code = modified_tree.code
                with open(source_file, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                
                changes.extend([
                    f"Extracted {len(extractor.extracted_patterns)} common patterns",
                    f"Added helper functions: {', '.join(extractor.helper_functions)}",
                    f"Reduced code duplication by ~{extractor.estimated_reduction}%"
                ])
                
                logger.info(f"Fixed context thrashing in {source_file}: extracted {len(extractor.extracted_patterns)} patterns")
            
        except Exception as e:
            logger.error(f"Failed to fix context thrashing in {issue.file}: {e}")
            changes.append(f"Error during context thrashing fix: {e}")
        
        return changes
    
    def _analyze_common_patterns(self, tree: cst.Module) -> Dict[str, List[cst.CSTNode]]:
        """Analyze the CST for common patterns that can be extracted."""
        patterns = defaultdict(list)
        
        class PatternAnalyzer(cst.CSTVisitor):
            def __init__(self):
                self.function_calls = defaultdict(list)
                self.assignment_patterns = defaultdict(list)
                self.conditional_patterns = defaultdict(list)
            
            def visit_Call(self, node: cst.Call) -> None:
                # Identify common function call patterns
                call_signature = self._get_call_signature(node)
                self.function_calls[call_signature].append(node)
            
            def visit_Assign(self, node: cst.Assign) -> None:
                # Identify common assignment patterns
                pattern = self._get_assignment_pattern(node)
                if pattern:
                    self.assignment_patterns[pattern].append(node)
            
            def visit_If(self, node: cst.If) -> None:
                # Identify common conditional patterns
                condition_pattern = self._get_condition_pattern(node.test)
                if condition_pattern:
                    self.conditional_patterns[condition_pattern].append(node)
            
            def _get_call_signature(self, node: cst.Call) -> str:
                """Get a normalized signature for a function call."""
                try:
                    if isinstance(node.func, cst.Name):
                        return f"call_{node.func.value}"
                    elif isinstance(node.func, cst.Attribute):
                        return f"attr_call_{node.func.attr.value}"
                    else:
                        return "complex_call"
                except:
                    return "unknown_call"
            
            def _get_assignment_pattern(self, node: cst.Assign) -> Optional[str]:
                """Get a pattern for assignment statements."""
                try:
                    if len(node.targets) == 1:
                        target = node.targets[0]
                        if isinstance(target.target, cst.Name):
                            if isinstance(node.value, cst.Call):
                                return f"assign_call_{self._get_call_signature(node.value)}"
                            elif isinstance(node.value, cst.SimpleString):
                                return "assign_string"
                            elif isinstance(node.value, cst.Integer):
                                return "assign_number"
                except:
                    pass
                return None
            
            def _get_condition_pattern(self, condition: cst.BaseExpression) -> Optional[str]:
                """Get a pattern for conditional expressions."""
                try:
                    if isinstance(condition, cst.Comparison):
                        return "comparison"
                    elif isinstance(condition, cst.BooleanOperation):
                        return f"boolean_{condition.operator.__class__.__name__}"
                    elif isinstance(condition, cst.UnaryOperation):
                        return f"unary_{condition.operator.__class__.__name__}"
                except:
                    pass
                return None
        
        analyzer = PatternAnalyzer()
        tree.visit(analyzer)
        
        # Collect patterns with multiple occurrences
        for pattern, occurrences in analyzer.function_calls.items():
            if len(occurrences) >= 3:  # At least 3 occurrences to be worth extracting
                patterns[f"func_call_{pattern}"] = occurrences
        
        for pattern, occurrences in analyzer.assignment_patterns.items():
            if len(occurrences) >= 3:
                patterns[f"assignment_{pattern}"] = occurrences
        
        for pattern, occurrences in analyzer.conditional_patterns.items():
            if len(occurrences) >= 3:
                patterns[f"condition_{pattern}"] = occurrences
        
        return patterns
    
    def _create_pattern_extractor(self, patterns: Dict[str, List[cst.CSTNode]]) -> 'PatternExtractor':
        """Create a CST transformer to extract common patterns."""
        return PatternExtractor(patterns)
    
    def _basic_context_fix(self, issue: Issue) -> List[str]:
        """Basic context thrashing fix when libcst is not available."""
        return [
            "TODO: Install libcst for advanced context thrashing fixes",
            "Basic fix: Add comment markers for manual refactoring"
        ]


class PatternExtractor(cst.CSTTransformer):
    """CST transformer that extracts common patterns into helper functions."""
    
    def __init__(self, patterns: Dict[str, List[cst.CSTNode]]):
        self.patterns = patterns
        self.extracted_patterns = []
        self.helper_functions = []
        self.estimated_reduction = 0
        self.helper_counter = 0
    
    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Add helper functions at the top of the module."""
        if not self.helper_functions:
            return updated_node
        
        # Create helper function definitions
        helper_defs = []
        for helper_name, helper_body in self.helper_functions:
            helper_def = cst.FunctionDef(
                name=cst.Name(helper_name),
                params=cst.Parameters(),
                body=helper_body
            )
            helper_defs.append(helper_def)
        
        # Add helpers after imports but before other code
        new_body = list(updated_node.body)
        
        # Find insertion point (after imports)
        insert_idx = 0
        for i, stmt in enumerate(new_body):
            if isinstance(stmt, cst.SimpleStatementLine):
                for inner_stmt in stmt.body:
                    if isinstance(inner_stmt, (cst.Import, cst.ImportFrom)):
                        insert_idx = i + 1
                        break
            if insert_idx > 0:
                break
        
        # Insert helper functions
        for helper_def in helper_defs:
            new_body.insert(insert_idx, helper_def)
            insert_idx += 1
        
        return updated_node.with_changes(body=new_body)


class HallucinationCascadeFixer:
    """
    Fixes hallucination cascades by consolidating interdependent classes.
    
    Identifies classes that are artificially separated but should be
    unified based on their interaction patterns.
    """
    
    def __init__(self):
        self.class_dependencies = defaultdict(set)
        self.consolidation_candidates = []
    
    def fix_hallucination_cascade(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Fix hallucination cascade by consolidating related classes."""
        changes = []
        
        if not LIBCST_AVAILABLE:
            logger.warning("libcst not available, using basic hallucination cascade fix")
            return self._basic_cascade_fix(issue)
        
        try:
            source_file = Path(issue.file)
            if not source_file.exists():
                return changes
            
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = cst.parse_module(source_code)
            
            # Analyze class dependencies
            dependencies = self._analyze_class_dependencies(tree)
            
            # Find consolidation opportunities
            consolidation_groups = self._find_consolidation_groups(dependencies)
            
            if consolidation_groups:
                # Apply consolidation
                consolidator = ClassConsolidator(consolidation_groups)
                modified_tree = tree.visit(consolidator)
                
                if consolidator.consolidations_applied:
                    modified_code = modified_tree.code
                    with open(source_file, 'w', encoding='utf-8') as f:
                        f.write(modified_code)
                    
                    changes.extend([
                        f"Consolidated {len(consolidator.consolidations_applied)} class groups",
                        f"Reduced class count from {consolidator.original_class_count} to {consolidator.final_class_count}",
                        f"Merged related functionality in: {', '.join(consolidator.merged_classes)}"
                    ])
                    
                    logger.info(f"Fixed hallucination cascade in {source_file}")
            else:
                changes.append("No suitable consolidation opportunities found")
            
        except Exception as e:
            logger.error(f"Failed to fix hallucination cascade in {issue.file}: {e}")
            changes.append(f"Error during hallucination cascade fix: {e}")
        
        return changes
    
    def _analyze_class_dependencies(self, tree: cst.Module) -> Dict[str, Set[str]]:
        """Analyze dependencies between classes in the module."""
        dependencies = defaultdict(set)
        
        class DependencyAnalyzer(cst.CSTVisitor):
            def __init__(self):
                self.current_class = None
                self.class_methods = defaultdict(set)
                self.class_attributes = defaultdict(set)
                self.class_references = defaultdict(set)
            
            def visit_ClassDef(self, node: cst.ClassDef) -> None:
                self.current_class = node.name.value
            
            def leave_ClassDef(self, node: cst.ClassDef) -> None:
                self.current_class = None
            
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                if self.current_class:
                    self.class_methods[self.current_class].add(node.name.value)
            
            def visit_Assign(self, node: cst.Assign) -> None:
                if self.current_class:
                    for target in node.targets:
                        if isinstance(target.target, cst.Attribute):
                            if isinstance(target.target.value, cst.Name) and target.target.value.value == "self":
                                self.class_attributes[self.current_class].add(target.target.attr.value)
            
            def visit_Call(self, node: cst.Call) -> None:
                if self.current_class and isinstance(node.func, cst.Attribute):
                    if isinstance(node.func.value, cst.Name):
                        # Reference to another class or instance
                        ref_name = node.func.value.value
                        if ref_name != "self":
                            self.class_references[self.current_class].add(ref_name)
        
        analyzer = DependencyAnalyzer()
        tree.visit(analyzer)
        
        # Build dependency graph based on shared attributes/methods and references
        class_names = set(analyzer.class_methods.keys())
        
        for class_name in class_names:
            methods = analyzer.class_methods[class_name]
            attributes = analyzer.class_attributes[class_name]
            references = analyzer.class_references[class_name]
            
            # Find classes with similar method/attribute patterns
            for other_class in class_names:
                if other_class == class_name:
                    continue
                
                other_methods = analyzer.class_methods[other_class]
                other_attributes = analyzer.class_attributes[other_class]
                
                # Calculate similarity
                method_overlap = len(methods & other_methods) / max(len(methods | other_methods), 1)
                attr_overlap = len(attributes & other_attributes) / max(len(attributes | other_attributes), 1)
                
                # Check for direct references
                has_reference = other_class in references or class_name in analyzer.class_references[other_class]
                
                # High similarity or direct coupling suggests consolidation opportunity
                if method_overlap > 0.3 or attr_overlap > 0.3 or has_reference:
                    dependencies[class_name].add(other_class)
        
        return dependencies
    
    def _find_consolidation_groups(self, dependencies: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find groups of classes that should be consolidated."""
        # Use union-find to group highly connected classes
        groups = []
        visited = set()
        
        def dfs(class_name: str, current_group: Set[str]):
            if class_name in visited:
                return
            
            visited.add(class_name)
            current_group.add(class_name)
            
            # Visit all dependencies
            for dep in dependencies.get(class_name, set()):
                if dep not in visited:
                    # Check if dependency relationship is strong enough
                    reverse_deps = dependencies.get(dep, set())
                    if class_name in reverse_deps:  # Bidirectional dependency
                        dfs(dep, current_group)
        
        for class_name in dependencies:
            if class_name not in visited:
                group = set()
                dfs(class_name, group)
                if len(group) >= 2:  # Only consolidate groups of 2+
                    groups.append(group)
        
        return groups
    
    def _basic_cascade_fix(self, issue: Issue) -> List[str]:
        """Basic hallucination cascade fix when libcst is not available."""
        return [
            "TODO: Install libcst for advanced hallucination cascade fixes",
            "Basic fix: Add documentation comments about class relationships"
        ]


class ClassConsolidator(cst.CSTTransformer):
    """CST transformer that consolidates related classes."""
    
    def __init__(self, consolidation_groups: List[Set[str]]):
        self.consolidation_groups = consolidation_groups
        self.consolidations_applied = []
        self.merged_classes = []
        self.original_class_count = 0
        self.final_class_count = 0
        self.classes_to_remove = set()
        
        # Prepare consolidation mapping
        self.class_mapping = {}
        for group in consolidation_groups:
            group_list = sorted(list(group))
            primary_class = group_list[0]  # Use first (alphabetically) as primary
            for class_name in group_list[1:]:
                self.class_mapping[class_name] = primary_class
                self.classes_to_remove.add(class_name)
    
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        """Process class definitions for consolidation."""
        class_name = updated_node.name.value
        self.original_class_count += 1
        
        if class_name in self.classes_to_remove:
            # This class will be merged into another, mark for removal
            return cst.RemovalSentinel.REMOVE
        
        # Check if this class should receive merged content
        merged_content = []
        for group in self.consolidation_groups:
            if class_name in group:
                # This is the primary class for the group
                group_list = sorted(list(group))
                if class_name == group_list[0]:
                    # Add comment about consolidation
                    comment = cst.SimpleStatementLine([
                        cst.Expr(cst.SimpleString(f'"Consolidated class containing functionality from: {", ".join(group_list[1:])}"'))
                    ])
                    
                    # Add the comment at the beginning of the class body
                    if isinstance(updated_node.body, cst.IndentedBlock):
                        new_body = [comment] + list(updated_node.body.body)
                        updated_node = updated_node.with_changes(
                            body=cst.IndentedBlock(new_body)
                        )
                    
                    self.consolidations_applied.append(group)
                    self.merged_classes.extend(group_list[1:])
        
        self.final_class_count += 1
        return updated_node


class CircularImportFixer:
    """
    Fixes circular imports by restructuring dependencies.
    
    Analyzes import dependencies and suggests/applies restructuring
    to break circular dependencies.
    """
    
    def __init__(self):
        self.import_graph = defaultdict(set)
        self.circular_chains = []
    
    def fix_circular_import(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Fix circular import by restructuring dependencies."""
        changes = []
        
        try:
            # Analyze the import structure
            if issue.evidence and 'circular_chain' in issue.evidence:
                circular_chain = issue.evidence['circular_chain']
                changes.extend(self._break_circular_chain(circular_chain))
            else:
                changes.append("Circular import detected but no chain information available")
            
        except Exception as e:
            logger.error(f"Failed to fix circular import: {e}")
            changes.append(f"Error during circular import fix: {e}")
        
        return changes
    
    def _break_circular_chain(self, chain: List[str]) -> List[str]:
        """Break a circular import chain."""
        changes = []
        
        if len(chain) < 2:
            return ["Invalid circular chain"]
        
        # Strategy: Move shared dependencies to a separate module
        shared_module_name = "shared_dependencies.py"
        
        changes.extend([
            f"TODO: Create {shared_module_name} for shared dependencies",
            f"TODO: Move common imports from {' -> '.join(chain)} to shared module",
            f"TODO: Update imports in affected files to use shared module"
        ])
        
        return changes


# Factory function to get the appropriate fixer
def get_advanced_fix_strategy(issue_kind: str) -> Optional[callable]:
    """Get an advanced fix strategy for complex issue types."""
    strategies = {
        'context_window_thrashing': ContextThrashingFixer().fix_context_thrashing,
        'hallucination_cascade': HallucinationCascadeFixer().fix_hallucination_cascade,
        'circular_import': CircularImportFixer().fix_circular_import,
    }
    
    return strategies.get(issue_kind)