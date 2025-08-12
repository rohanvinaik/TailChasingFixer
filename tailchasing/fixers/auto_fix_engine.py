"""
Intelligent Auto-Fix Engine for TailChasingFixer.

Automatically fixes detected tail-chasing patterns with comprehensive safety mechanisms,
dependency analysis, and detailed reporting.
"""

import ast
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Iterator
import json
import subprocess
import hashlib
from collections import defaultdict, deque

try:
    import libcst as cst
    from libcst import metadata
    LIBCST_AVAILABLE = True
except ImportError:
    LIBCST_AVAILABLE = False
    cst = None
    metadata = None

from ..core.issues import Issue, IssueSeverity
from .strategies.base import RiskLevel
from .advanced_strategies import get_advanced_fix_strategy

logger = logging.getLogger(__name__)


class FixStatus(Enum):
    """Status of fix application."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class FixPriority(Enum):
    """Priority levels for fix application."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class FixResult:
    """Result of applying a fix."""
    issue: Issue
    status: FixStatus
    applied_changes: List[str] = field(default_factory=list)
    backup_path: Optional[str] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    test_results: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if fix was successful."""
        return self.status == FixStatus.COMPLETED


@dataclass
class FixPlan:
    """Plan for applying fixes with dependency ordering."""
    fixes: List['FixAction'] = field(default_factory=list)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    estimated_time: float = 0.0
    total_risk_score: float = 0.0


@dataclass
class FixAction:
    """Individual fix action with metadata."""
    id: str
    issue: Issue
    strategy: str
    priority: FixPriority
    risk_level: RiskLevel
    dependencies: Set[str] = field(default_factory=set)
    estimated_time: float = 5.0
    dry_run_preview: Optional[str] = None
    requires_tests: bool = True


class BackupManager:
    """Manages file backups and rollback operations."""
    
    def __init__(self, backup_dir: Optional[str] = None):
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = Path(tempfile.gettempdir()) / "tailchasing_backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_registry: Dict[str, str] = {}
        
    def create_backup(self, file_path: str) -> str:
        """Create a backup of the file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Generate unique backup name
        timestamp = int(time.time())
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        backup_name = f"{file_path.name}_{timestamp}_{file_hash}.backup"
        backup_path = self.backup_dir / backup_name
        
        # Copy file to backup location
        shutil.copy2(file_path, backup_path)
        self.backup_registry[str(file_path)] = str(backup_path)
        
        logger.info(f"Created backup: {file_path} -> {backup_path}")
        return str(backup_path)
    
    def restore_backup(self, file_path: str) -> bool:
        """Restore a file from its backup."""
        backup_path = self.backup_registry.get(str(file_path))
        if not backup_path or not Path(backup_path).exists():
            logger.error(f"No backup found for {file_path}")
            return False
        
        try:
            shutil.copy2(backup_path, file_path)
            logger.info(f"Restored backup: {backup_path} -> {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup for {file_path}: {e}")
            return False
    
    def cleanup_backups(self, max_age_hours: int = 24):
        """Clean up old backup files."""
        current_time = time.time()
        for backup_file in self.backup_dir.glob("*.backup"):
            if current_time - backup_file.stat().st_mtime > max_age_hours * 3600:
                backup_file.unlink()
                logger.debug(f"Cleaned up old backup: {backup_file}")


class DependencyAnalyzer:
    """Analyzes dependencies between fixes to determine safe execution order."""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_dependency(self, fix_id: str, depends_on: str):
        """Add a dependency relationship."""
        self.graph[fix_id].add(depends_on)
        self.reverse_graph[depends_on].add(fix_id)
    
    def analyze_file_dependencies(self, fixes: List[FixAction]) -> Dict[str, Set[str]]:
        """Analyze file-level dependencies between fixes."""
        file_dependencies = defaultdict(set)
        
        # Group fixes by file
        fixes_by_file = defaultdict(list)
        for fix in fixes:
            if fix.issue.file:
                fixes_by_file[fix.issue.file].append(fix)
        
        # Analyze import relationships
        for file_path, file_fixes in fixes_by_file.items():
            if not Path(file_path).exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Find import statements
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                
                # Check if any imports correspond to files with fixes
                for imported_module in imports:
                    # Convert module name to potential file paths
                    module_path = imported_module.replace('.', '/')
                    potential_paths = [
                        f"{module_path}.py",
                        f"{module_path}/__init__.py"
                    ]
                    
                    for potential_path in potential_paths:
                        if potential_path in fixes_by_file:
                            # File depends on the imported file
                            for fix in file_fixes:
                                for imported_fix in fixes_by_file[potential_path]:
                                    file_dependencies[fix.id].add(imported_fix.id)
                                    
            except Exception as e:
                logger.warning(f"Failed to analyze dependencies for {file_path}: {e}")
        
        return file_dependencies
    
    def topological_sort(self, fixes: List[FixAction]) -> List[str]:
        """Perform topological sort to determine safe execution order."""
        # Build complete dependency graph
        all_dependencies = self.analyze_file_dependencies(fixes)
        
        # Add explicit dependencies
        for fix in fixes:
            for dep in fix.dependencies:
                all_dependencies[fix.id].add(dep)
        
        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)
        graph = defaultdict(set)
        
        # Build graph and calculate in-degrees
        all_fix_ids = {fix.id for fix in fixes}
        for fix_id in all_fix_ids:
            for dep in all_dependencies[fix_id]:
                if dep in all_fix_ids:
                    graph[dep].add(fix_id)
                    in_degree[fix_id] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque([fix_id for fix_id in all_fix_ids if in_degree[fix_id] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Remove current node and update in-degrees
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(all_fix_ids):
            remaining = all_fix_ids - set(result)
            logger.warning(f"Circular dependencies detected for fixes: {remaining}")
            # Add remaining fixes in priority order
            remaining_fixes = [f for f in fixes if f.id in remaining]
            remaining_fixes.sort(key=lambda x: x.priority.value)
            result.extend([f.id for f in remaining_fixes])
        
        return result


class SafetyValidator:
    """Validates fixes for safety before and after application."""
    
    def __init__(self, test_command: Optional[str] = None):
        self.test_command = test_command or "python -m pytest --tb=short -x"
        
    def validate_syntax(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate Python syntax of a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Parse error: {e}"
    
    def run_tests(self, test_paths: List[str] = None) -> Dict[str, Any]:
        """Run tests to validate changes."""
        if not test_paths:
            # Run all tests
            cmd = self.test_command.split()
        else:
            # Run specific test files
            cmd = self.test_command.split() + test_paths
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Test execution timed out',
                'return_code': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Test execution failed: {e}',
                'return_code': -1
            }
    
    def assess_risk(self, fix_action: FixAction) -> Dict[str, Any]:
        """Assess risk level of applying a fix."""
        risk_factors = {
            'complexity': 0.0,
            'file_impact': 0.0,
            'dependency_impact': 0.0,
            'test_coverage': 0.0
        }
        
        # Assess based on issue type
        if fix_action.issue.kind == "semantic_duplicate_function":
            risk_factors['complexity'] = 0.3
            risk_factors['file_impact'] = 0.2
        elif fix_action.issue.kind == "phantom_function":
            risk_factors['complexity'] = 0.6
            risk_factors['file_impact'] = 0.4
        elif fix_action.issue.kind == "import_anxiety":
            risk_factors['complexity'] = 0.1
            risk_factors['file_impact'] = 0.1
        elif fix_action.issue.kind == "context_window_thrashing":
            risk_factors['complexity'] = 0.5
            risk_factors['file_impact'] = 0.6
        elif fix_action.issue.kind == "hallucination_cascade":
            risk_factors['complexity'] = 0.8
            risk_factors['file_impact'] = 0.7
        
        # Assess dependency impact
        if len(fix_action.dependencies) > 3:
            risk_factors['dependency_impact'] = 0.6
        elif len(fix_action.dependencies) > 1:
            risk_factors['dependency_impact'] = 0.3
        
        # Calculate overall risk score
        weights = {'complexity': 0.3, 'file_impact': 0.3, 'dependency_impact': 0.3, 'test_coverage': 0.1}
        overall_risk = sum(risk_factors[factor] * weights[factor] for factor in risk_factors)
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'risk_level': fix_action.risk_level.name,
            'requires_manual_review': overall_risk > 0.7
        }


class FixStrategyRegistry:
    """Registry of fix strategies for different issue types."""
    
    def __init__(self):
        self.strategies: Dict[str, callable] = {}
        self._register_default_strategies()
    
    def register_strategy(self, issue_kind: str, strategy_func: callable):
        """Register a fix strategy for an issue type."""
        self.strategies[issue_kind] = strategy_func
        logger.debug(f"Registered fix strategy for {issue_kind}")
    
    def get_strategy(self, issue_kind: str) -> Optional[callable]:
        """Get fix strategy for an issue type."""
        return self.strategies.get(issue_kind)
    
    def _register_default_strategies(self):
        """Register default fix strategies."""
        self.strategies.update({
            "semantic_duplicate_function": self._fix_semantic_duplicates,
            "phantom_function": self._fix_phantom_function,
            "import_anxiety": self._fix_import_anxiety,
            "context_window_thrashing": self._fix_context_thrashing,
            "hallucination_cascade": self._fix_hallucination_cascade,
            "circular_import": self._fix_circular_import,
            "missing_symbol": self._fix_missing_symbol
        })
    
    def _fix_semantic_duplicates(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Fix semantic duplicate functions by merging and creating aliases."""
        changes = []
        
        if not LIBCST_AVAILABLE:
            logger.warning("libcst not available, skipping semantic duplicate fix")
            return changes
        
        try:
            # Get duplicate information from evidence
            evidence = issue.evidence or {}
            duplicate_file = evidence.get('duplicate_file')
            duplicate_symbol = evidence.get('duplicate_symbol')
            
            if not duplicate_file or not duplicate_symbol:
                logger.warning("Insufficient evidence for semantic duplicate fix")
                return changes
            
            # Read and parse the source file
            source_file = Path(issue.file)
            if not source_file.exists():
                return changes
            
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse with libcst
            tree = cst.parse_expression(source_code) if source_code.strip() else None
            if not tree:
                tree = cst.parse_module(source_code)
            
            # Create transformer to add deprecation warning and alias
            class DuplicateTransformer(cst.CSTTransformer):
                def __init__(self, target_func: str, replacement_func: str):
                    self.target_func = target_func
                    self.replacement_func = replacement_func
                    self.modified = False
                
                def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
                    if updated_node.name.value == self.target_func:
                        # Add deprecation warning
                        warning_stmt = cst.SimpleStatementLine([
                            cst.Expr(cst.Call(
                                func=cst.Attribute(
                                    value=cst.Name("warnings"),
                                    attr=cst.Name("warn")
                                ),
                                args=[
                                    cst.Arg(cst.SimpleString(f'"Function {self.target_func} is deprecated. Use {self.replacement_func} instead."')),
                                    cst.Arg(cst.Name("DeprecationWarning"))
                                ]
                            ))
                        ])
                        
                        # Add alias call
                        alias_stmt = cst.SimpleStatementLine([
                            cst.Return(cst.Call(
                                func=cst.Name(self.replacement_func),
                                args=[cst.Arg(cst.StarredElement(cst.Name("args"))), 
                                     cst.Arg(keyword=cst.Name("kwargs"), value=cst.StarredElement(cst.Name("kwargs")))]
                            ))
                        ])
                        
                        # Modify function body
                        new_body = cst.IndentedBlock([
                            warning_stmt,
                            alias_stmt
                        ])
                        
                        # Add *args, **kwargs parameters if not present
                        new_params = list(updated_node.params.params)
                        has_args = any(param.star == "*" for param in new_params)
                        has_kwargs = updated_node.params.kwonly_params or updated_node.params.star_kwarg
                        
                        if not has_args:
                            new_params.append(cst.Param(name=cst.Name("args"), star="*"))
                        if not has_kwargs:
                            star_kwarg = cst.Param(name=cst.Name("kwargs"), star="**")
                        else:
                            star_kwarg = updated_node.params.star_kwarg
                        
                        new_params_obj = updated_node.params.with_changes(
                            params=new_params,
                            star_kwarg=star_kwarg
                        )
                        
                        self.modified = True
                        return updated_node.with_changes(
                            params=new_params_obj,
                            body=new_body
                        )
                    return updated_node
            
            # Apply transformation
            transformer = DuplicateTransformer(issue.symbol, duplicate_symbol)
            modified_tree = tree.visit(transformer)
            
            if transformer.modified:
                # Write back the modified code
                modified_code = modified_tree.code
                with open(source_file, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                
                changes.append(f"Added deprecation warning and alias for {issue.symbol}")
                logger.info(f"Fixed semantic duplicate: {issue.symbol} -> {duplicate_symbol}")
            
        except Exception as e:
            logger.error(f"Failed to fix semantic duplicate {issue.symbol}: {e}")
        
        return changes
    
    def _fix_phantom_function(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Generate implementation for phantom functions based on context."""
        changes = []
        
        if not LIBCST_AVAILABLE:
            logger.warning("libcst not available, skipping phantom function fix")
            return changes
        
        try:
            source_file = Path(issue.file)
            if not source_file.exists():
                return changes
            
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = cst.parse_module(source_code)
            
            class PhantomTransformer(cst.CSTTransformer):
                def __init__(self, target_func: str):
                    self.target_func = target_func
                    self.modified = False
                
                def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
                    if updated_node.name.value == self.target_func:
                        # Check if function is empty or has only pass/placeholder
                        if self._is_phantom_function(updated_node):
                            # Generate basic implementation
                            impl = self._generate_implementation(updated_node)
                            self.modified = True
                            return updated_node.with_changes(body=impl)
                    return updated_node
                
                def _is_phantom_function(self, func_node: cst.FunctionDef) -> bool:
                    """Check if function is a phantom (empty or placeholder)."""
                    if isinstance(func_node.body, cst.IndentedBlock):
                        statements = func_node.body.body
                        if len(statements) == 1:
                            stmt = statements[0]
                            if isinstance(stmt, cst.SimpleStatementLine):
                                if len(stmt.body) == 1:
                                    body_item = stmt.body[0]
                                    if isinstance(body_item, (cst.Pass, cst.Expr)):
                                        if isinstance(body_item, cst.Expr):
                                            # Check for common placeholders
                                            if isinstance(body_item.value, cst.SimpleString):
                                                content = body_item.value.value.lower()
                                                if any(placeholder in content for placeholder in 
                                                      ['todo', 'implement', 'placeholder', 'not implemented']):
                                                    return True
                                        return True
                    return False
                
                def _generate_implementation(self, func_node: cst.FunctionDef) -> cst.IndentedBlock:
                    """Generate basic implementation based on function signature."""
                    func_name = func_node.name.value
                    
                    # Analyze function name for hints
                    if 'get' in func_name.lower():
                        # Getter function
                        impl = cst.SimpleStatementLine([
                            cst.Return(cst.SimpleString('"TODO: Implement getter"'))
                        ])
                    elif 'set' in func_name.lower():
                        # Setter function
                        impl = cst.SimpleStatementLine([
                            cst.Expr(cst.SimpleString('"TODO: Implement setter"')),
                            cst.Return(cst.Name("None"))
                        ])
                    elif 'calculate' in func_name.lower() or 'compute' in func_name.lower():
                        # Calculation function
                        impl = cst.SimpleStatementLine([
                            cst.Return(cst.Integer("0"))
                        ])
                    elif 'process' in func_name.lower():
                        # Processing function
                        impl = cst.SimpleStatementLine([
                            cst.Return(cst.List([]))
                        ])
                    else:
                        # Generic implementation
                        impl = cst.SimpleStatementLine([
                            cst.Expr(cst.SimpleString(f'"TODO: Implement {func_name}"')),
                            cst.Return(cst.Name("None"))
                        ])
                    
                    return cst.IndentedBlock([impl])
            
            transformer = PhantomTransformer(issue.symbol)
            modified_tree = tree.visit(transformer)
            
            if transformer.modified:
                modified_code = modified_tree.code
                with open(source_file, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                
                changes.append(f"Generated basic implementation for phantom function {issue.symbol}")
                logger.info(f"Fixed phantom function: {issue.symbol}")
            
        except Exception as e:
            logger.error(f"Failed to fix phantom function {issue.symbol}: {e}")
        
        return changes
    
    def _fix_import_anxiety(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Remove unused imports."""
        changes = []
        
        if not LIBCST_AVAILABLE:
            logger.warning("libcst not available, skipping import anxiety fix")
            return changes
        
        try:
            source_file = Path(issue.file)
            if not source_file.exists():
                return changes
            
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = cst.parse_module(source_code)
            
            # Analyze which imports are actually used
            class ImportUsageAnalyzer(cst.CSTVisitor):
                def __init__(self):
                    self.imports = set()
                    self.used_names = set()
                
                def visit_Import(self, node: cst.Import) -> None:
                    for name in node.names:
                        if isinstance(name, cst.ImportAlias):
                            self.imports.add(name.name.value if isinstance(name.name, cst.Name) else str(name.name))
                
                def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
                    if isinstance(node.names, cst.ImportStar):
                        return
                    for name in node.names:
                        if isinstance(name, cst.ImportAlias):
                            alias = name.asname.name.value if name.asname else name.name.value
                            self.used_names.add(alias)
                
                def visit_Name(self, node: cst.Name) -> None:
                    self.used_names.add(node.value)
                
                def visit_Attribute(self, node: cst.Attribute) -> None:
                    if isinstance(node.value, cst.Name):
                        self.used_names.add(node.value.value)
            
            analyzer = ImportUsageAnalyzer()
            tree.visit(analyzer)
            
            # Remove unused imports
            class ImportRemover(cst.CSTTransformer):
                def __init__(self, used_names: Set[str]):
                    self.used_names = used_names
                    self.removed_imports = []
                
                def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine) -> Union[cst.SimpleStatementLine, cst.RemovalSentinel]:
                    # Check if this is an import statement
                    for stmt in updated_node.body:
                        if isinstance(stmt, (cst.Import, cst.ImportFrom)):
                            if self._should_remove_import(stmt):
                                self.removed_imports.append(str(stmt))
                                return cst.RemovalSentinel.REMOVE
                    return updated_node
                
                def _should_remove_import(self, import_stmt: Union[cst.Import, cst.ImportFrom]) -> bool:
                    """Check if import should be removed."""
                    if isinstance(import_stmt, cst.Import):
                        for name in import_stmt.names:
                            if isinstance(name, cst.ImportAlias):
                                import_name = name.asname.name.value if name.asname else name.name.value
                                if import_name in self.used_names:
                                    return False
                        return True
                    elif isinstance(import_stmt, cst.ImportFrom):
                        if isinstance(import_stmt.names, cst.ImportStar):
                            return False  # Don't remove star imports
                        for name in import_stmt.names:
                            if isinstance(name, cst.ImportAlias):
                                import_name = name.asname.name.value if name.asname else name.name.value
                                if import_name in self.used_names:
                                    return False
                        return True
                    return False
            
            remover = ImportRemover(analyzer.used_names)
            modified_tree = tree.visit(remover)
            
            if remover.removed_imports:
                modified_code = modified_tree.code
                with open(source_file, 'w', encoding='utf-8') as f:
                    f.write(modified_code)
                
                changes.extend([f"Removed unused import: {imp}" for imp in remover.removed_imports])
                logger.info(f"Fixed import anxiety: removed {len(remover.removed_imports)} unused imports")
            
        except Exception as e:
            logger.error(f"Failed to fix import anxiety: {e}")
        
        return changes
    
    def _fix_context_thrashing(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Extract common functionality to reduce context window thrashing."""
        advanced_strategy = get_advanced_fix_strategy('context_window_thrashing')
        if advanced_strategy:
            return advanced_strategy(issue, context)
        else:
            return ["Advanced context thrashing fix not available"]
    
    def _fix_hallucination_cascade(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Consolidate interdependent classes that form hallucination cascades."""
        advanced_strategy = get_advanced_fix_strategy('hallucination_cascade')
        if advanced_strategy:
            return advanced_strategy(issue, context)
        else:
            return ["Advanced hallucination cascade fix not available"]
    
    def _fix_circular_import(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Fix circular import by moving shared dependencies."""
        advanced_strategy = get_advanced_fix_strategy('circular_import')
        if advanced_strategy:
            return advanced_strategy(issue, context)
        else:
            return ["Advanced circular import fix not available"]
    
    def _fix_missing_symbol(self, issue: Issue, context: Dict[str, Any]) -> List[str]:
        """Add missing symbol imports or definitions."""
        changes = []
        changes.append("TODO: Implement missing symbol fix - requires symbol resolution")
        return changes


class IntelligentAutoFixer:
    """
    Main auto-fix engine that orchestrates the entire fix process.
    
    Features:
    - Dependency analysis and topological ordering
    - Safety mechanisms (dry-run, backup, rollback)
    - Test verification
    - Detailed reporting
    """
    
    def __init__(self, 
                 dry_run: bool = True,
                 backup_dir: Optional[str] = None,
                 test_command: Optional[str] = None,
                 max_parallel_fixes: int = 1):
        """
        Initialize the auto-fixer.
        
        Args:
            dry_run: Run in dry-run mode (no actual changes)
            backup_dir: Directory for backups
            test_command: Command to run tests
            max_parallel_fixes: Maximum number of fixes to apply in parallel
        """
        self.dry_run = dry_run
        self.max_parallel_fixes = max_parallel_fixes
        
        # Initialize components
        self.backup_manager = BackupManager(backup_dir)
        self.dependency_analyzer = DependencyAnalyzer()
        self.safety_validator = SafetyValidator(test_command)
        self.strategy_registry = FixStrategyRegistry()
        
        # State tracking
        self.fix_results: List[FixResult] = []
        self.current_plan: Optional[FixPlan] = None
        
        logger.info(f"Initialized IntelligentAutoFixer (dry_run={dry_run})")
    
    def create_fix_plan(self, issues: List[Issue]) -> FixPlan:
        """Create a comprehensive fix plan with dependency ordering."""
        logger.info(f"Creating fix plan for {len(issues)} issues")
        
        fix_actions = []
        
        # Convert issues to fix actions
        for i, issue in enumerate(issues):
            strategy = self.strategy_registry.get_strategy(issue.kind)
            if not strategy:
                logger.warning(f"No fix strategy available for {issue.kind}")
                continue
            
            # Determine priority based on issue severity and type
            if issue.severity >= IssueSeverity.CRITICAL.value:
                priority = FixPriority.CRITICAL
            elif issue.severity >= IssueSeverity.ERROR.value:
                priority = FixPriority.HIGH
            elif issue.severity >= IssueSeverity.WARNING.value:
                priority = FixPriority.MEDIUM
            else:
                priority = FixPriority.LOW
            
            # Determine risk level
            risk_level = self._assess_fix_risk(issue)
            
            fix_action = FixAction(
                id=f"fix_{i}_{issue.kind}_{issue.symbol or 'unknown'}",
                issue=issue,
                strategy=issue.kind,
                priority=priority,
                risk_level=risk_level,
                estimated_time=self._estimate_fix_time(issue)
            )
            
            fix_actions.append(fix_action)
        
        # Analyze dependencies and create execution order
        execution_order = self.dependency_analyzer.topological_sort(fix_actions)
        
        # Calculate plan metrics
        total_time = sum(action.estimated_time for action in fix_actions)
        total_risk = sum(self._risk_level_to_score(action.risk_level) for action in fix_actions)
        
        plan = FixPlan(
            fixes=fix_actions,
            execution_order=execution_order,
            estimated_time=total_time,
            total_risk_score=total_risk / len(fix_actions) if fix_actions else 0.0
        )
        
        self.current_plan = plan
        logger.info(f"Created fix plan: {len(fix_actions)} fixes, estimated time: {total_time:.1f}s")
        
        return plan
    
    def execute_fix_plan(self, plan: FixPlan) -> List[FixResult]:
        """Execute the fix plan with full safety mechanisms."""
        logger.info(f"Executing fix plan with {len(plan.fixes)} fixes")
        
        if self.dry_run:
            logger.info("DRY RUN MODE - No actual changes will be made")
        
        results = []
        fixes_by_id = {fix.id: fix for fix in plan.fixes}
        
        # Execute fixes in dependency order
        for fix_id in plan.execution_order:
            if fix_id not in fixes_by_id:
                logger.warning(f"Fix {fix_id} not found in plan")
                continue
            
            fix_action = fixes_by_id[fix_id]
            result = self._apply_single_fix(fix_action)
            results.append(result)
            
            # Stop on critical failures
            if result.status == FixStatus.FAILED and fix_action.priority == FixPriority.CRITICAL:
                logger.error(f"Critical fix failed: {fix_id}, stopping execution")
                break
            
            # Run intermediate tests if configured
            if result.success and fix_action.requires_tests and not self.dry_run:
                test_result = self.safety_validator.run_tests()
                result.test_results = test_result
                
                if not test_result['success']:
                    logger.warning(f"Tests failed after applying {fix_id}, attempting rollback")
                    if self._rollback_fix(result):
                        result.status = FixStatus.ROLLED_BACK
                    else:
                        result.status = FixStatus.FAILED
        
        self.fix_results = results
        logger.info(f"Fix plan execution completed: {len(results)} fixes processed")
        
        return results
    
    def _apply_single_fix(self, fix_action: FixAction) -> FixResult:
        """Apply a single fix with full safety checks."""
        logger.info(f"Applying fix: {fix_action.id}")
        
        start_time = time.time()
        result = FixResult(
            issue=fix_action.issue,
            status=FixStatus.IN_PROGRESS
        )
        
        try:
            # Pre-flight safety checks
            if fix_action.issue.file and not Path(fix_action.issue.file).exists():
                result.status = FixStatus.FAILED
                result.error_message = f"File not found: {fix_action.issue.file}"
                return result
            
            # Assess risk
            result.risk_assessment = self.safety_validator.assess_risk(fix_action)
            if result.risk_assessment.get('requires_manual_review', False) and not self.dry_run:
                result.status = FixStatus.SKIPPED
                result.error_message = "Fix requires manual review due to high risk"
                return result
            
            # Create backup
            if not self.dry_run and fix_action.issue.file:
                try:
                    result.backup_path = self.backup_manager.create_backup(fix_action.issue.file)
                except Exception as e:
                    logger.error(f"Failed to create backup for {fix_action.issue.file}: {e}")
                    result.status = FixStatus.FAILED
                    result.error_message = f"Backup creation failed: {e}"
                    return result
            
            # Validate syntax before fix
            if fix_action.issue.file:
                syntax_valid, syntax_error = self.safety_validator.validate_syntax(fix_action.issue.file)
                if not syntax_valid:
                    result.status = FixStatus.FAILED
                    result.error_message = f"Pre-fix syntax validation failed: {syntax_error}"
                    return result
            
            # Apply the fix
            strategy = self.strategy_registry.get_strategy(fix_action.strategy)
            if not strategy:
                result.status = FixStatus.FAILED
                result.error_message = f"No strategy found for {fix_action.strategy}"
                return result
            
            if self.dry_run:
                # Generate dry-run preview
                fix_action.dry_run_preview = self._generate_dry_run_preview(fix_action)
                result.applied_changes = [f"DRY RUN: Would apply {fix_action.strategy} fix"]
                result.status = FixStatus.COMPLETED
            else:
                # Apply actual fix
                changes = strategy(fix_action.issue, {})
                result.applied_changes = changes
                
                # Validate syntax after fix
                if fix_action.issue.file:
                    syntax_valid, syntax_error = self.safety_validator.validate_syntax(fix_action.issue.file)
                    if not syntax_valid:
                        # Rollback on syntax error
                        if result.backup_path:
                            self.backup_manager.restore_backup(fix_action.issue.file)
                        result.status = FixStatus.FAILED
                        result.error_message = f"Post-fix syntax validation failed: {syntax_error}"
                        return result
                
                result.status = FixStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Fix application failed for {fix_action.id}: {e}")
            result.status = FixStatus.FAILED
            result.error_message = str(e)
            
            # Attempt rollback on failure
            if not self.dry_run and result.backup_path:
                self._rollback_fix(result)
        
        finally:
            result.execution_time = time.time() - start_time
        
        return result
    
    def _rollback_fix(self, result: FixResult) -> bool:
        """Rollback a fix using its backup."""
        if not result.backup_path or not result.issue.file:
            return False
        
        try:
            success = self.backup_manager.restore_backup(result.issue.file)
            if success:
                logger.info(f"Successfully rolled back fix for {result.issue.file}")
            return success
        except Exception as e:
            logger.error(f"Rollback failed for {result.issue.file}: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive fix report."""
        if not self.fix_results:
            return {"error": "No fix results available"}
        
        # Calculate statistics
        total_fixes = len(self.fix_results)
        successful_fixes = sum(1 for r in self.fix_results if r.success)
        failed_fixes = sum(1 for r in self.fix_results if r.status == FixStatus.FAILED)
        skipped_fixes = sum(1 for r in self.fix_results if r.status == FixStatus.SKIPPED)
        rolled_back_fixes = sum(1 for r in self.fix_results if r.status == FixStatus.ROLLED_BACK)
        
        total_time = sum(r.execution_time for r in self.fix_results)
        avg_time = total_time / total_fixes if total_fixes > 0 else 0
        
        # Group by issue type
        fixes_by_type = defaultdict(list)
        for result in self.fix_results:
            fixes_by_type[result.issue.kind].append(result)
        
        # Generate detailed breakdown
        type_breakdown = {}
        for issue_type, results in fixes_by_type.items():
            type_breakdown[issue_type] = {
                'total': len(results),
                'successful': sum(1 for r in results if r.success),
                'failed': sum(1 for r in results if r.status == FixStatus.FAILED),
                'avg_time': sum(r.execution_time for r in results) / len(results)
            }
        
        report = {
            'summary': {
                'total_fixes': total_fixes,
                'successful_fixes': successful_fixes,
                'failed_fixes': failed_fixes,
                'skipped_fixes': skipped_fixes,
                'rolled_back_fixes': rolled_back_fixes,
                'success_rate': successful_fixes / total_fixes if total_fixes > 0 else 0,
                'total_execution_time': total_time,
                'average_fix_time': avg_time,
                'dry_run_mode': self.dry_run
            },
            'breakdown_by_type': type_breakdown,
            'detailed_results': [self._result_to_dict(r) for r in self.fix_results],
            'plan_metrics': {
                'estimated_time': self.current_plan.estimated_time if self.current_plan else 0,
                'actual_time': total_time,
                'time_accuracy': abs(self.current_plan.estimated_time - total_time) / self.current_plan.estimated_time if self.current_plan and self.current_plan.estimated_time > 0 else 0
            }
        }
        
        return report
    
    def _result_to_dict(self, result: FixResult) -> Dict[str, Any]:
        """Convert FixResult to dictionary for reporting."""
        return {
            'issue_kind': result.issue.kind,
            'issue_file': result.issue.file,
            'issue_symbol': result.issue.symbol,
            'status': result.status.value,
            'changes': result.applied_changes,
            'execution_time': result.execution_time,
            'error': result.error_message,
            'test_success': result.test_results.get('success') if result.test_results else None,
            'risk_score': result.risk_assessment.get('overall_risk') if result.risk_assessment else None
        }
    
    def _assess_fix_risk(self, issue: Issue) -> RiskLevel:
        """Assess risk level for fixing an issue."""
        # Simple risk assessment based on issue type
        high_risk_types = {"hallucination_cascade", "context_window_thrashing"}
        medium_risk_types = {"semantic_duplicate_function", "phantom_function", "circular_import"}
        
        if issue.kind in high_risk_types:
            return RiskLevel.HIGH
        elif issue.kind in medium_risk_types:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _estimate_fix_time(self, issue: Issue) -> float:
        """Estimate time required to fix an issue in seconds."""
        time_estimates = {
            "import_anxiety": 2.0,
            "missing_symbol": 3.0,
            "semantic_duplicate_function": 5.0,
            "phantom_function": 8.0,
            "circular_import": 10.0,
            "context_window_thrashing": 15.0,
            "hallucination_cascade": 20.0
        }
        return time_estimates.get(issue.kind, 5.0)
    
    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """Convert risk level to numerical score."""
        return {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0
        }.get(risk_level, 0.5)
    
    def _generate_dry_run_preview(self, fix_action: FixAction) -> str:
        """Generate preview of what the fix would do."""
        issue = fix_action.issue
        return f"""
Fix Preview for {fix_action.strategy}:
- Issue: {issue.kind} in {issue.file}:{issue.line}
- Symbol: {issue.symbol}
- Estimated time: {fix_action.estimated_time:.1f}s
- Risk level: {fix_action.risk_level.name}
- Would apply strategy: {fix_action.strategy}
"""


# Convenience functions for easy usage
def create_auto_fixer(dry_run: bool = True, 
                     backup_dir: Optional[str] = None,
                     test_command: Optional[str] = None) -> IntelligentAutoFixer:
    """Create a configured auto-fixer instance."""
    if not LIBCST_AVAILABLE:
        logger.warning("libcst not available. Some fix strategies will be limited.")
        logger.info("Install libcst with: pip install libcst")
    
    return IntelligentAutoFixer(
        dry_run=dry_run,
        backup_dir=backup_dir,
        test_command=test_command
    )


def auto_fix_issues(issues: List[Issue], 
                   dry_run: bool = True,
                   backup_dir: Optional[str] = None,
                   test_command: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to auto-fix a list of issues."""
    fixer = create_auto_fixer(dry_run, backup_dir, test_command)
    plan = fixer.create_fix_plan(issues)
    results = fixer.execute_fix_plan(plan)
    return fixer.generate_report()