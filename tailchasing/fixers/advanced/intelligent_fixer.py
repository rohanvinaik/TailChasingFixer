"""
Intelligent Auto-Fixing Framework for TailChasingFixer.

This module provides the main IntelligentAutoFixer class that orchestrates
automated fix generation, impact estimation, and rollback planning for 
detected tail-chasing patterns.
"""

from __future__ import annotations
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import tempfile
import shutil
import os

from ...core.issues import Issue
from ...core.ast_analyzer import ASTAnalyzer
from ...analyzers.base import AnalysisContext
from .fix_strategies import (
    BaseFixStrategy, FixResult, FixPlan, FixAction,
    ImportResolutionStrategy, DuplicateMergeStrategy, 
    PlaceholderImplementationStrategy, CircularDependencyBreaker,
    StrategySelector
)

logger = logging.getLogger(__name__)


class FixImpactLevel(Enum):
    """Impact levels for fixes."""
    MINIMAL = 1      # Single line changes, imports
    LOW = 2          # Function-level changes
    MODERATE = 3     # Multi-function refactoring
    HIGH = 4         # Architectural changes
    CRITICAL = 5     # Cross-module restructuring


@dataclass
class FixImpact:
    """Represents the estimated impact of applying a fix."""
    
    level: FixImpactLevel
    affected_files: Set[str] = field(default_factory=set)
    affected_functions: Set[str] = field(default_factory=set)
    lines_changed: int = 0
    breaking_changes: List[str] = field(default_factory=list)
    dependencies_modified: Set[str] = field(default_factory=set)
    
    # Risk assessment
    risk_score: float = 0.0  # 0-1 scale
    confidence: float = 0.0  # 0-1 scale
    reversibility: float = 1.0  # 0-1 scale (1.0 = fully reversible)
    
    # Performance impact
    performance_impact: Optional[str] = None
    test_coverage_impact: Optional[float] = None
    
    def is_safe_to_apply(self, risk_threshold: float = 0.3) -> bool:
        """Check if fix is safe to apply based on risk assessment."""
        return (self.risk_score <= risk_threshold and 
                self.confidence >= 0.7 and 
                self.reversibility >= 0.8)


@dataclass
class RollbackPlan:
    """Plan for rolling back applied fixes."""
    
    plan_id: str
    fix_plan_id: str
    rollback_actions: List[FixAction] = field(default_factory=list)
    backup_files: Dict[str, str] = field(default_factory=dict)  # original_path -> backup_path
    
    # Rollback metadata
    creation_timestamp: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)
    
    def is_complete(self) -> bool:
        """Check if rollback plan is complete and executable."""
        return (len(self.rollback_actions) > 0 and 
                all(os.path.exists(backup) for backup in self.backup_files.values()))


class IntelligentAutoFixer:
    """
    Main intelligent auto-fixing orchestrator.
    
    This class coordinates fix strategy selection, impact estimation,
    and rollback planning to provide safe automated fixes for detected
    tail-chasing patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the intelligent fixer."""
        self.config = config or {}
        
        # Core components
        self.strategy_selector = StrategySelector(self.config.get('strategy_config', {}))
        self.ast_analyzer = ASTAnalyzer()
        
        # Fix planning configuration
        self.risk_threshold = self.config.get('risk_threshold', 0.3)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.backup_enabled = self.config.get('backup_enabled', True)
        
        # Caches and state
        self._strategy_cache: Dict[str, BaseFixStrategy] = {}
        self._impact_cache: Dict[str, FixImpact] = {}
        self._rollback_plans: Dict[str, RollbackPlan] = {}
        
        # Temporary directories for backups
        self.backup_dir = Path(tempfile.mkdtemp(prefix="tailchasing_backup_"))
        
        logger.info(f"IntelligentAutoFixer initialized with backup dir: {self.backup_dir}")
    
    def generate_fix_plan(
        self, 
        issues: List[Issue], 
        ctx: AnalysisContext,
        options: Optional[Dict[str, Any]] = None
    ) -> FixPlan:
        """
        Create comprehensive fix plans for detected issues.
        
        Analyzes the provided issues and generates a coordinated fix plan
        that considers dependencies, conflicts, and optimal application order.
        
        Args:
            issues: List of detected issues to fix
            ctx: Analysis context with AST index and configuration
            options: Additional options for fix generation
            
        Returns:
            FixPlan with ordered fix actions and metadata
        """
        options = options or {}
        
        try:
            logger.info(f"Generating fix plan for {len(issues)} issues")
            
            # Step 1: Filter and prioritize issues
            fixable_issues = self._filter_fixable_issues(issues, ctx)
            priority_issues = self._prioritize_issues(fixable_issues)
            
            logger.debug(f"Found {len(fixable_issues)} fixable issues out of {len(issues)}")
            
            # Step 2: Select strategies for each issue
            issue_strategies = {}
            for issue in priority_issues:
                strategy = self.strategy_selector.select_strategy(issue, ctx)
                if strategy:
                    issue_strategies[issue] = strategy
                    logger.debug(f"Selected {strategy.__class__.__name__} for issue {issue.kind}")
            
            # Step 3: Generate individual fix actions
            all_actions = []
            action_dependencies = {}
            
            for issue, strategy in issue_strategies.items():
                try:
                    # Get fix actions from strategy
                    fix_result = strategy.generate_fix(issue, ctx)
                    if fix_result.success and fix_result.actions:
                        for action in fix_result.actions:
                            # Add issue context to action
                            action.metadata['source_issue'] = issue.kind
                            action.metadata['strategy'] = strategy.__class__.__name__
                            
                        all_actions.extend(fix_result.actions)
                        
                        # Track dependencies between actions
                        if len(fix_result.actions) > 1:
                            for i in range(len(fix_result.actions) - 1):
                                action_dependencies[fix_result.actions[i+1]] = fix_result.actions[i]
                
                except Exception as e:
                    logger.error(f"Error generating fix for {issue.kind}: {e}", exc_info=True)
                    continue
            
            # Step 4: Resolve conflicts and optimize order
            ordered_actions = self._resolve_action_conflicts(all_actions, action_dependencies)
            optimized_actions = self._optimize_action_order(ordered_actions, ctx)
            
            # Step 5: Create comprehensive fix plan
            plan_id = f"fix_plan_{hashlib.md5(str(len(issues)).encode()).hexdigest()[:8]}"
            
            fix_plan = FixPlan(
                plan_id=plan_id,
                actions=optimized_actions,
                metadata={
                    'total_issues': len(issues),
                    'fixable_issues': len(fixable_issues),
                    'strategies_used': list(set(s.__class__.__name__ for s in issue_strategies.values())),
                    'generation_config': options,
                    'risk_threshold': self.risk_threshold,
                    'backup_enabled': self.backup_enabled
                }
            )
            
            # Add validation steps
            fix_plan.validation_steps = self._generate_validation_steps(optimized_actions, ctx)
            
            logger.info(f"Generated fix plan {plan_id} with {len(optimized_actions)} actions")
            return fix_plan
            
        except Exception as e:
            logger.error(f"Error generating fix plan: {e}", exc_info=True)
            
            # Return empty plan on error
            return FixPlan(
                plan_id="error_plan", 
                actions=[],
                metadata={'error': str(e)}
            )
    
    def estimate_fix_impact(
        self, 
        fix_plan: FixPlan, 
        ctx: AnalysisContext
    ) -> FixImpact:
        """
        Predict impact before applying fixes.
        
        Analyzes the fix plan to estimate the scope of changes, potential
        risks, and reversibility before any modifications are made.
        
        Args:
            fix_plan: Fix plan to analyze
            ctx: Analysis context
            
        Returns:
            FixImpact with detailed impact assessment
        """
        try:
            logger.info(f"Estimating impact for fix plan {fix_plan.plan_id}")
            
            # Check cache first
            cache_key = f"{fix_plan.plan_id}_{len(fix_plan.actions)}"
            if cache_key in self._impact_cache:
                return self._impact_cache[cache_key]
            
            impact = FixImpact(level=FixImpactLevel.MINIMAL)
            
            # Step 1: Analyze file and function scope
            affected_files = set()
            affected_functions = set()
            dependencies_modified = set()
            
            for action in fix_plan.actions:
                if action.file_path:
                    affected_files.add(action.file_path)
                
                if action.target_symbol:
                    affected_functions.add(action.target_symbol)
                
                # Check for dependency modifications
                if action.action_type in ['add_import', 'remove_import', 'modify_import']:
                    dependencies_modified.add(action.file_path)
            
            impact.affected_files = affected_files
            impact.affected_functions = affected_functions
            impact.dependencies_modified = dependencies_modified
            
            # Step 2: Estimate lines changed
            total_lines_changed = 0
            for action in fix_plan.actions:
                if action.action_type == 'replace_text' and action.old_content and action.new_content:
                    old_lines = len(action.old_content.splitlines())
                    new_lines = len(action.new_content.splitlines())
                    total_lines_changed += max(old_lines, new_lines)
                elif action.action_type in ['add_function', 'remove_function']:
                    total_lines_changed += 10  # Estimate for function changes
                else:
                    total_lines_changed += 1   # Single line change
            
            impact.lines_changed = total_lines_changed
            
            # Step 3: Determine impact level based on scope
            impact.level = self._calculate_impact_level(
                len(affected_files),
                len(affected_functions), 
                total_lines_changed,
                len(dependencies_modified)
            )
            
            # Step 4: Assess breaking changes
            breaking_changes = []
            for action in fix_plan.actions:
                if self._is_breaking_change(action, ctx):
                    breaking_changes.append(f"{action.action_type} in {action.file_path}")
            
            impact.breaking_changes = breaking_changes
            
            # Step 5: Calculate risk score
            impact.risk_score = self._calculate_risk_score(
                impact.level,
                len(breaking_changes),
                len(dependencies_modified),
                total_lines_changed
            )
            
            # Step 6: Estimate confidence
            impact.confidence = self._calculate_confidence(fix_plan, ctx)
            
            # Step 7: Assess reversibility
            impact.reversibility = self._assess_reversibility(fix_plan)
            
            # Step 8: Check performance impact
            impact.performance_impact = self._assess_performance_impact(fix_plan, ctx)
            
            # Step 9: Estimate test coverage impact
            impact.test_coverage_impact = self._estimate_test_coverage_impact(
                affected_files, affected_functions, ctx
            )
            
            # Cache the result
            self._impact_cache[cache_key] = impact
            
            logger.info(f"Impact assessment complete: {impact.level.name} level, "
                       f"risk={impact.risk_score:.2f}, confidence={impact.confidence:.2f}")
            
            return impact
            
        except Exception as e:
            logger.error(f"Error estimating fix impact: {e}", exc_info=True)
            
            # Return high-risk impact on error
            return FixImpact(
                level=FixImpactLevel.CRITICAL,
                risk_score=1.0,
                confidence=0.0,
                reversibility=0.0
            )
    
    def generate_rollback_plan(
        self, 
        fix_plan: FixPlan, 
        ctx: AnalysisContext
    ) -> RollbackPlan:
        """
        Create safety rollback plans.
        
        Generates comprehensive rollback plans that can undo applied fixes,
        including file backups and reverse transformations.
        
        Args:
            fix_plan: Fix plan to create rollback for
            ctx: Analysis context
            
        Returns:
            RollbackPlan with rollback actions and file backups
        """
        try:
            logger.info(f"Generating rollback plan for fix plan {fix_plan.plan_id}")
            
            plan_id = f"rollback_{fix_plan.plan_id}"
            rollback_plan = RollbackPlan(
                plan_id=plan_id,
                fix_plan_id=fix_plan.plan_id,
                creation_timestamp=str(hash(str(fix_plan.actions)))
            )
            
            # Step 1: Create file backups if enabled
            if self.backup_enabled:
                affected_files = set()
                for action in fix_plan.actions:
                    if action.file_path:
                        affected_files.add(action.file_path)
                
                for file_path in affected_files:
                    if os.path.exists(file_path):
                        backup_path = self._create_file_backup(file_path, plan_id)
                        if backup_path:
                            rollback_plan.backup_files[file_path] = backup_path
                            logger.debug(f"Created backup: {file_path} -> {backup_path}")
            
            # Step 2: Generate reverse actions (in reverse order)
            rollback_actions = []
            for action in reversed(fix_plan.actions):
                reverse_action = self._create_reverse_action(action, rollback_plan.backup_files)
                if reverse_action:
                    rollback_actions.append(reverse_action)
            
            rollback_plan.rollback_actions = rollback_actions
            
            # Step 3: Add verification steps
            rollback_plan.verification_steps = [
                "Verify all files are restored from backup",
                "Run syntax check on modified files",
                "Verify imports are resolved",
                "Run basic functionality tests"
            ]
            
            # Step 4: Track dependencies
            for action in fix_plan.actions:
                if action.action_type in ['add_import', 'remove_import']:
                    rollback_plan.dependencies.append(action.file_path)
            
            # Store rollback plan for reference
            self._rollback_plans[plan_id] = rollback_plan
            
            logger.info(f"Generated rollback plan {plan_id} with {len(rollback_actions)} actions")
            return rollback_plan
            
        except Exception as e:
            logger.error(f"Error generating rollback plan: {e}", exc_info=True)
            
            # Return minimal rollback plan on error
            return RollbackPlan(
                plan_id=f"error_rollback_{fix_plan.plan_id}",
                fix_plan_id=fix_plan.plan_id
            )
    
    # Helper methods for fix plan generation
    
    def _filter_fixable_issues(self, issues: List[Issue], ctx: AnalysisContext) -> List[Issue]:
        """Filter issues that can be automatically fixed."""
        fixable = []
        
        fixable_kinds = {
            'duplicate_function', 'semantic_duplicate', 'missing_symbol',
            'circular_import', 'phantom_function', 'import_anxiety',
            'hallucination_cascade', 'context_window_thrashing'
        }
        
        for issue in issues:
            if issue.kind in fixable_kinds and issue.confidence >= 0.6:
                fixable.append(issue)
        
        return fixable
    
    def _prioritize_issues(self, issues: List[Issue]) -> List[Issue]:
        """Prioritize issues by severity and confidence."""
        return sorted(issues, key=lambda i: (i.severity, i.confidence), reverse=True)
    
    def _resolve_action_conflicts(
        self, 
        actions: List[FixAction], 
        dependencies: Dict[FixAction, FixAction]
    ) -> List[FixAction]:
        """Resolve conflicts between actions."""
        # Simple conflict resolution - remove duplicate actions on same file/line
        seen_actions = set()
        resolved_actions = []
        
        for action in actions:
            action_key = (action.file_path, action.line_number, action.action_type)
            if action_key not in seen_actions:
                resolved_actions.append(action)
                seen_actions.add(action_key)
        
        return resolved_actions
    
    def _optimize_action_order(self, actions: List[FixAction], ctx: AnalysisContext) -> List[FixAction]:
        """Optimize the order of actions for safe application."""
        # Group actions by file and type for optimal ordering
        import_actions = []
        function_actions = []
        text_actions = []
        
        for action in actions:
            if action.action_type in ['add_import', 'remove_import', 'modify_import']:
                import_actions.append(action)
            elif action.action_type in ['add_function', 'remove_function', 'move_function']:
                function_actions.append(action)
            else:
                text_actions.append(action)
        
        # Optimal order: imports first, then functions, then text replacements
        return import_actions + function_actions + text_actions
    
    def _generate_validation_steps(self, actions: List[FixAction], ctx: AnalysisContext) -> List[str]:
        """Generate validation steps for the fix plan."""
        steps = ["Syntax validation of modified files"]
        
        has_imports = any(a.action_type.endswith('import') for a in actions)
        has_functions = any(a.action_type.endswith('function') for a in actions)
        
        if has_imports:
            steps.append("Import resolution verification")
        
        if has_functions:
            steps.append("Function signature compatibility check")
        
        steps.append("Run affected unit tests if available")
        return steps
    
    # Helper methods for impact estimation
    
    def _calculate_impact_level(
        self, 
        file_count: int, 
        function_count: int, 
        lines_changed: int, 
        dependency_count: int
    ) -> FixImpactLevel:
        """Calculate impact level based on change metrics."""
        if file_count <= 1 and lines_changed <= 5:
            return FixImpactLevel.MINIMAL
        elif file_count <= 2 and function_count <= 3:
            return FixImpactLevel.LOW
        elif file_count <= 5 and function_count <= 10:
            return FixImpactLevel.MODERATE
        elif file_count <= 10 or dependency_count > 5:
            return FixImpactLevel.HIGH
        else:
            return FixImpactLevel.CRITICAL
    
    def _is_breaking_change(self, action: FixAction, ctx: AnalysisContext) -> bool:
        """Check if an action represents a breaking change."""
        breaking_types = {'remove_function', 'modify_signature', 'remove_import'}
        return action.action_type in breaking_types
    
    def _calculate_risk_score(
        self, 
        impact_level: FixImpactLevel, 
        breaking_changes: int, 
        dependency_changes: int, 
        lines_changed: int
    ) -> float:
        """Calculate risk score for the fix."""
        base_risk = impact_level.value / 5.0
        breaking_risk = min(0.3, breaking_changes * 0.1)
        dependency_risk = min(0.2, dependency_changes * 0.05)
        complexity_risk = min(0.2, lines_changed * 0.001)
        
        return min(1.0, base_risk + breaking_risk + dependency_risk + complexity_risk)
    
    def _calculate_confidence(self, fix_plan: FixPlan, ctx: AnalysisContext) -> float:
        """Calculate confidence in the fix plan."""
        if not fix_plan.actions:
            return 0.0
        
        # Average confidence of strategies used
        strategy_confidence = 0.8  # Base confidence for known strategies
        
        # Reduce confidence for complex changes
        complexity_penalty = min(0.3, len(fix_plan.actions) * 0.02)
        
        return max(0.0, strategy_confidence - complexity_penalty)
    
    def _assess_reversibility(self, fix_plan: FixPlan) -> float:
        """Assess how easily fixes can be reversed."""
        if not fix_plan.actions:
            return 1.0
        
        reversible_actions = 0
        for action in fix_plan.actions:
            # Text replacements with old content are highly reversible
            if action.action_type == 'replace_text' and action.old_content:
                reversible_actions += 1
            # Import changes are generally reversible
            elif action.action_type.endswith('import'):
                reversible_actions += 1
            # Function additions are reversible
            elif action.action_type == 'add_function':
                reversible_actions += 1
        
        return reversible_actions / len(fix_plan.actions)
    
    def _assess_performance_impact(self, fix_plan: FixPlan, ctx: AnalysisContext) -> Optional[str]:
        """Assess potential performance impact."""
        function_changes = sum(1 for a in fix_plan.actions 
                             if a.action_type in ['add_function', 'remove_function'])
        
        if function_changes > 10:
            return "Potential performance impact from function restructuring"
        elif any(a.action_type == 'modify_import' for a in fix_plan.actions):
            return "Minor impact from import changes"
        
        return None
    
    def _estimate_test_coverage_impact(
        self, 
        affected_files: Set[str], 
        affected_functions: Set[str], 
        ctx: AnalysisContext
    ) -> Optional[float]:
        """Estimate impact on test coverage."""
        # Simple heuristic - if we're modifying many functions, coverage might drop
        if len(affected_functions) > 5:
            return 0.8  # Assume 80% of original coverage maintained
        elif len(affected_functions) > 2:
            return 0.9  # Assume 90% of original coverage maintained
        
        return None  # No significant impact expected
    
    # Helper methods for rollback plan generation
    
    def _create_file_backup(self, file_path: str, plan_id: str) -> Optional[str]:
        """Create backup of a file."""
        try:
            source = Path(file_path)
            if not source.exists():
                return None
            
            backup_name = f"{plan_id}_{source.name}.backup"
            backup_path = self.backup_dir / backup_name
            
            # Ensure backup directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to backup location
            shutil.copy2(source, backup_path)
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Error creating backup for {file_path}: {e}")
            return None
    
    def _create_reverse_action(
        self, 
        action: FixAction, 
        backup_files: Dict[str, str]
    ) -> Optional[FixAction]:
        """Create reverse action for rollback."""
        try:
            # For text replacements, swap old and new content
            if action.action_type == 'replace_text' and action.old_content:
                return FixAction(
                    action_type='replace_text',
                    file_path=action.file_path,
                    line_number=action.line_number,
                    old_content=action.new_content,
                    new_content=action.old_content,
                    metadata={'rollback_for': action.metadata.get('action_id')}
                )
            
            # For file restoration, use backup
            elif action.file_path in backup_files:
                return FixAction(
                    action_type='restore_file',
                    file_path=action.file_path,
                    new_content=backup_files[action.file_path],
                    metadata={'backup_source': backup_files[action.file_path]}
                )
            
            # For additions, create removal
            elif action.action_type.startswith('add_'):
                reverse_type = action.action_type.replace('add_', 'remove_')
                return FixAction(
                    action_type=reverse_type,
                    file_path=action.file_path,
                    target_symbol=action.target_symbol,
                    line_number=action.line_number
                )
            
            # For removals, this would require restoration from backup
            elif action.action_type.startswith('remove_'):
                # This requires the backup file content
                if action.file_path in backup_files:
                    return FixAction(
                        action_type='restore_from_backup',
                        file_path=action.file_path,
                        new_content=backup_files[action.file_path]
                    )
            
        except Exception as e:
            logger.error(f"Error creating reverse action for {action.action_type}: {e}")
        
        return None
    
    def cleanup_backups(self, plan_id: Optional[str] = None) -> None:
        """Clean up backup files."""
        try:
            if plan_id:
                # Clean up specific plan backups
                for backup_file in self.backup_dir.glob(f"{plan_id}_*.backup"):
                    backup_file.unlink()
                    logger.debug(f"Cleaned up backup: {backup_file}")
            else:
                # Clean up all backups
                if self.backup_dir.exists():
                    shutil.rmtree(self.backup_dir)
                    logger.info(f"Cleaned up backup directory: {self.backup_dir}")
        
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
    
    # Orchestration methods for applying, validating, and rolling back fixes
    
    def apply_fixes(
        self,
        fix_plan: FixPlan,
        ctx: AnalysisContext,
        dry_run: bool = False,
        validate_before: bool = True,
        validate_after: bool = True
    ) -> Dict[str, Any]:
        """
        Apply fixes from a fix plan with comprehensive validation and error handling.
        
        Args:
            fix_plan: Fix plan to apply
            ctx: Analysis context
            dry_run: If True, simulate application without making changes
            validate_before: Validate fix plan before application
            validate_after: Validate results after application
            
        Returns:
            Dict with application results, including success status and details
        """
        application_id = f"apply_{fix_plan.plan_id}_{hash(str(fix_plan.actions))}"
        
        result = {
            'application_id': application_id,
            'fix_plan_id': fix_plan.plan_id,
            'dry_run': dry_run,
            'success': False,
            'actions_applied': 0,
            'actions_failed': 0,
            'validation_results': {},
            'errors': [],
            'rollback_plan_id': None,
            'files_modified': set(),
            'backup_created': False
        }
        
        try:
            logger.info(f"{'[DRY RUN] ' if dry_run else ''}Applying fix plan {fix_plan.plan_id}")
            
            # Step 1: Pre-application validation
            if validate_before:
                logger.debug("Running pre-application validation")
                validation_result = self.validate_fixes(fix_plan, ctx, phase='before')
                result['validation_results']['before'] = validation_result
                
                if not validation_result.get('valid', False):
                    result['errors'].append("Pre-application validation failed")
                    logger.error(f"Pre-application validation failed: {validation_result.get('errors', [])}")
                    return result
            
            # Step 2: Create rollback plan
            rollback_plan = None
            if not dry_run:
                logger.debug("Creating rollback plan")
                rollback_plan = self.generate_rollback_plan(fix_plan, ctx)
                result['rollback_plan_id'] = rollback_plan.plan_id
                result['backup_created'] = rollback_plan.is_complete()
            
            # Step 3: Apply actions sequentially
            applied_actions = []
            failed_actions = []
            
            for i, action in enumerate(fix_plan.actions):
                try:
                    logger.debug(f"{'[DRY RUN] ' if dry_run else ''}Applying action {i+1}/{len(fix_plan.actions)}: {action.action_type}")
                    
                    if dry_run:
                        # Simulate application
                        simulation_result = self._simulate_action(action, ctx)
                        if simulation_result.get('would_succeed', False):
                            applied_actions.append(action)
                            result['files_modified'].add(action.file_path)
                        else:
                            failed_actions.append(action)
                            result['errors'].append(f"Action {action.action_type} would fail: {simulation_result.get('reason', 'Unknown')}")
                    else:
                        # Actually apply the action
                        apply_result = self._apply_single_action(action, ctx)
                        if apply_result.get('success', False):
                            applied_actions.append(action)
                            result['files_modified'].add(action.file_path)
                        else:
                            failed_actions.append(action)
                            result['errors'].append(f"Failed to apply {action.action_type}: {apply_result.get('error', 'Unknown error')}")
                            
                            # Stop on first failure for safety
                            logger.warning(f"Stopping application due to failed action: {action.action_type}")
                            break
                    
                except Exception as e:
                    logger.error(f"Error applying action {action.action_type}: {e}", exc_info=True)
                    failed_actions.append(action)
                    result['errors'].append(f"Exception applying {action.action_type}: {str(e)}")
                    break
            
            result['actions_applied'] = len(applied_actions)
            result['actions_failed'] = len(failed_actions)
            
            # Step 4: Post-application validation
            if validate_after and applied_actions and not dry_run:
                logger.debug("Running post-application validation")
                validation_result = self.validate_fixes(fix_plan, ctx, phase='after')
                result['validation_results']['after'] = validation_result
                
                if not validation_result.get('valid', False):
                    result['errors'].append("Post-application validation failed")
                    logger.warning(f"Post-application validation failed: {validation_result.get('errors', [])}")
                    
                    # Consider automatic rollback if validation fails
                    if rollback_plan and self.config.get('auto_rollback_on_validation_failure', True):
                        logger.info("Automatically rolling back due to validation failure")
                        rollback_result = self.rollback_fixes(rollback_plan, ctx)
                        result['auto_rollback'] = rollback_result
            
            # Step 5: Determine overall success
            result['success'] = (len(applied_actions) > 0 and 
                               len(failed_actions) == 0 and 
                               result['validation_results'].get('after', {}).get('valid', True))
            
            logger.info(f"Fix application {'simulation' if dry_run else 'complete'}: "
                       f"{result['actions_applied']} applied, {result['actions_failed']} failed, "
                       f"success: {result['success']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error during fix application: {e}", exc_info=True)
            result['errors'].append(f"Critical error: {str(e)}")
            return result
    
    def validate_fixes(
        self,
        fix_plan: FixPlan,
        ctx: AnalysisContext,
        phase: str = 'before'
    ) -> Dict[str, Any]:
        """
        Validate fixes for correctness and safety.
        
        Args:
            fix_plan: Fix plan to validate
            ctx: Analysis context
            phase: Validation phase ('before', 'after', 'standalone')
            
        Returns:
            Dict with validation results and any detected issues
        """
        validation_id = f"validate_{fix_plan.plan_id}_{phase}_{hash(str(fix_plan.actions))}"
        
        result = {
            'validation_id': validation_id,
            'fix_plan_id': fix_plan.plan_id,
            'phase': phase,
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks_performed': [],
            'files_validated': set(),
            'syntax_valid': True,
            'semantics_preserved': True,
            'imports_resolved': True,
            'tests_passing': None  # Will be populated if tests are run
        }
        
        try:
            logger.info(f"Validating fix plan {fix_plan.plan_id} (phase: {phase})")
            
            # Import the validator (will be created next)
            from .fix_validator import FixValidator
            validator = FixValidator(ctx)
            
            # Get affected files
            affected_files = set()
            for action in fix_plan.actions:
                if action.file_path:
                    affected_files.add(action.file_path)
            
            result['files_validated'] = affected_files
            
            # Step 1: Syntax validation
            logger.debug("Performing syntax validation")
            result['checks_performed'].append('syntax')
            syntax_results = validator.validate_syntax(list(affected_files), fix_plan.actions if phase == 'before' else None)
            result['syntax_valid'] = syntax_results.get('valid', False)
            
            if not result['syntax_valid']:
                result['errors'].extend(syntax_results.get('errors', []))
            
            # Step 2: Semantic preservation check (for 'after' phase)
            if phase == 'after':
                logger.debug("Performing semantic preservation check")
                result['checks_performed'].append('semantics')
                semantic_results = validator.validate_semantic_preservation(list(affected_files), fix_plan)
                result['semantics_preserved'] = semantic_results.get('preserved', False)
                
                if not result['semantics_preserved']:
                    result['warnings'].extend(semantic_results.get('warnings', []))
            
            # Step 3: Import resolution validation
            logger.debug("Performing import resolution validation")
            result['checks_performed'].append('imports')
            import_results = validator.validate_import_resolution(list(affected_files))
            result['imports_resolved'] = import_results.get('resolved', False)
            
            if not result['imports_resolved']:
                result['errors'].extend(import_results.get('errors', []))
            
            # Step 4: Test validation (if tests are available and phase is 'after')
            if phase == 'after' and self.config.get('run_tests_after_fix', True):
                logger.debug("Running test suite validation")
                result['checks_performed'].append('tests')
                test_results = validator.validate_test_suite(list(affected_files))
                result['tests_passing'] = test_results.get('passing', None)
                
                if result['tests_passing'] is False:
                    result['warnings'].extend(test_results.get('failures', []))
            
            # Step 5: Determine overall validity
            result['valid'] = (result['syntax_valid'] and 
                              result['imports_resolved'] and
                              (result['tests_passing'] is not False))  # None or True is acceptable
            
            if result['valid']:
                logger.info(f"Validation passed for fix plan {fix_plan.plan_id} (phase: {phase})")
            else:
                logger.warning(f"Validation failed for fix plan {fix_plan.plan_id}: "
                             f"syntax={result['syntax_valid']}, imports={result['imports_resolved']}, "
                             f"tests={result['tests_passing']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            result['valid'] = False
            result['errors'].append(f"Validation error: {str(e)}")
            return result
    
    def rollback_fixes(
        self,
        rollback_plan: RollbackPlan,
        ctx: AnalysisContext,
        validate_after: bool = True
    ) -> Dict[str, Any]:
        """
        Roll back applied fixes using a rollback plan.
        
        Args:
            rollback_plan: Plan for rolling back fixes
            ctx: Analysis context
            validate_after: Validate results after rollback
            
        Returns:
            Dict with rollback results and status
        """
        rollback_id = f"rollback_{rollback_plan.plan_id}_{hash(str(rollback_plan.rollback_actions))}"
        
        result = {
            'rollback_id': rollback_id,
            'rollback_plan_id': rollback_plan.plan_id,
            'success': False,
            'actions_applied': 0,
            'actions_failed': 0,
            'files_restored': set(),
            'validation_results': {},
            'errors': [],
            'backups_used': []
        }
        
        try:
            logger.info(f"Rolling back fixes using plan {rollback_plan.plan_id}")
            
            # Step 1: Verify rollback plan is complete
            if not rollback_plan.is_complete():
                result['errors'].append("Rollback plan is incomplete - missing backups or actions")
                logger.error("Cannot execute rollback: plan is incomplete")
                return result
            
            # Step 2: Apply rollback actions
            applied_actions = []
            failed_actions = []
            
            for i, action in enumerate(rollback_plan.rollback_actions):
                try:
                    logger.debug(f"Applying rollback action {i+1}/{len(rollback_plan.rollback_actions)}: {action.action_type}")
                    
                    rollback_result = self._apply_rollback_action(action, rollback_plan.backup_files)
                    
                    if rollback_result.get('success', False):
                        applied_actions.append(action)
                        if action.file_path:
                            result['files_restored'].add(action.file_path)
                    else:
                        failed_actions.append(action)
                        result['errors'].append(f"Failed to apply rollback action {action.action_type}: {rollback_result.get('error', 'Unknown')}")
                    
                except Exception as e:
                    logger.error(f"Error applying rollback action {action.action_type}: {e}", exc_info=True)
                    failed_actions.append(action)
                    result['errors'].append(f"Exception during rollback {action.action_type}: {str(e)}")
            
            # Step 3: Restore files from backups
            for original_path, backup_path in rollback_plan.backup_files.items():
                try:
                    if os.path.exists(backup_path):
                        logger.debug(f"Restoring {original_path} from backup {backup_path}")
                        shutil.copy2(backup_path, original_path)
                        result['files_restored'].add(original_path)
                        result['backups_used'].append(backup_path)
                    else:
                        result['errors'].append(f"Backup file not found: {backup_path}")
                
                except Exception as e:
                    logger.error(f"Error restoring {original_path} from backup: {e}", exc_info=True)
                    result['errors'].append(f"Failed to restore {original_path}: {str(e)}")
            
            result['actions_applied'] = len(applied_actions)
            result['actions_failed'] = len(failed_actions)
            
            # Step 4: Post-rollback validation
            if validate_after and len(result['files_restored']) > 0:
                logger.debug("Running post-rollback validation")
                
                # Create a minimal fix plan for validation purposes
                validation_plan = FixPlan(
                    plan_id=f"rollback_validation_{rollback_plan.plan_id}",
                    actions=[],
                    metadata={'rollback_validation': True}
                )
                
                validation_result = self.validate_fixes(validation_plan, ctx, phase='after')
                result['validation_results']['after'] = validation_result
                
                if not validation_result.get('valid', False):
                    result['errors'].append("Post-rollback validation failed")
                    logger.warning(f"Post-rollback validation issues: {validation_result.get('errors', [])}")
            
            # Step 5: Determine overall success
            result['success'] = (len(applied_actions) > 0 and 
                               len(failed_actions) == 0 and
                               len(result['files_restored']) > 0 and
                               result['validation_results'].get('after', {}).get('valid', True))
            
            logger.info(f"Rollback complete: {result['actions_applied']} actions applied, "
                       f"{len(result['files_restored'])} files restored, success: {result['success']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error during rollback: {e}", exc_info=True)
            result['errors'].append(f"Critical rollback error: {str(e)}")
            return result
    
    # Helper methods for orchestration
    
    def _simulate_action(self, action: FixAction, ctx: AnalysisContext) -> Dict[str, Any]:
        """Simulate applying an action without making changes."""
        try:
            result = {'would_succeed': True, 'reason': None}
            
            # Check if file exists
            if action.file_path and not os.path.exists(action.file_path):
                result['would_succeed'] = False
                result['reason'] = f"Target file does not exist: {action.file_path}"
                return result
            
            # Check if target content exists for replacement actions
            if action.action_type == 'replace_text' and action.old_content:
                with open(action.file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                if action.old_content not in file_content:
                    result['would_succeed'] = False
                    result['reason'] = "Target content not found in file"
                    return result
            
            # Additional simulation checks could be added here
            
            return result
            
        except Exception as e:
            return {'would_succeed': False, 'reason': f"Simulation error: {str(e)}"}
    
    def _apply_single_action(self, action: FixAction, ctx: AnalysisContext) -> Dict[str, Any]:
        """Apply a single fix action."""
        try:
            logger.debug(f"Applying {action.action_type} to {action.file_path}")
            
            if action.action_type == 'replace_text':
                return self._apply_text_replacement(action)
            elif action.action_type == 'add_import':
                return self._apply_import_addition(action)
            elif action.action_type == 'remove_import':
                return self._apply_import_removal(action)
            elif action.action_type == 'add_function':
                return self._apply_function_addition(action)
            elif action.action_type == 'remove_function':
                return self._apply_function_removal(action)
            else:
                return {'success': False, 'error': f"Unknown action type: {action.action_type}"}
        
        except Exception as e:
            logger.error(f"Error applying action {action.action_type}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _apply_rollback_action(self, action: FixAction, backup_files: Dict[str, str]) -> Dict[str, Any]:
        """Apply a rollback action."""
        try:
            if action.action_type == 'restore_file':
                backup_path = action.new_content
                if backup_path and os.path.exists(backup_path):
                    shutil.copy2(backup_path, action.file_path)
                    return {'success': True}
                else:
                    return {'success': False, 'error': f"Backup not found: {backup_path}"}
            
            # For other rollback actions, use the same logic as normal actions
            return self._apply_single_action(action, None)
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_text_replacement(self, action: FixAction) -> Dict[str, Any]:
        """Apply text replacement action."""
        try:
            with open(action.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if action.old_content and action.old_content not in content:
                return {'success': False, 'error': 'Target content not found'}
            
            if action.old_content:
                new_content = content.replace(action.old_content, action.new_content, 1)
            else:
                # Insert at specific line
                lines = content.splitlines()
                if action.line_number and 0 <= action.line_number <= len(lines):
                    lines.insert(action.line_number, action.new_content)
                    new_content = '\n'.join(lines)
                else:
                    new_content = content + '\n' + action.new_content
            
            with open(action.file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return {'success': True}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_import_addition(self, action: FixAction) -> Dict[str, Any]:
        """Apply import addition action."""
        try:
            with open(action.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find appropriate location for import (after existing imports)
            import_line = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(('import ', 'from ')):
                    import_line = i + 1
            
            lines.insert(import_line, action.new_content + '\n')
            
            with open(action.file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return {'success': True}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_import_removal(self, action: FixAction) -> Dict[str, Any]:
        """Apply import removal action."""
        try:
            with open(action.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Remove lines matching the old content
            new_lines = []
            for line in lines:
                if action.old_content and action.old_content.strip() not in line.strip():
                    new_lines.append(line)
            
            with open(action.file_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            
            return {'success': True}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_function_addition(self, action: FixAction) -> Dict[str, Any]:
        """Apply function addition action."""
        try:
            with open(action.file_path, 'a', encoding='utf-8') as f:
                f.write('\n\n' + action.new_content + '\n')
            
            return {'success': True}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_function_removal(self, action: FixAction) -> Dict[str, Any]:
        """Apply function removal action."""
        try:
            with open(action.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if action.old_content and action.old_content in content:
                new_content = content.replace(action.old_content, '', 1)
                
                with open(action.file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                return {'success': True}
            else:
                return {'success': False, 'error': 'Function content not found'}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
