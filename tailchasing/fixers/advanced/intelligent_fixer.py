"""
Intelligent Auto-Fix System

Automatically fixes detected tail-chasing patterns with smart strategies.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from ...core.issues import Issue
from .fix_strategies import (
    SemanticDuplicateFixer,
    PhantomImplementationFixer,
    CircularImportFixer,
    ImportAnxietyFixer
)


@dataclass
class Fix:
    """Represents a fix to be applied."""
    type: str
    file: str
    line: Optional[int]
    original_code: str
    fixed_code: str
    description: str
    confidence: float
    impact: Dict[str, Any]


@dataclass
class FixPlan:
    """Comprehensive fix plan for detected patterns."""
    fixes: List[Fix]
    execution_order: List[str]
    estimated_impact: Dict[str, Any]
    rollback_plan: List[Dict[str, Any]]
    confidence: float


class IntelligentAutoFixer:
    """Automatically fix detected tail-chasing patterns."""
    
    def __init__(self):
        self.fix_strategies = {
            'semantic_duplicate': SemanticDuplicateFixer(),
            'enhanced_semantic_duplicate': SemanticDuplicateFixer(),
            'phantom_function': PhantomImplementationFixer(),
            'circular_import': CircularImportFixer(),
            'import_anxiety': ImportAnxietyFixer(),
            'hallucination_cascade': PhantomImplementationFixer(),  # Use same strategy
            'context_window_thrashing': SemanticDuplicateFixer(),  # Use same strategy
        }
        
        # Fix priority order
        self.fix_order = [
            'circular_import',  # Fix these first as they block other fixes
            'import_anxiety',   # Clean up imports
            'semantic_duplicate',  # Merge duplicates
            'enhanced_semantic_duplicate',
            'context_window_thrashing',
            'phantom_function',  # Implement stubs last
            'hallucination_cascade'
        ]
    
    def generate_fix_plan(self, issues: List[Issue]) -> FixPlan:
        """Generate comprehensive fix plan for detected issues."""
        # Group issues by type
        grouped = defaultdict(list)
        for issue in issues:
            grouped[issue.kind].append(issue)
        
        # Generate fixes in optimal order
        all_fixes = []
        execution_order = []
        
        for issue_type in self.fix_order:
            if issue_type in grouped and issue_type in self.fix_strategies:
                fixer = self.fix_strategies[issue_type]
                fixes = fixer.generate_fixes(grouped[issue_type])
                
                if fixes:
                    all_fixes.extend(fixes)
                    execution_order.append(issue_type)
        
        # Estimate impact
        estimated_impact = self._estimate_fix_impact(all_fixes)
        
        # Generate rollback plan
        rollback_plan = self._generate_rollback_plan(all_fixes)
        
        # Calculate overall confidence
        confidence = self._calculate_plan_confidence(all_fixes, estimated_impact)
        
        return FixPlan(
            fixes=all_fixes,
            execution_order=execution_order,
            estimated_impact=estimated_impact,
            rollback_plan=rollback_plan,
            confidence=confidence
        )
    
    def _estimate_fix_impact(self, fixes: List[Fix]) -> Dict[str, Any]:
        """Estimate the impact of applying fixes."""
        impact = {
            'files_affected': set(),
            'functions_modified': 0,
            'imports_changed': 0,
            'lines_changed': 0,
            'risk_level': 'low',
            'estimated_time_minutes': 0,
            'potential_issues': []
        }
        
        for fix in fixes:
            impact['files_affected'].add(fix.file)
            
            if fix.type in ['merge_duplicates', 'implement_phantom']:
                impact['functions_modified'] += 1
            elif fix.type in ['clean_imports', 'break_circular_import']:
                impact['imports_changed'] += 1
            
            # Estimate lines changed
            original_lines = fix.original_code.count('\n') + 1
            fixed_lines = fix.fixed_code.count('\n') + 1
            impact['lines_changed'] += abs(fixed_lines - original_lines)
        
        # Calculate risk level
        if impact['functions_modified'] > 10:
            impact['risk_level'] = 'high'
            impact['potential_issues'].append("Many function modifications may affect behavior")
        elif impact['functions_modified'] > 5:
            impact['risk_level'] = 'medium'
        
        if impact['imports_changed'] > 20:
            if impact['risk_level'] == 'low':
                impact['risk_level'] = 'medium'
            impact['potential_issues'].append("Extensive import changes may cause resolution issues")
        
        # Estimate time (rough approximation)
        impact['estimated_time_minutes'] = (
            impact['functions_modified'] * 2 +
            impact['imports_changed'] * 0.5 +
            impact['lines_changed'] * 0.1
        )
        
        # Convert set to list for JSON serialization
        impact['files_affected'] = list(impact['files_affected'])
        
        return impact
    
    def _generate_rollback_plan(self, fixes: List[Fix]) -> List[Dict[str, Any]]:
        """Generate plan to rollback fixes if needed."""
        rollback_plan = []
        
        # Group by file for efficient rollback
        fixes_by_file = defaultdict(list)
        for fix in fixes:
            fixes_by_file[fix.file].append(fix)
        
        for file, file_fixes in fixes_by_file.items():
            rollback = {
                'file': file,
                'backup_needed': True,
                'fixes_count': len(file_fixes),
                'restore_command': f"git checkout HEAD -- {file}",
                'manual_restore': []
            }
            
            # Add manual restore instructions for each fix
            for fix in file_fixes:
                rollback['manual_restore'].append({
                    'line': fix.line,
                    'type': fix.type,
                    'description': f"Undo: {fix.description}"
                })
            
            rollback_plan.append(rollback)
        
        return rollback_plan
    
    def _calculate_plan_confidence(self, fixes: List[Fix], impact: Dict[str, Any]) -> float:
        """Calculate overall confidence in the fix plan."""
        if not fixes:
            return 0.0
        
        # Average fix confidence
        avg_confidence = sum(fix.confidence for fix in fixes) / len(fixes)
        
        # Adjust based on risk
        risk_multiplier = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.6
        }.get(impact['risk_level'], 0.5)
        
        # Adjust based on number of files affected
        files_penalty = min(1.0, 5.0 / max(len(impact['files_affected']), 1))
        
        return avg_confidence * risk_multiplier * files_penalty
    
    def apply_fixes(self, fix_plan: FixPlan, dry_run: bool = True) -> Dict[str, Any]:
        """Apply the fixes from a fix plan."""
        results = {
            'success': [],
            'failed': [],
            'skipped': [],
            'dry_run': dry_run
        }
        
        if dry_run:
            # In dry run mode, just validate fixes
            for fix in fix_plan.fixes:
                if self._validate_fix(fix):
                    results['success'].append({
                        'file': fix.file,
                        'type': fix.type,
                        'description': fix.description
                    })
                else:
                    results['failed'].append({
                        'file': fix.file,
                        'type': fix.type,
                        'reason': 'Validation failed'
                    })
        else:
            # Apply fixes (would need actual file manipulation)
            results['skipped'] = [{
                'reason': 'Actual file modification not implemented in this version'
            }]
        
        return results
    
    def _validate_fix(self, fix: Fix) -> bool:
        """Validate that a fix can be safely applied."""
        # Basic validation
        if not fix.fixed_code:
            return False
        
        # Check confidence threshold
        if fix.confidence < 0.5:
            return False
        
        # More validation would go here
        return True
