"""Playbook generator for different types of code issues and clusters."""

from typing import Dict, List, Any
import re

from .fix_playbooks import (
    FixPlaybook, PlaybookStep, CodeChange, SafetyCheck, 
    ChangeRisk, PlaybookEngine
)
from .issues import Issue
from ..analyzers.root_cause_clustering import IssueCluster
from ..analyzers.phantom_triage import PhantomStub, PhantomPriority


class PlaybookGenerator:
    """Generates fix playbooks for different types of issues and clusters."""
    
    def __init__(self):
        self.engine = PlaybookEngine()
        self.playbook_templates = self._load_playbook_templates()
    
    def _load_playbook_templates(self) -> Dict[str, Any]:
        """Load playbook templates for different issue types."""
        return {
            "phantom_stub_security": {
                "risk_level": ChangeRisk.CRITICAL,
                "requires_review": True,
                "safety_checks": [
                    {"check_id": "syntax_valid", "description": "Ensure syntax is valid"},
                    {"check_id": "no_security_issues", "description": "Check for security issues"},
                    {"check_id": "tests_pass", "description": "Ensure tests pass"}
                ]
            },
            "phantom_stub_functional": {
                "risk_level": ChangeRisk.MEDIUM,
                "requires_review": True,
                "safety_checks": [
                    {"check_id": "syntax_valid", "description": "Ensure syntax is valid"},
                    {"check_id": "imports_resolvable", "description": "Check imports"},
                    {"check_id": "tests_pass", "description": "Ensure tests pass"}
                ]
            },
            "duplicate_function_removal": {
                "risk_level": ChangeRisk.HIGH,
                "requires_review": True,
                "safety_checks": [
                    {"check_id": "syntax_valid", "description": "Ensure syntax is valid"},
                    {"check_id": "imports_resolvable", "description": "Check imports"},
                    {"check_id": "tests_pass", "description": "Ensure tests pass"},
                    {"check_id": "lint_clean", "description": "Code quality checks"}
                ]
            },
            "circular_import_fix": {
                "risk_level": ChangeRisk.HIGH,
                "requires_review": True,
                "safety_checks": [
                    {"check_id": "syntax_valid", "description": "Ensure syntax is valid"},
                    {"check_id": "imports_resolvable", "description": "Check imports"},
                    {"check_id": "tests_pass", "description": "Ensure tests pass"}
                ]
            }
        }
    
    def generate_cluster_playbook(self, cluster: IssueCluster, issues: List[Issue]) -> FixPlaybook:
        """Generate a playbook for fixing a root cause cluster."""
        playbook_id = f"cluster_{cluster.cluster_id}"
        
        # Determine the primary issue type
        issue_kinds = [issue.kind for issue in issues]
        primary_kind = max(set(issue_kinds), key=issue_kinds.count)
        
        if primary_kind == "phantom_function":
            return self._generate_phantom_stub_playbook(cluster, issues)
        elif primary_kind in ["duplicate_function", "semantic_duplicate_function"]:
            return self._generate_duplicate_removal_playbook(cluster, issues)
        elif primary_kind == "circular_import":
            return self._generate_circular_import_playbook(cluster, issues)
        else:
            return self._generate_generic_cluster_playbook(cluster, issues)
    
    def generate_phantom_stub_playbook(self, stubs: List[PhantomStub]) -> FixPlaybook:
        """Generate a playbook specifically for phantom stub fixes."""
        # Group stubs by priority
        p0_stubs = [s for s in stubs if s.priority == PhantomPriority.P0_SECURITY]
        p1_stubs = [s for s in stubs if s.priority == PhantomPriority.P1_FUNCTIONAL]
        p3_stubs = [s for s in stubs if s.priority == PhantomPriority.P3_EXPERIMENTAL]
        
        playbook_id = f"phantom_stubs_{len(stubs)}"
        
        # Determine overall risk level
        if p0_stubs:
            risk_level = ChangeRisk.CRITICAL
            template = self.playbook_templates["phantom_stub_security"]
        else:
            risk_level = ChangeRisk.MEDIUM
            template = self.playbook_templates["phantom_stub_functional"]
        
        steps = []
        
        # Step 1: Handle P0 Security stubs (if any)
        if p0_stubs:
            security_step = self._create_security_stub_step(p0_stubs)
            steps.append(security_step)
        
        # Step 2: Handle P1 Functional stubs
        if p1_stubs:
            functional_step = self._create_functional_stub_step(p1_stubs)
            steps.append(functional_step)
        
        # Step 3: Handle P3 Experimental stubs (optional)
        if p3_stubs:
            experimental_step = self._create_experimental_stub_step(p3_stubs)
            experimental_step.is_optional = True
            steps.append(experimental_step)
        
        # Create safety checks
        safety_checks = []
        for check_def in template["safety_checks"]:
            safety_checks.append(SafetyCheck(
                check_id=check_def["check_id"],
                description=check_def["description"],
                check_function=check_def["check_id"],
                is_blocking=True
            ))
        
        return FixPlaybook(
            playbook_id=playbook_id,
            name=f"Phantom Stub Fixes ({len(stubs)} stubs)",
            description=f"Fix {len(p0_stubs)} P0, {len(p1_stubs)} P1, and {len(p3_stubs)} P3 phantom stubs",
            cluster_id="phantom_stubs",
            steps=steps,
            risk_level=risk_level,
            safety_checks=safety_checks,
            requires_review=template["requires_review"],
            estimated_time_minutes=len(stubs) * 2,
            metadata={
                "stub_counts": {"P0": len(p0_stubs), "P1": len(p1_stubs), "P3": len(p3_stubs)},
                "blocked_stubs": len([s for s in stubs if s.is_blocked]),
                "allowed_stubs": len([s for s in stubs if s.is_allowed])
            }
        )
    
    def _generate_phantom_stub_playbook(self, cluster: IssueCluster, issues: List[Issue]) -> FixPlaybook:
        """Generate playbook for phantom stub cluster."""
        playbook_id = f"cluster_{cluster.cluster_id}_phantom_stubs"
        
        steps = []
        
        # Create implementation step
        changes = []
        for issue in issues:
            if 'generated_implementation' in issue.evidence:
                change = CodeChange(
                    file_path=issue.file,
                    line_start=issue.line,
                    line_end=issue.line + 1,  # Assuming single-line stub
                    old_content="    pass  # TODO: implement",
                    new_content=issue.evidence['generated_implementation'],
                    change_type='replace',
                    description=f"Implement {issue.evidence.get('function_name', 'function')}",
                    risk_level=ChangeRisk.HIGH if 'P0' in issue.message else ChangeRisk.MEDIUM
                )
                changes.append(change)
        
        if changes:
            step = PlaybookStep(
                step_id="implement_phantoms",
                name="Implement Phantom Functions",
                description=f"Replace phantom stubs with proper implementations ({len(changes)} functions)",
                changes=changes,
                safety_checks=[
                    SafetyCheck("syntax_valid", "Check syntax validity", "syntax_valid"),
                    SafetyCheck("tests_pass", "Run tests", "tests_pass", is_blocking=False)
                ]
            )
            steps.append(step)
        
        template = self.playbook_templates["phantom_stub_functional"]
        safety_checks = [
            SafetyCheck(check["check_id"], check["description"], check["check_id"])
            for check in template["safety_checks"]
        ]
        
        return FixPlaybook(
            playbook_id=playbook_id,
            name=f"Phantom Stub Cluster Fix",
            description=cluster.root_cause_guess,
            cluster_id=cluster.cluster_id,
            steps=steps,
            risk_level=ChangeRisk.HIGH,
            safety_checks=safety_checks,
            requires_review=True,
            estimated_time_minutes=len(issues) * 3
        )
    
    def _generate_duplicate_removal_playbook(self, cluster: IssueCluster, issues: List[Issue]) -> FixPlaybook:
        """Generate playbook for duplicate function removal."""
        playbook_id = f"cluster_{cluster.cluster_id}_duplicate_removal"
        
        # Group duplicates by function name
        duplicates_by_name = {}
        for issue in issues:
            func_name = self._extract_function_name(issue)
            if func_name not in duplicates_by_name:
                duplicates_by_name[func_name] = []
            duplicates_by_name[func_name].append(issue)
        
        steps = []
        
        for func_name, func_issues in duplicates_by_name.items():
            if len(func_issues) < 2:
                continue
            
            # Choose canonical implementation (first one or highest line count)
            canonical_issue = max(func_issues, key=lambda i: i.evidence.get('line_count', 1))
            duplicates = [i for i in func_issues if i != canonical_issue]
            
            # Step 1: Update imports to point to canonical
            import_changes = self._create_import_redirect_changes(canonical_issue, duplicates)
            
            # Step 2: Remove duplicate implementations
            removal_changes = self._create_duplicate_removal_changes(duplicates)
            
            if import_changes:
                import_step = PlaybookStep(
                    step_id=f"redirect_imports_{func_name}",
                    name=f"Redirect Imports for {func_name}",
                    description=f"Update imports to use canonical implementation of {func_name}",
                    changes=import_changes,
                    safety_checks=[
                        SafetyCheck("imports_resolvable", "Check imports resolve", "imports_resolvable")
                    ]
                )
                steps.append(import_step)
            
            if removal_changes:
                removal_step = PlaybookStep(
                    step_id=f"remove_duplicates_{func_name}",
                    name=f"Remove Duplicate {func_name}",
                    description=f"Remove duplicate implementations of {func_name}",
                    changes=removal_changes,
                    safety_checks=[
                        SafetyCheck("tests_pass", "Ensure tests still pass", "tests_pass")
                    ],
                    dependencies=[f"redirect_imports_{func_name}"] if import_changes else []
                )
                steps.append(removal_step)
        
        template = self.playbook_templates["duplicate_function_removal"]
        safety_checks = [
            SafetyCheck(check["check_id"], check["description"], check["check_id"])
            for check in template["safety_checks"]
        ]
        
        return FixPlaybook(
            playbook_id=playbook_id,
            name=f"Duplicate Function Removal",
            description=f"Remove {len(issues)} duplicate functions across {len(duplicates_by_name)} names",
            cluster_id=cluster.cluster_id,
            steps=steps,
            risk_level=ChangeRisk.HIGH,
            safety_checks=safety_checks,
            requires_review=True,
            estimated_time_minutes=len(steps) * 5
        )
    
    def _generate_circular_import_playbook(self, cluster: IssueCluster, issues: List[Issue]) -> FixPlaybook:
        """Generate playbook for circular import fixes."""
        playbook_id = f"cluster_{cluster.cluster_id}_circular_imports"
        
        steps = []
        
        # Extract modules involved in circular imports
        modules = set()
        for issue in issues:
            if 'modules' in issue.evidence:
                modules.update(issue.evidence['modules'])
        
        # Step 1: Create shared module if beneficial
        if len(modules) > 2:
            shared_changes = self._create_shared_module_changes(list(modules), issues)
            if shared_changes:
                shared_step = PlaybookStep(
                    step_id="create_shared_module",
                    name="Create Shared Module",
                    description="Extract shared symbols to break circular imports",
                    changes=shared_changes,
                    safety_checks=[
                        SafetyCheck("syntax_valid", "Check syntax", "syntax_valid"),
                        SafetyCheck("imports_resolvable", "Check imports", "imports_resolvable")
                    ]
                )
                steps.append(shared_step)
        
        # Step 2: Convert to function-scope imports
        function_scope_changes = self._create_function_scope_import_changes(issues)
        if function_scope_changes:
            function_step = PlaybookStep(
                step_id="function_scope_imports",
                name="Convert to Function-Scope Imports",
                description="Move imports inside functions to break cycles",
                changes=function_scope_changes,
                safety_checks=[
                    SafetyCheck("imports_resolvable", "Check imports", "imports_resolvable"),
                    SafetyCheck("tests_pass", "Run tests", "tests_pass")
                ],
                dependencies=["create_shared_module"] if len(modules) > 2 else []
            )
            steps.append(function_step)
        
        template = self.playbook_templates["circular_import_fix"]
        safety_checks = [
            SafetyCheck(check["check_id"], check["description"], check["check_id"])
            for check in template["safety_checks"]
        ]
        
        return FixPlaybook(
            playbook_id=playbook_id,
            name="Circular Import Resolution",
            description=f"Resolve circular imports between {len(modules)} modules",
            cluster_id=cluster.cluster_id,
            steps=steps,
            risk_level=ChangeRisk.HIGH,
            safety_checks=safety_checks,
            requires_review=True,
            estimated_time_minutes=len(steps) * 10
        )
    
    def _generate_generic_cluster_playbook(self, cluster: IssueCluster, issues: List[Issue]) -> FixPlaybook:
        """Generate a generic playbook for miscellaneous clusters."""
        playbook_id = f"cluster_{cluster.cluster_id}_generic"
        
        # Create a simple fix step
        changes = []
        for issue in issues:
            if issue.suggestions:
                # Use the first suggestion as a change
                suggestion = issue.suggestions[0]
                change = CodeChange(
                    file_path=issue.file,
                    line_start=issue.line,
                    line_end=issue.line,
                    old_content="",  # Will be determined during execution
                    new_content=suggestion,
                    change_type='replace',
                    description=f"Apply fix for {issue.kind}",
                    risk_level=ChangeRisk.MEDIUM
                )
                changes.append(change)
        
        steps = []
        if changes:
            step = PlaybookStep(
                step_id="apply_fixes",
                name="Apply Suggested Fixes",
                description=f"Apply {len(changes)} suggested fixes from cluster analysis",
                changes=changes,
                safety_checks=[
                    SafetyCheck("syntax_valid", "Check syntax", "syntax_valid"),
                    SafetyCheck("tests_pass", "Run tests", "tests_pass", is_blocking=False)
                ]
            )
            steps.append(step)
        
        return FixPlaybook(
            playbook_id=playbook_id,
            name=f"Generic Cluster Fix ({cluster.cluster_id})",
            description=cluster.root_cause_guess,
            cluster_id=cluster.cluster_id,
            steps=steps,
            risk_level=ChangeRisk.MEDIUM,
            safety_checks=[
                SafetyCheck("syntax_valid", "Check syntax", "syntax_valid")
            ],
            requires_review=True,
            estimated_time_minutes=len(issues) * 2
        )
    
    def _create_security_stub_step(self, stubs: List[PhantomStub]) -> PlaybookStep:
        """Create step for handling P0 security stubs."""
        changes = []
        
        for stub in stubs:
            if stub.is_blocked:
                # Create explicit failure implementation
                new_content = f'''def {stub.function_name}{stub.signature}:
    """SECURITY CRITICAL: This function is blocked from use."""
    raise NotImplementedError(
        "SECURITY CRITICAL: This function must be properly implemented before use"
    )'''
            else:
                # Create secure scaffolding
                new_content = f'''def {stub.function_name}{stub.signature}:
    """Security-critical function requiring proper implementation."""
    import warnings
    warnings.warn(
        "Using stub implementation for security-critical function: {stub.function_name}",
        UserWarning,
        stacklevel=2
    )
    return False  # Safe default: fail closed'''
            
            change = CodeChange(
                file_path=stub.file_path,
                line_start=stub.line_number,
                line_end=stub.line_number + 2,  # Estimate stub size
                old_content="    pass",
                new_content=new_content,
                change_type='replace',
                description=f"Secure implementation for {stub.function_name}",
                risk_level=ChangeRisk.CRITICAL
            )
            changes.append(change)
        
        return PlaybookStep(
            step_id="security_stubs",
            name="Handle Security-Critical Stubs",
            description=f"Implement or block {len(stubs)} security-critical phantom stubs",
            changes=changes,
            safety_checks=[
                SafetyCheck("no_security_issues", "Security check", "no_security_issues"),
                SafetyCheck("syntax_valid", "Syntax check", "syntax_valid")
            ]
        )
    
    def _create_functional_stub_step(self, stubs: List[PhantomStub]) -> PlaybookStep:
        """Create step for handling P1 functional stubs."""
        changes = []
        
        for stub in stubs:
            # Generate minimal implementation based on function name patterns
            if 'get' in stub.function_name.lower() and 'config' in stub.function_name.lower():
                new_content = f'''def {stub.function_name}{stub.signature}:
    """Minimal configuration getter."""
    import os
    return os.environ.get('CONFIG_' + '{stub.function_name}'.upper(), {{}})'''
            
            elif 'id' in stub.function_name.lower():
                new_content = f'''def {stub.function_name}{stub.signature}:
    """Minimal ID generator."""
    import uuid
    return str(uuid.uuid4())[:8]'''
            
            else:
                new_content = f'''def {stub.function_name}{stub.signature}:
    """Functional stub - minimal implementation."""
    import warnings
    warnings.warn(f"Using stub for {{stub.function_name}}", UserWarning)
    return None'''
            
            change = CodeChange(
                file_path=stub.file_path,
                line_start=stub.line_number,
                line_end=stub.line_number + 2,
                old_content="    pass",
                new_content=new_content,
                change_type='replace',
                description=f"Minimal implementation for {stub.function_name}",
                risk_level=ChangeRisk.MEDIUM
            )
            changes.append(change)
        
        return PlaybookStep(
            step_id="functional_stubs",
            name="Implement Functional Stubs",
            description=f"Add minimal implementations for {len(stubs)} functional stubs",
            changes=changes,
            safety_checks=[
                SafetyCheck("syntax_valid", "Syntax check", "syntax_valid"),
                SafetyCheck("imports_resolvable", "Import check", "imports_resolvable")
            ]
        )
    
    def _create_experimental_stub_step(self, stubs: List[PhantomStub]) -> PlaybookStep:
        """Create step for handling P3 experimental stubs."""
        changes = []
        
        for stub in stubs:
            new_content = f'''def {stub.function_name}{stub.signature}:
    """Experimental feature - implementation pending."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Called experimental function: {stub.function_name}")
    return None  # Placeholder for experimental feature'''
            
            change = CodeChange(
                file_path=stub.file_path,
                line_start=stub.line_number,
                line_end=stub.line_number + 2,
                old_content="    pass",
                new_content=new_content,
                change_type='replace',
                description=f"Experimental placeholder for {stub.function_name}",
                risk_level=ChangeRisk.LOW
            )
            changes.append(change)
        
        return PlaybookStep(
            step_id="experimental_stubs",
            name="Handle Experimental Stubs",
            description=f"Add placeholders for {len(stubs)} experimental features",
            changes=changes,
            safety_checks=[
                SafetyCheck("syntax_valid", "Syntax check", "syntax_valid")
            ]
        )
    
    def _extract_function_name(self, issue: Issue) -> str:
        """Extract function name from issue."""
        if 'function_name' in issue.evidence:
            return issue.evidence['function_name']
        
        # Try to extract from message
        match = re.search(r'function[s]?\s+[\'"]([^"\']+)[\'"]', issue.message)
        if match:
            return match.group(1)
        
        return f"unknown_{issue.line}"
    
    def _create_import_redirect_changes(self, canonical: Issue, duplicates: List[Issue]) -> List[CodeChange]:
        """Create changes to redirect imports to canonical implementation."""
        # This would need to analyze actual import usage
        # Simplified for now
        return []
    
    def _create_duplicate_removal_changes(self, duplicates: List[Issue]) -> List[CodeChange]:
        """Create changes to remove duplicate implementations."""
        changes = []
        
        for issue in duplicates:
            # Remove the entire function (simplified)
            change = CodeChange(
                file_path=issue.file,
                line_start=issue.line,
                line_end=issue.line + issue.evidence.get('line_count', 5),
                old_content="# Function to be removed",
                new_content="",
                change_type='delete',
                description=f"Remove duplicate function",
                risk_level=ChangeRisk.HIGH
            )
            changes.append(change)
        
        return changes
    
    def _create_shared_module_changes(self, modules: List[str], issues: List[Issue]) -> List[CodeChange]:
        """Create changes to extract shared module."""
        # This would create a new shared module file
        # Simplified for now
        return []
    
    def _create_function_scope_import_changes(self, issues: List[Issue]) -> List[CodeChange]:
        """Create changes to move imports to function scope."""
        changes = []
        
        for issue in issues:
            if 'import_statement' in issue.evidence:
                import_stmt = issue.evidence['import_statement']
                
                # Move import inside function (simplified)
                change = CodeChange(
                    file_path=issue.file,
                    line_start=issue.line,
                    line_end=issue.line,
                    old_content=import_stmt,
                    new_content=f"    {import_stmt}  # Moved to function scope",
                    change_type='replace',
                    description="Move import to function scope",
                    risk_level=ChangeRisk.MEDIUM
                )
                changes.append(change)
        
        return changes
    
    def preview_playbook(self, playbook: FixPlaybook) -> str:
        """Generate a preview of the playbook."""
        return self.engine.preview_playbook(playbook)
    
    def execute_playbook(self, playbook: FixPlaybook, dry_run: bool = True):
        """Execute a playbook with optional dry run."""
        return self.engine.execute_playbook(playbook, dry_run=dry_run)