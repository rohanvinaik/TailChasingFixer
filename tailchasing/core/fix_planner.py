"""
Fix planner that generates actionable, executable fix plans for issues.
"""

import ast
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import textwrap
import difflib
import hashlib

from ..core.issues import Issue, IssueCollection


@dataclass
class FixAction:
    """Represents a single fix action."""
    action_type: str  # delete_file, replace_content, add_import, etc.
    target_file: str
    description: str
    confidence: float  # 0.0 to 1.0
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    backup_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Other actions this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class FixPlan:
    """Complete fix plan for a set of issues."""
    actions: List[FixAction]
    issues_addressed: List[Issue]
    estimated_risk: str  # low, medium, high
    total_confidence: float
    backup_dir: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_executable_script(self) -> str:
        """Generate an executable Python script for the fix plan."""
        return FixScriptGenerator().generate(self)
        
    def get_summary(self) -> str:
        """Get a human-readable summary of the fix plan."""
        summary = []
        summary.append(f"Fix Plan Summary")
        summary.append("=" * 50)
        summary.append(f"Issues addressed: {len(self.issues_addressed)}")
        summary.append(f"Total actions: {len(self.actions)}")
        summary.append(f"Estimated risk: {self.estimated_risk}")
        summary.append(f"Average confidence: {self.total_confidence:.1%}")
        summary.append(f"Backup directory: {self.backup_dir}")
        summary.append("")
        
        # Group actions by type
        action_types = {}
        for action in self.actions:
            action_types.setdefault(action.action_type, []).append(action)
            
        summary.append("Actions by type:")
        for action_type, actions in sorted(action_types.items()):
            summary.append(f"  - {action_type}: {len(actions)}")
            
        return "\n".join(summary)


class FixPlanner:
    """Plans fixes for detected issues."""
    
    def __init__(
        self,
        root_dir: Path,
        backup_dir: Optional[Path] = None,
        interactive: bool = False,
        dry_run: bool = False
    ):
        """
        Initialize fix planner.
        
        Args:
            root_dir: Root directory of the project
            backup_dir: Directory for backups (default: .tailchasing_backups)
            interactive: Whether to run in interactive mode
            dry_run: Whether to generate plan without executing
        """
        self.root_dir = root_dir
        self.backup_dir = backup_dir or root_dir / ".tailchasing_backups"
        self.interactive = interactive
        self.dry_run = dry_run
        
        # Strategy handlers for each issue type
        self.strategies = {
            "duplicate_function": self._plan_duplicate_fix,
            "semantic_duplicate_function": self._plan_duplicate_fix,
            "phantom_function": self._plan_phantom_fix,
            "missing_symbol": self._plan_missing_symbol_fix,
            "circular_import": self._plan_circular_import_fix,
            "placeholder": self._plan_placeholder_fix,
            "context_window_thrashing": self._plan_context_thrashing_fix,
            "import_anxiety": self._plan_import_anxiety_fix,
        }
        
    def create_fix_plan(self, issues: List[Issue]) -> FixPlan:
        """
        Create a complete fix plan for the given issues.
        
        Args:
            issues: List of issues to fix
            
        Returns:
            Complete fix plan
        """
        actions = []
        issues_addressed = []
        
        # Group issues by type for batch processing
        issues_by_type = self._group_issues_by_type(issues)
        
        # Process each issue type
        for issue_type, type_issues in issues_by_type.items():
            if issue_type in self.strategies:
                strategy = self.strategies[issue_type]
                type_actions = strategy(type_issues)
                actions.extend(type_actions)
                issues_addressed.extend(type_issues)
                
        # Deduplicate and order actions
        actions = self._deduplicate_actions(actions)
        actions = self._order_actions(actions)
        
        # Calculate risk and confidence
        estimated_risk = self._estimate_risk(actions)
        total_confidence = self._calculate_total_confidence(actions)
        
        # Create backup directory path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = str(self.backup_dir / f"backup_{timestamp}")
        
        return FixPlan(
            actions=actions,
            issues_addressed=issues_addressed,
            estimated_risk=estimated_risk,
            total_confidence=total_confidence,
            backup_dir=backup_dir
        )
        
    def _group_issues_by_type(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group issues by their type."""
        grouped = {}
        for issue in issues:
            grouped.setdefault(issue.kind, []).append(issue)
        return grouped
        
    def _plan_duplicate_fix(self, issues: List[Issue]) -> List[FixAction]:
        """Plan fixes for duplicate functions."""
        actions = []
        
        # Group duplicates by function signature
        duplicate_groups = self._group_duplicates(issues)
        
        for signature, duplicates in duplicate_groups.items():
            if len(duplicates) < 2:
                continue
                
            # Choose canonical version (prefer shortest file path, then earliest)
            canonical = min(duplicates, key=lambda i: (len(i.file or ""), i.file or ""))
            
            # Create actions to remove duplicates and update imports
            for dup in duplicates:
                if dup == canonical:
                    continue
                    
                confidence = 0.8 if len(duplicates) == 2 else 0.6
                
                # Get function name from issue
                func_name = dup.symbol or "unknown_function"
                
                # Action to remove duplicate function
                actions.append(FixAction(
                    action_type="remove_function",
                    target_file=dup.file,
                    description=f"Remove duplicate function '{func_name}' from {dup.file}",
                    confidence=confidence,
                    metadata={
                        "function_name": func_name,
                        "line_number": dup.line,
                        "canonical_file": canonical.file,
                        "canonical_line": canonical.line
                    }
                ))
                
                # Action to update imports
                actions.append(FixAction(
                    action_type="update_imports",
                    target_file=dup.file,
                    description=f"Update imports to use '{func_name}' from {canonical.file}",
                    confidence=confidence * 0.9,
                    metadata={
                        "function_name": func_name,
                        "old_location": dup.file,
                        "new_location": canonical.file
                    },
                    dependencies=[f"remove_function_{dup.file}_{func_name}"]
                ))
                
        return actions
        
    def _plan_phantom_fix(self, issues: List[Issue]) -> List[FixAction]:
        """Plan fixes for phantom/stub functions."""
        actions = []
        
        for issue in issues:
            func_name = issue.symbol or "unknown_function"
            
            # Generate implementation based on function name and context
            implementation = self._generate_implementation(func_name, issue)
            
            actions.append(FixAction(
                action_type="replace_function",
                target_file=issue.file,
                description=f"Replace stub function '{func_name}' with implementation",
                confidence=0.6,
                old_content=self._get_function_content(issue.file, issue.line),
                new_content=implementation,
                metadata={
                    "function_name": func_name,
                    "line_number": issue.line
                }
            ))
            
        return actions
        
    def _plan_missing_symbol_fix(self, issues: List[Issue]) -> List[FixAction]:
        """Plan fixes for missing symbols."""
        actions = []
        
        # Group by symbol name
        symbols = {}
        for issue in issues:
            symbol = issue.symbol or issue.message.split("'")[1] if "'" in issue.message else "unknown"
            symbols.setdefault(symbol, []).append(issue)
            
        for symbol_name, symbol_issues in symbols.items():
            # Determine where to add the symbol
            target_file = self._determine_symbol_location(symbol_name, symbol_issues)
            
            # Generate skeleton implementation
            skeleton = self._generate_skeleton(symbol_name, symbol_issues)
            
            actions.append(FixAction(
                action_type="add_symbol",
                target_file=target_file,
                description=f"Add missing symbol '{symbol_name}' to {target_file}",
                confidence=0.7,
                new_content=skeleton,
                metadata={
                    "symbol_name": symbol_name,
                    "usage_count": len(symbol_issues),
                    "usage_files": list(set(i.file for i in symbol_issues if i.file))
                }
            ))
            
        return actions
        
    def _plan_circular_import_fix(self, issues: List[Issue]) -> List[FixAction]:
        """Plan fixes for circular imports."""
        actions = []
        
        for issue in issues:
            # Extract cycle information
            cycle_info = issue.evidence.get("cycle", []) if issue.evidence else []
            
            if not cycle_info:
                continue
                
            # Find the weakest link in the cycle
            weakest_link = self._find_weakest_import_link(cycle_info)
            
            if weakest_link:
                actions.append(FixAction(
                    action_type="lazy_import",
                    target_file=weakest_link[0],
                    description=f"Convert import of {weakest_link[1]} to lazy import",
                    confidence=0.7,
                    metadata={
                        "import_from": weakest_link[0],
                        "import_to": weakest_link[1],
                        "cycle": cycle_info
                    }
                ))
                
        return actions
        
    def _plan_placeholder_fix(self, issues: List[Issue]) -> List[FixAction]:
        """Plan fixes for placeholder functions."""
        actions = []
        
        for issue in issues:
            func_name = issue.symbol or "unknown_function"
            
            # Determine implementation based on function name and usage
            if "_validate_" in func_name or "_check_" in func_name:
                implementation = self._generate_validation_impl(func_name, issue)
            elif "_process_" in func_name or "_handle_" in func_name:
                implementation = self._generate_processor_impl(func_name, issue)
            else:
                implementation = self._generate_generic_impl(func_name, issue)
                
            actions.append(FixAction(
                action_type="implement_placeholder",
                target_file=issue.file,
                description=f"Implement placeholder function '{func_name}'",
                confidence=0.5,
                old_content=self._get_function_content(issue.file, issue.line),
                new_content=implementation,
                metadata={
                    "function_name": func_name,
                    "line_number": issue.line,
                    "pattern_type": issue.evidence.get("pattern") if issue.evidence else None
                }
            ))
            
        return actions
        
    def _plan_context_thrashing_fix(self, issues: List[Issue]) -> List[FixAction]:
        """Plan fixes for context window thrashing."""
        actions = []
        
        for issue in issues:
            # Extract reimplemented functions
            reimplemented = issue.evidence.get("reimplemented", []) if issue.evidence else []
            
            for func_info in reimplemented:
                actions.append(FixAction(
                    action_type="consolidate_implementation",
                    target_file=func_info.get("file"),
                    description=f"Consolidate reimplemented function '{func_info.get('name')}'",
                    confidence=0.6,
                    metadata=func_info
                ))
                
        return actions
        
    def _plan_import_anxiety_fix(self, issues: List[Issue]) -> List[FixAction]:
        """Plan fixes for import anxiety."""
        actions = []
        
        for issue in issues:
            unused_imports = issue.evidence.get("unused_imports", []) if issue.evidence else []
            
            for import_name in unused_imports:
                actions.append(FixAction(
                    action_type="remove_import",
                    target_file=issue.file,
                    description=f"Remove unused import '{import_name}'",
                    confidence=0.9,
                    metadata={
                        "import_name": import_name,
                        "line_number": issue.line
                    }
                ))
                
        return actions
        
    def _generate_implementation(self, func_name: str, issue: Issue) -> str:
        """Generate implementation for a function."""
        # Extract function signature if available
        signature = issue.evidence.get("signature", "") if issue.evidence else ""
        
        if not signature:
            signature = f"def {func_name}(*args, **kwargs)"
            
        impl = f"""{signature}:
    \"\"\"
    Implementation for {func_name}.
    Auto-generated to replace stub.
    \"\"\"
    # TODO: Implement actual logic
    raise NotImplementedError(f"{func_name} needs implementation")
"""
        return impl
        
    def _generate_skeleton(self, symbol_name: str, issues: List[Issue]) -> str:
        """Generate skeleton for missing symbol."""
        # Analyze usage to determine if it's a function, class, or variable
        usage_contexts = []
        for issue in issues:
            if issue.evidence:
                usage_contexts.append(issue.evidence.get("usage_context", ""))
                
        # Determine type based on usage
        if any("(" in ctx for ctx in usage_contexts):
            # It's called like a function
            return self._generate_function_skeleton(symbol_name, issues)
        elif any("." in ctx for ctx in usage_contexts):
            # It's used like a class
            return self._generate_class_skeleton(symbol_name, issues)
        else:
            # It's used like a variable
            return f"# Missing variable\n{symbol_name} = None  # TODO: Set appropriate value\n"
            
    def _generate_function_skeleton(self, func_name: str, issues: List[Issue]) -> str:
        """Generate function skeleton."""
        # Analyze call patterns to infer parameters
        params = self._infer_parameters(func_name, issues)
        
        skeleton = f"""def {func_name}({', '.join(params)}):
    \"\"\"
    Skeleton implementation for {func_name}.
    
    TODO: Add proper documentation
    \"\"\"
    # TODO: Implement function logic
    raise NotImplementedError(f"{func_name} is not yet implemented")
"""
        return skeleton
        
    def _generate_class_skeleton(self, class_name: str, issues: List[Issue]) -> str:
        """Generate class skeleton."""
        skeleton = f"""class {class_name}:
    \"\"\"
    Skeleton implementation for {class_name}.
    
    TODO: Add proper documentation
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize {class_name}.\"\"\"
        # TODO: Add initialization logic
        pass
        
    # TODO: Add methods based on usage
"""
        return skeleton
        
    def _generate_validation_impl(self, func_name: str, issue: Issue) -> str:
        """Generate validation function implementation."""
        signature = issue.evidence.get("signature", f"def {func_name}(value)") if issue.evidence else f"def {func_name}(value)"
        
        impl = f"""{signature}:
    \"\"\"
    Validation function for {func_name}.
    
    Args:
        value: Value to validate
        
    Returns:
        bool: True if valid, False otherwise
    \"\"\"
    if value is None:
        return False
        
    # TODO: Add actual validation logic
    # Example validations:
    # - Type checking
    # - Range checking
    # - Format validation
    
    return True  # Placeholder - implement actual validation
"""
        return impl
        
    def _generate_processor_impl(self, func_name: str, issue: Issue) -> str:
        """Generate processor function implementation."""
        signature = issue.evidence.get("signature", f"def {func_name}(data)") if issue.evidence else f"def {func_name}(data)"
        
        impl = f"""{signature}:
    \"\"\"
    Processor function for {func_name}.
    
    Args:
        data: Data to process
        
    Returns:
        Processed data
    \"\"\"
    if data is None:
        return None
        
    # TODO: Add actual processing logic
    # Example processing:
    # - Data transformation
    # - Filtering
    # - Aggregation
    
    result = data  # Placeholder - implement actual processing
    
    return result
"""
        return impl
        
    def _generate_generic_impl(self, func_name: str, issue: Issue) -> str:
        """Generate generic function implementation."""
        signature = issue.evidence.get("signature", f"def {func_name}()") if issue.evidence else f"def {func_name}()"
        
        impl = f"""{signature}:
    \"\"\"
    Implementation for {func_name}.
    
    TODO: Add proper documentation
    \"\"\"
    # TODO: Implement function logic
    
    # Placeholder implementation
    pass
"""
        return impl
        
    def _get_function_content(self, file_path: str, line_number: int) -> str:
        """Get the current content of a function."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Find function boundaries
            start_line = line_number - 1
            end_line = start_line
            
            # Find end of function (next function or class definition)
            indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())
            for i in range(start_line + 1, len(lines)):
                line = lines[i]
                if line.strip() and not line.startswith(' ' * (indent_level + 1)):
                    end_line = i
                    break
            else:
                end_line = len(lines)
                
            return ''.join(lines[start_line:end_line])
        except:
            return ""
            
    def _group_duplicates(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group duplicate issues by function signature."""
        groups = {}
        for issue in issues:
            # Use function name and file as key
            key = issue.evidence.get("signature", issue.symbol) if issue.evidence else issue.symbol
            if key:
                groups.setdefault(key, []).append(issue)
        return groups
        
    def _determine_symbol_location(self, symbol_name: str, issues: List[Issue]) -> str:
        """Determine best location to add missing symbol."""
        # Find most common usage location
        usage_files = {}
        for issue in issues:
            if issue.file:
                usage_files[issue.file] = usage_files.get(issue.file, 0) + 1
                
        if usage_files:
            # Add to most frequently using file
            return max(usage_files.items(), key=lambda x: x[1])[0]
        else:
            # Default to a utility module
            return str(self.root_dir / "utils.py")
            
    def _find_weakest_import_link(self, cycle: List[str]) -> Optional[Tuple[str, str]]:
        """Find the weakest link in an import cycle."""
        # Simple heuristic: prefer to break imports of utility/helper modules
        for i in range(len(cycle)):
            curr = cycle[i]
            next_mod = cycle[(i + 1) % len(cycle)]
            
            # Prefer to break imports of modules with certain patterns
            if any(pattern in next_mod.lower() for pattern in ["util", "helper", "common"]):
                return (curr, next_mod)
                
        # Default to first link
        if cycle:
            return (cycle[0], cycle[1] if len(cycle) > 1 else cycle[0])
        return None
        
    def _infer_parameters(self, func_name: str, issues: List[Issue]) -> List[str]:
        """Infer function parameters from usage."""
        param_counts = {}
        
        for issue in issues:
            if issue.evidence:
                call_info = issue.evidence.get("call_info", {})
                param_count = call_info.get("param_count", 0)
                param_counts[param_count] = param_counts.get(param_count, 0) + 1
                
        if param_counts:
            # Use most common parameter count
            max_count = max(param_counts.keys())
            if max_count > 0:
                return [f"arg{i+1}" for i in range(max_count)]
                
        return []
        
    def _deduplicate_actions(self, actions: List[FixAction]) -> List[FixAction]:
        """Remove duplicate actions."""
        seen = set()
        unique_actions = []
        
        for action in actions:
            # Create unique key for action
            key = (action.action_type, action.target_file, action.metadata.get("function_name", ""))
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)
                
        return unique_actions
        
    def _order_actions(self, actions: List[FixAction]) -> List[FixAction]:
        """Order actions based on dependencies and risk."""
        # Group by action type priority
        priority = {
            "remove_import": 1,
            "add_symbol": 2,
            "implement_placeholder": 3,
            "replace_function": 4,
            "remove_function": 5,
            "update_imports": 6,
            "lazy_import": 7,
            "consolidate_implementation": 8,
        }
        
        return sorted(actions, key=lambda a: priority.get(a.action_type, 99))
        
    def _estimate_risk(self, actions: List[FixAction]) -> str:
        """Estimate overall risk of fix plan."""
        if not actions:
            return "low"
            
        # High risk actions
        high_risk = ["remove_function", "consolidate_implementation", "remove_file"]
        medium_risk = ["replace_function", "update_imports", "lazy_import"]
        
        high_count = sum(1 for a in actions if a.action_type in high_risk)
        medium_count = sum(1 for a in actions if a.action_type in medium_risk)
        
        if high_count > 2 or high_count > len(actions) * 0.3:
            return "high"
        elif medium_count > 5 or medium_count > len(actions) * 0.5:
            return "medium"
        else:
            return "low"
            
    def _calculate_total_confidence(self, actions: List[FixAction]) -> float:
        """Calculate average confidence across all actions."""
        if not actions:
            return 0.0
            
        return sum(a.confidence for a in actions) / len(actions)


class FixScriptGenerator:
    """Generates executable Python scripts from fix plans."""
    
    def generate(self, plan: FixPlan) -> str:
        """Generate executable Python script for fix plan."""
        script = []
        
        # Header
        script.append("#!/usr/bin/env python3")
        script.append('"""')
        script.append("Auto-generated fix script for tail-chasing issues.")
        script.append(f"Generated: {plan.timestamp.isoformat()}")
        script.append(f"Total actions: {len(plan.actions)}")
        script.append(f"Risk level: {plan.estimated_risk}")
        script.append('"""')
        script.append("")
        
        # Imports
        script.append("import os")
        script.append("import shutil")
        script.append("import ast")
        script.append("import re")
        script.append("from pathlib import Path")
        script.append("from datetime import datetime")
        script.append("")
        
        # Configuration
        script.append("# Configuration")
        script.append(f'BACKUP_DIR = "{plan.backup_dir}"')
        script.append("DRY_RUN = False  # Set to True to preview changes without applying")
        script.append("VERBOSE = True   # Set to False to reduce output")
        script.append("")
        
        # Helper functions
        script.extend(self._generate_helpers())
        
        # Backup function
        script.extend(self._generate_backup_function())
        
        # Action implementations
        script.extend(self._generate_action_functions())
        
        # Main function
        script.extend(self._generate_main(plan))
        
        # Entry point
        script.append("")
        script.append('if __name__ == "__main__":')
        script.append("    main()")
        
        return "\n".join(script)
        
    def _generate_helpers(self) -> List[str]:
        """Generate helper functions."""
        helpers = []
        
        helpers.append("# Helper functions")
        helpers.append("")
        helpers.append("def log(message, level='INFO'):")
        helpers.append("    if VERBOSE or level in ['ERROR', 'WARNING']:")
        helpers.append("        print(f'[{level}] {message}')")
        helpers.append("")
        
        helpers.append("def read_file(filepath):")
        helpers.append("    \"\"\"Read file content.\"\"\"")
        helpers.append("    with open(filepath, 'r', encoding='utf-8') as f:")
        helpers.append("        return f.read()")
        helpers.append("")
        
        helpers.append("def write_file(filepath, content):")
        helpers.append("    \"\"\"Write content to file.\"\"\"")
        helpers.append("    if DRY_RUN:")
        helpers.append("        log(f'Would write to {filepath}', 'DRY_RUN')")
        helpers.append("        return")
        helpers.append("    with open(filepath, 'w', encoding='utf-8') as f:")
        helpers.append("        f.write(content)")
        helpers.append("    log(f'Updated {filepath}')")
        helpers.append("")
        
        return helpers
        
    def _generate_backup_function(self) -> List[str]:
        """Generate backup function."""
        backup = []
        
        backup.append("def create_backup(filepath):")
        backup.append("    \"\"\"Create backup of file.\"\"\"")
        backup.append("    if DRY_RUN:")
        backup.append("        return")
        backup.append("    ")
        backup.append("    backup_path = Path(BACKUP_DIR) / Path(filepath).name")
        backup.append("    backup_path.parent.mkdir(parents=True, exist_ok=True)")
        backup.append("    ")
        backup.append("    if Path(filepath).exists():")
        backup.append("        shutil.copy2(filepath, backup_path)")
        backup.append("        log(f'Backed up {filepath} to {backup_path}')")
        backup.append("        return str(backup_path)")
        backup.append("    return None")
        backup.append("")
        
        return backup
        
    def _generate_action_functions(self) -> List[str]:
        """Generate functions for each action type."""
        functions = []
        
        # Remove function action
        functions.append("def remove_function(filepath, func_name, line_number):")
        functions.append("    \"\"\"Remove a function from a file.\"\"\"")
        functions.append("    create_backup(filepath)")
        functions.append("    content = read_file(filepath)")
        functions.append("    ")
        functions.append("    # Parse AST and remove function")
        functions.append("    tree = ast.parse(content)")
        functions.append("    new_body = []")
        functions.append("    for node in tree.body:")
        functions.append("        if not (isinstance(node, ast.FunctionDef) and node.name == func_name):")
        functions.append("            new_body.append(node)")
        functions.append("    ")
        functions.append("    tree.body = new_body")
        functions.append("    new_content = ast.unparse(tree)")
        functions.append("    write_file(filepath, new_content)")
        functions.append("    log(f'Removed function {func_name} from {filepath}')")
        functions.append("")
        
        # Update imports action
        functions.append("def update_imports(filepath, old_module, new_module, symbol):")
        functions.append("    \"\"\"Update import statements.\"\"\"")
        functions.append("    create_backup(filepath)")
        functions.append("    content = read_file(filepath)")
        functions.append("    ")
        functions.append("    # Update import statements")
        functions.append("    old_import = f'from {old_module} import {symbol}'")
        functions.append("    new_import = f'from {new_module} import {symbol}'")
        functions.append("    ")
        functions.append("    if old_import in content:")
        functions.append("        content = content.replace(old_import, new_import)")
        functions.append("        write_file(filepath, content)")
        functions.append("        log(f'Updated imports in {filepath}')")
        functions.append("")
        
        # Add more action functions as needed
        functions.append("def add_symbol(filepath, symbol_content):")
        functions.append("    \"\"\"Add a new symbol to a file.\"\"\"")
        functions.append("    create_backup(filepath)")
        functions.append("    ")
        functions.append("    if Path(filepath).exists():")
        functions.append("        content = read_file(filepath)")
        functions.append("        content += '\\n\\n' + symbol_content")
        functions.append("    else:")
        functions.append("        content = symbol_content")
        functions.append("    ")
        functions.append("    write_file(filepath, content)")
        functions.append("    log(f'Added symbol to {filepath}')")
        functions.append("")
        
        return functions
        
    def _generate_main(self, plan: FixPlan) -> List[str]:
        """Generate main function with all actions."""
        main = []
        
        main.append("def main():")
        main.append("    \"\"\"Execute all fix actions.\"\"\"")
        main.append("    print('='*60)")
        main.append("    print('Tail-Chasing Fix Script')")
        main.append("    print('='*60)")
        main.append(f"    print('Total actions: {len(plan.actions)}')")
        main.append(f"    print('Risk level: {plan.estimated_risk}')")
        main.append(f"    print('Confidence: {plan.total_confidence:.1%}')")
        main.append("    print()")
        main.append("    ")
        main.append("    if DRY_RUN:")
        main.append("        print('DRY RUN MODE - No changes will be made')")
        main.append("        print()")
        main.append("    ")
        main.append("    # Create backup directory")
        main.append("    if not DRY_RUN:")
        main.append("        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)")
        main.append("    ")
        main.append("    success_count = 0")
        main.append("    error_count = 0")
        main.append("    ")
        
        # Generate action calls
        for i, action in enumerate(plan.actions, 1):
            main.append(f"    # Action {i}: {action.description}")
            main.append(f"    try:")
            
            if action.action_type == "remove_function":
                main.append(f"        remove_function(")
                main.append(f"            '{action.target_file}',")
                main.append(f"            '{action.metadata.get('function_name', 'unknown')}',")
                main.append(f"            {action.metadata.get('line_number', 0)}")
                main.append(f"        )")
                
            elif action.action_type == "update_imports":
                main.append(f"        update_imports(")
                main.append(f"            '{action.target_file}',")
                main.append(f"            '{action.metadata.get('old_location', '')}',")
                main.append(f"            '{action.metadata.get('new_location', '')}',")
                main.append(f"            '{action.metadata.get('function_name', '')}'")
                main.append(f"        )")
                
            elif action.action_type == "add_symbol":
                # Escape the new content properly
                content = action.new_content or ""
                content = content.replace("\\", "\\\\").replace('"', '\\"').replace('\n', '\\n')
                main.append(f"        add_symbol(")
                main.append(f"            '{action.target_file}',")
                main.append(f'            "{content}"')
                main.append(f"        )")
                
            else:
                main.append(f"        # TODO: Implement {action.action_type}")
                main.append(f"        log('Action {action.action_type} not yet implemented', 'WARNING')")
                
            main.append("        success_count += 1")
            main.append("    except Exception as e:")
            main.append(f"        log(f'Error in action {i}: {{e}}', 'ERROR')")
            main.append("        error_count += 1")
            main.append("    ")
            
        # Summary
        main.append("    # Print summary")
        main.append("    print()")
        main.append("    print('='*60)")
        main.append("    print('Fix Script Complete')")
        main.append("    print(f'Successful actions: {success_count}')")
        main.append("    print(f'Failed actions: {error_count}')")
        main.append("    ")
        main.append("    if not DRY_RUN:")
        main.append(f"        print(f'Backups saved to: {{BACKUP_DIR}}')")
        main.append("    ")
        main.append("    return error_count == 0")
        
        return main


class InteractiveFixReviewer:
    """Interactive review of fix actions."""
    
    def __init__(self, plan: FixPlan):
        """Initialize reviewer with fix plan."""
        self.plan = plan
        self.approved_actions = []
        self.rejected_actions = []
        
    def review(self) -> Tuple[List[FixAction], List[FixAction]]:
        """
        Interactively review each action.
        
        Returns:
            Tuple of (approved_actions, rejected_actions)
        """
        print("=" * 60)
        print("Interactive Fix Review")
        print("=" * 60)
        print(f"Total actions to review: {len(self.plan.actions)}")
        print()
        
        for i, action in enumerate(self.plan.actions, 1):
            print(f"\nAction {i}/{len(self.plan.actions)}")
            print("-" * 40)
            self._display_action(action)
            
            while True:
                response = input("\nApprove this action? [y/n/d/q]: ").lower()
                
                if response == 'y':
                    self.approved_actions.append(action)
                    print("✓ Action approved")
                    break
                elif response == 'n':
                    self.rejected_actions.append(action)
                    print("✗ Action rejected")
                    break
                elif response == 'd':
                    self._show_diff(action)
                elif response == 'q':
                    print("\nReview cancelled")
                    return self.approved_actions, self.rejected_actions
                else:
                    print("Invalid response. Use 'y' for yes, 'n' for no, 'd' for diff, 'q' to quit")
                    
        self._show_summary()
        return self.approved_actions, self.rejected_actions
        
    def _display_action(self, action: FixAction):
        """Display action details."""
        print(f"Type: {action.action_type}")
        print(f"File: {action.target_file}")
        print(f"Description: {action.description}")
        print(f"Confidence: {action.confidence:.1%}")
        
        if action.metadata:
            print("Details:")
            for key, value in action.metadata.items():
                if key not in ['old_content', 'new_content']:
                    print(f"  - {key}: {value}")
                    
    def _show_diff(self, action: FixAction):
        """Show diff for the action."""
        if action.old_content and action.new_content:
            print("\nDiff:")
            print("-" * 40)
            
            old_lines = action.old_content.splitlines()
            new_lines = action.new_content.splitlines()
            
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"{action.target_file} (before)",
                tofile=f"{action.target_file} (after)",
                lineterm=""
            )
            
            for line in diff:
                if line.startswith('+'):
                    print(f"\033[92m{line}\033[0m")  # Green
                elif line.startswith('-'):
                    print(f"\033[91m{line}\033[0m")  # Red
                else:
                    print(line)
        else:
            print("No diff available for this action")
            
    def _show_summary(self):
        """Show review summary."""
        print("\n" + "=" * 60)
        print("Review Summary")
        print("=" * 60)
        print(f"Approved actions: {len(self.approved_actions)}")
        print(f"Rejected actions: {len(self.rejected_actions)}")
        
        if self.approved_actions:
            print("\nApproved action types:")
            action_types = {}
            for action in self.approved_actions:
                action_types[action.action_type] = action_types.get(action.action_type, 0) + 1
            for action_type, count in sorted(action_types.items()):
                print(f"  - {action_type}: {count}")