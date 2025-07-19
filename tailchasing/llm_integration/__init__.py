"""
LLM Feedback Integration - Automatically generate prompts to prevent tail-chasing.

This module generates corrective prompts that can be fed back to LLMs to
prevent tail-chasing patterns before they occur.
"""

from typing import List, Dict, Optional
from ..core.issues import Issue


class TailChasingFeedbackGenerator:
    """
    Generates structured feedback for LLMs to prevent tail-chasing.
    """
    
    def generate_corrective_prompt(self, issues: List[Issue]) -> str:
        """
        Generate a prompt that can be prepended to the next LLM interaction
        to prevent recurring tail-chasing patterns.
        """
        if not issues:
            return ""
        
        prompt_parts = [
            "IMPORTANT: Previous analysis detected tail-chasing patterns. "
            "Please avoid the following issues:\n"
        ]
        
        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            issues_by_type.setdefault(issue.kind, []).append(issue)
        
        # Generate specific guidance for each issue type
        if 'semantic_duplicate_function' in issues_by_type:
            duplicates = issues_by_type['semantic_duplicate_function']
            prompt_parts.append(
                f"\n❌ SEMANTIC DUPLICATES DETECTED ({len(duplicates)} instances):\n"
            )
            for issue in duplicates[:3]:  # Limit to top 3
                pair = issue.evidence.get('pair', [])
                if pair:
                    prompt_parts.append(
                        f"  - {pair[0]['name']} and {pair[1]['name']} have identical behavior\n"
                        f"    DO NOT create new implementations. Reuse existing: {pair[0]['name']}\n"
                    )
        
        if 'phantom_function' in issues_by_type:
            phantoms = issues_by_type['phantom_function']
            prompt_parts.append(
                f"\n❌ PHANTOM IMPLEMENTATIONS ({len(phantoms)} stubs):\n"
                f"  - Do not create pass-only or NotImplementedError stubs\n"
                f"  - Either implement fully or don't create the function\n"
            )
        
        if 'missing_symbol' in issues_by_type:
            missing = issues_by_type['missing_symbol']
            prompt_parts.append(
                f"\n❌ HALLUCINATED SYMBOLS ({len(missing)} missing):\n"
            )
            for issue in missing[:3]:
                prompt_parts.append(
                    f"  - {issue.symbol} does not exist - do not import or call it\n"
                )
        
        if 'rename_cascade_chain' in issues_by_type:
            prompt_parts.append(
                "\n❌ RENAME CASCADE DETECTED:\n"
                "  - Do not rename functions to 'fix' errors\n"
                "  - Address the root cause instead of renaming\n"
            )
        
        prompt_parts.append(
            "\n✅ INSTEAD, PLEASE:\n"
            "1. Check if functionality already exists before implementing\n"
            "2. Fix root causes rather than symptoms\n"
            "3. Complete implementations rather than creating stubs\n"
            "4. Verify imports and function calls exist\n\n"
        )
        
        return "".join(prompt_parts)
    
    def generate_architecture_summary(self, symbol_table, issues: List[Issue]) -> str:
        """
        Generate a summary of existing architecture to provide context to LLM.
        """
        summary_parts = ["EXISTING ARCHITECTURE CONTEXT:\n\n"]
        
        # List key functions by module
        functions_by_file = {}
        for func_name, entries in symbol_table.functions.items():
            for entry in entries:
                file = entry['file']
                functions_by_file.setdefault(file, []).append(func_name)
        
        summary_parts.append("Available Functions by Module:\n")
        for file, funcs in sorted(functions_by_file.items())[:10]:  # Limit
            summary_parts.append(f"  {file}:\n")
            for func in sorted(funcs)[:5]:  # Limit functions per file
                summary_parts.append(f"    - {func}()\n")
        
        # Highlight semantic clusters if available
        semantic_issues = [i for i in issues if i.kind == 'semantic_duplicate_function']
        if semantic_issues:
            summary_parts.append("\nSemantic Function Groups (use one per group):\n")
            seen_funcs = set()
            for issue in semantic_issues:
                pair = issue.evidence.get('pair', [])
                if pair:
                    group_funcs = [p['name'] for p in pair if p['name'] not in seen_funcs]
                    if group_funcs:
                        summary_parts.append(f"  - {' ≈ '.join(group_funcs)}\n")
                        seen_funcs.update(group_funcs)
        
        return "".join(summary_parts)


class AutoFixGenerator:
    """
    Generates automated fixes for common tail-chasing patterns.
    """
    
    def generate_fix_script(self, issues: List[Issue]) -> Dict[str, str]:
        """
        Generate a Python script that fixes common issues.
        
        Returns dict of {filename: fixed_content}
        """
        fixes = {}
        
        # Group by file
        issues_by_file = {}
        for issue in issues:
            if issue.file:
                issues_by_file.setdefault(issue.file, []).append(issue)
        
        for file, file_issues in issues_by_file.items():
            # TODO: Implement AST-based fixes
            # - Remove phantom functions
            # - Replace duplicate calls with canonical function
            # - Remove circular imports
            pass
        
        return fixes
    
    def generate_refactoring_plan(self, issues: List[Issue]) -> str:
        """
        Generate a high-level refactoring plan for addressing systemic issues.
        """
        plan_parts = ["RECOMMENDED REFACTORING PLAN:\n\n"]
        
        # Count issue types
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.kind] = issue_counts.get(issue.kind, 0) + 1
        
        # Generate recommendations based on patterns
        if issue_counts.get('prototype_fragmentation', 0) > 3:
            plan_parts.append(
                "1. CONSOLIDATE FRAGMENTED IMPLEMENTATIONS\n"
                "   - Create a single source of truth for each concept\n"
                "   - Move all variants to a common module\n"
                "   - Use parameters/config instead of multiple functions\n\n"
            )
        
        if issue_counts.get('semantic_duplicate_function', 0) > 5:
            plan_parts.append(
                "2. ELIMINATE SEMANTIC DUPLICATES\n"
                "   - Identify the most complete implementation\n"
                "   - Replace all calls to use the canonical version\n"
                "   - Delete redundant implementations\n\n"
            )
        
        if issue_counts.get('circular_import', 0) > 0:
            plan_parts.append(
                "3. RESOLVE CIRCULAR DEPENDENCIES\n"
                "   - Extract shared interfaces to separate modules\n"
                "   - Use dependency injection patterns\n"
                "   - Consider reorganizing module structure\n\n"
            )
        
        return "".join(plan_parts)