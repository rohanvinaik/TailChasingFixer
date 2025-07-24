"""
Natural Language Explainer for tail-chasing patterns.
Generates human-readable explanations of detected issues.
"""

from typing import Dict, List, Optional
from ...core.issues import Issue


class TailChasingExplainer:
    """Generate human-readable explanations of tail-chasing patterns."""
    
    def __init__(self):
        self.explanation_templates = {
            'semantic_duplicate': self._explain_semantic_duplicate,
            'semantic_duplicate_multimodal': self._explain_semantic_duplicate_multimodal,
            'phantom_implementation': self._explain_phantom_implementation,
            'circular_import': self._explain_circular_import,
            'import_anxiety': self._explain_import_anxiety,
            'context_window_thrashing': self._explain_context_thrashing,
            'hallucination_cascade': self._explain_hallucination_cascade,
            'fix_induced_regression': self._explain_fix_induced_regression,
        }
    
    def explain_issue(self, issue: Issue) -> str:
        """Generate a comprehensive explanation for a tail-chasing issue."""
        explainer = self.explanation_templates.get(issue.kind)
        if explainer:
            return explainer(issue)
        else:
            return self._explain_generic(issue)
    
    def generate_summary_report(self, issues: List[Issue]) -> str:
        """Generate a comprehensive summary report of all issues."""
        if not issues:
            return "🎉 **No tail-chasing patterns detected!** Your code appears to be free of LLM-induced anti-patterns."
        
        # Group issues by type
        issue_groups = {}
        for issue in issues:
            issue_groups.setdefault(issue.kind, []).append(issue)
        
        report_sections = []
        
        # Executive summary
        total_risk = sum(issue.severity for issue in issues)
        report_sections.append(f"""
# 🔍 Tail-Chasing Analysis Report

**Overall Risk Score:** {total_risk}/100
**Issues Detected:** {len(issues)}
**Issue Types:** {len(issue_groups)}

## 📊 Executive Summary

Your codebase shows signs of LLM-assisted development with some tail-chasing patterns. 
These patterns suggest that an AI assistant may have been making superficial fixes 
without addressing root causes, leading to code duplication, phantom implementations, 
and circular dependencies.
""")
        
        # Detailed explanations by type
        for issue_type, type_issues in issue_groups.items():
            report_sections.append(f"## 🎯 {issue_type.replace('_', ' ').title()} ({len(type_issues)} issues)")
            
            # Add explanation for the first issue as example
            if type_issues:
                explanation = self.explain_issue(type_issues[0])
                report_sections.append(explanation)
                
                if len(type_issues) > 1:
                    report_sections.append(f"*{len(type_issues) - 1} additional similar issues detected.*")
        
        # Recommendations
        report_sections.append(self._generate_recommendations(issues))
        
        return "\n\n".join(report_sections)
    
    def _explain_semantic_duplicate(self, issue: Issue) -> str:
        """Explain semantic duplicate functions."""
        func1 = issue.evidence.get('function1', 'Function A')
        func2 = issue.evidence.get('function2', 'Function B')
        similarity = issue.evidence.get('similarity', 0)
        
        return f"""
### 🔄 What happened:
The AI detected two functions (`{func1}` and `{func2}`) that do essentially the same thing, 
but with different names or slight variations. This typically happens when an LLM:
1. Encounters an error in existing code
2. Creates a new function instead of fixing the original
3. Gradually drifts away from the original intent

### 🎯 Similarity Score: {similarity:.0%}

### ⚠️ Why this is problematic:
- **Code Bloat**: Unnecessary duplication increases maintenance burden
- **Inconsistency**: Different implementations may behave differently
- **Confusion**: Developers won't know which function to use
- **Bug Propagation**: Fixes might only be applied to one version

### 🔧 Root cause:
The LLM likely couldn't see both functions in its context window, or was asked to 
"create a function that..." without being aware an equivalent already existed.

### ✅ Recommended fix:
1. Compare both implementations carefully
2. Choose the better version (more complete, better tested, clearer name)
3. Update all references to use the chosen version
4. Remove or deprecate the duplicate
"""
    
    def _explain_semantic_duplicate_multimodal(self, issue: Issue) -> str:
        """Explain multimodal semantic duplicates."""
        func1 = issue.evidence.get('function1', 'Function A')
        func2 = issue.evidence.get('function2', 'Function B')
        similarity = issue.evidence.get('similarity', 0)
        channels = issue.evidence.get('channels', {})
        
        return f"""
### 🧠 Advanced Semantic Analysis Detected Duplication

Functions `{func1}` and `{func2}` show {similarity:.0%} semantic similarity across multiple dimensions:

**Channel Analysis:**
""" + "\n".join([f"- **{channel.replace('_', ' ').title()}**: {score:.0%}" 
                 for channel, score in channels.items()]) + f"""

### 🔍 What this means:
These functions may look different on the surface but follow the same logical patterns:
- Similar data flow
- Comparable error handling
- Equivalent control structures
- Related return patterns

### 🎯 This is often a sign of:
- Context window limitations causing re-implementation
- Gradual feature drift during iterative development
- Different naming conventions masking functional similarity

### ✅ Recommended approach:
1. Analyze the core business logic of both functions
2. Identify which implementation is more robust
3. Consider extracting common patterns into shared utilities
4. Consolidate functionality while preserving all necessary features
"""
    
    def _explain_phantom_implementation(self, issue: Issue) -> str:
        """Explain phantom implementations."""
        symbol = issue.symbol or 'function'
        
        return f"""
### 👻 Phantom Implementation Detected

### 🔍 What happened:
The function `{symbol}` is a placeholder (contains only `pass` or `NotImplementedError`) 
that was likely created to satisfy an import error or API expectation, but never properly implemented.

### 🎯 Common LLM pattern:
1. Code tries to import or call a function that doesn't exist
2. LLM creates a stub to make the error go away
3. LLM gets distracted by other issues and never returns to implement it
4. Stub becomes permanent placeholder

### ⚠️ Why this is problematic:
- **Silent Failures**: Code appears to work but does nothing
- **Runtime Errors**: May crash when actually called
- **Incomplete Features**: Functionality is promised but not delivered
- **Technical Debt**: Accumulates as more stubs are created

### 🔧 Root cause:
This often indicates the LLM was fixing import errors or trying to match an expected 
API without understanding what the function should actually do.

### ✅ Recommended fix:
1. Analyze where this function is called from
2. Determine what it should actually do based on calling context
3. Implement proper functionality or remove if not needed
4. Add appropriate tests to ensure it works correctly
"""
    
    def _explain_circular_import(self, issue: Issue) -> str:
        """Explain circular import issues."""
        cycle = issue.evidence.get('cycle', [])
        
        return f"""
### 🔄 Circular Import Dependency

### 🔍 What happened:
A circular import cycle was detected: {' → '.join(cycle)}

This means these modules are trying to import from each other, creating a dependency loop.

### 🎯 How this typically occurs with LLMs:
1. LLM adds an import to fix an undefined symbol error
2. Later, adds another import in the opposite direction
3. Each fix seems logical in isolation but creates a cycle
4. System works until Python's import system encounters the loop

### ⚠️ Why this is problematic:
- **Import Errors**: Python can't resolve the circular dependency
- **Unpredictable Behavior**: Import order becomes critical
- **Difficult Debugging**: Errors may be intermittent
- **Architectural Smell**: Suggests poor separation of concerns

### 🔧 Root cause:
LLMs often lack the global view needed to understand import dependencies and 
may add imports reactively without considering the broader architecture.

### ✅ Recommended fix:
1. Move shared code to a separate module
2. Use local imports (inside functions) where appropriate
3. Restructure code to respect dependency hierarchy
4. Consider using dependency injection or event patterns
"""
    
    def _explain_import_anxiety(self, issue: Issue) -> str:
        """Explain import anxiety patterns."""
        module = issue.evidence.get('module', 'unknown')
        imported_count = issue.evidence.get('imported_count', 0)
        used_count = issue.evidence.get('used_count', 0)
        pattern = issue.evidence.get('pattern', 'unknown')
        
        return f"""
### 😰 Import Anxiety Pattern

### 🔍 What happened:
The code imports {imported_count} items from `{module}` but only uses {used_count}. 
Pattern detected: **{pattern.replace('_', ' ').title()}**

### 🎯 This suggests:
The LLM encountered import errors and responded by importing "everything that might be needed" 
as a defensive measure, rather than understanding what was actually required.

### ⚠️ Why this is problematic:
- **Namespace Pollution**: Too many symbols in the namespace
- **Slower Imports**: Loading unnecessary code
- **Hidden Dependencies**: Makes it unclear what's actually needed
- **Refactoring Difficulty**: Hard to know what can be safely changed

### 🔧 Root cause:
Import anxiety typically occurs when an LLM:
- Encounters undefined symbol errors
- Lacks confidence about what imports are needed
- Applies a "better safe than sorry" approach
- Copies imports from examples without understanding them

### ✅ Recommended fix:
1. Remove unused imports (automated tools can help)
2. Import only what you need: `from module import specific_item`
3. Use qualified imports where appropriate: `import module; module.item`
4. Consider using IDE features to optimize imports
"""
    
    def _explain_context_thrashing(self, issue: Issue) -> str:
        """Explain context window thrashing."""
        func1 = issue.evidence.get('function1', 'Function A')
        func2 = issue.evidence.get('function2', 'Function B')
        distance = issue.evidence.get('distance', 0)
        similarity = issue.evidence.get('similarity', 0)
        
        return f"""
### 🧠 Context Window Thrashing

### 🔍 What happened:
Functions `{func1}` (line {issue.evidence.get('line1', '?')}) and `{func2}` (line {issue.evidence.get('line2', '?')}) 
are {similarity:.0%} similar but separated by {distance} lines.

### 🎯 This indicates:
The LLM likely "forgot" about the first function due to context window limitations and 
reimplemented similar functionality later in the file.

### ⚠️ Classic signs of context thrashing:
- Similar functions far apart in the same file
- Gradual naming drift (get_data → fetch_data → retrieve_data)
- Slight implementation variations for the same purpose
- Multiple attempts at solving the same problem

### 🔧 Root cause:
LLMs have limited context windows. When working on large files, they may lose track 
of functions defined earlier and recreate similar functionality.

### ✅ Recommended fix:
1. Compare the implementations to understand differences
2. Merge into a single, well-designed function
3. Extract common functionality if both variants are needed
4. Consider breaking large files into smaller, focused modules
5. Use better naming conventions to avoid future confusion
"""
    
    def _explain_hallucination_cascade(self, issue: Issue) -> str:
        """Explain hallucination cascade patterns."""
        components = issue.evidence.get('components', [])
        external_refs = issue.evidence.get('external_refs', 0)
        
        return f"""
### 🌀 Hallucination Cascade

### 🔍 What happened:
Detected a group of {len(components)} related classes/modules that were created together 
but have minimal connections to the rest of the codebase (only {external_refs} external references).

**Involved components:** {', '.join(components[:5])}{'...' if len(components) > 5 else ''}

### 🎯 This pattern suggests:
The LLM created an entire fictional subsystem to solve a problem, where each new component 
was created to support the previous one, leading to a cascade of hallucinated dependencies.

### ⚠️ How cascades develop:
1. LLM encounters an error
2. Creates Class A to solve it
3. Class A needs Class B, so creates Class B
4. Class B needs Class C, so creates Class C
5. Results in a self-contained but unnecessary subsystem

### 🔧 Root cause:
Instead of understanding and using existing functionality, the LLM invents new abstractions 
that seem logical but duplicate or overcomplicate existing solutions.

### ✅ Recommended fix:
1. **Map the cascade**: Understand what this subsystem was meant to accomplish
2. **Check existing solutions**: Can current code handle this use case?
3. **Evaluate necessity**: Is this functionality actually needed?
4. **Gradual removal**: If removing, do it incrementally with testing
5. **Extract value**: Keep any genuinely useful abstractions
"""
    
    def _explain_fix_induced_regression(self, issue: Issue) -> str:
        """Explain fix-induced regressions."""
        changed_functions = issue.evidence.get('changed_functions', [])
        failing_tests = issue.evidence.get('failing_tests', [])
        
        return f"""
### 🐛 Fix-Induced Regression

### 🔍 What happened:
Changes to {', '.join(changed_functions)} caused {len(failing_tests)} tests to fail:
{chr(10).join(f'- {test}' for test in failing_tests[:5])}{'...' if len(failing_tests) > 5 else ''}

### 🎯 This indicates:
A "fix" for one issue inadvertently broke other functionality, suggesting the LLM 
made changes without understanding the full impact.

### ⚠️ Common causes:
- **Narrow focus**: Fixing immediate error without considering side effects
- **Incomplete context**: Not understanding how the function is used elsewhere
- **Overly aggressive changes**: Modifying more than necessary
- **Missing integration tests**: Changes looked safe in isolation

### 🔧 Root cause:
LLMs often make changes based on local error messages without understanding 
the broader system architecture and usage patterns.

### ✅ Recommended fix:
1. **Revert the changes** if possible
2. **Analyze the failing tests** to understand what broke
3. **Identify the root cause** of the original issue
4. **Make minimal, targeted changes** that don't affect other functionality
5. **Run comprehensive tests** before considering the fix complete
"""
    
    def _explain_generic(self, issue: Issue) -> str:
        """Generic explanation for unknown issue types."""
        return f"""
### 🔍 {issue.kind.replace('_', ' ').title()}

**Issue:** {issue.message}
**Severity:** {issue.severity}/5

This appears to be a tail-chasing pattern where the AI made changes that may not address 
the root cause of the problem. Consider reviewing the change that introduced this pattern 
and whether a simpler, more direct solution exists.

**Evidence:** {issue.evidence}

**Suggestions:**
""" + "\n".join(f"- {suggestion}" for suggestion in issue.suggestions)
    
    def _generate_recommendations(self, issues: List[Issue]) -> str:
        """Generate overall recommendations based on all issues."""
        high_severity = [i for i in issues if i.severity >= 4]
        medium_severity = [i for i in issues if 2 <= i.severity < 4]
        
        recommendations = ["## 🎯 Recommendations"]
        
        if high_severity:
            recommendations.append(f"""
### 🚨 High Priority ({len(high_severity)} issues)
These issues require immediate attention as they may cause runtime failures or significant technical debt:
- Focus on fixing circular imports and hallucination cascades first
- Review phantom implementations for critical functionality
- Test thoroughly after making changes
""")
        
        if medium_severity:
            recommendations.append(f"""
### ⚠️ Medium Priority ({len(medium_severity)} issues)
These issues should be addressed during regular refactoring:
- Consolidate duplicate functions
- Clean up unnecessary imports
- Improve code organization to prevent future context thrashing
""")
        
        recommendations.append("""
### 💡 Prevention Strategies
To avoid future tail-chasing patterns:
1. **Provide better context** to LLMs with more complete code snippets
2. **Review AI suggestions** before implementing them
3. **Run tests frequently** to catch regressions early
4. **Use static analysis tools** to detect issues automatically
5. **Break large files** into smaller, more manageable modules
6. **Maintain clear naming conventions** to avoid confusion
""")
        
        return "\n".join(recommendations)
