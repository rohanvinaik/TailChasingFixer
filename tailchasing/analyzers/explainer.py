"""
Generate human-readable explanations of tail-chasing patterns.
Helps developers understand why something is problematic and how to fix it.
"""

import ast
from typing import List, Dict, Optional
from textwrap import dedent
from ..core.issues import Issue


class TailChasingExplainer:
    """Generate natural language explanations for tail-chasing patterns."""
    
    def __init__(self):
        self.pattern_explanations = {
            'phantom_function': self._explain_phantom_function,
            'circular_import': self._explain_circular_import,
            'duplicate_function': self._explain_duplicate_function,
            'semantic_duplicate_function': self._explain_semantic_duplicate,
            'missing_symbol': self._explain_missing_symbol,
            'wrapper_abstraction': self._explain_wrapper_abstraction,
            'hallucination_cascade': self._explain_hallucination_cascade,
            'context_window_thrashing': self._explain_context_thrashing,
            'import_anxiety': self._explain_import_anxiety,
            'mirror_test': self._explain_mirror_test,
            'brittle_test_assertions': self._explain_brittle_test,
            'cargo_cult': self._explain_cargo_cult
        }
        
    def explain_issue(self, issue: Issue) -> str:
        """Generate a comprehensive explanation for a single issue."""
        # Find the appropriate explainer
        explainer = None
        for pattern, func in self.pattern_explanations.items():
            if pattern in issue.kind:
                explainer = func
                break
        
        if not explainer:
            explainer = self._explain_generic
        
        return explainer(issue)
    
    def generate_summary_explanation(self, issues: List[Issue]) -> str:
        """Generate a summary explanation for multiple related issues."""
        if not issues:
            return "No tail-chasing patterns detected."
        
        # Group by pattern type
        pattern_counts = {}
        for issue in issues:
            base_kind = issue.kind.split('_')[0]  # Get base pattern type
            pattern_counts[base_kind] = pattern_counts.get(base_kind, 0) + 1
        
        # Generate summary
        summary = f"""
🔍 **Tail-Chasing Analysis Summary**

I've detected {len(issues)} tail-chasing patterns in your code. Here's what's happening:

"""
        
        # Add pattern-specific summaries
        if 'phantom' in pattern_counts:
            summary += self._summarize_phantom_pattern(pattern_counts['phantom'])
        
        if 'circular' in pattern_counts:
            summary += self._summarize_circular_pattern(pattern_counts['circular'])
        
        if 'duplicate' in pattern_counts or 'semantic' in pattern_counts:
            dup_count = pattern_counts.get('duplicate', 0) + pattern_counts.get('semantic', 0)
            summary += self._summarize_duplication_pattern(dup_count)
        
        # Add root cause analysis
        summary += "\n## 🎯 Root Cause Analysis\n\n"
        summary += self._analyze_root_causes(issues)
        
        # Add actionable recommendations
        summary += "\n## 💡 Recommended Actions\n\n"
        summary += self._generate_action_plan(issues)
        
        return summary
    
    def _explain_phantom_function(self, issue: Issue) -> str:
        """Explain phantom function pattern."""
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Phantom Function**
        
        **What happened:**
        The LLM created a stub function `{issue.symbol}` that contains no real implementation 
        (just `pass` or `raise NotImplementedError`). This typically happens when the LLM 
        encounters an import error and creates a placeholder to satisfy it, but then never 
        returns to implement the actual functionality.
        
        **Why this is problematic:**
        - The function exists but does nothing useful
        - Code calling this function will fail at runtime
        - It masks the real problem (why was this function needed?)
        - Future LLM iterations might try to "fix" code that calls this phantom
        
        **Root cause:**
        The LLM likely saw an error like "NameError: name '{issue.symbol}' is not defined" 
        and created an empty function instead of understanding why the reference existed.
        
        **Recommended fix:**
        1. Delete the phantom function
        2. Investigate why it was referenced in the first place
        3. Either implement it properly or remove all references to it
        4. If it's truly needed, provide the LLM with clear requirements for what it should do
        
        **Similar patterns in your codebase:**
        Look for other functions that only contain `pass`, `...`, or `NotImplementedError`.
        """).strip()
    
    def _explain_circular_import(self, issue: Issue) -> str:
        """Explain circular import pattern."""
        evidence = issue.evidence or {}
        cycle = evidence.get('cycle', [])
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Circular Import**
        
        **What happened:**
        The LLM created a circular dependency chain: {' → '.join(cycle[:3])}{'...' if len(cycle) > 3 else ''}
        Each module tries to import from another in the chain, creating an infinite loop.
        
        **Why this is problematic:**
        - Python cannot resolve circular imports at module level
        - The code will crash with ImportError when any module in the cycle is imported
        - It indicates poor separation of concerns
        
        **Root cause:**
        The LLM tried to fix an import error by adding imports without understanding the 
        module structure. Each "fix" created a new dependency that eventually circled back.
        
        **Recommended fix:**
        1. Identify the shared functionality causing the circular dependency
        2. Extract it to a separate module that others can import from
        3. Use import-at-function-level as a temporary workaround if needed
        4. Restructure the code to have clear, one-way dependencies
        
        **Prevention tip:**
        When prompting the LLM, provide clear module structure: "Module A should not import from Module B"
        """).strip()
    
    def _explain_duplicate_function(self, issue: Issue) -> str:
        """Explain duplicate function pattern."""
        evidence = issue.evidence or {}
        locations = evidence.get('locations', [])
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Duplicate Implementation**
        
        **What happened:**
        The LLM created multiple versions of essentially the same function in different locations:
        {self._format_locations(locations[:3])}
        
        **Why this is problematic:**
        - Violates DRY (Don't Repeat Yourself) principle
        - Bug fixes must be applied multiple times
        - Increases maintenance burden
        - Confuses which version should be used
        
        **Root cause:**
        Instead of finding and importing an existing function, the LLM recreated it. This often 
        happens when the LLM loses track of what's already implemented or works with limited context.
        
        **Recommended fix:**
        1. Choose the most complete/documented version as the canonical implementation
        2. Delete all other versions
        3. Update all imports to use the canonical version
        4. Consider moving to a shared utility module if used across multiple files
        
        **Prevention tip:**
        Tell the LLM explicitly: "Check if this functionality already exists before implementing"
        """).strip()
    
    def _explain_semantic_duplicate(self, issue: Issue) -> str:
        """Explain semantic duplicate pattern."""
        evidence = issue.evidence or {}
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Semantic Duplicate**
        
        **What happened:**
        The LLM created functions that do the same thing but with different names or slight 
        implementation variations. These functions are {evidence.get('similarity', 0):.0%} similar 
        in their behavior despite looking different.
        
        **Why this is problematic:**
        - Hidden duplication is harder to maintain than obvious duplication
        - Different names for the same concept create confusion
        - Teams waste time understanding why there are multiple ways to do the same thing
        
        **Root cause:**
        The LLM implemented the same logic multiple times, possibly with different prompts or 
        in different sessions, without recognizing the existing implementation.
        
        **Recommended fix:**
        1. Review all semantically similar functions
        2. Identify the core functionality they share
        3. Create one well-named function that handles all use cases
        4. Replace all variations with calls to the unified function
        
        **Detection insight:**
        Our semantic analysis looked beyond names and syntax to identify these duplicates by 
        analyzing data flow, return patterns, and overall structure.
        """).strip()
    
    def _explain_missing_symbol(self, issue: Issue) -> str:
        """Explain missing symbol pattern."""
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Missing Symbol Import**
        
        **What happened:**
        The code tries to import `{issue.symbol}` but it doesn't exist in the target module. 
        This is likely a hallucinated import where the LLM assumed something exists when it doesn't.
        
        **Why this is problematic:**
        - The code will fail with ImportError at runtime
        - It might trigger the LLM to create phantom implementations
        - It indicates the LLM is guessing rather than checking
        
        **Root cause:**
        The LLM either misremembered the module structure or tried to import something that 
        seemed logical but doesn't actually exist.
        
        **Recommended fix:**
        1. Remove the invalid import
        2. Find where this symbol is actually defined (if anywhere)
        3. If it doesn't exist, either implement it properly or refactor to not need it
        4. Verify all imports against actual module contents
        
        **Prevention tip:**
        Provide the LLM with accurate module documentation or have it check available exports first
        """).strip()
    
    def _explain_wrapper_abstraction(self, issue: Issue) -> str:
        """Explain wrapper abstraction pattern."""
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Trivial Wrapper**
        
        **What happened:**
        The function `{issue.symbol}` is just a thin wrapper that calls another function 
        without adding any value (no validation, transformation, or additional logic).
        
        **Why this is problematic:**
        - Adds unnecessary indirection
        - Makes code harder to follow
        - No real abstraction benefit
        - Often indicates the LLM is creating structure without purpose
        
        **Root cause:**
        The LLM might be over-engineering or trying to maintain a pattern (like always having 
        a service layer) even when it adds no value.
        
        **Recommended fix:**
        1. Remove the wrapper function
        2. Have callers use the wrapped function directly
        3. Only keep wrappers that add real value (validation, logging, transformation)
        
        **Code smell indicator:**
        Wrappers should have a clear purpose. If you can't explain why the wrapper exists 
        in one sentence, it probably shouldn't exist.
        """).strip()
    
    def _explain_hallucination_cascade(self, issue: Issue) -> str:
        """Explain hallucination cascade pattern."""
        evidence = issue.evidence or {}
        components = evidence.get('locations', [])
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Hallucination Cascade**
        
        **What happened:**
        The LLM created an entire fictional subsystem with multiple interconnected classes/functions 
        that reference each other but have no connection to the rest of your codebase. 
        Components involved: {len(components)} interdependent pieces.
        
        **Why this is problematic:**
        - Large amount of useless code that looks legitimate
        - These components only make sense together, not individually
        - High complexity with zero actual functionality
        - Future modifications might try to "fix" this fictional system
        
        **Root cause:**
        The LLM encountered an error and started creating supporting infrastructure for its fix, 
        then needed more infrastructure for that infrastructure, creating a cascade of fictional code.
        
        **Recommended fix:**
        1. DELETE THE ENTIRE HALLUCINATED SUBSYSTEM
        2. Go back to the original error that triggered this
        3. Find the correct solution that uses existing code
        4. Be very suspicious of large amounts of new code for simple fixes
        
        **Red flag:**
        When an LLM creates 5+ new classes/functions to fix a simple error, it's likely hallucinating.
        """).strip()
    
    def _explain_context_thrashing(self, issue: Issue) -> str:
        """Explain context window thrashing pattern."""
        evidence = issue.evidence or {}
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Context Window Thrashing**
        
        **What happened:**
        The LLM implemented similar functionality multiple times in the same file, separated by 
        {evidence.get('line_distance', 'many')} lines. The implementations are {evidence.get('similarity', 0):.0%} 
        similar, suggesting the LLM forgot about the earlier implementation.
        
        **Why this is problematic:**
        - Duplicate logic in the same file
        - Indicates the LLM lost track of context
        - Future edits might update one version but not the other
        - Makes the file unnecessarily long and complex
        
        **Root cause:**
        The LLM's context window couldn't hold the entire file, so when asked to implement 
        something, it didn't realize it had already done so earlier.
        
        **Recommended fix:**
        1. Identify all similar implementations
        2. Keep the best version, delete the others
        3. Consider breaking large files into smaller, focused modules
        4. For LLM sessions, work on smaller chunks at a time
        
        **Prevention strategy:**
        When working with LLMs on large files, explicitly remind it of existing functions: 
        "We already have function X that does Y"
        """).strip()
    
    def _explain_import_anxiety(self, issue: Issue) -> str:
        """Explain import anxiety pattern."""
        evidence = issue.evidence or {}
        unused_ratio = evidence.get('unused_ratio', 0)
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Import Anxiety**
        
        **What happened:**
        The LLM imported many more items than needed - {unused_ratio:.0%} of imports are unused. 
        This "defensive importing" happens when the LLM isn't sure what's needed so imports everything.
        
        **Why this is problematic:**
        - Slower startup time
        - Unclear dependencies
        - Potential naming conflicts
        - Makes it harder to understand what the code actually uses
        
        **Root cause:**
        After encountering an import error, the LLM overcorrected by importing everything that 
        might be related, just to be "safe".
        
        **Recommended fix:**
        1. Remove all unused imports
        2. Only import what's actually used
        3. Use tools like `isort` and `autoflake` to manage imports
        4. Be specific in prompts: "Only import what's needed"
        
        **Pattern recognition:**
        Import anxiety often follows this pattern: Error → Add one import → Add related imports → 
        Add everything from that module "just in case"
        """).strip()
    
    def _explain_mirror_test(self, issue: Issue) -> str:
        """Explain mirror test pattern."""
        evidence = issue.evidence or {}
        tested_function = evidence.get('tested_function', 'unknown')
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Mirror Test**
        
        **What happened:**
        The test `{issue.symbol}` duplicates the implementation logic of `{tested_function}` 
        instead of testing its behavior. The test is {evidence.get('similarity_score', 0):.0%} 
        similar to the implementation itself.
        
        **Why this is problematic:**
        - The test will pass even if the implementation is wrong
        - Changes to implementation require identical changes to tests
        - No actual validation of correctness
        - Provides false confidence
        
        **Root cause:**
        The LLM misunderstood testing principles and created a test that proves the code does 
        what it does, rather than proving it does what it should do.
        
        **Recommended fix:**
        1. Rewrite the test to check expected outcomes, not implementation steps
        2. Test the contract/interface, not the internals
        3. Add edge cases and error conditions
        4. Ask: "What should this function do?" not "What does this function do?"
        
        **Good test principles:**
        - Test behavior, not implementation
        - Test edge cases and errors, not just happy path
        - Tests should fail if the implementation is wrong
        """).strip()
    
    def _explain_brittle_test(self, issue: Issue) -> str:
        """Explain brittle test pattern."""
        evidence = issue.evidence or {}
        patterns = evidence.get('patterns', [])
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Brittle Test Assertions**
        
        **What happened:**
        The test `{issue.symbol}` contains overly specific assertions that will break with minor, 
        inconsequential changes. Brittle patterns found: {', '.join(patterns)}.
        
        **Why this is problematic:**
        - Tests break for the wrong reasons
        - Maintenance nightmare - every small change requires test updates
        - Tests become a burden rather than a safety net
        - Developers start ignoring test failures
        
        **Root cause:**
        The LLM created tests that check exact implementation details rather than essential behavior, 
        possibly by copying current output verbatim.
        
        **Recommended fix:**
        1. Replace exact string matches with pattern matching or key checks
        2. Use `assertIn` instead of `assertEqual` for flexible validation
        3. Test properties and invariants, not exact values
        4. Focus on what matters, ignore what doesn't
        
        **Example transformation:**
        Bad:  `assertEqual(result, "User John Doe created at 2024-01-15 10:30:45")`
        Good: `assertIn("John Doe", result)` and `assertRegex(result, r"created at \d{4}-")`
        """).strip()
    
    def _explain_cargo_cult(self, issue: Issue) -> str:
        """Explain cargo cult pattern."""
        evidence = issue.evidence or {}
        pattern_type = evidence.get('type', 'unknown')
        
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: Cargo Cult Programming**
        
        **What happened:**
        The LLM copied a common pattern or boilerplate without understanding its purpose. 
        Specific pattern: {pattern_type}. This is "cargo cult" programming - mimicking form 
        without understanding function.
        
        **Why this is problematic:**
        - Unnecessary code that serves no purpose
        - Increases complexity without benefit
        - Shows lack of understanding of the codebase
        - May introduce bugs or performance issues
        
        **Root cause:**
        The LLM has seen this pattern many times in training data and applies it even when 
        it's not appropriate, like adding `super().__init__()` when there's no parent class.
        
        **Recommended fix:**
        1. Remove the unnecessary boilerplate
        2. Only include code that has a clear purpose
        3. Question every line: "What does this accomplish?"
        4. Don't copy patterns without understanding why they exist
        
        **Learning opportunity:**
        Each piece of code should exist for a reason. If you can't explain why something is there, 
        it probably shouldn't be.
        """).strip()
    
    def _explain_generic(self, issue: Issue) -> str:
        """Generic explanation for unknown patterns."""
        return dedent(f"""
        🔍 **Tail-Chasing Pattern Detected: {issue.kind}**
        
        **What happened:**
        {issue.message}
        
        **Location:**
        {issue.file}:{issue.line} - {issue.symbol or 'N/A'}
        
        **Why this matters:**
        This pattern indicates the LLM is making changes without fully understanding the context, 
        leading to a cycle of fixes that create new problems.
        
        **Recommended actions:**
        {chr(10).join(f"- {s}" for s in (issue.suggestions or ["Review and fix manually"]))}
        
        **General advice:**
        When you see tail-chasing patterns, it's a sign to stop the automated fixes and 
        understand the root cause manually.
        """).strip()
    
    def _summarize_phantom_pattern(self, count: int) -> str:
        """Summarize phantom pattern issues."""
        return dedent(f"""
        ### 👻 Phantom Functions ({count} found)
        The LLM created {count} empty stub functions that don't actually do anything. These are 
        like "TODO" notes that never got completed. Each one is a potential runtime error waiting 
        to happen.
        
        """)
    
    def _summarize_circular_pattern(self, count: int) -> str:
        """Summarize circular import issues."""
        return dedent(f"""
        ### 🔄 Circular Dependencies ({count} found)
        There are {count} circular import chains where modules try to import each other in a loop. 
        This is like two people saying "after you" at a door forever - neither can proceed.
        
        """)
    
    def _summarize_duplication_pattern(self, count: int) -> str:
        """Summarize duplication issues."""
        return dedent(f"""
        ### 👥 Code Duplication ({count} found)
        The LLM created {count} duplicate implementations of the same functionality. Instead of 
        finding and reusing existing code, it kept recreating it in different places.
        
        """)
    
    def _analyze_root_causes(self, issues: List[Issue]) -> str:
        """Analyze root causes across all issues."""
        # Look for patterns in the issues
        file_counts = {}
        for issue in issues:
            if issue.file:
                file_counts[issue.file] = file_counts.get(issue.file, 0) + 1
        
        hotspot_files = [f for f, count in file_counts.items() if count > 2]
        
        analysis = "Based on the patterns detected:\n\n"
        
        if hotspot_files:
            analysis += f"- **Hotspot files**: {', '.join(hotspot_files[:3])} have multiple issues\n"
        
        if any('phantom' in i.kind for i in issues) and any('missing' in i.kind for i in issues):
            analysis += "- **Import-driven development**: The LLM is creating code to satisfy imports rather than actual requirements\n"
        
        if any('circular' in i.kind for i in issues):
            analysis += "- **Poor module structure**: The codebase lacks clear separation of concerns\n"
        
        if any('duplicate' in i.kind or 'semantic' in i.kind for i in issues):
            analysis += "- **Context loss**: The LLM is working without awareness of existing code\n"
        
        return analysis
    
    def _generate_action_plan(self, issues: List[Issue]) -> str:
        """Generate prioritized action plan."""
        plan = "1. **Immediate actions** (prevent runtime errors):\n"
        
        # Critical issues first
        critical = [i for i in issues if i.severity >= 3]
        if critical:
            plan += f"   - Fix {len(critical)} critical issues that will cause immediate failures\n"
            plan += f"   - Start with circular imports and missing symbols\n"
        
        plan += "\n2. **Cleanup actions** (improve code quality):\n"
        plan += "   - Remove phantom functions and replace with real implementations\n"
        plan += "   - Consolidate duplicate code into shared utilities\n"
        plan += "   - Delete unnecessary wrappers and abstractions\n"
        
        plan += "\n3. **Prevention strategies**:\n"
        plan += "   - Break large files into smaller, focused modules\n"
        plan += "   - Provide LLMs with clear context about existing code\n"
        plan += "   - Review LLM changes before accepting them\n"
        plan += "   - Use 'tail-chasing' detection in your CI/CD pipeline\n"
        
        return plan
    
    def _format_locations(self, locations: List[tuple]) -> str:
        """Format file locations nicely."""
        if not locations:
            return "No locations available"
        
        formatted = []
        for loc in locations[:5]:  # Limit to 5
            if isinstance(loc, tuple) and len(loc) >= 2:
                formatted.append(f"- {loc[0]}:{loc[1]}")
            else:
                formatted.append(f"- {loc}")
        
        if len(locations) > 5:
            formatted.append(f"- ... and {len(locations) - 5} more")
        
        return "\n".join(formatted)
