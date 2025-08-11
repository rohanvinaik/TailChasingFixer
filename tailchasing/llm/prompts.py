"""
Intelligent prompting system for tail-chasing scenarios.

Provides context-aware prompts with error history, AST-safe transformations,
and few-shot examples for optimal LLM performance.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import json

from ..core.issues import Issue


@dataclass
class PromptContext:
    """Context information for generating intelligent prompts."""
    current_issue: Issue
    related_issues: List[Issue] = None
    previous_attempts: List[Dict[str, Any]] = None
    source_code: Optional[str] = None
    file_context: Optional[Dict[str, str]] = None  # filename -> content
    codebase_patterns: Optional[Dict[str, Any]] = None
    confidence_threshold: float = 0.7


class TailChasingPrompts:
    """Generates intelligent prompts for different tail-chasing scenarios."""
    
    def __init__(self):
        self.few_shot_examples = self._load_few_shot_examples()
        self.ast_safety_rules = self._get_ast_safety_rules()
    
    def generate_fix_prompt(self, context: PromptContext) -> str:
        """Generate a comprehensive fix prompt with context and examples."""
        
        prompt_parts = [
            self._get_role_prompt(),
            self._get_issue_analysis(context),
            self._get_context_information(context),
            self._get_previous_attempts_analysis(context),
            self._get_few_shot_examples_for_issue(context.current_issue.kind),
            self._get_ast_safety_requirements(),
            self._get_output_format_requirements(context.confidence_threshold)
        ]
        
        return "\n\n".join(filter(None, prompt_parts))
    
    def generate_validation_prompt(self, 
                                 original_code: str,
                                 proposed_fix: str, 
                                 issue: Issue,
                                 context: Optional[PromptContext] = None) -> str:
        """Generate prompt for validating a proposed fix."""
        
        prompt_parts = [
            "You are validating a proposed fix for a tail-chasing pattern.",
            "",
            f"**ORIGINAL ISSUE**: {issue.kind}",
            f"**DESCRIPTION**: {issue.message}",
            f"**SEVERITY**: {issue.severity}",
            "",
            "**ORIGINAL CODE**:",
            f"```python\n{original_code}\n```",
            "",
            "**PROPOSED FIX**:",
            f"```python\n{proposed_fix}\n```",
        ]
        
        if context and context.previous_attempts:
            prompt_parts.extend([
                "",
                "**PREVIOUS FAILED ATTEMPTS**:",
                self._format_previous_attempts(context.previous_attempts)
            ])
        
        prompt_parts.extend([
            "",
            "**VALIDATION CRITERIA**:",
            "1. Does the fix eliminate the tail-chasing pattern completely?",
            "2. Is the code syntactically correct and executable?", 
            "3. Are all AST transformations safe?",
            "4. Is existing functionality preserved?",
            "5. Are any new issues introduced?",
            "6. Does the fix address the root cause, not just symptoms?",
            "",
            "**RESPONSE FORMAT**:",
            "```json",
            "{",
            '  "valid": boolean,',
            '  "confidence": 0.0-1.0,', 
            '  "issues_found": ["list", "of", "problems"],',
            '  "recommendations": ["suggested", "improvements"],',
            '  "explanation": "detailed reasoning"',
            "}",
            "```"
        ])
        
        return "\n".join(prompt_parts)
    
    def generate_explanation_prompt(self, 
                                  issue: Issue,
                                  context: Optional[PromptContext] = None) -> str:
        """Generate prompt for explaining why a tail-chasing pattern occurred."""
        
        prompt_parts = [
            "You are an expert at analyzing why tail-chasing patterns emerge in code.",
            "",
            f"**TAIL-CHASING PATTERN**: {issue.kind}",
            f"**DESCRIPTION**: {issue.message}",
        ]
        
        if issue.file and issue.line:
            prompt_parts.append(f"**LOCATION**: {issue.file}:{issue.line}")
        
        if issue.evidence:
            prompt_parts.extend([
                "",
                "**EVIDENCE**:",
                f"```json\n{json.dumps(issue.evidence, indent=2)}\n```"
            ])
        
        if context and context.source_code:
            prompt_parts.extend([
                "",
                "**SOURCE CODE**:",
                f"```python\n{context.source_code}\n```"
            ])
        
        if context and context.codebase_patterns:
            prompt_parts.extend([
                "",
                "**CODEBASE PATTERNS**:",
                json.dumps(context.codebase_patterns, indent=2)
            ])
        
        prompt_parts.extend([
            "",
            "**ANALYSIS REQUESTED**:",
            "",
            "1. **Root Cause Analysis**: What specific conditions led to this pattern?",
            "2. **Developer/AI Psychology**: Why might someone create this pattern?",
            "3. **Systemic Issues**: Are there broader codebase issues contributing?",
            "4. **Prevention Strategy**: How can this pattern be prevented in future?",
            "5. **Warning Signs**: What early indicators suggest this pattern is emerging?",
            "",
            "Provide a thorough analysis that helps prevent recurrence of this pattern."
        ])
        
        return "\n".join(prompt_parts)
    
    def _get_role_prompt(self) -> str:
        """Get the expert role definition."""
        return """You are an expert Python developer and code quality specialist focused on eliminating tail-chasing patterns. You excel at:

- Providing complete, working implementations (never stubs or placeholders)
- AST-safe code transformations that maintain syntactic correctness
- Root cause analysis and systematic fixes
- Consolidating duplicate functionality intelligently
- Preventing regression and new pattern introduction

Your solutions are always practical, complete, and eliminate the problematic pattern entirely."""
    
    def _get_issue_analysis(self, context: PromptContext) -> str:
        """Generate detailed issue analysis section."""
        issue = context.current_issue
        
        analysis_parts = [
            f"**CURRENT ISSUE**: {issue.kind}",
            f"**DESCRIPTION**: {issue.message}",
            f"**SEVERITY**: {issue.severity}/5"
        ]
        
        if issue.file and issue.line:
            analysis_parts.append(f"**LOCATION**: {issue.file}:{issue.line}")
        
        if issue.symbol:
            analysis_parts.append(f"**SYMBOL**: {issue.symbol}")
        
        if issue.evidence:
            analysis_parts.extend([
                "",
                "**EVIDENCE**:",
                f"```json\n{json.dumps(issue.evidence, indent=2)}\n```"
            ])
        
        # Add related issues if present
        if context.related_issues:
            related_info = []
            for related in context.related_issues[:3]:  # Limit to 3 most relevant
                related_info.append(f"- {related.kind}: {related.message}")
            
            analysis_parts.extend([
                "",
                "**RELATED ISSUES**:",
                *related_info
            ])
        
        return "\n".join(analysis_parts)
    
    def _get_context_information(self, context: PromptContext) -> str:
        """Generate context information section."""
        if not (context.source_code or context.file_context or context.codebase_patterns):
            return ""
        
        context_parts = ["**CONTEXT INFORMATION**:"]
        
        if context.source_code:
            context_parts.extend([
                "",
                "**CURRENT SOURCE CODE**:",
                f"```python\n{context.source_code}\n```"
            ])
        
        if context.file_context:
            context_parts.append("")
            context_parts.append("**RELATED FILES**:")
            for filename, content in list(context.file_context.items())[:3]:
                context_parts.extend([
                    f"*{filename}*:",
                    f"```python\n{content[:500]}{'...' if len(content) > 500 else ''}\n```"
                ])
        
        if context.codebase_patterns:
            context_parts.extend([
                "",
                "**CODEBASE PATTERNS**:",
                json.dumps(context.codebase_patterns, indent=2)
            ])
        
        return "\n".join(context_parts)
    
    def _get_previous_attempts_analysis(self, context: PromptContext) -> str:
        """Analyze previous failed attempts."""
        if not context.previous_attempts:
            return ""
        
        analysis_parts = [
            "**PREVIOUS ATTEMPTS** (Learn from these failures):",
            ""
        ]
        
        for i, attempt in enumerate(context.previous_attempts[-3:], 1):  # Last 3 attempts
            analysis_parts.extend([
                f"**Attempt {i}**:",
                f"- Approach: {attempt.get('approach', 'Unknown')}",
                f"- Result: {attempt.get('result', 'Failed')}",
                f"- Issue: {attempt.get('failure_reason', 'No reason provided')}",
                ""
            ])
        
        analysis_parts.extend([
            "**LEARNING**: Avoid the above failed approaches. Try a fundamentally different strategy.",
            ""
        ])
        
        return "\n".join(analysis_parts)
    
    def _get_few_shot_examples_for_issue(self, issue_kind: str) -> str:
        """Get relevant few-shot examples for the issue type."""
        if issue_kind not in self.few_shot_examples:
            return ""
        
        examples = self.few_shot_examples[issue_kind]
        
        example_parts = [
            f"**EXAMPLES** (Similar {issue_kind} fixes):",
            ""
        ]
        
        for i, example in enumerate(examples[:2], 1):  # Limit to 2 examples
            example_parts.extend([
                f"**Example {i}**:",
                "BEFORE:",
                f"```python\n{example['before']}\n```",
                "",
                "AFTER:",
                f"```python\n{example['after']}\n```",
                "",
                f"APPROACH: {example['approach']}",
                ""
            ])
        
        return "\n".join(example_parts)
    
    def _get_ast_safety_requirements(self) -> str:
        """Get AST safety requirements."""
        safety_parts = [
            "**AST-SAFE TRANSFORMATION REQUIREMENTS**:",
            ""
        ]
        
        safety_parts.extend([f"- {rule}" for rule in self.ast_safety_rules])
        
        return "\n".join(safety_parts)
    
    def _get_output_format_requirements(self, confidence_threshold: float) -> str:
        """Get output format requirements."""
        return f"""**RESPONSE FORMAT**:

Provide your solution in JSON format:

```json
{{
  "analysis": "Brief analysis of the tail-chasing pattern and root cause",
  "approach": "Your strategy for fixing this issue", 
  "code": "Complete, working Python code that eliminates the pattern",
  "confidence": 0.0-1.0,
  "rationale": "Explanation of your solution and why it works",
  "ast_safe": true/false,
  "preserves_functionality": true/false,
  "eliminates_pattern": true/false
}}
```

**QUALITY REQUIREMENTS**:
- Confidence must be >= {confidence_threshold} for production use
- Code must be complete and executable (no stubs or TODOs)
- Must eliminate the tail-chasing pattern completely
- All transformations must be AST-safe
- Preserve all existing functionality"""
    
    def _format_previous_attempts(self, attempts: List[Dict[str, Any]]) -> str:
        """Format previous attempts for display."""
        formatted = []
        for i, attempt in enumerate(attempts[-3:], 1):
            formatted.extend([
                f"**Attempt {i}**: {attempt.get('approach', 'Unknown approach')}",
                f"Failed because: {attempt.get('failure_reason', 'No reason provided')}",
                ""
            ])
        return "\n".join(formatted)
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict[str, str]]]:
        """Load few-shot examples for different issue types."""
        return {
            "phantom_function": [
                {
                    "before": """def calculate_metrics(data):
    pass  # TODO: implement later""",
                    "after": """def calculate_metrics(data):
    return {
        'mean': sum(data) / len(data) if data else 0,
        'count': len(data),
        'sum': sum(data)
    }""",
                    "approach": "Implemented complete functionality based on function name and context"
                },
                {
                    "before": """def validate_input(value):
    raise NotImplementedError("Validation not implemented")""",
                    "after": """# Function removed - not used anywhere in codebase""",
                    "approach": "Removed phantom function that was never called"
                }
            ],
            
            "duplicate_function": [
                {
                    "before": """def format_name(first, last):
    return f"{first} {last}"

def create_full_name(first_name, last_name):
    return f"{first_name} {last_name}"

def get_user_display_name(f, l):
    return f + " " + l""",
                    "after": """def format_name(first, last):
    return f"{first} {last}"

# All calls updated to use format_name()""",
                    "approach": "Consolidated to the most clearly named function, updated all references"
                }
            ],
            
            "circular_import": [
                {
                    "before": """# models.py
from utils import helper
class User:
    def process(self): return helper(self)

# utils.py
from models import User  
def helper(user: User): return user.name""",
                    "after": """# models.py
class User:
    def process(self): 
        from utils import helper  # Local import
        return helper(self)

# utils.py
def helper(user): return user.name  # Use duck typing""",
                    "approach": "Moved import local and removed type annotation to break cycle"
                }
            ]
        }
    
    def _get_ast_safety_rules(self) -> List[str]:
        """Get list of AST safety rules."""
        return [
            "All generated code must parse successfully with ast.parse()",
            "Maintain proper Python syntax and indentation",
            "Preserve import statements and dependencies",
            "Don't break existing function signatures unless explicitly required",
            "Maintain proper scope and variable accessibility",
            "Ensure all brackets, parentheses, and quotes are properly matched",
            "Don't create unreachable code or invalid control flow",
            "Preserve docstrings and maintain API compatibility where possible"
        ]


def create_context_aware_prompt(issue: Issue,
                               source_code: Optional[str] = None,
                               previous_attempts: Optional[List[Dict[str, Any]]] = None,
                               related_issues: Optional[List[Issue]] = None,
                               codebase_info: Optional[Dict[str, Any]] = None) -> str:
    """Create a context-aware prompt for the given issue."""
    
    context = PromptContext(
        current_issue=issue,
        source_code=source_code,
        previous_attempts=previous_attempts,
        related_issues=related_issues,
        codebase_patterns=codebase_info
    )
    
    prompter = TailChasingPrompts()
    return prompter.generate_fix_prompt(context)