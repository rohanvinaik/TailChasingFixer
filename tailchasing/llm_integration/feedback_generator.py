"""
Feedback generator for LLMs to prevent tail-chasing patterns.

Generates system prompts, context warnings, and negative examples
to help LLMs avoid creating tail-chasing anti-patterns.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from ..core.issues import Issue
from ..analyzers.explainer import TailChasingExplainer

logger = logging.getLogger(__name__)


@dataclass
class LLMFeedback:
    """Feedback package for an LLM."""
    
    system_prompt_additions: List[str] = field(default_factory=list)
    context_warnings: List[str] = field(default_factory=list)
    negative_examples: List[Dict[str, Any]] = field(default_factory=list)
    positive_examples: List[Dict[str, Any]] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'system_prompt_additions': self.system_prompt_additions,
            'context_warnings': self.context_warnings,
            'negative_examples': self.negative_examples,
            'positive_examples': self.positive_examples,
            'rules': self.rules,
            'metadata': self.metadata
        }


class FeedbackGenerator:
    """
    Generate actionable feedback for LLMs to prevent tail-chasing patterns.
    
    Creates system prompts, warnings, and examples based on detected issues.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feedback generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.explainer = TailChasingExplainer(config)
        
        # Configuration
        self.max_negative_examples = self.config.get('max_negative_examples', 5)
        self.max_positive_examples = self.config.get('max_positive_examples', 3)
        self.include_explanations = self.config.get('include_explanations', True)
        self.severity_threshold = self.config.get('severity_threshold', 2)
        
        # Pattern-specific prompt templates
        self.prompt_templates = self._initialize_prompt_templates()
        
        # General rules for preventing tail-chasing
        self.general_rules = self._initialize_general_rules()
    
    def generate_feedback(
        self,
        issues: List[Issue],
        codebase_context: Optional[Dict[str, Any]] = None
    ) -> LLMFeedback:
        """
        Generate comprehensive feedback based on detected issues.
        
        Args:
            issues: List of detected issues
            codebase_context: Optional context about the codebase
            
        Returns:
            LLMFeedback object with actionable guidance
        """
        feedback = LLMFeedback()
        
        # Add metadata
        feedback.metadata = {
            'generated_at': datetime.now().isoformat(),
            'issue_count': len(issues),
            'codebase_context': codebase_context or {}
        }
        
        if not issues:
            # No issues - provide positive reinforcement
            feedback.system_prompt_additions.append(
                "Great job! No tail-chasing patterns detected. Continue following best practices."
            )
            feedback.rules.extend(self.general_rules)
            return feedback
        
        # Group issues by pattern type
        pattern_groups = self._group_issues_by_pattern(issues)
        
        # Generate system prompt additions
        feedback.system_prompt_additions = self._generate_system_prompts(pattern_groups)
        
        # Generate context warnings
        feedback.context_warnings = self._generate_context_warnings(pattern_groups, issues)
        
        # Generate negative examples
        feedback.negative_examples = self._generate_negative_examples(
            issues[:self.max_negative_examples]
        )
        
        # Generate positive examples (corrections)
        feedback.positive_examples = self._generate_positive_examples(pattern_groups)
        
        # Add specific rules based on detected patterns
        feedback.rules = self._generate_pattern_rules(pattern_groups)
        
        return feedback
    
    def generate_prevention_prompt(
        self,
        detected_patterns: Set[str]
    ) -> str:
        """
        Generate a prevention-focused system prompt based on detected patterns.
        
        Args:
            detected_patterns: Set of pattern types detected
            
        Returns:
            System prompt text
        """
        prompt_parts = [
            "## Code Quality Guidelines\n",
            "You are assisting with code development. Please be aware of and avoid these common anti-patterns:\n"
        ]
        
        # Add pattern-specific guidelines
        for pattern in detected_patterns:
            if pattern in self.prompt_templates:
                prompt_parts.append(f"\n### {self._humanize_pattern_name(pattern)}")
                prompt_parts.append(self.prompt_templates[pattern]['prevention'])
        
        # Add general best practices
        prompt_parts.append("\n## General Best Practices:")
        prompt_parts.extend([f"- {rule}" for rule in self.general_rules[:5]])
        
        return "\n".join(prompt_parts)
    
    def generate_context_alert(
        self,
        current_file: str,
        current_function: Optional[str],
        recent_issues: List[Issue]
    ) -> Optional[str]:
        """
        Generate a context-specific alert for the current code location.
        
        Args:
            current_file: Current file being edited
            current_function: Current function being edited
            recent_issues: Recently detected issues
            
        Returns:
            Alert message if risks detected, None otherwise
        """
        relevant_issues = []
        
        for issue in recent_issues:
            # Check if issue is relevant to current context
            if issue.file == current_file:
                relevant_issues.append(issue)
            elif current_function and issue.symbol == current_function:
                relevant_issues.append(issue)
        
        if not relevant_issues:
            return None
        
        # Generate alert
        alert_parts = ["⚠️ **Context Alert**: Be aware of these existing issues in this area:\n"]
        
        for issue in relevant_issues[:3]:  # Limit to top 3
            pattern = self._classify_pattern(issue)
            alert_parts.append(f"- {self._humanize_pattern_name(pattern)}: {issue.message[:100]}")
        
        alert_parts.append("\nPlease ensure your changes don't exacerbate these patterns.")
        
        return "\n".join(alert_parts)
    
    def _initialize_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize pattern-specific prompt templates."""
        return {
            'duplicate_function': {
                'prevention': (
                    "**Avoid creating duplicate functions:**\n"
                    "- First search for existing functions with similar names or purposes\n"
                    "- Use descriptive, unique function names\n"
                    "- If you find similar functionality, extend or refactor it rather than duplicating\n"
                    "- Ask: 'Does this functionality already exist somewhere?'"
                ),
                'warning': "Similar functions already exist in this codebase. Consider reusing or extending them.",
                'rule': "Before creating a new function, search for existing similar implementations"
            },
            
            'circular_import': {
                'prevention': (
                    "**Prevent circular imports:**\n"
                    "- Maintain a clear dependency hierarchy (high-level → low-level)\n"
                    "- Avoid importing from modules that import from the current module\n"
                    "- Use local imports (inside functions) when necessary\n"
                    "- Consider if shared code should be in a separate utility module"
                ),
                'warning': "This module has import dependencies that could create cycles.",
                'rule': "Check import dependencies before adding new imports"
            },
            
            'phantom_function': {
                'prevention': (
                    "**Avoid creating placeholder functions:**\n"
                    "- Only create functions when you can implement them fully\n"
                    "- If a stub is necessary, add a clear TODO comment with requirements\n"
                    "- Don't create empty functions just to satisfy imports\n"
                    "- Raise NotImplementedError with a descriptive message if truly needed"
                ),
                'warning': "There are existing stub functions that need implementation.",
                'rule': "Implement functions completely or clearly mark them as intentional stubs"
            },
            
            'hallucination_cascade': {
                'prevention': (
                    "**Avoid over-engineering:**\n"
                    "- Start with the simplest solution that works\n"
                    "- Don't create abstract base classes unless there are multiple implementations\n"
                    "- Avoid creating entire subsystems for simple problems\n"
                    "- Check if existing libraries or code already solve the problem"
                ),
                'warning': "Complex abstraction detected. Consider if simpler solutions exist.",
                'rule': "Prefer simple, direct solutions over complex abstractions"
            },
            
            'context_window_thrashing': {
                'prevention': (
                    "**Be aware of context limitations:**\n"
                    "- Review the entire file before adding new functions\n"
                    "- Look for existing similar implementations, especially in large files\n"
                    "- Use consistent naming patterns\n"
                    "- Consider breaking large files into smaller, focused modules"
                ),
                'warning': "This is a large file. Check for existing similar functions before adding new ones.",
                'rule': "Review existing code thoroughly before implementing new features"
            },
            
            'import_anxiety': {
                'prevention': (
                    "**Import only what you need:**\n"
                    "- Import specific items: `from module import specific_function`\n"
                    "- Avoid wildcard imports: `from module import *`\n"
                    "- Don't import 'just in case' - add imports as needed\n"
                    "- Remove unused imports after refactoring"
                ),
                'warning': "This file has many imports. Only add what's actually needed.",
                'rule': "Import only the specific items you actually use"
            },
            
            'semantic_duplicate': {
                'prevention': (
                    "**Check for semantic equivalents:**\n"
                    "- Functions with different names might do the same thing\n"
                    "- Look for similar logic patterns, not just similar names\n"
                    "- Consider if slight variations could be parameterized\n"
                    "- Test if existing functions already meet your needs"
                ),
                'warning': "Semantically similar functions exist. Consider if they meet your needs.",
                'rule': "Look for functionally equivalent code, not just name matches"
            }
        }
    
    def _initialize_general_rules(self) -> List[str]:
        """Initialize general rules for preventing tail-chasing."""
        return [
            "Always search for existing implementations before creating new functions",
            "Maintain a clear import hierarchy to avoid circular dependencies",
            "Implement functions completely - avoid creating empty stubs",
            "Start with simple solutions and only add complexity when necessary",
            "Import only what you actually use in the code",
            "Use descriptive, unique names to avoid confusion",
            "Consider the broader codebase context before making changes",
            "Test existing functions before reimplementing functionality",
            "Break large files into smaller, focused modules",
            "Document why code exists if its purpose isn't obvious",
            "Review related code before making modifications",
            "Prefer extending existing code over duplicating it",
            "Use consistent patterns throughout the codebase",
            "Avoid defensive programming without clear requirements",
            "Consider maintenance burden when adding abstractions"
        ]
    
    def _group_issues_by_pattern(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group issues by pattern type."""
        groups = defaultdict(list)
        for issue in issues:
            pattern = self._classify_pattern(issue)
            groups[pattern].append(issue)
        return dict(groups)
    
    def _classify_pattern(self, issue: Issue) -> str:
        """Classify the pattern type from an issue."""
        kind = issue.kind.lower()
        
        if 'duplicate' in kind and 'semantic' in kind:
            return 'semantic_duplicate'
        elif 'duplicate' in kind:
            return 'duplicate_function'
        elif 'circular' in kind:
            return 'circular_import'
        elif 'phantom' in kind:
            return 'phantom_function'
        elif 'hallucination' in kind:
            return 'hallucination_cascade'
        elif 'context' in kind or 'thrashing' in kind:
            return 'context_window_thrashing'
        elif 'import' in kind or 'anxiety' in kind:
            return 'import_anxiety'
        else:
            return 'unknown'
    
    def _generate_system_prompts(
        self,
        pattern_groups: Dict[str, List[Issue]]
    ) -> List[str]:
        """Generate system prompt additions based on detected patterns."""
        prompts = []
        
        # Add header
        prompts.append(
            "## ⚠️ Code Quality Alert\n"
            "Recent analysis has detected patterns that suggest tail-chasing in this codebase. "
            "Please be extra careful to avoid these anti-patterns:"
        )
        
        # Add pattern-specific prompts
        for pattern, issues in pattern_groups.items():
            if pattern in self.prompt_templates:
                count = len(issues)
                severity = max(i.severity for i in issues)
                
                prompt = f"\n### {self._humanize_pattern_name(pattern)} ({count} instance{'s' if count != 1 else ''})"
                
                # Add prevention guidance
                prompt += f"\n{self.prompt_templates[pattern]['prevention']}"
                
                # Add specific examples if high severity
                if severity >= 3:
                    prompt += f"\n**Example issue:** {issues[0].message[:150]}"
                
                prompts.append(prompt)
        
        return prompts
    
    def _generate_context_warnings(
        self,
        pattern_groups: Dict[str, List[Issue]],
        all_issues: List[Issue]
    ) -> List[str]:
        """Generate context-specific warnings."""
        warnings = []
        
        # Identify high-risk files
        file_issue_counts = defaultdict(int)
        for issue in all_issues:
            if issue.file:
                file_issue_counts[issue.file] += 1
        
        high_risk_files = [
            file for file, count in file_issue_counts.items()
            if count >= 3
        ]
        
        if high_risk_files:
            warnings.append(
                f"⚠️ **High-risk files** with multiple issues: {', '.join(high_risk_files[:5])}\n"
                "Exercise extra caution when modifying these files."
            )
        
        # Add pattern-specific warnings
        for pattern, issues in pattern_groups.items():
            if pattern in self.prompt_templates and len(issues) >= 2:
                warnings.append(self.prompt_templates[pattern]['warning'])
        
        # Add size warnings
        large_file_patterns = pattern_groups.get('context_window_thrashing', [])
        if large_file_patterns:
            warnings.append(
                "📏 **Large file warning**: Some files exceed optimal size for LLM context. "
                "Consider splitting them into smaller modules."
            )
        
        return warnings
    
    def _generate_negative_examples(
        self,
        issues: List[Issue]
    ) -> List[Dict[str, Any]]:
        """Generate negative examples from detected issues."""
        examples = []
        
        for issue in issues:
            if issue.severity < self.severity_threshold:
                continue
            
            pattern = self._classify_pattern(issue)
            
            example = {
                'pattern': pattern,
                'description': f"❌ **Don't do this:** {issue.message[:200]}",
                'file': issue.file,
                'line': issue.line,
                'why_bad': self._explain_why_bad(pattern, issue),
                'evidence': {}
            }
            
            # Add relevant evidence
            if hasattr(issue, 'evidence'):
                evidence = issue.evidence
                if 'duplicate_count' in evidence:
                    example['evidence']['duplicates'] = evidence['duplicate_count']
                if 'similarity' in evidence:
                    example['evidence']['similarity'] = f"{evidence['similarity']:.1%}"
                if 'cycle' in evidence:
                    example['evidence']['import_cycle'] = ' → '.join(evidence['cycle'][:5])
            
            # Add explanation if enabled
            if self.include_explanations:
                explanation = self.explainer.explain_issue_enhanced(issue)
                example['explanation'] = explanation.summary
                example['root_cause'] = explanation.root_causes[0] if explanation.root_causes else "Unknown"
            
            examples.append(example)
        
        return examples
    
    def _generate_positive_examples(
        self,
        pattern_groups: Dict[str, List[Issue]]
    ) -> List[Dict[str, Any]]:
        """Generate positive examples (corrections) for detected patterns."""
        examples = []
        
        corrections = {
            'duplicate_function': {
                'description': "✅ **Do this instead:** Consolidate duplicate functions",
                'example': (
                    "# Instead of having process_data() and processData() doing the same thing:\n"
                    "def process_data(input_data, validate=True):\n"
                    "    \"\"\"Single, well-documented function for data processing.\"\"\"\n"
                    "    if validate:\n"
                    "        validate_input(input_data)\n"
                    "    return transform(input_data)\n\n"
                    "# Use parameters to handle variations"
                ),
                'principle': "One function, parameterized for flexibility"
            },
            
            'circular_import': {
                'description': "✅ **Do this instead:** Use proper dependency hierarchy",
                'example': (
                    "# models/base.py - Shared base (no imports from app)\n"
                    "class BaseModel: ...\n\n"
                    "# models/user.py - Imports only from base\n"
                    "from .base import BaseModel\n"
                    "class User(BaseModel): ...\n\n"
                    "# app/service.py - Imports from models (not vice versa)\n"
                    "from models.user import User"
                ),
                'principle': "Clear dependency flow: shared → models → services → app"
            },
            
            'import_anxiety': {
                'description': "✅ **Do this instead:** Import only what you need",
                'example': (
                    "# Good: Specific imports\n"
                    "from datetime import datetime\n"
                    "from typing import List, Optional\n\n"
                    "# Bad: Wildcard and unused imports\n"
                    "# from datetime import *\n"
                    "# from typing import *  # when only using List"
                ),
                'principle': "Import specific items you actually use"
            },
            
            'phantom_function': {
                'description': "✅ **Do this instead:** Implement or remove",
                'example': (
                    "# Either implement the function:\n"
                    "def calculate_metrics(data):\n"
                    "    \"\"\"Calculate and return metrics.\"\"\"\n"
                    "    return {\n"
                    "        'mean': sum(data) / len(data),\n"
                    "        'max': max(data),\n"
                    "        'min': min(data)\n"
                    "    }\n\n"
                    "# Or remove it if not needed"
                ),
                'principle': "Only create functions you can implement"
            }
        }
        
        # Generate examples for detected patterns
        for pattern in pattern_groups:
            if pattern in corrections and len(examples) < self.max_positive_examples:
                examples.append(corrections[pattern])
        
        return examples
    
    def _generate_pattern_rules(
        self,
        pattern_groups: Dict[str, List[Issue]]
    ) -> List[str]:
        """Generate specific rules based on detected patterns."""
        rules = []
        
        # Add general rules first
        rules.extend(self.general_rules[:5])
        
        # Add pattern-specific rules
        for pattern in pattern_groups:
            if pattern in self.prompt_templates:
                rule = self.prompt_templates[pattern].get('rule')
                if rule and rule not in rules:
                    rules.append(rule)
        
        # Add severity-based rules
        high_severity_issues = [
            issue for issues in pattern_groups.values()
            for issue in issues if issue.severity >= 3
        ]
        
        if high_severity_issues:
            rules.insert(0, "🔴 CRITICAL: Fix high-severity issues before adding new features")
        
        return rules
    
    def _explain_why_bad(self, pattern: str, issue: Issue) -> str:
        """Explain why a pattern is problematic."""
        explanations = {
            'duplicate_function': "Creates maintenance burden and potential inconsistencies",
            'circular_import': "Can cause import errors and makes code hard to understand",
            'phantom_function': "Provides no functionality and can cause runtime errors",
            'hallucination_cascade': "Adds unnecessary complexity without solving real problems",
            'context_window_thrashing': "Wastes effort reimplementing existing functionality",
            'import_anxiety': "Clutters namespace and makes dependencies unclear",
            'semantic_duplicate': "Hidden duplication that's hard to maintain"
        }
        
        return explanations.get(pattern, "Creates technical debt and maintenance issues")
    
    def _humanize_pattern_name(self, pattern: str) -> str:
        """Convert pattern type to human-readable name."""
        names = {
            'duplicate_function': 'Duplicate Functions',
            'semantic_duplicate': 'Semantic Duplicates',
            'circular_import': 'Circular Imports',
            'phantom_function': 'Phantom Functions',
            'hallucination_cascade': 'Hallucination Cascades',
            'context_window_thrashing': 'Context Window Thrashing',
            'import_anxiety': 'Import Anxiety',
            'unknown': 'Unknown Pattern'
        }
        return names.get(pattern, pattern.replace('_', ' ').title())
    
    def generate_learning_summary(
        self,
        issues: List[Issue],
        improvements: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a learning summary for the LLM.
        
        Args:
            issues: Issues that were detected
            improvements: Improvements that were made
            
        Returns:
            Learning summary text
        """
        summary_parts = ["## 📚 Learning Summary\n"]
        
        if not issues and not improvements:
            summary_parts.append("No significant patterns to learn from in this session.")
            return "\n".join(summary_parts)
        
        # Summarize issues
        if issues:
            pattern_counts = defaultdict(int)
            for issue in issues:
                pattern = self._classify_pattern(issue)
                pattern_counts[pattern] += 1
            
            summary_parts.append("### Patterns to Avoid:")
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
                summary_parts.append(f"- {self._humanize_pattern_name(pattern)}: {count} instance(s)")
        
        # Summarize improvements
        if improvements:
            summary_parts.append("\n### Successful Improvements:")
            for improvement in improvements[:5]:
                summary_parts.append(f"- {improvement.get('description', 'Improvement made')}")
        
        # Add key learnings
        summary_parts.append("\n### Key Learnings:")
        learnings = self._extract_key_learnings(issues)
        for learning in learnings[:5]:
            summary_parts.append(f"- {learning}")
        
        return "\n".join(summary_parts)
    
    def _extract_key_learnings(self, issues: List[Issue]) -> List[str]:
        """Extract key learnings from issues."""
        learnings = []
        
        pattern_groups = self._group_issues_by_pattern(issues)
        
        if 'duplicate_function' in pattern_groups:
            learnings.append("Always search for existing functions before creating new ones")
        
        if 'circular_import' in pattern_groups:
            learnings.append("Maintain clear dependency hierarchies to avoid circular imports")
        
        if 'phantom_function' in pattern_groups:
            learnings.append("Implement functions completely or clearly mark intentional stubs")
        
        if 'hallucination_cascade' in pattern_groups:
            learnings.append("Start with simple solutions and avoid over-engineering")
        
        if 'context_window_thrashing' in pattern_groups:
            learnings.append("Review entire files before adding new implementations")
        
        if 'import_anxiety' in pattern_groups:
            learnings.append("Import only what you actually use in the code")
        
        return learnings