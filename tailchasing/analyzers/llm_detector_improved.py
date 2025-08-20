"""
Improved LLM artifact detection with context awareness.

This analyzer distinguishes between:
- Legitimate sequential documentation and actual LLM artifacts
- Intentional patterns vs suspicious uniformity
- Standard conventions vs generated boilerplate
"""

import ast
import re
from typing import List

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


class ImprovedLLMDetector(BaseAnalyzer):
    """Detects actual LLM-generated artifacts while avoiding false positives."""
    
    name = "llm_detector_improved"
    
    def __init__(self):
        super().__init__()
        # Legitimate patterns that should NOT be flagged
        self.legitimate_patterns = {
            # Configuration and setup sequences
            r'^\d+\.\s+(Install|Configure|Setup|Update|Initialize)',
            r'^\d+\.\s+(First|Second|Third|Next|Then|Finally)',
            # Standard documentation patterns  
            r'^(Step|Phase|Stage)\s+\d+:',
            r'^[A-Z]\.\s+',  # A. B. C. style lists
            # Command sequences
            r'^\$\s+',  # Shell commands
            r'^>>>\s+',  # Python REPL
            # Version histories
            r'^v?\d+\.\d+(\.\d+)?:?\s+',
            # TODO/FIXME/NOTE patterns
            r'^(TODO|FIXME|NOTE|WARNING|DEPRECATED):',
        }
        
        # Actual LLM indicators
        self.llm_indicators = {
            'repetitive_hedging': [
                r'(?i)(perhaps|maybe|possibly|potentially|could be|might be|seems to be|appears to be){3,}',
                r'(?i)(it (seems|appears) that|one could argue|it might be said){2,}',
            ],
            'excessive_politeness': [
                r'(?i)(please note that|kindly|would you mind|if you could){3,}',
                r'(?i)(I apologize|sorry for|excuse me|pardon){3,}',
            ],
            'meta_commentary': [
                r'(?i)as an AI|as a language model|I cannot|I am unable to',
                r'(?i)my training data|my knowledge cutoff',
            ],
            'filler_phrases': [
                r'(?i)(in general|basically|essentially|fundamentally){4,}',
                r'(?i)(to be honest|to be fair|to be clear){3,}',
            ],
            'uniform_structure': [
                # Exactly repeated sentence structures (not just numbered lists)
                r'(.{20,})\n\1\n\1',  # Same line repeated 3+ times
            ]
        }
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Detect LLM artifacts with improved accuracy."""
        issues = []
        
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
            
            # Check file-level patterns
            file_issues = self._check_file_patterns(file, tree, ctx)
            issues.extend(file_issues)
            
            # Check function and class level
            visitor = LLMPatternVisitor(file, self)
            visitor.visit(tree)
            issues.extend(visitor.issues)
        
        return issues
    
    def _check_file_patterns(self, file: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Check for file-level LLM patterns."""
        issues = []
        
        # Read the actual file content for pattern matching
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
        except:
            return issues
        
        # Check for legitimate patterns first (to exclude them)
        legitimate_lines = set()
        for i, line in enumerate(lines):
            for pattern in self.legitimate_patterns:
                if re.match(pattern, line.strip()):
                    legitimate_lines.add(i)
                    break
        
        # Now check for actual LLM indicators
        for indicator_type, patterns in self.llm_indicators.items():
            for pattern in patterns:
                # Check full content for pattern
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    # Get line number of match
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Skip if this is a legitimate pattern
                    if line_num - 1 in legitimate_lines:
                        continue
                    
                    # Skip if in comments or docstrings
                    if self._is_in_comment_or_docstring(tree, line_num):
                        continue
                    
                    issue = Issue(
                        kind=f"llm_artifact_{indicator_type}",
                        message=f"Possible LLM-generated content: {indicator_type.replace('_', ' ')}",
                        severity=1,  # Low severity
                        file=file,
                        line=line_num,
                        evidence={
                            "pattern": pattern,
                            "matched_text": match.group(0)[:100]
                        },
                        suggestions=[
                            "Review and rewrite in project style",
                            "Remove unnecessary hedging or filler",
                            "Simplify language to be more direct"
                        ],
                        confidence=0.6  # Lower confidence to reduce false positives
                    )
                    issues.append(issue)
        
        return issues
    
    def _is_in_comment_or_docstring(self, tree: ast.AST, line_num: int) -> bool:
        """Check if a line number is within a comment or docstring."""
        for node in ast.walk(tree):
            # Check docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring and hasattr(node, 'lineno'):
                    # Rough check - docstrings are usually in first few lines of function/class
                    if node.lineno <= line_num <= node.lineno + len(docstring.split('\n')) + 1:
                        return True
            
            # Check string literals that might be comments
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if hasattr(node, 'lineno') and node.lineno == line_num:
                    return True
        
        return False
    
    def is_legitimate_sequence(self, text: str) -> bool:
        """Check if text is a legitimate sequential pattern."""
        # Check against legitimate patterns
        for pattern in self.legitimate_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
        
        # Check for specific legitimate contexts
        legitimate_contexts = [
            'installation', 'setup', 'configuration', 'deployment',
            'migration', 'upgrade', 'changelog', 'release notes',
            'api documentation', 'user guide', 'tutorial'
        ]
        
        text_lower = text.lower()
        for context in legitimate_contexts:
            if context in text_lower:
                return True
        
        return False


class LLMPatternVisitor(ast.NodeVisitor):
    """Visit AST nodes looking for LLM patterns."""
    
    def __init__(self, file: str, detector: ImprovedLLMDetector):
        self.file = file
        self.detector = detector
        self.issues: List[Issue] = []
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function for LLM patterns."""
        self._check_function(node)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Check async function for LLM patterns."""
        self._check_function(node)
        self.generic_visit(node)
    
    def _check_function(self, node: ast.FunctionDef):
        """Check if function shows LLM patterns."""
        docstring = ast.get_docstring(node)
        if not docstring:
            return
        
        # Skip if it's a legitimate sequence
        if self.detector.is_legitimate_sequence(docstring):
            return
        
        # Check for excessive uniform structure in docstring
        lines = docstring.split('\n')
        if len(lines) > 5:
            # Check for suspiciously uniform line lengths
            line_lengths = [len(line.strip()) for line in lines if line.strip()]
            if line_lengths:
                avg_length = sum(line_lengths) / len(line_lengths)
                variance = sum((l - avg_length) ** 2 for l in line_lengths) / len(line_lengths)
                
                # Very low variance in line lengths is suspicious
                if variance < 25 and avg_length > 40:  # Lines are suspiciously uniform
                    issue = Issue(
                        kind="llm_artifact_uniform_docstring",
                        message=f"Suspiciously uniform docstring structure in {node.name}",
                        severity=1,
                        file=self.file,
                        line=safe_get_lineno(node),
                        symbol=node.name,
                        evidence={
                            "average_line_length": avg_length,
                            "variance": variance,
                            "line_count": len(lines)
                        },
                        suggestions=[
                            "Rewrite docstring with natural variation",
                            "Focus on actual function behavior, not generic descriptions"
                        ],
                        confidence=0.5  # Low confidence
                    )
                    self.issues.append(issue)