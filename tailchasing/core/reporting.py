"""Report generation for tail-chasing analysis."""

import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

from .issues import Issue
from .scoring import RiskScorer


class ReportFormatter:
    """Helper class for creating professional formatted reports."""
    
    @staticmethod
    def create_header(title: str, width: int = 60) -> List[str]:
        """Create a professional header with box-drawing characters."""
        lines = []
        # Calculate padding for centered title
        padding = (width - len(title) - 2) // 2
        padded_title = f"{' ' * padding}{title}{' ' * (width - len(title) - padding - 2)}"
        
        lines.append(f"╔{'═' * (width - 2)}╗")
        lines.append(f"║{padded_title}║")
        lines.append(f"╚{'═' * (width - 2)}╝")
        return lines
    
    @staticmethod
    def create_summary_box(stats: Dict[str, Any], risk_level: str, 
                          fixable_count: int = 0) -> List[str]:
        """Create a formatted summary box with emoji indicators."""
        lines = []
        
        # Risk level emoji and color
        risk_emoji = {
            "CRITICAL": "🔴",
            "WARNING": "⚠️",
            "OK": "✅"
        }.get(risk_level, "📊")
        
        lines.append("\n📊 SUMMARY")
        lines.append("├─ Files Analyzed: " + str(stats.get('files_analyzed', 0)))
        lines.append("├─ Issues Found: " + str(stats.get('total_issues', 0)))
        lines.append(f"├─ Risk Score: {stats.get('risk_score', 0):.2f} ({risk_emoji} {risk_level})")
        
        if fixable_count > 0:
            lines.append(f"├─ Fixable Issues: {fixable_count} ✨")
        
        lines.append(f"└─ Affected Modules: {stats.get('affected_modules', 0)}")
        
        return lines
    
    @staticmethod
    def format_severity_distribution(distribution: Dict[int, int]) -> List[str]:
        """Format severity distribution with emoji indicators."""
        lines = []
        lines.append("\n📈 SEVERITY DISTRIBUTION")
        
        severity_info = {
            4: ("🔴 Critical", "red"),
            3: ("🟠 High", "orange"),
            2: ("🟡 Medium", "yellow"),
            1: ("🟢 Low", "green")
        }
        
        for severity in sorted(distribution.keys(), reverse=True):
            count = distribution[severity]
            label, _ = severity_info.get(severity, (f"Level {severity}", "gray"))
            
            # Create a simple bar chart
            bar_length = min(30, count)
            bar = "█" * bar_length
            
            lines.append(f"  {label:15} {count:4} {bar}")
        
        return lines
    
    @staticmethod
    def format_module_risk_table(top_modules: List[Tuple[str, float]], limit: int = 10) -> List[str]:
        """Format top risk modules as a table."""
        lines = []
        lines.append("\n🎯 TOP RISK MODULES")
        lines.append("┌" + "─" * 58 + "┐")
        lines.append("│ " + "Score".ljust(10) + "│ " + "Module".ljust(45) + "│")
        lines.append("├" + "─" * 10 + "┼" + "─" * 47 + "┤")
        
        for module, score in top_modules[:limit]:
            # Truncate module name if too long
            if len(module) > 44:
                module = "..." + module[-41:]
            
            # Add risk indicator
            if score >= 50:
                indicator = "🔴"
            elif score >= 30:
                indicator = "🟠"
            elif score >= 20:
                indicator = "🟡"
            else:
                indicator = "🟢"
                
            lines.append(f"│ {indicator} {score:6.1f} │ {module:45} │")
        
        lines.append("└" + "─" * 10 + "┴" + "─" * 47 + "┘")
        return lines


class PathSanitizer:
    """Utility for sanitizing file paths for privacy and readability."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize path sanitizer.
        
        Args:
            project_root: The project root directory. If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        # Store both resolved and non-resolved versions for flexibility
        self.project_root_resolved = self.project_root.resolve()
        
    def sanitize(self, file_path: Optional[str]) -> str:
        """
        Sanitize a file path to show relative path from project root.
        
        Args:
            file_path: The file path to sanitize (can be absolute or relative)
            
        Returns:
            Sanitized relative path like ./src/module.py
        """
        if not file_path:
            return ""
            
        try:
            path = Path(file_path)
            
            # If it's already relative and doesn't go outside project, keep it
            if not path.is_absolute():
                # Make sure it starts with ./ for consistency
                path_str = str(path)
                if not path_str.startswith('./') and not path_str.startswith('../'):
                    return f"./{path_str}"
                return path_str
            
            # Try to make it relative to project root
            try:
                # Try with resolved path first
                rel_path = path.relative_to(self.project_root_resolved)
            except ValueError:
                try:
                    # Try with non-resolved path
                    rel_path = path.relative_to(self.project_root)
                except ValueError:
                    # Path is outside project root - sanitize it for privacy
                    # Replace home directory with ~
                    path_str = str(path)
                    home = str(Path.home())
                    if path_str.startswith(home):
                        path_str = "~" + path_str[len(home):]
                    
                    # Further sanitize by only showing last few components
                    parts = Path(path_str).parts
                    if len(parts) > 3:
                        return f".../{'/'.join(parts[-3:])}"
                    return path_str
            
            # Format relative path with ./ prefix
            return f"./{rel_path}"
            
        except Exception:
            # If anything goes wrong, return the original but with home replaced
            path_str = str(file_path)
            home = str(Path.home())
            if path_str.startswith(home):
                return "~" + path_str[len(home):]
            return path_str
    
    def sanitize_issue(self, issue: Issue) -> Issue:
        """
        Create a copy of an issue with sanitized file path.
        
        Args:
            issue: The issue to sanitize
            
        Returns:
            New issue with sanitized path
        """
        if issue.file:
            # Create a new issue with sanitized path
            sanitized = Issue(
                kind=issue.kind,
                message=issue.message,
                severity=issue.severity,
                file=self.sanitize(issue.file),
                line=issue.line,
                end_line=issue.end_line,
                confidence=issue.confidence,
                evidence=issue.evidence,
                suggestions=issue.suggestions
            )
            return sanitized
        return issue


class IssueExplainer:
    """Provides detailed explanations and fixes for each issue type."""
    
    ISSUE_DETAILS = {
        'duplicate_function': {
            'problem': 'Identical function implementation found',
            'impact': 'Code duplication increases maintenance burden, can lead to inconsistent bug fixes, and wastes developer time',
            'fix': 'Extract common functionality to a shared module or base class'
        },
        'semantic_duplicate_function': {
            'problem': 'Semantically similar function detected',
            'impact': 'Multiple implementations of the same logic can diverge over time, creating subtle bugs and confusion',
            'fix': 'Consolidate similar functions into a single parameterized implementation'
        },
        'circular_import': {
            'problem': 'Circular dependency between modules',
            'impact': 'Can cause import failures, makes code harder to test and refactor, violates architectural boundaries',
            'fix': 'Extract shared code to a separate module or use dependency injection'
        },
        'missing_symbol': {
            'problem': 'Reference to undefined function or variable',
            'impact': 'Will cause runtime errors when code path is executed, indicates incomplete implementation',
            'fix': 'Implement the missing function/variable or remove the reference'
        },
        'phantom_function': {
            'problem': 'Function contains only placeholder implementation',
            'impact': 'Silently fails or returns incorrect results, creates false sense of completeness',
            'fix': 'Implement the actual logic or explicitly raise NotImplementedError'
        },
        'hallucination_cascade': {
            'problem': 'Fictional or inconsistent subsystem detected',
            'impact': 'Entire code sections may be non-functional, compounds errors across multiple files',
            'fix': 'Review and reimplement the entire subsystem with correct architecture'
        },
        'context_window_thrashing': {
            'problem': 'Function reimplemented multiple times',
            'impact': 'Wastes context window in LLM interactions, creates inconsistent implementations',
            'fix': 'Use the most complete implementation and remove duplicates'
        },
        'import_anxiety': {
            'problem': 'Excessive or unused imports detected',
            'impact': 'Slows down module loading, indicates confusion about dependencies, may hide circular dependencies',
            'fix': 'Remove unused imports and organize remaining imports logically'
        },
        'function_coupling_risk': {
            'problem': 'High coupling detected between functions',
            'impact': 'Changes to one function likely break the other, makes refactoring difficult',
            'fix': 'Reduce coupling through interface segregation or dependency injection'
        },
        'llm_filler_text': {
            'problem': 'Lorem ipsum or synthetic placeholder text detected',
            'impact': 'Placeholder text in production code indicates incomplete implementation',
            'fix': 'Replace with real data or remove if not needed'
        },
        'llm_filler_sequence': {
            'problem': 'Sequential placeholder data pattern detected (item1, item2, etc)',
            'impact': 'Generic test data may not represent real-world scenarios',
            'fix': 'Use realistic data that matches actual use cases'
        },
        'llm_filler_dict': {
            'problem': 'Dictionary with generic placeholder keys/values',
            'impact': 'Placeholder data structures indicate incomplete implementation',
            'fix': 'Define proper data models with meaningful field names'
        },
        'llm_filler_docstring': {
            'problem': 'Generic placeholder documentation or TODO',
            'impact': 'Poor documentation makes code harder to understand and maintain',
            'fix': 'Write specific, meaningful documentation'
        },
        'llm_filler_json': {
            'problem': 'JSON data contains placeholder content',
            'impact': 'Mock data in production can cause unexpected behavior',
            'fix': 'Replace with actual configuration or data'
        },
        'wrapper_abstraction': {
            'problem': 'Unnecessary wrapper function that adds no value',
            'impact': 'Adds complexity without benefit, makes debugging harder',
            'fix': 'Remove the wrapper and call the underlying function directly'
        },
        'prototype_fragmentation': {
            'problem': 'Multiple incomplete implementations of the same concept',
            'impact': 'Confusion about which implementation to use, wasted effort on parallel development',
            'fix': 'Choose the best implementation and remove others'
        },
        'cargo_cult': {
            'problem': 'Code copied without understanding its purpose',
            'impact': 'May not work correctly in new context, perpetuates misunderstandings',
            'fix': 'Understand the code\'s purpose and rewrite appropriately for your use case'
        },
        'tdd_antipattern': {
            'problem': 'Test-driven development antipattern detected',
            'impact': 'Tests may not actually verify functionality, gives false confidence',
            'fix': 'Rewrite tests to properly verify behavior, not just structure'
        }
    }
    
    @classmethod
    def get_explanation(cls, issue_kind: str) -> Dict[str, str]:
        """Get detailed explanation for an issue type."""
        # Handle issue kinds with suffixes (e.g., 'duplicate_function_risk')
        base_kind = issue_kind
        for known_kind in cls.ISSUE_DETAILS:
            if issue_kind.startswith(known_kind):
                base_kind = known_kind
                break
                
        return cls.ISSUE_DETAILS.get(base_kind, {
            'problem': f'{issue_kind} detected in code',
            'impact': 'May affect code quality and maintainability',
            'fix': 'Review the code and apply appropriate refactoring'
        })


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def format_issue_for_console(issue: Issue, sanitizer: Optional[PathSanitizer] = None) -> str:
    """Format a single issue for console output with explanation.
    
    Args:
        issue: The issue to format
        sanitizer: Optional path sanitizer for privacy
        
    Returns:
        Formatted string for console output
    """
    if sanitizer:
        issue = sanitizer.sanitize_issue(issue)
    
    location = f"{issue.file}:{issue.line}" if issue.file else "global"
    explanation = IssueExplainer.get_explanation(issue.kind)
    
    lines = []
    lines.append(f"[{issue.kind}] {location}")
    lines.append(f"  Problem: {explanation['problem']}")
    lines.append(f"  Details: {issue.message}")
    lines.append(f"  Impact:  {explanation['impact']}")
    lines.append(f"  Fix:     {explanation['fix']}")
    
    if issue.suggestions:
        lines.append("  Specific suggestions:")
        for suggestion in issue.suggestions[:2]:
            lines.append(f"    • {suggestion}")
    
    return "\n".join(lines)


class Reporter:
    """Generate reports in various formats."""
    
    def __init__(self, config: Dict[str, Any], project_root: Optional[Path] = None):
        self.config = config
        self.scorer = RiskScorer(config.get("scoring_weights"))
        self.sanitizer = PathSanitizer(project_root)
        
    def generate_reports(
        self, 
        issues: List[Issue], 
        output_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """Generate reports in all configured formats.
        
        Returns:
            Dict mapping format to output content/path
        """
        output_dir = output_dir or Path(self.config.get("report", {}).get("output_dir", "."))
        formats = self.config.get("report", {}).get("formats", ["text"])
        
        # Always save text report to file if there are many issues
        if len(issues) > 50 and "text" not in formats:
            formats.append("text")
        
        results = {}
        
        for fmt in formats:
            if fmt == "text":
                content = self.render_text(issues)
                results["text"] = content
                
                # Also save to file if output_dir specified
                if output_dir != Path("."):
                    path = output_dir / "tailchasing_report.txt"
                    path.write_text(content)
                    results["text_file"] = str(path)
                    
            elif fmt == "json":
                content = self.render_json(issues)
                results["json"] = content
                
                if output_dir != Path("."):
                    path = output_dir / "tailchasing_report.json"
                    path.write_text(content)
                    results["json_file"] = str(path)
                    
            elif fmt == "html":
                content = self.render_html(issues)
                results["html"] = content
                
                if output_dir != Path("."):
                    path = output_dir / "tailchasing_report.html"
                    path.write_text(content)
                    results["html_file"] = str(path)
                    
        return results
        
    def render_text(self, issues: List[Issue], files_analyzed: int = 0) -> str:
        """Render a professionally formatted text report.
        
        Args:
            issues: List of issues found
            files_analyzed: Number of files analyzed (optional)
            
        Returns:
            Formatted text report
        """
        lines = []
        
        # Professional header with box drawing
        lines.extend(ReportFormatter.create_header("TAIL-CHASING ANALYSIS REPORT"))
        lines.append(f"\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate scores
        module_scores, global_score = self.scorer.calculate_scores(issues)
        risk_level = self.scorer.get_risk_level(
            global_score, 
            self.config.get("risk_thresholds", {})
        )
        
        # Count fixable issues
        fixable_types = {
            'duplicate_function', 'semantic_duplicate_function', 
            'phantom_function', 'missing_symbol', 'circular_import',
            'wrapper_abstraction', 'import_anxiety'
        }
        fixable_count = sum(1 for issue in issues if issue.kind in fixable_types)
        
        # Professional summary box
        summary_stats = {
            'files_analyzed': files_analyzed,
            'total_issues': len(issues),
            'risk_score': global_score,
            'affected_modules': len(module_scores)
        }
        lines.extend(ReportFormatter.create_summary_box(summary_stats, risk_level, fixable_count))
        
        # Severity distribution with emoji and bars
        severity_dist = defaultdict(int)
        for issue in issues:
            severity_dist[issue.severity] += 1
        
        if severity_dist:
            lines.extend(ReportFormatter.format_severity_distribution(severity_dist))
        
        # Group issues by category with severity breakdown
        lines.append("\n🔍 ISSUES BY CATEGORY")
        
        # Define issue categories
        categories = {
            'Coupling Problems': ['circular_import', 'function_coupling_risk', 'import_anxiety'],
            'Duplicate Functions': ['duplicate_function', 'semantic_duplicate_function', 'context_window_thrashing'],
            'Phantom Functions': ['phantom_function', 'phantom_stub_triage', 'missing_symbol'],
            'Code Quality': ['hallucination_cascade', 'wrapper_abstraction', 'prototype_fragmentation', 'cargo_cult', 'tdd_antipattern'],
            'LLM Filler Data': ['llm_filler_text', 'llm_filler_sequence', 'llm_filler_dict', 'llm_filler_docstring', 'llm_filler_json']
        }
        
        distribution = self.scorer.get_issue_distribution(issues)
        
        for category_name, issue_types in categories.items():
            # Count issues in this category
            category_issues = []
            for issue in issues:
                if issue.kind in issue_types:
                    category_issues.append(issue)
            
            if category_issues:
                # Count by severity
                severity_counts = defaultdict(int)
                for issue in category_issues:
                    severity_counts[issue.severity] += 1
                
                # Format the category line
                total = len(category_issues)
                lines.append(f"├─ {category_name} ({total} issues)")
                
                # Show severity breakdown
                if severity_counts.get(4, 0) > 0:  # Critical
                    lines.append(f"│  ├─ 🔴 Critical: {severity_counts[4]} issues")
                if severity_counts.get(3, 0) > 0:  # High
                    lines.append(f"│  ├─ 🟠 High: {severity_counts[3]} issues")
                if severity_counts.get(2, 0) > 0:  # Medium
                    lines.append(f"│  ├─ 🟡 Medium: {severity_counts[2]} issues")
                if severity_counts.get(1, 0) > 0:  # Low
                    lines.append(f"│  └─ 🟢 Low: {severity_counts[1]} issues")
        
        # Add any uncategorized issues
        categorized_types = set()
        for types in categories.values():
            categorized_types.update(types)
        
        uncategorized = []
        for issue in issues:
            if issue.kind not in categorized_types:
                uncategorized.append(issue)
        
        if uncategorized:
            lines.append(f"└─ Other Issues ({len(uncategorized)} issues)")
        
        # Top risky modules with formatted table
        top_modules = self.scorer.get_top_modules(module_scores, limit=10)
        if top_modules:
            # Sanitize module paths for the table
            sanitized_modules = []
            for module, score in top_modules:
                sanitized_path = self.sanitizer.sanitize(module)
                sanitized_modules.append((sanitized_path, score))
            lines.extend(ReportFormatter.format_module_risk_table(sanitized_modules))
        
        # Detailed issues section with emoji headers
        lines.append("\n\n🔍 DETAILED ISSUES")
        lines.append("─" * 60)
        
        # Group by severity
        by_severity = defaultdict(list)
        for issue in issues:
            by_severity[issue.severity].append(issue)
            
        severity_info = {
            4: ("🔴 CRITICAL", "═"),
            3: ("🟠 HIGH", "─"),
            2: ("🟡 MEDIUM", "─"),
            1: ("🟢 LOW", "·")
        }
        
        for severity in sorted(by_severity.keys(), reverse=True):
            severity_issues = by_severity[severity]
            label, separator = severity_info.get(severity, (f"SEVERITY-{severity}", "─"))
            
            lines.append(f"\n{label} ({len(severity_issues)} issues)")
            lines.append(separator * 50)
            
            for issue in severity_issues[:20]:  # Limit to first 20 per severity
                # Sanitize the issue paths
                sanitized_issue = self.sanitizer.sanitize_issue(issue)
                location = f"{sanitized_issue.file}:{sanitized_issue.line}" if sanitized_issue.file else "global"
                explanation = IssueExplainer.get_explanation(issue.kind)
                
                # Main issue header
                lines.append(f"\n[{issue.kind}] {location}")
                
                # Enhanced formatting with problem, impact, and fix
                lines.append(f"  Problem: {explanation['problem']}")
                lines.append(f"  Details: {issue.message}")
                lines.append(f"  Impact:  {explanation['impact']}")
                lines.append(f"  Fix:     {explanation['fix']}")
                
                # Add specific suggestions if available
                if issue.suggestions:
                    lines.append("  Specific suggestions:")
                    for suggestion in issue.suggestions[:2]:
                        lines.append(f"    • {suggestion}")
                        
            if len(severity_issues) > 20:
                lines.append(f"\n  ... and {len(severity_issues) - 20} more {label} issues")
                
        # Footer
        lines.append("")
        lines.append("=" * 60)
        lines.append("Run with --json for machine-readable output")
        
        return "\n".join(lines)
        
    def render_json(self, issues: List[Issue]) -> str:
        """Render a JSON report with enhanced issue details."""
        module_scores, global_score = self.scorer.calculate_scores(issues)
        
        # Enhance each issue with explanations and sanitized paths
        enhanced_issues = []
        for issue in issues:
            # Sanitize the issue path first
            sanitized_issue = self.sanitizer.sanitize_issue(issue)
            issue_dict = sanitized_issue.to_dict()
            explanation = IssueExplainer.get_explanation(issue.kind)
            issue_dict['explanation'] = {
                'problem': explanation['problem'],
                'impact': explanation['impact'],
                'fix': explanation['fix']
            }
            enhanced_issues.append(issue_dict)
        
        report = {
            "metadata": {
                "version": "1.1",
                "generated": datetime.now().isoformat(),
                "tool": "tail-chasing-detector"
            },
            "summary": {
                "total_issues": len(issues),
                "global_score": global_score,
                "risk_level": self.scorer.get_risk_level(
                    global_score,
                    self.config.get("risk_thresholds", {})
                ),
                "affected_modules": len(module_scores)
            },
            "distribution": self.scorer.get_issue_distribution(issues),
            "module_scores": module_scores,
            "issues": enhanced_issues
        }
        
        return json.dumps(report, indent=2, cls=NumpyJSONEncoder)
        
    def render_html(self, issues: List[Issue]) -> str:
        """Render an HTML report."""
        module_scores, global_score = self.scorer.calculate_scores(issues)
        risk_level = self.scorer.get_risk_level(
            global_score,
            self.config.get("risk_thresholds", {})
        )
        
        # Basic HTML template
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tail-Chasing Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .risk-critical {{ color: #d32f2f; font-weight: bold; }}
        .risk-warning {{ color: #f57c00; font-weight: bold; }}
        .risk-ok {{ color: #388e3c; font-weight: bold; }}
        .issue {{ border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .severity-4 {{ border-left: 5px solid #d32f2f; }}
        .severity-3 {{ border-left: 5px solid #f57c00; }}
        .severity-2 {{ border-left: 5px solid #fbc02d; }}
        .severity-1 {{ border-left: 5px solid #689f38; }}
        .suggestion {{ font-style: italic; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f0f0f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Tail-Chasing Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Issues: <strong>{len(issues)}</strong></p>
        <p>Global Risk Score: <span class="risk-{risk_level.lower()}">{global_score}</span></p>
        <p>Risk Level: <span class="risk-{risk_level.lower()}">{risk_level}</span></p>
    </div>
"""
        
        # Issue distribution
        distribution = self.scorer.get_issue_distribution(issues)
        if distribution:
            html += """
    <h2>Issue Distribution</h2>
    <table>
        <tr><th>Issue Type</th><th>Count</th></tr>
"""
            for issue_type, count in sorted(distribution.items(), key=lambda x: -x[1]):
                html += f"        <tr><td>{issue_type}</td><td>{count}</td></tr>\n"
            html += "    </table>\n"
            
        # Top modules
        top_modules = self.scorer.get_top_modules(module_scores, limit=10)
        if top_modules:
            html += """
    <h2>Top Risk Modules</h2>
    <table>
        <tr><th>Module</th><th>Risk Score</th></tr>
"""
            for module, score in top_modules:
                html += f"        <tr><td>{module}</td><td>{score:.1f}</td></tr>\n"
            html += "    </table>\n"
            
        # Issues with enhanced formatting
        html += "    <h2>Issues</h2>\n"
        
        for issue in sorted(issues, key=lambda i: -i.severity)[:50]:
            # Sanitize paths for privacy
            sanitized_issue = self.sanitizer.sanitize_issue(issue)
            location = f"{sanitized_issue.file}:{sanitized_issue.line}" if sanitized_issue.file else "global"
            explanation = IssueExplainer.get_explanation(issue.kind)
            severity_label = {
                4: "CRITICAL", 3: "HIGH", 2: "MEDIUM", 1: "LOW"
            }.get(issue.severity, f"SEV-{issue.severity}")
            
            html += f"""
    <div class="issue severity-{issue.severity}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong>[{issue.kind}]</strong> {location}
            <span class="severity-label" style="padding: 2px 8px; background: #eee; border-radius: 3px;">{severity_label}</span>
        </div>
        <div style="margin-top: 10px;">
            <div><strong>Problem:</strong> {explanation['problem']}</div>
            <div><strong>Details:</strong> {issue.message}</div>
            <div><strong>Impact:</strong> <span style="color: #d32f2f;">{explanation['impact']}</span></div>
            <div><strong>Fix:</strong> <span style="color: #388e3c;">{explanation['fix']}</span></div>
"""
            if issue.suggestions:
                html += "            <div style='margin-top: 5px;'><strong>Specific suggestions:</strong><ul style='margin: 5px 0;'>"
                for suggestion in issue.suggestions[:2]:
                    html += f"<li>{suggestion}</li>"
                html += "</ul></div>\n"
            html += "        </div>\n    </div>\n"
            
        if len(issues) > 50:
            html += f"    <p><em>... and {len(issues) - 50} more issues</em></p>\n"
            
        html += """
</body>
</html>"""
        
        return html
