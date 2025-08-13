"""Report generation for tail-chasing analysis."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

from .issues import Issue, IssueCollection
from .scoring import RiskScorer


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


def format_issue_for_console(issue: Issue) -> str:
    """Format a single issue for console output with explanation."""
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
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scorer = RiskScorer(config.get("scoring_weights"))
        
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
        
    def render_text(self, issues: List[Issue]) -> str:
        """Render a text report."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("TAIL-CHASING ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Calculate scores
        module_scores, global_score = self.scorer.calculate_scores(issues)
        risk_level = self.scorer.get_risk_level(
            global_score, 
            self.config.get("risk_thresholds", {})
        )
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 30)
        lines.append(f"Total Issues: {len(issues)}")
        lines.append(f"Global Risk Score: {global_score} ({risk_level})")
        lines.append(f"Affected Modules: {len(module_scores)}")
        lines.append("")
        
        # Issue distribution
        distribution = self.scorer.get_issue_distribution(issues)
        if distribution:
            lines.append("ISSUE DISTRIBUTION")
            lines.append("-" * 30)
            for issue_type, count in sorted(distribution.items(), key=lambda x: -x[1]):
                lines.append(f"  {issue_type}: {count}")
            lines.append("")
        
        # Top risky modules
        top_modules = self.scorer.get_top_modules(module_scores, limit=10)
        if top_modules:
            lines.append("TOP RISK MODULES")
            lines.append("-" * 30)
            for module, score in top_modules:
                lines.append(f"  {score:6.1f} - {module}")
            lines.append("")
        
        # Detailed issues with enhanced formatting
        lines.append("DETAILED ISSUES")
        lines.append("-" * 30)
        
        # Group by severity
        by_severity = defaultdict(list)
        for issue in issues:
            by_severity[issue.severity].append(issue)
            
        for severity in sorted(by_severity.keys(), reverse=True):
            severity_issues = by_severity[severity]
            severity_label = {
                4: "CRITICAL",
                3: "HIGH",
                2: "MEDIUM",
                1: "LOW"
            }.get(severity, f"SEVERITY-{severity}")
            
            lines.append(f"\n{severity_label} ({len(severity_issues)} issues):")
            lines.append("=" * 50)
            
            for issue in severity_issues[:20]:  # Limit to first 20 per severity
                location = f"{issue.file}:{issue.line}" if issue.file else "global"
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
                lines.append(f"\n  ... and {len(severity_issues) - 20} more {severity_label} issues")
                
        # Footer
        lines.append("")
        lines.append("=" * 60)
        lines.append("Run with --json for machine-readable output")
        
        return "\n".join(lines)
        
    def render_json(self, issues: List[Issue]) -> str:
        """Render a JSON report with enhanced issue details."""
        module_scores, global_score = self.scorer.calculate_scores(issues)
        
        # Enhance each issue with explanations
        enhanced_issues = []
        for issue in issues:
            issue_dict = issue.to_dict()
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
            location = f"{issue.file}:{issue.line}" if issue.file else "global"
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
