"""Report generation for tail-chasing analysis."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from .issues import Issue, IssueCollection
from .scoring import RiskScorer


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
        
        # Detailed issues
        lines.append("DETAILED ISSUES")
        lines.append("-" * 30)
        
        # Group by severity
        by_severity = defaultdict(list)
        for issue in issues:
            by_severity[issue.severity].append(issue)
            
        for severity in sorted(by_severity.keys(), reverse=True):
            severity_issues = by_severity[severity]
            lines.append(f"\nSeverity {severity} ({len(severity_issues)} issues):")
            
            for issue in severity_issues[:20]:  # Limit to first 20 per severity
                location = f"{issue.file}:{issue.line}" if issue.file else "global"
                lines.append(f"\n  [{issue.kind}] {location}")
                lines.append(f"  {issue.message}")
                
                if issue.suggestions:
                    lines.append("  Suggestions:")
                    for suggestion in issue.suggestions[:2]:
                        lines.append(f"    - {suggestion}")
                        
            if len(severity_issues) > 20:
                lines.append(f"\n  ... and {len(severity_issues) - 20} more")
                
        # Footer
        lines.append("")
        lines.append("=" * 60)
        lines.append("Run with --json for machine-readable output")
        
        return "\n".join(lines)
        
    def render_json(self, issues: List[Issue]) -> str:
        """Render a JSON report."""
        module_scores, global_score = self.scorer.calculate_scores(issues)
        
        report = {
            "metadata": {
                "version": "1.0",
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
            "issues": [issue.to_dict() for issue in issues]
        }
        
        return json.dumps(report, indent=2)
        
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
            
        # Issues
        html += "    <h2>Issues</h2>\n"
        
        for issue in sorted(issues, key=lambda i: -i.severity)[:50]:
            location = f"{issue.file}:{issue.line}" if issue.file else "global"
            html += f"""
    <div class="issue severity-{issue.severity}">
        <strong>[{issue.kind}]</strong> {location}<br>
        {issue.message}<br>
"""
            if issue.suggestions:
                html += "        <div class='suggestion'>Suggestions: "
                html += "; ".join(issue.suggestions[:2])
                html += "</div>\n"
            html += "    </div>\n"
            
        if len(issues) > 50:
            html += f"    <p><em>... and {len(issues) - 50} more issues</em></p>\n"
            
        html += """
</body>
</html>"""
        
        return html
