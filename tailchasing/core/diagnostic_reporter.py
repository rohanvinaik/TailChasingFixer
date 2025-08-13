"""
Diagnostic report generator for comprehensive issue analysis.

Generates human-readable reports, CSV files, and JSON databases
for any codebase analysis.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

from ..core.issues import Issue


class DiagnosticReporter:
    """Generates comprehensive diagnostic reports for analysis results."""
    
    def __init__(self, root_path: Path, output_dir: Optional[Path] = None):
        """
        Initialize the diagnostic reporter.
        
        Args:
            root_path: Root path of analyzed codebase
            output_dir: Output directory for reports (defaults to root_path)
        """
        self.root_path = root_path
        self.output_dir = output_dir or root_path
        self.timestamp = datetime.now().isoformat()
    
    def generate_all_reports(
        self,
        issues: List[Issue],
        risk_score: float,
        module_scores: Dict[str, float],
        fixable_issues: List[Dict[str, Any]],
        analysis_time: float = 0
    ) -> Dict[str, Path]:
        """
        Generate all diagnostic reports.
        
        Args:
            issues: List of detected issues
            risk_score: Global risk score
            module_scores: Risk scores by module
            fixable_issues: List of fixable issues
            analysis_time: Time taken for analysis
            
        Returns:
            Dict mapping report type to file path
        """
        reports = {}
        
        # Generate each report type
        reports['markdown'] = self._generate_markdown_report(
            issues, risk_score, module_scores, fixable_issues, analysis_time
        )
        reports['csv'] = self._generate_csv_report(issues)
        reports['json'] = self._generate_json_database(
            issues, risk_score, module_scores, fixable_issues, analysis_time
        )
        
        return reports
    
    def _generate_markdown_report(
        self,
        issues: List[Issue],
        risk_score: float,
        module_scores: Dict[str, float],
        fixable_issues: List[Dict[str, Any]],
        analysis_time: float
    ) -> Path:
        """Generate human-readable markdown report."""
        output_path = self.output_dir / "ISSUE_REPORT.md"
        
        # Categorize issues
        categories = self._categorize_issues(issues)
        
        # Calculate statistics
        total_issues = len(issues)
        fixable_count = len(fixable_issues)
        affected_modules = len(module_scores)
        
        # Build report content
        lines = [
            f"# Code Analysis Report - {self.root_path.name}",
            f"*Generated: {self.timestamp}*",
            "",
            "## Executive Summary",
            f"- **Total Issues Found**: {total_issues}",
            f"- **Fixable Issues**: {fixable_count} ({fixable_count/total_issues*100:.1f}%)",
            f"- **Risk Score**: {risk_score:.2f}",
            f"- **Affected Modules**: {affected_modules}",
            f"- **Analysis Time**: {analysis_time:.2f}s",
            "",
            "## Issues by Category",
            ""
        ]
        
        # Add categorized issues
        for category, cat_issues in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            lines.append(f"### {self._humanize_category(category)} ({len(cat_issues)} issues)")
            
            # Group by file
            by_file = defaultdict(list)
            for issue in cat_issues[:20]:  # Show top 20 per category
                by_file[issue.file].append(issue)
            
            for file, file_issues in sorted(by_file.items()):
                rel_path = Path(file).relative_to(self.root_path) if Path(file).is_absolute() else file
                lines.append(f"**{rel_path}**")
                for issue in file_issues[:5]:  # Show top 5 per file
                    lines.append(f"- Line {issue.line}: {issue.message}")
                if len(file_issues) > 5:
                    lines.append(f"- ... and {len(file_issues) - 5} more")
                lines.append("")
        
        # Add top affected files
        lines.extend([
            "## Most Affected Files",
            ""
        ])
        
        file_counts = defaultdict(int)
        for issue in issues:
            file_counts[issue.file] += 1
        
        for file, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            rel_path = Path(file).relative_to(self.root_path) if Path(file).is_absolute() else file
            lines.append(f"1. **{rel_path}**: {count} issues")
        
        # Add priority actions
        lines.extend([
            "",
            "## Priority Actions",
            "",
            "### High Priority",
            "1. Fix duplicate and placeholder functions",
            "2. Remove unused imports and clean up import structure",
            "3. Consolidate semantic duplicates",
            "",
            "### Medium Priority", 
            "1. Refactor context window thrashing patterns",
            "2. Implement missing function stubs",
            "3. Clean up LLM-generated filler content",
            "",
            "### Low Priority",
            "1. Improve documentation and docstrings",
            "2. Optimize code organization",
            "3. Add comprehensive test coverage",
            "",
            "## Next Steps",
            "",
            "1. Review this report to understand issue patterns",
            "2. Use `--generate-fixes` to create automated fix scripts",
            "3. Start with high-priority issues for maximum impact",
            "4. Run analysis regularly to track progress",
            "",
            "---",
            f"*Analysis performed on: {self.root_path}*",
            f"*TailChasingFixer Version: 1.0*"
        ])
        
        # Write report
        output_path.write_text("\n".join(lines))
        return output_path
    
    def _generate_csv_report(self, issues: List[Issue]) -> Path:
        """Generate CSV report for spreadsheet analysis."""
        output_path = self.output_dir / "DETAILED_ISSUES.csv"
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'File', 'Line', 'End_Line', 'Column', 'Function/Symbol',
                'Issue_Type', 'Severity', 'Confidence', 'Message', 
                'Suggestions', 'Evidence'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for issue in issues:
                # Make file path relative for readability
                file_path = issue.file
                try:
                    if Path(file_path).is_absolute():
                        file_path = str(Path(file_path).relative_to(self.root_path))
                except:
                    pass  # Keep absolute if relative conversion fails
                
                row = {
                    'File': file_path,
                    'Line': issue.line or 0,
                    'End_Line': issue.end_line or issue.line or 0,
                    'Column': issue.column or 0,
                    'Function/Symbol': issue.symbol or '',
                    'Issue_Type': issue.kind,
                    'Severity': issue.severity,
                    'Confidence': f"{issue.confidence:.2f}" if issue.confidence else '1.00',
                    'Message': issue.message,
                    'Suggestions': ' | '.join(issue.suggestions) if issue.suggestions else '',
                    'Evidence': json.dumps(issue.evidence) if issue.evidence else ''
                }
                writer.writerow(row)
        
        return output_path
    
    def _generate_json_database(
        self,
        issues: List[Issue],
        risk_score: float,
        module_scores: Dict[str, float],
        fixable_issues: List[Dict[str, Any]],
        analysis_time: float
    ) -> Path:
        """Generate comprehensive JSON database."""
        output_path = self.output_dir / "issues_database.json"
        
        # Categorize issues
        categories = self._categorize_issues(issues)
        
        # Build category statistics
        category_stats = {}
        for category, cat_issues in categories.items():
            subcats = defaultdict(int)
            for issue in cat_issues:
                subcats[issue.kind] += 1
            
            category_stats[category] = {
                'total': len(cat_issues),
                'subcategories': dict(subcats),
                'severity_distribution': self._get_severity_distribution(cat_issues)
            }
        
        # Build file statistics
        file_stats = defaultdict(lambda: {'count': 0, 'severity_sum': 0, 'types': set()})
        for issue in issues:
            file_stats[issue.file]['count'] += 1
            file_stats[issue.file]['severity_sum'] += issue.severity
            file_stats[issue.file]['types'].add(issue.kind)
        
        # Convert sets to lists for JSON serialization
        for file_data in file_stats.values():
            file_data['types'] = list(file_data['types'])
            file_data['avg_severity'] = file_data['severity_sum'] / file_data['count']
        
        # Build complete database
        database = {
            'metadata': {
                'analysis_timestamp': self.timestamp,
                'root_path': str(self.root_path),
                'total_issues': len(issues),
                'fixable_issues': len(fixable_issues),
                'risk_score': risk_score,
                'affected_modules': len(module_scores),
                'analysis_time_seconds': analysis_time
            },
            'statistics': {
                'issues_by_category': category_stats,
                'top_issue_types': self._get_top_issue_types(issues, 10),
                'severity_breakdown': self._get_severity_distribution(issues),
                'fixability_rate': len(fixable_issues) / len(issues) * 100 if issues else 0
            },
            'file_analysis': {
                file: stats for file, stats in 
                sorted(file_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:50]
            },
            'module_risk_scores': module_scores,
            'issues': [
                {
                    'file': issue.file,
                    'line': issue.line,
                    'end_line': issue.end_line,
                    'column': issue.column,
                    'symbol': issue.symbol,
                    'kind': issue.kind,
                    'message': issue.message,
                    'severity': issue.severity,
                    'confidence': issue.confidence,
                    'suggestions': issue.suggestions,
                    'evidence': issue.evidence,
                    'is_fixable': any(
                        fix.get('file') == issue.file and fix.get('line') == issue.line 
                        for fix in fixable_issues
                    )
                }
                for issue in sorted(issues, key=lambda x: (x.severity, x.confidence), reverse=True)[:1000]
            ]
        }
        
        # Write database
        with open(output_path, 'w') as f:
            json.dump(database, f, indent=2, default=str)
        
        return output_path
    
    def _categorize_issues(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Categorize issues by type patterns."""
        categories = defaultdict(list)
        
        for issue in issues:
            kind = issue.kind
            
            # Categorize based on patterns
            if 'duplicate' in kind or 'semantic' in kind:
                categories['duplicates'].append(issue)
            elif 'placeholder' in kind or 'phantom' in kind or 'stub' in kind:
                categories['placeholders'].append(issue)
            elif 'import' in kind or 'circular' in kind:
                categories['imports'].append(issue)
            elif 'llm' in kind or 'filler' in kind or 'generated' in kind:
                categories['llm_artifacts'].append(issue)
            elif 'thrashing' in kind or 'context' in kind:
                categories['context_issues'].append(issue)
            elif 'missing' in kind or 'undefined' in kind:
                categories['missing_symbols'].append(issue)
            elif 'coupling' in kind or 'complexity' in kind:
                categories['code_quality'].append(issue)
            else:
                categories['other'].append(issue)
        
        return dict(categories)
    
    def _humanize_category(self, category: str) -> str:
        """Convert category key to human-readable title."""
        mappings = {
            'duplicates': 'Duplicate & Semantic Duplicates',
            'placeholders': 'Placeholder & Stub Functions',
            'imports': 'Import & Dependency Issues',
            'llm_artifacts': 'LLM-Generated Artifacts',
            'context_issues': 'Context Window Thrashing',
            'missing_symbols': 'Missing & Undefined Symbols',
            'code_quality': 'Code Quality & Complexity',
            'other': 'Other Issues'
        }
        return mappings.get(category, category.replace('_', ' ').title())
    
    def _get_severity_distribution(self, issues: List[Issue]) -> Dict[str, int]:
        """Get distribution of issues by severity."""
        distribution = defaultdict(int)
        for issue in issues:
            if issue.severity >= 4:
                distribution['critical'] += 1
            elif issue.severity >= 3:
                distribution['high'] += 1
            elif issue.severity >= 2:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        return dict(distribution)
    
    def _get_top_issue_types(self, issues: List[Issue], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top issue types by count."""
        type_counts = defaultdict(int)
        for issue in issues:
            type_counts[issue.kind] += 1
        
        return [
            {'type': kind, 'count': count}
            for kind, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]