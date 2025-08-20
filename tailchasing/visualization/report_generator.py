"""
Comprehensive report generation for tail-chasing analysis.

Generates HTML reports with embedded visualizations, JSON for programmatic access,
Markdown summaries, and GraphML for network analysis tools.
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import base64
import logging

from .tail_chase_visualizer import TailChaseVisualizer
from ..core.issues import Issue

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section in the report."""
    
    title: str
    content: str
    section_type: str = 'text'  # 'text', 'visualization', 'table', 'code'
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher priority sections appear first


@dataclass
class ReportStatistics:
    """Statistics for the report."""
    
    total_issues: int = 0
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    
    total_files: int = 0
    affected_files: int = 0
    
    # Pattern counts
    duplicate_functions: int = 0
    circular_imports: int = 0
    phantom_functions: int = 0
    hallucination_cascades: int = 0
    context_thrashing: int = 0
    import_anxiety: int = 0
    
    # Risk metrics
    overall_risk_score: float = 0.0
    max_risk_score: float = 0.0
    avg_risk_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ReportGenerator:
    """
    Generate comprehensive reports for tail-chasing analysis.
    
    Supports multiple output formats including HTML, JSON, Markdown, and GraphML.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize report generator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.issues: List[Issue] = []
        self.sections: List[ReportSection] = []
        self.statistics = ReportStatistics()
        self.visualizer = TailChaseVisualizer(config)
        self.metadata: Dict[str, Any] = {}
    
    def add_issues(self, issues: List[Issue]) -> None:
        """Add issues to the report."""
        self.issues.extend(issues)
        self._update_statistics()
    
    def add_section(
        self,
        title: str,
        content: str,
        section_type: str = 'text',
        priority: int = 0,
        **kwargs
    ) -> None:
        """Add a custom section to the report."""
        section = ReportSection(
            title=title,
            content=content,
            section_type=section_type,
            priority=priority,
            metadata=kwargs
        )
        self.sections.append(section)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set report metadata."""
        self.metadata[key] = value
    
    def _update_statistics(self) -> None:
        """Update statistics based on current issues."""
        self.statistics = ReportStatistics()
        
        if not self.issues:
            return
        
        self.statistics.total_issues = len(self.issues)
        
        # Count by severity
        for issue in self.issues:
            if issue.severity >= 4:
                self.statistics.critical_issues += 1
            elif issue.severity == 3:
                self.statistics.high_issues += 1
            elif issue.severity == 2:
                self.statistics.medium_issues += 1
            else:
                self.statistics.low_issues += 1
        
        # Count by pattern type
        pattern_counts = {
            'duplicate': 0,
            'circular': 0,
            'phantom': 0,
            'hallucination': 0,
            'context': 0,
            'import': 0
        }
        
        for issue in self.issues:
            kind = issue.kind.lower()
            if 'duplicate' in kind:
                pattern_counts['duplicate'] += 1
            elif 'circular' in kind:
                pattern_counts['circular'] += 1
            elif 'phantom' in kind:
                pattern_counts['phantom'] += 1
            elif 'hallucination' in kind:
                pattern_counts['hallucination'] += 1
            elif 'context' in kind or 'thrashing' in kind:
                pattern_counts['context'] += 1
            elif 'import' in kind or 'anxiety' in kind:
                pattern_counts['import'] += 1
        
        self.statistics.duplicate_functions = pattern_counts['duplicate']
        self.statistics.circular_imports = pattern_counts['circular']
        self.statistics.phantom_functions = pattern_counts['phantom']
        self.statistics.hallucination_cascades = pattern_counts['hallucination']
        self.statistics.context_thrashing = pattern_counts['context']
        self.statistics.import_anxiety = pattern_counts['import']
        
        # File statistics
        affected_files = set()
        risk_scores = []
        
        for issue in self.issues:
            if issue.file:
                affected_files.add(issue.file)
            if hasattr(issue, 'confidence'):
                risk_scores.append(issue.confidence)
        
        self.statistics.affected_files = len(affected_files)
        
        # Risk metrics
        if risk_scores:
            self.statistics.overall_risk_score = sum(risk_scores)
            self.statistics.max_risk_score = max(risk_scores)
            self.statistics.avg_risk_score = sum(risk_scores) / len(risk_scores)
    
    def generate_html_report(
        self,
        output_path: Optional[Path] = None,
        include_visualizations: bool = True,
        embed_data: bool = True
    ) -> str:
        """
        Generate comprehensive HTML report with embedded visualizations.
        
        Args:
            output_path: Optional path to save the report
            include_visualizations: Include interactive visualizations
            embed_data: Embed raw data for interactive exploration
            
        Returns:
            HTML string
        """
        timestamp = datetime.now().isoformat()
        
        # Build dependency graph if we have issues
        if include_visualizations and self.issues:
            self._build_dependency_graph()
        
        # Generate visualizations
        viz_sections = []
        if include_visualizations:
            # Dependency graph
            dep_graph = self.visualizer.generate_dependency_graph(
                title="Issue Dependency Graph"
            )
            viz_sections.append(('Dependency Graph', dep_graph, 'full'))
            
            # Similarity heatmap if available
            if self.visualizer.similarity_matrix is not None:
                heatmap = self.visualizer.generate_similarity_heatmap(
                    title="Semantic Similarity Heatmap"
                )
                viz_sections.append(('Similarity Heatmap', heatmap, 'full'))
            
            # Temporal animation if available
            if self.visualizer.temporal_data:
                temporal = self.visualizer.generate_temporal_animation(
                    title="Temporal Evolution"
                )
                viz_sections.append(('Temporal Evolution', temporal, 'full'))
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Tail-Chasing Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f7fa;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 0;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        header .subtitle {{
            opacity: 0.9;
            font-size: 1.1em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .stat-card.critical {{
            border-left: 4px solid #f44336;
        }}
        
        .stat-card.high {{
            border-left: 4px solid #ff9800;
        }}
        
        .stat-card.medium {{
            border-left: 4px solid #ffc107;
        }}
        
        .stat-card.low {{
            border-left: 4px solid #4caf50;
        }}
        
        .section {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        .issue-list {{
            list-style: none;
        }}
        
        .issue-item {{
            padding: 15px;
            margin-bottom: 10px;
            border-left: 4px solid #ddd;
            background: #f9f9f9;
            border-radius: 4px;
            transition: all 0.3s;
        }}
        
        .issue-item:hover {{
            background: #f5f5f5;
            transform: translateX(5px);
        }}
        
        .issue-item.severity-4 {{
            border-left-color: #f44336;
        }}
        
        .issue-item.severity-3 {{
            border-left-color: #ff9800;
        }}
        
        .issue-item.severity-2 {{
            border-left-color: #ffc107;
        }}
        
        .issue-item.severity-1 {{
            border-left-color: #4caf50;
        }}
        
        .issue-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .issue-title {{
            font-weight: bold;
            color: #333;
        }}
        
        .issue-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .badge-critical {{
            background: #f44336;
            color: white;
        }}
        
        .badge-high {{
            background: #ff9800;
            color: white;
        }}
        
        .badge-medium {{
            background: #ffc107;
            color: #333;
        }}
        
        .badge-low {{
            background: #4caf50;
            color: white;
        }}
        
        .issue-location {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }}
        
        .issue-message {{
            color: #555;
            margin-bottom: 10px;
        }}
        
        .issue-suggestions {{
            background: #e8f5e9;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
        }}
        
        .issue-suggestions h4 {{
            color: #4caf50;
            margin-bottom: 5px;
            font-size: 0.9em;
        }}
        
        .issue-suggestions ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .issue-suggestions li {{
            padding: 3px 0;
            font-size: 0.9em;
            color: #555;
        }}
        
        .issue-suggestions li:before {{
            content: "→ ";
            color: #4caf50;
            font-weight: bold;
        }}
        
        .visualization-container {{
            margin: 20px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .visualization-container iframe {{
            width: 100%;
            height: 600px;
            border: none;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #f0f0f0;
        }}
        
        .tab {{
            padding: 10px 20px;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1em;
            color: #666;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
        }}
        
        .tab:hover {{
            color: #333;
        }}
        
        .tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }}
        
        .summary-card h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px 0;
            color: #666;
            font-size: 0.9em;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            header h1 {{
                font-size: 1.8em;
            }}
            
            .container {{
                padding: 10px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>🔍 Tail-Chasing Analysis Report</h1>
            <div class="subtitle">Generated on {timestamp}</div>
        </div>
    </header>
    
    <div class="container">
        <!-- Statistics Summary -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{self.statistics.total_issues}</div>
                <div class="label">Total Issues</div>
            </div>
            <div class="stat-card critical">
                <div class="value">{self.statistics.critical_issues}</div>
                <div class="label">Critical</div>
            </div>
            <div class="stat-card high">
                <div class="value">{self.statistics.high_issues}</div>
                <div class="label">High</div>
            </div>
            <div class="stat-card medium">
                <div class="value">{self.statistics.medium_issues}</div>
                <div class="label">Medium</div>
            </div>
            <div class="stat-card low">
                <div class="value">{self.statistics.low_issues}</div>
                <div class="label">Low</div>
            </div>
            <div class="stat-card">
                <div class="value">{self.statistics.affected_files}</div>
                <div class="label">Affected Files</div>
            </div>
        </div>
        
        <!-- Pattern Summary -->
        <div class="section">
            <h2>📊 Pattern Analysis</h2>
            <div class="summary-grid">
                {self._generate_pattern_summary_cards()}
            </div>
        </div>
        
        <!-- Visualizations -->
        {self._generate_visualization_section(viz_sections) if viz_sections else ''}
        
        <!-- Issues by Severity -->
        <div class="section">
            <h2>⚠️ Issues by Severity</h2>
            <div class="tabs">
                <button class="tab active" onclick="showTab('all')">All Issues</button>
                <button class="tab" onclick="showTab('critical')">Critical</button>
                <button class="tab" onclick="showTab('high')">High</button>
                <button class="tab" onclick="showTab('medium')">Medium</button>
                <button class="tab" onclick="showTab('low')">Low</button>
            </div>
            
            <div id="all" class="tab-content active">
                {self._generate_issues_html(self.issues)}
            </div>
            <div id="critical" class="tab-content">
                {self._generate_issues_html([i for i in self.issues if i.severity >= 4])}
            </div>
            <div id="high" class="tab-content">
                {self._generate_issues_html([i for i in self.issues if i.severity == 3])}
            </div>
            <div id="medium" class="tab-content">
                {self._generate_issues_html([i for i in self.issues if i.severity == 2])}
            </div>
            <div id="low" class="tab-content">
                {self._generate_issues_html([i for i in self.issues if i.severity <= 1])}
            </div>
        </div>
        
        <!-- Custom Sections -->
        {self._generate_custom_sections()}
        
        <!-- Raw Data (if requested) -->
        {self._generate_data_section() if embed_data else ''}
    </div>
    
    <footer class="footer">
        <div class="container">
            <p>Generated by <a href="https://github.com/yourusername/tailchasingfixer">TailChasingFixer</a> 
            • Report Version 1.0.0 • {timestamp}</p>
        </div>
    </footer>
    
    <script>
        function showTab(tabName) {{
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }}
        
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'start'
                    }});
                }}
            }});
        }});
    </script>
</body>
</html>
"""
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(html, encoding='utf-8')
            logger.info(f"HTML report saved to {output_path}")
        
        return html
    
    def generate_json_report(
        self,
        output_path: Optional[Path] = None,
        pretty: bool = True
    ) -> str:
        """
        Generate JSON report for programmatic access.
        
        Args:
            output_path: Optional path to save the report
            pretty: Pretty print the JSON
            
        Returns:
            JSON string
        """
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0.0',
                'generator': 'TailChasingFixer',
                **self.metadata
            },
            'statistics': self.statistics.to_dict(),
            'issues': [
                {
                    'kind': issue.kind,
                    'message': issue.message,
                    'severity': issue.severity,
                    'file': issue.file,
                    'line': issue.line,
                    'symbol': issue.symbol,
                    'confidence': getattr(issue, 'confidence', None),
                    'evidence': getattr(issue, 'evidence', {}),
                    'suggestions': issue.suggestions
                }
                for issue in self.issues
            ],
            'sections': [
                {
                    'title': section.title,
                    'content': section.content,
                    'type': section.section_type,
                    'metadata': section.metadata
                }
                for section in self.sections
            ]
        }
        
        json_str = json.dumps(report_data, indent=2 if pretty else None, default=str)
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(json_str, encoding='utf-8')
            logger.info(f"JSON report saved to {output_path}")
        
        return json_str
    
    def generate_markdown_report(
        self,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate Markdown summary report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Markdown string
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        md = f"""# Tail-Chasing Analysis Report

Generated: {timestamp}

## Executive Summary

Total Issues Found: **{self.statistics.total_issues}**

### Severity Breakdown

| Severity | Count | Percentage |
|----------|-------|------------|
| Critical | {self.statistics.critical_issues} | {self._percentage(self.statistics.critical_issues)} |
| High | {self.statistics.high_issues} | {self._percentage(self.statistics.high_issues)} |
| Medium | {self.statistics.medium_issues} | {self._percentage(self.statistics.medium_issues)} |
| Low | {self.statistics.low_issues} | {self._percentage(self.statistics.low_issues)} |

### Pattern Distribution

| Pattern Type | Count |
|--------------|-------|
| Duplicate Functions | {self.statistics.duplicate_functions} |
| Circular Imports | {self.statistics.circular_imports} |
| Phantom Functions | {self.statistics.phantom_functions} |
| Hallucination Cascades | {self.statistics.hallucination_cascades} |
| Context Window Thrashing | {self.statistics.context_thrashing} |
| Import Anxiety | {self.statistics.import_anxiety} |

### Risk Metrics

- **Overall Risk Score:** {self.statistics.overall_risk_score:.2f}
- **Maximum Risk Score:** {self.statistics.max_risk_score:.2f}
- **Average Risk Score:** {self.statistics.avg_risk_score:.2f}
- **Affected Files:** {self.statistics.affected_files}

## Top Issues

"""
        
        # Add top 10 critical/high issues
        top_issues = sorted(
            [i for i in self.issues if i.severity >= 3],
            key=lambda x: x.severity,
            reverse=True
        )[:10]
        
        for i, issue in enumerate(top_issues, 1):
            severity_label = self._get_severity_label(issue.severity)
            md += f"""
### {i}. [{severity_label}] {issue.kind}

**File:** `{issue.file}:{issue.line}` {'(' + issue.symbol + ')' if issue.symbol else ''}
**Message:** {issue.message}
"""
            if getattr(issue, 'confidence', None):
                md += f"**Confidence:** {issue.confidence:.2%}\n"
            
            if issue.suggestions:
                md += "\n**Suggestions:**\n"
                for suggestion in issue.suggestions[:3]:
                    md += f"- {suggestion}\n"
            
            md += "\n---\n"
        
        # Add custom sections
        for section in sorted(self.sections, key=lambda s: -s.priority):
            if section.section_type == 'text':
                md += f"\n## {section.title}\n\n{section.content}\n"
        
        # Add recommendations
        md += self._generate_recommendations_markdown()
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(md, encoding='utf-8')
            logger.info(f"Markdown report saved to {output_path}")
        
        return md
    
    def generate_graphml_report(
        self,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate GraphML for network analysis tools.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            GraphML XML string
        """
        # Create root element
        graphml = ET.Element('graphml', xmlns="http://graphml.graphdrawing.org/xmlns")
        graphml.set('xmlns:xsi', "http://www.w3.org/2001/XMLSchema-instance")
        graphml.set('xsi:schemaLocation', 
                   "http://graphml.graphdrawing.org/xmlns "
                   "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd")
        
        # Define keys for node and edge attributes
        keys = [
            ('node', 'label', 'string', 'd0'),
            ('node', 'type', 'string', 'd1'),
            ('node', 'risk_score', 'double', 'd2'),
            ('node', 'file_path', 'string', 'd3'),
            ('node', 'line_number', 'int', 'd4'),
            ('edge', 'weight', 'double', 'd5'),
            ('edge', 'type', 'string', 'd6')
        ]
        
        for target, name, attr_type, id_val in keys:
            key = ET.SubElement(graphml, 'key')
            key.set('for', target)
            key.set('id', id_val)
            key.set('attr.name', name)
            key.set('attr.type', attr_type)
        
        # Create graph
        graph = ET.SubElement(graphml, 'graph')
        graph.set('id', 'TailChasingGraph')
        graph.set('edgedefault', 'directed')
        
        # Add nodes
        for node in self.visualizer.nodes:
            n = ET.SubElement(graph, 'node')
            n.set('id', node.id)
            
            # Add node data
            data_elements = [
                ('d0', node.label),
                ('d1', node.type),
                ('d2', str(node.risk_score)),
                ('d3', node.file_path),
                ('d4', str(node.line_number))
            ]
            
            for key_id, value in data_elements:
                data = ET.SubElement(n, 'data')
                data.set('key', key_id)
                data.text = value
        
        # Add edges
        for i, edge in enumerate(self.visualizer.edges):
            e = ET.SubElement(graph, 'edge')
            e.set('id', f'e{i}')
            e.set('source', edge.source)
            e.set('target', edge.target)
            
            # Add edge data
            data_elements = [
                ('d5', str(edge.weight)),
                ('d6', edge.type)
            ]
            
            for key_id, value in data_elements:
                data = ET.SubElement(e, 'data')
                data.set('key', key_id)
                data.text = value
        
        # Convert to string with pretty printing
        xml_str = minidom.parseString(ET.tostring(graphml)).toprettyxml(indent="  ")
        
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(xml_str, encoding='utf-8')
            logger.info(f"GraphML report saved to {output_path}")
        
        return xml_str
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph from issues."""
        # Clear existing graph
        self.visualizer.clear()
        
        # Track unique nodes
        node_ids = set()
        
        for issue in self.issues:
            # Create node for the issue location
            if issue.file and issue.symbol:
                node_id = f"{issue.file}:{issue.symbol}"
                if node_id not in node_ids:
                    self.visualizer.add_node(
                        node_id=node_id,
                        label=issue.symbol or 'Unknown',
                        node_type='function' if '(' in str(issue.symbol) else 'module',
                        risk_score=getattr(issue, 'confidence', 0.5),
                        file_path=issue.file,
                        line_number=issue.line or 0
                    )
                    node_ids.add(node_id)
            
            # Add edges based on evidence
            if hasattr(issue, 'evidence') and issue.evidence:
                evidence = issue.evidence
                
                # Look for related functions/modules
                if 'related_functions' in evidence:
                    for related in evidence['related_functions']:
                        related_id = f"{issue.file}:{related}"
                        if related_id not in node_ids:
                            self.visualizer.add_node(
                                node_id=related_id,
                                label=related,
                                node_type='function',
                                risk_score=0.3,
                                file_path=issue.file
                            )
                            node_ids.add(related_id)
                        
                        self.visualizer.add_edge(
                            source=node_id,
                            target=related_id,
                            weight=0.8,
                            edge_type='similarity'
                        )
    
    def _generate_pattern_summary_cards(self) -> str:
        """Generate pattern summary cards HTML."""
        patterns = [
            ('Duplicate Functions', self.statistics.duplicate_functions, '🔁'),
            ('Circular Imports', self.statistics.circular_imports, '🔄'),
            ('Phantom Functions', self.statistics.phantom_functions, '👻'),
            ('Hallucination Cascades', self.statistics.hallucination_cascades, '🌀'),
            ('Context Thrashing', self.statistics.context_thrashing, '💭'),
            ('Import Anxiety', self.statistics.import_anxiety, '📦')
        ]
        
        html = ""
        for name, count, icon in patterns:
            if count > 0:
                percentage = (count / self.statistics.total_issues * 100) if self.statistics.total_issues > 0 else 0
                html += f"""
                <div class="summary-card">
                    <h3>{icon} {name}</h3>
                    <div style="font-size: 2em; color: #667eea; margin: 10px 0;">{count}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {percentage}%"></div>
                    </div>
                    <div style="color: #666; font-size: 0.9em;">{percentage:.1f}% of issues</div>
                </div>
                """
        
        return html
    
    def _generate_visualization_section(self, viz_sections: List[Tuple[str, str, str]]) -> str:
        """Generate visualization section HTML."""
        if not viz_sections:
            return ""
        
        html = """
        <div class="section">
            <h2>📈 Interactive Visualizations</h2>
        """
        
        for title, viz_html, viz_type in viz_sections:
            # Embed visualization in iframe
            viz_data = base64.b64encode(viz_html.encode()).decode()
            html += f"""
            <div class="visualization-container">
                <h3 style="padding: 15px; margin: 0; background: #f9f9f9; border-bottom: 1px solid #e0e0e0;">
                    {title}
                </h3>
                <iframe src="data:text/html;base64,{viz_data}"></iframe>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_issues_html(self, issues: List[Issue]) -> str:
        """Generate HTML for issues list."""
        if not issues:
            return '<p style="color: #666; text-align: center; padding: 20px;">No issues found in this category.</p>'
        
        html = '<ul class="issue-list">'
        
        for issue in issues:
            severity_class = f"severity-{issue.severity}"
            severity_label = self._get_severity_label(issue.severity)
            badge_class = self._get_badge_class(issue.severity)
            
            html += f"""
            <li class="issue-item {severity_class}">
                <div class="issue-header">
                    <span class="issue-title">{issue.kind}</span>
                    <span class="issue-badge {badge_class}">{severity_label}</span>
                </div>
                <div class="issue-location">
                    📁 {issue.file}:{issue.line} 
                    {('• ' + issue.symbol) if issue.symbol else ''}
                </div>
                <div class="issue-message">{issue.message}</div>
            """
            
            if issue.suggestions:
                html += """
                <div class="issue-suggestions">
                    <h4>Suggestions:</h4>
                    <ul>
                """
                for suggestion in issue.suggestions[:3]:
                    html += f"<li>{suggestion}</li>"
                html += """
                    </ul>
                </div>
                """
            
            html += "</li>"
        
        html += "</ul>"
        return html
    
    def _generate_custom_sections(self) -> str:
        """Generate custom sections HTML."""
        if not self.sections:
            return ""
        
        html = ""
        for section in sorted(self.sections, key=lambda s: -s.priority):
            if section.section_type == 'text':
                html += f"""
                <div class="section">
                    <h2>{section.title}</h2>
                    <div>{section.content}</div>
                </div>
                """
        
        return html
    
    def _generate_data_section(self) -> str:
        """Generate embedded data section."""
        data = {
            'issues': [
                {
                    'kind': i.kind,
                    'severity': i.severity,
                    'file': i.file,
                    'line': i.line,
                    'message': i.message
                }
                for i in self.issues
            ],
            'statistics': self.statistics.to_dict()
        }
        
        return f"""
        <div class="section">
            <h2>📊 Raw Data</h2>
            <details>
                <summary style="cursor: pointer; padding: 10px; background: #f9f9f9; border-radius: 4px;">
                    Click to view raw data (JSON)
                </summary>
                <pre style="background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; margin-top: 10px;">
{json.dumps(data, indent=2, default=str)}
                </pre>
            </details>
        </div>
        """
    
    def _generate_recommendations_markdown(self) -> str:
        """Generate recommendations based on analysis."""
        md = "\n## Recommendations\n\n"
        
        recommendations = []
        
        if self.statistics.duplicate_functions > 5:
            recommendations.append(
                "**High number of duplicate functions detected.** "
                "Consider extracting common functionality into shared utilities."
            )
        
        if self.statistics.circular_imports > 0:
            recommendations.append(
                "**Circular imports found.** "
                "Refactor module dependencies to create a clear hierarchy."
            )
        
        if self.statistics.hallucination_cascades > 0:
            recommendations.append(
                "**Hallucination cascades detected.** "
                "Review and simplify over-engineered abstractions."
            )
        
        if self.statistics.context_thrashing > 3:
            recommendations.append(
                "**Context window thrashing patterns found.** "
                "Consolidate similar implementations and improve code organization."
            )
        
        if self.statistics.import_anxiety > 10:
            recommendations.append(
                "**Import anxiety patterns detected.** "
                "Remove unused imports and standardize import conventions."
            )
        
        if not recommendations:
            recommendations.append(
                "Code quality is generally good. "
                "Continue monitoring for tail-chasing patterns as the codebase grows."
            )
        
        for i, rec in enumerate(recommendations, 1):
            md += f"{i}. {rec}\n\n"
        
        return md
    
    def _get_severity_label(self, severity: int) -> str:
        """Get label for severity level."""
        if severity >= 4:
            return "CRITICAL"
        elif severity == 3:
            return "HIGH"
        elif severity == 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_badge_class(self, severity: int) -> str:
        """Get CSS class for severity badge."""
        if severity >= 4:
            return "badge-critical"
        elif severity == 3:
            return "badge-high"
        elif severity == 2:
            return "badge-medium"
        else:
            return "badge-low"
    
    def _percentage(self, count: int) -> str:
        """Calculate percentage of total issues."""
        if self.statistics.total_issues == 0:
            return "0%"
        return f"{(count / self.statistics.total_issues * 100):.1f}%"
    
    def clear(self) -> None:
        """Clear all report data."""
        self.issues = []
        self.sections = []
        self.statistics = ReportStatistics()
        self.visualizer.clear()
        self.metadata = {}