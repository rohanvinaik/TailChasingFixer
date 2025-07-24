"""
Visualization components for tail-chasing patterns.
Generates interactive visualizations of code issues and dependencies.
"""

import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
import networkx as nx

from ..core.issues import Issue


class TailChasingVisualizer:
    """Create visual representations of tail-chasing patterns."""
    
    def __init__(self):
        self.color_scheme = {
            'semantic_duplicate': '#FF6B6B',
            'phantom_implementation': '#4ECDC4',
            'circular_import': '#45B7D1',
            'import_anxiety': '#96CEB4',
            'context_window_thrashing': '#FFEAA7',
            'hallucination_cascade': '#DDA0DD',
            'fix_induced_regression': '#FF7675'
        }
    
    def generate_dependency_graph(self, issues: List[Issue], codebase_files: List[str]) -> Dict[str, Any]:
        """Generate an interactive dependency graph showing issues."""
        graph_data = {
            'nodes': [],
            'edges': [],
            'clusters': [],
            'metadata': {
                'total_issues': len(issues),
                'total_files': len(codebase_files),
                'issue_types': list(set(issue.kind for issue in issues))
            }
        }
        
        # Create nodes for each file
        file_issues = defaultdict(list)
        for issue in issues:
            if issue.file:
                file_issues[issue.file].append(issue)
        
        for file in codebase_files:
            issues_in_file = file_issues.get(file, [])
            severity_sum = sum(issue.severity for issue in issues_in_file)
            
            # Determine node color based on most severe issue type
            node_color = '#E0E0E0'  # Default gray
            if issues_in_file:
                most_severe_issue = max(issues_in_file, key=lambda x: x.severity)
                node_color = self.color_scheme.get(most_severe_issue.kind, '#E0E0E0')
            
            graph_data['nodes'].append({
                'id': file,
                'label': file.split('/')[-1],  # Just filename
                'fullPath': file,
                'size': max(10, severity_sum * 2),  # Size based on total severity
                'color': node_color,
                'issues': len(issues_in_file),
                'totalSeverity': severity_sum,
                'issueTypes': list(set(issue.kind for issue in issues_in_file))
            })
        
        # Create edges for dependencies and issues
        self._add_import_edges(graph_data, issues)
        self._add_issue_edges(graph_data, issues)
        
        # Identify clusters
        graph_data['clusters'] = self._identify_clusters(issues)
        
        return graph_data
    
    def generate_timeline_data(self, issues: List[Issue], git_history: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate timeline data showing issue evolution."""
        timeline_data = {
            'events': [],
            'patterns': [],
            'trends': {}
        }
        
        if git_history:
            # Process git history to show issue evolution
            for issue in issues:
                timeline_data['events'].append({
                    'date': git_history.get(f"{issue.file}:{issue.line}", '2023-01-01'),
                    'type': issue.kind,
                    'severity': issue.severity,
                    'file': issue.file,
                    'message': issue.message
                })
        
        # Identify patterns over time
        issue_types = defaultdict(int)
        for issue in issues:
            issue_types[issue.kind] += 1
        
        timeline_data['patterns'] = [
            {'type': issue_type, 'count': count, 'percentage': count / len(issues) * 100}
            for issue_type, count in issue_types.items()
        ]
        
        return timeline_data
    
    def generate_risk_heatmap(self, issues: List[Issue], codebase_files: List[str]) -> Dict[str, Any]:
        """Generate a risk heatmap of the codebase."""
        heatmap_data = {
            'files': [],
            'scale': {
                'min': 0,
                'max': 0,
                'colors': ['#E8F5E8', '#FFF3CD', '#F8D7DA', '#F5C6CB', '#D1ECF1']
            }
        }
        
        # Calculate risk scores for each file
        file_risks = defaultdict(int)
        for issue in issues:
            if issue.file:
                file_risks[issue.file] += issue.severity
        
        max_risk = max(file_risks.values()) if file_risks else 0
        heatmap_data['scale']['max'] = max_risk
        
        # Create heatmap entries
        for file in codebase_files:
            risk_score = file_risks.get(file, 0)
            risk_level = self._calculate_risk_level(risk_score, max_risk)
            
            heatmap_data['files'].append({
                'path': file,
                'name': file.split('/')[-1],
                'riskScore': risk_score,
                'riskLevel': risk_level,
                'color': self._get_risk_color(risk_level),
                'issues': [issue.kind for issue in issues if issue.file == file]
            })
        
        return heatmap_data
    
    def generate_semantic_similarity_matrix(self, issues: List[Issue]) -> Dict[str, Any]:
        """Generate a similarity matrix for semantic duplicates."""
        semantic_issues = [issue for issue in issues if 'semantic_duplicate' in issue.kind]
        
        matrix_data = {
            'functions': [],
            'similarities': [],
            'clusters': []
        }
        
        # Extract function pairs and similarities
        function_pairs = {}
        for issue in semantic_issues:
            func1 = issue.evidence.get('function1', 'unknown')
            func2 = issue.evidence.get('function2', 'unknown')
            similarity = issue.evidence.get('similarity', 0)
            
            if func1 not in matrix_data['functions']:
                matrix_data['functions'].append(func1)
            if func2 not in matrix_data['functions']:
                matrix_data['functions'].append(func2)
            
            function_pairs[(func1, func2)] = similarity
        
        # Create similarity matrix
        n_functions = len(matrix_data['functions'])
        for i in range(n_functions):
            row = []
            for j in range(n_functions):
                func1 = matrix_data['functions'][i]
                func2 = matrix_data['functions'][j]
                
                if i == j:
                    similarity = 1.0
                elif (func1, func2) in function_pairs:
                    similarity = function_pairs[(func1, func2)]
                elif (func2, func1) in function_pairs:
                    similarity = function_pairs[(func2, func1)]
                else:
                    similarity = 0.0
                
                row.append(similarity)
            matrix_data['similarities'].append(row)
        
        return matrix_data
    
    def generate_html_report(self, issues: List[Issue], codebase_files: List[str], 
                           output_path: str = "tail_chasing_report.html"):
        """Generate a complete HTML report with interactive visualizations."""
        # Generate all visualization data
        dependency_data = self.generate_dependency_graph(issues, codebase_files)
        timeline_data = self.generate_timeline_data(issues)
        heatmap_data = self.generate_risk_heatmap(issues, codebase_files)
        similarity_data = self.generate_semantic_similarity_matrix(issues)
        
        # Create HTML template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tail-Chasing Analysis Report</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            min-width: 150px;
            margin: 10px;
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
        }}
        .visualization {{
            margin: 40px 0;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
        }}
        .issue-list {{
            margin: 20px 0;
        }}
        .issue-item {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .issue-severity-4, .issue-severity-5 {{
            border-left-color: #dc3545;
        }}
        .issue-severity-3 {{
            border-left-color: #ffc107;
        }}
        .issue-severity-1, .issue-severity-2 {{
            border-left-color: #28a745;
        }}
        #dependencyGraph, #timeline, #heatmap, #similarityMatrix {{
            width: 100%;
            height: 500px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Tail-Chasing Analysis Report</h1>
            <p>Automated detection of LLM-induced anti-patterns</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(issues)}</div>
                <div>Total Issues</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(codebase_files)}</div>
                <div>Files Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sum(issue.severity for issue in issues)}</div>
                <div>Risk Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(issue.kind for issue in issues))}</div>
                <div>Pattern Types</div>
            </div>
        </div>
        
        <div class="visualization">
            <h3>📊 Dependency Graph</h3>
            <p>Interactive visualization of file dependencies and issues</p>
            <div id="dependencyGraph"></div>
        </div>
        
        <div class="visualization">
            <h3>📈 Issue Timeline</h3>
            <p>Distribution and evolution of tail-chasing patterns</p>
            <div id="timeline"></div>
        </div>
        
        <div class="visualization">
            <h3>🔥 Risk Heatmap</h3>
            <p>File-level risk assessment</p>
            <div id="heatmap"></div>
        </div>
        
        <div class="visualization">
            <h3>🧬 Semantic Similarity Matrix</h3>
            <p>Function similarity analysis</p>
            <div id="similarityMatrix"></div>
        </div>
        
        <div class="issue-list">
            <h3>📋 Detailed Issues</h3>
            {self._generate_issue_html(issues)}
        </div>
    </div>
    
    <script>
        // Data from Python
        const dependencyData = {json.dumps(dependency_data)};
        const timelineData = {json.dumps(timeline_data)};
        const heatmapData = {json.dumps(heatmap_data)};
        const similarityData = {json.dumps(similarity_data)};
        
        // Create dependency graph
        function createDependencyGraph() {{
            const trace = {{
                x: dependencyData.nodes.map((n, i) => Math.cos(i * 2 * Math.PI / dependencyData.nodes.length)),
                y: dependencyData.nodes.map((n, i) => Math.sin(i * 2 * Math.PI / dependencyData.nodes.length)),
                mode: 'markers+text',
                type: 'scatter',
                text: dependencyData.nodes.map(n => n.label),
                textposition: 'middle center',
                marker: {{
                    size: dependencyData.nodes.map(n => n.size),
                    color: dependencyData.nodes.map(n => n.color),
                    line: {{ width: 2, color: 'white' }}
                }},
                hovertemplate: '<b>%{{text}}</b><br>Issues: %{{customdata.issues}}<br>Severity: %{{customdata.totalSeverity}}<extra></extra>',
                customdata: dependencyData.nodes
            }};
            
            const layout = {{
                title: 'File Dependency Network',
                showlegend: false,
                xaxis: {{ showgrid: false, showticklabels: false, zeroline: false }},
                yaxis: {{ showgrid: false, showticklabels: false, zeroline: false }},
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            }};
            
            Plotly.newPlot('dependencyGraph', [trace], layout);
        }}
        
        // Create timeline
        function createTimeline() {{
            const trace = {{
                x: timelineData.patterns.map(p => p.type),
                y: timelineData.patterns.map(p => p.count),
                type: 'bar',
                marker: {{ color: '#667eea' }}
            }};
            
            const layout = {{
                title: 'Issue Distribution by Type',
                xaxis: {{ title: 'Issue Type' }},
                yaxis: {{ title: 'Count' }}
            }};
            
            Plotly.newPlot('timeline', [trace], layout);
        }}
        
        // Create heatmap
        function createHeatmap() {{
            const files = heatmapData.files.slice(0, 20); // Limit to first 20 files
            const trace = {{
                x: files.map(f => f.name),
                y: ['Risk Score'],
                z: [files.map(f => f.riskScore)],
                type: 'heatmap',
                colorscale: [[0, '#E8F5E8'], [0.25, '#FFF3CD'], [0.5, '#F8D7DA'], [0.75, '#F5C6CB'], [1, '#D1ECF1']]
            }};
            
            const layout = {{
                title: 'File Risk Heatmap',
                xaxis: {{ title: 'Files' }},
                yaxis: {{ title: '' }}
            }};
            
            Plotly.newPlot('heatmap', [trace], layout);
        }}
        
        // Create similarity matrix
        function createSimilarityMatrix() {{
            if (similarityData.functions.length > 0) {{
                const trace = {{
                    z: similarityData.similarities,
                    x: similarityData.functions,
                    y: similarityData.functions,
                    type: 'heatmap',
                    colorscale: 'Viridis'
                }};
                
                const layout = {{
                    title: 'Function Similarity Matrix',
                    xaxis: {{ title: 'Functions' }},
                    yaxis: {{ title: 'Functions' }}
                }};
                
                Plotly.newPlot('similarityMatrix', [trace], layout);
            }} else {{
                document.getElementById('similarityMatrix').innerHTML = '<p>No semantic duplicates detected.</p>';
            }}
        }}
        
        // Initialize all visualizations
        document.addEventListener('DOMContentLoaded', function() {{
            createDependencyGraph();
            createTimeline();
            createHeatmap();
            createSimilarityMatrix();
        }});
    </script>
</body>
</html>
"""
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _add_import_edges(self, graph_data: Dict, issues: List[Issue]):
        """Add edges for import relationships."""
        for issue in issues:
            if issue.kind == 'circular_import':
                cycle = issue.evidence.get('cycle', [])
                for i in range(len(cycle) - 1):
                    graph_data['edges'].append({
                        'source': cycle[i],
                        'target': cycle[i + 1],
                        'type': 'circular_import',
                        'color': '#FF6B6B',
                        'weight': 3
                    })
    
    def _add_issue_edges(self, graph_data: Dict, issues: List[Issue]):
        """Add edges representing issue relationships."""
        for issue in issues:
            if 'semantic_duplicate' in issue.kind:
                file1 = issue.file
                file2 = issue.evidence.get('file2')
                if file1 and file2 and file1 != file2:
                    graph_data['edges'].append({
                        'source': file1,
                        'target': file2,
                        'type': 'semantic_duplicate',
                        'color': '#4ECDC4',
                        'weight': 2
                    })
    
    def _identify_clusters(self, issues: List[Issue]) -> List[Dict]:
        """Identify clusters of related issues."""
        clusters = []
        
        # Group by issue type
        issue_groups = defaultdict(list)
        for issue in issues:
            issue_groups[issue.kind].append(issue)
        
        for issue_type, type_issues in issue_groups.items():
            if len(type_issues) > 1:
                clusters.append({
                    'type': issue_type,
                    'count': len(type_issues),
                    'files': list(set(issue.file for issue in type_issues if issue.file)),
                    'totalSeverity': sum(issue.severity for issue in type_issues)
                })
        
        return clusters
    
    def _calculate_risk_level(self, risk_score: int, max_risk: int) -> str:
        """Calculate risk level based on score."""
        if max_risk == 0:
            return 'low'
        
        ratio = risk_score / max_risk
        if ratio >= 0.8:
            return 'critical'
        elif ratio >= 0.6:
            return 'high'
        elif ratio >= 0.4:
            return 'medium'
        elif ratio >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def _get_risk_color(self, risk_level: str) -> str:
        """Get color for risk level."""
        colors = {
            'minimal': '#E8F5E8',
            'low': '#FFF3CD',
            'medium': '#F8D7DA',
            'high': '#F5C6CB',
            'critical': '#D1ECF1'
        }
        return colors.get(risk_level, '#E0E0E0')
    
    def _generate_issue_html(self, issues: List[Issue]) -> str:
        """Generate HTML for issue list."""
        html_parts = []
        
        for issue in issues:
            severity_class = f"issue-severity-{issue.severity}"
            location = f"{issue.file}:{issue.line}" if issue.file and issue.line else "Unknown location"
            
            html_parts.append(f"""
            <div class="issue-item {severity_class}">
                <h4>{issue.kind.replace('_', ' ').title()}</h4>
                <p><strong>Message:</strong> {issue.message}</p>
                <p><strong>Location:</strong> {location}</p>
                <p><strong>Severity:</strong> {issue.severity}/5</p>
                {f'<p><strong>Suggestions:</strong></p><ul>{"".join(f"<li>{s}</li>" for s in issue.suggestions)}</ul>' if issue.suggestions else ''}
            </div>
            """)
        
        return "".join(html_parts)
