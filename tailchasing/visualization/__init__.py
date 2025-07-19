"""
Visualization module for tail-chasing patterns.

Generates interactive visualizations to help understand and communicate
tail-chasing patterns in codebases.
"""

from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

from ..core.issues import Issue


class TailChasingVisualizer:
    """
    Generate various visualizations of tail-chasing patterns.
    """
    
    def generate_semantic_similarity_heatmap(self, 
                                           similarity_pairs: List[Tuple],
                                           output_path: Path) -> None:
        """
        Generate an interactive heatmap showing semantic similarities between functions.
        
        This creates an HTML file with D3.js visualization.
        """
        html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Semantic Similarity Heatmap</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .cell:hover { stroke: #000; stroke-width: 2px; }
        .tooltip { position: absolute; background: rgba(0,0,0,0.8); 
                  color: white; padding: 5px; border-radius: 3px; }
        .axis { font-size: 12px; }
        .legend { font-size: 14px; }
    </style>
</head>
<body>
    <h2>Function Semantic Similarity Heatmap</h2>
    <div id="heatmap"></div>
    <script>
        const data = {data_json};
        // D3.js heatmap implementation
        // ... (would include full D3 visualization code)
    </script>
</body>
</html>
'''
        # Convert similarity data to JSON
        # ... implementation
    
    def generate_dependency_graph(self, 
                                 issues: List[Issue],
                                 output_path: Path) -> None:
        """
        Generate an interactive dependency graph showing circular imports
        and function relationships.
        
        Uses vis.js for network visualization.
        """
        nodes = []
        edges = []
        
        # Build graph from issues
        for issue in issues:
            if issue.kind == 'circular_import':
                cycle = issue.evidence.get('cycle', [])
                for i in range(len(cycle) - 1):
                    edges.append({
                        'from': cycle[i],
                        'to': cycle[i + 1],
                        'color': 'red',
                        'label': 'circular'
                    })
            
            elif issue.kind == 'semantic_duplicate_function':
                pair = issue.evidence.get('pair', [])
                if len(pair) == 2:
                    edges.append({
                        'from': f"{pair[0]['file']}:{pair[0]['name']}",
                        'to': f"{pair[1]['file']}:{pair[1]['name']}",
                        'color': 'orange',
                        'label': f"z={issue.evidence.get('z_score', 0):.2f}",
                        'dashes': True
                    })
        
        # Generate HTML with vis.js
        # ... implementation
    
    def generate_drift_timeline(self,
                               drift_data: Dict,
                               output_path: Path) -> None:
        """
        Generate a timeline visualization showing how functions drift
        semantically over time.
        
        Uses Chart.js for time series visualization.
        """
        # ... implementation
        pass
    
    def generate_3d_semantic_space(self,
                                  hypervectors: Dict[str, List[float]],
                                  output_path: Path) -> None:
        """
        Generate a 3D visualization of the semantic space using t-SNE
        dimensionality reduction.
        
        This helps visualize clusters and relationships.
        """
        html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>3D Semantic Space</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h2>Function Semantic Space (t-SNE projection)</h2>
    <div id="plot"></div>
    <script>
        // Would use plotly.js for 3D scatter plot
        // Points colored by cluster, sized by complexity
    </script>
</body>
</html>
'''
        # ... implementation


class MetricsDashboard:
    """
    Generate a comprehensive metrics dashboard.
    """
    
    def generate_dashboard(self, 
                          issues: List[Issue],
                          history: Optional[List[Dict]] = None) -> str:
        """
        Generate an HTML dashboard with key metrics and trends.
        """
        dashboard_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Tail-Chasing Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .metric-card {
            display: inline-block;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value { font-size: 48px; font-weight: bold; }
        .metric-label { color: #666; }
        .critical { background: #fee; color: #c33; }
        .warning { background: #ffd; color: #cc0; }
        .ok { background: #efe; color: #3c3; }
    </style>
</head>
<body>
    <h1>Tail-Chasing Analysis Dashboard</h1>
    
    <div class="metrics-row">
        <div class="metric-card {risk_class}">
            <div class="metric-value">{total_score}</div>
            <div class="metric-label">Risk Score</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value">{semantic_duplicates}</div>
            <div class="metric-label">Semantic Duplicates</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-value">{fragmentation}</div>
            <div class="metric-label">Prototype Fragments</div>
        </div>
    </div>
    
    <h2>Issue Distribution</h2>
    <canvas id="issueChart"></canvas>
    
    <h2>Risk Trend</h2>
    <canvas id="trendChart"></canvas>
    
    <script>
        // Chart.js implementation for visualizations
    </script>
</body>
</html>
'''
        # ... implementation
        return dashboard_template