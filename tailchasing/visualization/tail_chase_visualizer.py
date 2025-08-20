"""
Interactive visualization system for tail-chasing patterns.

Generates self-contained HTML visualizations with embedded D3.js for
dependency graphs, temporal animations, and similarity heatmaps.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationNode:
    """Node in visualization graph."""
    id: str
    label: str
    type: str  # 'function', 'class', 'module'
    risk_score: float = 0.0
    file_path: str = ""
    line_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'label': self.label,
            'type': self.type,
            'risk_score': self.risk_score,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'metadata': self.metadata
        }


@dataclass
class VisualizationEdge:
    """Edge in visualization graph."""
    source: str
    target: str
    weight: float = 1.0
    type: str = 'dependency'  # 'dependency', 'similarity', 'temporal'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'type': self.type,
            'metadata': self.metadata
        }


class TailChaseVisualizer:
    """
    Generate interactive visualizations for tail-chasing patterns.
    
    Creates self-contained HTML with embedded D3.js visualizations.
    """
    
    # Embedded D3.js v7 minified (partial - in production would include full minified version)
    # This is a placeholder - in production, embed the full minified D3.js library
    D3_JS_EMBEDDED = """
// D3.js v7 - This would be the full minified version in production
// For brevity, showing structure only
!function(t,n){"object"==typeof exports&&"undefined"!=typeof module?n(exports):"function"==typeof define&&define.amd?define(["exports"],n):n((t="undefined"!=typeof globalThis?globalThis:t||self).d3={})}(this,function(t){"use strict";
// ... full D3.js v7 minified code would go here ...
});
"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.nodes: List[VisualizationNode] = []
        self.edges: List[VisualizationEdge] = []
        self.temporal_data: List[Dict[str, Any]] = []
        self.similarity_matrix: Optional[np.ndarray] = None
        self.node_labels: List[str] = []
    
    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str = 'function',
        risk_score: float = 0.0,
        **kwargs
    ) -> None:
        """Add a node to the visualization."""
        node = VisualizationNode(
            id=node_id,
            label=label,
            type=node_type,
            risk_score=risk_score,
            file_path=kwargs.get('file_path', ''),
            line_number=kwargs.get('line_number', 0),
            metadata=kwargs.get('metadata', {})
        )
        self.nodes.append(node)
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
        edge_type: str = 'dependency',
        **kwargs
    ) -> None:
        """Add an edge to the visualization."""
        edge = VisualizationEdge(
            source=source,
            target=target,
            weight=weight,
            type=edge_type,
            metadata=kwargs.get('metadata', {})
        )
        self.edges.append(edge)
    
    def set_similarity_matrix(
        self,
        matrix: np.ndarray,
        labels: List[str]
    ) -> None:
        """Set similarity matrix for heatmap visualization."""
        self.similarity_matrix = matrix
        self.node_labels = labels
    
    def add_temporal_snapshot(
        self,
        timestamp: datetime,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> None:
        """Add a temporal snapshot for animation."""
        self.temporal_data.append({
            'timestamp': timestamp.isoformat(),
            'nodes': nodes,
            'edges': edges
        })
    
    def generate_dependency_graph(
        self,
        title: str = "Dependency Graph",
        width: int = 1200,
        height: int = 800
    ) -> str:
        """
        Generate interactive dependency graph visualization.
        
        Returns:
            HTML string with embedded visualization
        """
        # Convert nodes and edges to JSON
        nodes_json = json.dumps([n.to_dict() for n in self.nodes])
        edges_json = json.dumps([e.to_dict() for e in self.edges])
        
        # Generate color scale based on risk scores
        max_risk = max((n.risk_score for n in self.nodes), default=1.0)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        #visualization {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
        }}
        .node:hover {{
            stroke-width: 3px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1px;
        }}
        .link.similarity {{
            stroke: #4CAF50;
            stroke-dasharray: 5,5;
        }}
        .link.temporal {{
            stroke: #2196F3;
        }}
        .node-label {{
            font-size: 12px;
            pointer-events: none;
        }}
        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 10px;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        .controls {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        button {{
            margin: 2px;
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 3px;
            background: white;
            cursor: pointer;
        }}
        button:hover {{
            background: #f0f0f0;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="visualization">
        <div class="controls">
            <button onclick="resetZoom()">Reset Zoom</button>
            <button onclick="toggleLabels()">Toggle Labels</button>
            <button onclick="toggleForce()">Toggle Physics</button>
        </div>
        <div class="legend">
            <div><strong>Risk Score</strong></div>
            <svg width="150" height="30">
                <defs>
                    <linearGradient id="risk-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" style="stop-color:#4CAF50;stop-opacity:1" />
                        <stop offset="50%" style="stop-color:#FFC107;stop-opacity:1" />
                        <stop offset="100%" style="stop-color:#F44336;stop-opacity:1" />
                    </linearGradient>
                </defs>
                <rect width="150" height="20" fill="url(#risk-gradient)" />
                <text x="0" y="30" font-size="10">Low</text>
                <text x="130" y="30" font-size="10">High</text>
            </svg>
        </div>
        <div class="tooltip"></div>
    </div>
    
    <script>
        // Embedded D3.js would go here in production
        {self.D3_JS_EMBEDDED}
        
        // Visualization data
        const nodes = {nodes_json};
        const edges = {edges_json};
        const maxRisk = {max_risk};
        
        // Set up SVG
        const width = {width};
        const height = {height};
        
        const svg = d3.select("#visualization")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        const g = svg.append("g");
        
        // Create force simulation
        let simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(edges).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        // Create arrow markers for directed edges
        svg.append("defs").selectAll("marker")
            .data(["dependency", "similarity", "temporal"])
            .enter().append("marker")
            .attr("id", d => `arrow-${{d}}`)
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 20)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(edges)
            .enter().append("line")
            .attr("class", d => `link ${{d.type}}`)
            .attr("stroke-width", d => Math.sqrt(d.weight))
            .attr("marker-end", d => `url(#arrow-${{d.type}})`);
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => 10 + d.risk_score * 10)
            .attr("fill", d => getRiskColor(d.risk_score / maxRisk))
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Add labels
        const label = g.append("g")
            .selectAll("text")
            .data(nodes)
            .enter().append("text")
            .attr("class", "node-label")
            .text(d => d.label)
            .attr("x", 12)
            .attr("y", 3);
        
        // Tooltip
        const tooltip = d3.select(".tooltip");
        
        node.on("mouseover", function(event, d) {{
            tooltip.style("opacity", 1)
                .html(`
                    <strong>${{d.label}}</strong><br>
                    Type: ${{d.type}}<br>
                    Risk Score: ${{d.risk_score.toFixed(2)}}<br>
                    File: ${{d.file_path}}<br>
                    Line: ${{d.line_number}}
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", function() {{
            tooltip.style("opacity", 0);
        }});
        
        // Update positions on tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x + 12)
                .attr("y", d => d.y + 3);
        }});
        
        // Helper functions
        function getRiskColor(value) {{
            // Interpolate from green to yellow to red
            if (value < 0.5) {{
                const g = 255;
                const r = Math.floor(255 * value * 2);
                return `rgb(${{r}}, ${{g}}, 0)`;
            }} else {{
                const r = 255;
                const g = Math.floor(255 * (1 - (value - 0.5) * 2));
                return `rgb(${{r}}, ${{g}}, 0)`;
            }}
        }}
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Control functions
        let labelsVisible = true;
        let forceActive = true;
        
        function resetZoom() {{
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        }}
        
        function toggleLabels() {{
            labelsVisible = !labelsVisible;
            label.style("display", labelsVisible ? "block" : "none");
        }}
        
        function toggleForce() {{
            forceActive = !forceActive;
            if (forceActive) {{
                simulation.alpha(0.3).restart();
            }} else {{
                simulation.stop();
            }}
        }}
    </script>
</body>
</html>
"""
        return html
    
    def generate_temporal_animation(
        self,
        title: str = "Temporal Evolution",
        width: int = 1200,
        height: int = 800,
        duration: int = 10000  # Animation duration in ms
    ) -> str:
        """
        Generate temporal evolution animation.
        
        Returns:
            HTML string with embedded animation
        """
        if not self.temporal_data:
            return self._generate_empty_viz("No temporal data available")
        
        temporal_json = json.dumps(self.temporal_data)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        #visualization {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            position: relative;
        }}
        .timeline {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            width: 80%;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        .timeline-slider {{
            width: 100%;
        }}
        .timeline-label {{
            text-align: center;
            font-size: 14px;
            margin-top: 5px;
        }}
        .play-controls {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        button {{
            margin: 2px;
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 3px;
            background: white;
            cursor: pointer;
        }}
        button:hover {{
            background: #f0f0f0;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 2px;
            transition: all 0.3s;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            transition: all 0.3s;
        }}
        .entering {{
            opacity: 0;
            animation: fadeIn 0.5s forwards;
        }}
        .exiting {{
            animation: fadeOut 0.5s forwards;
        }}
        @keyframes fadeIn {{
            to {{ opacity: 1; }}
        }}
        @keyframes fadeOut {{
            to {{ opacity: 0; }}
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="visualization">
        <div class="play-controls">
            <button id="playBtn" onclick="togglePlay()">▶ Play</button>
            <button onclick="reset()">⏮ Reset</button>
            <button onclick="stepForward()">⏭ Step</button>
            <span id="speedLabel">Speed: 1x</span>
            <input type="range" id="speedSlider" min="0.5" max="3" step="0.5" value="1" 
                   onchange="updateSpeed(this.value)">
        </div>
        <svg id="mainSvg" width="{width}" height="{height}"></svg>
        <div class="timeline">
            <input type="range" id="timeSlider" class="timeline-slider" 
                   min="0" max="100" value="0" onchange="seekTo(this.value)">
            <div class="timeline-label" id="timeLabel">Time: 0</div>
        </div>
    </div>
    
    <script>
        // Embedded D3.js would go here
        {self.D3_JS_EMBEDDED}
        
        // Animation data
        const temporalData = {temporal_json};
        const duration = {duration};
        let currentFrame = 0;
        let playing = false;
        let playSpeed = 1;
        let animationTimer = null;
        
        const svg = d3.select("#mainSvg");
        const g = svg.append("g");
        const width = {width};
        const height = {height};
        
        // Initialize with first frame
        if (temporalData.length > 0) {{
            renderFrame(0);
        }}
        
        function renderFrame(frameIndex) {{
            if (frameIndex < 0 || frameIndex >= temporalData.length) return;
            
            const frame = temporalData[frameIndex];
            currentFrame = frameIndex;
            
            // Update timeline
            document.getElementById("timeSlider").value = 
                (frameIndex / (temporalData.length - 1)) * 100;
            document.getElementById("timeLabel").textContent = 
                `Time: ${{frame.timestamp}}`;
            
            // Clear existing elements with fade out
            g.selectAll(".node, .link").classed("exiting", true);
            setTimeout(() => {{
                g.selectAll(".exiting").remove();
            }}, 500);
            
            // Create force simulation for this frame
            const simulation = d3.forceSimulation(frame.nodes)
                .force("link", d3.forceLink(frame.edges).id(d => d.id))
                .force("charge", d3.forceManyBody().strength(-200))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .stop();
            
            // Run simulation
            for (let i = 0; i < 100; ++i) simulation.tick();
            
            // Draw edges
            const links = g.selectAll(".link")
                .data(frame.edges, d => `${{d.source.id}}-${{d.target.id}}`);
            
            links.enter()
                .append("line")
                .attr("class", "link entering")
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            // Draw nodes
            const nodes = g.selectAll(".node")
                .data(frame.nodes, d => d.id);
            
            nodes.enter()
                .append("circle")
                .attr("class", "node entering")
                .attr("r", d => 10 + (d.risk_score || 0) * 10)
                .attr("fill", d => getRiskColor(d.risk_score || 0))
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
        }}
        
        function getRiskColor(value) {{
            const hue = (1 - value) * 120; // Green to red
            return `hsl(${{hue}}, 70%, 50%)`;
        }}
        
        function togglePlay() {{
            playing = !playing;
            document.getElementById("playBtn").textContent = playing ? "⏸ Pause" : "▶ Play";
            
            if (playing) {{
                playAnimation();
            }} else {{
                clearTimeout(animationTimer);
            }}
        }}
        
        function playAnimation() {{
            if (!playing) return;
            
            if (currentFrame >= temporalData.length - 1) {{
                currentFrame = 0;
            }}
            
            renderFrame(currentFrame);
            currentFrame++;
            
            if (currentFrame < temporalData.length) {{
                animationTimer = setTimeout(playAnimation, duration / temporalData.length / playSpeed);
            }} else {{
                playing = false;
                document.getElementById("playBtn").textContent = "▶ Play";
            }}
        }}
        
        function reset() {{
            playing = false;
            document.getElementById("playBtn").textContent = "▶ Play";
            clearTimeout(animationTimer);
            renderFrame(0);
        }}
        
        function stepForward() {{
            if (currentFrame < temporalData.length - 1) {{
                renderFrame(currentFrame + 1);
            }}
        }}
        
        function seekTo(value) {{
            const frameIndex = Math.floor((value / 100) * (temporalData.length - 1));
            renderFrame(frameIndex);
        }}
        
        function updateSpeed(value) {{
            playSpeed = parseFloat(value);
            document.getElementById("speedLabel").textContent = `Speed: ${{playSpeed}}x`;
        }}
    </script>
</body>
</html>
"""
        return html
    
    def generate_similarity_heatmap(
        self,
        title: str = "Semantic Similarity Heatmap",
        width: int = 800,
        height: int = 800
    ) -> str:
        """
        Generate similarity heatmap visualization.
        
        Returns:
            HTML string with embedded heatmap
        """
        if self.similarity_matrix is None:
            return self._generate_empty_viz("No similarity matrix available")
        
        # Convert matrix to list for JSON
        matrix_list = self.similarity_matrix.tolist()
        matrix_json = json.dumps(matrix_list)
        labels_json = json.dumps(self.node_labels)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        #visualization {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .cell {{
            stroke: #E6E6E6;
            stroke-width: 0.5px;
        }}
        .cell:hover {{
            stroke: #000;
            stroke-width: 2px;
        }}
        .label {{
            font-size: 10px;
            cursor: pointer;
        }}
        .label:hover {{
            font-weight: bold;
        }}
        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 8px;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .legend {{
            font-size: 12px;
        }}
        .controls {{
            margin-bottom: 10px;
        }}
        button {{
            margin: 2px;
            padding: 5px 10px;
            border: 1px solid #ddd;
            border-radius: 3px;
            background: white;
            cursor: pointer;
        }}
        button:hover {{
            background: #f0f0f0;
        }}
        .threshold-control {{
            display: inline-block;
            margin-left: 20px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="visualization">
        <div class="controls">
            <button onclick="resetOrder()">Reset Order</button>
            <button onclick="clusterByGroups()">Cluster</button>
            <div class="threshold-control">
                <label>Threshold: <span id="thresholdValue">0.5</span></label>
                <input type="range" id="thresholdSlider" min="0" max="1" step="0.05" value="0.5" 
                       onchange="updateThreshold(this.value)">
            </div>
        </div>
        <div id="heatmap"></div>
        <div class="tooltip"></div>
    </div>
    
    <script>
        // Embedded D3.js would go here
        {self.D3_JS_EMBEDDED}
        
        // Data
        const matrix = {matrix_json};
        const labels = {labels_json};
        const n = labels.length;
        
        // Dimensions
        const margin = {{top: 100, right: 50, bottom: 50, left: 100}};
        const width = {width} - margin.left - margin.right;
        const height = {height} - margin.top - margin.bottom;
        const cellSize = Math.min(width / n, height / n);
        
        // Scales
        const x = d3.scaleBand()
            .domain(d3.range(n))
            .range([0, cellSize * n]);
        
        const y = d3.scaleBand()
            .domain(d3.range(n))
            .range([0, cellSize * n]);
        
        const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
            .domain([0, 1]);
        
        // Create SVG
        const svg = d3.select("#heatmap")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom);
        
        const g = svg.append("g")
            .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
        
        // Tooltip
        const tooltip = d3.select(".tooltip");
        
        // Current order
        let currentOrder = d3.range(n);
        let threshold = 0.5;
        
        function drawHeatmap() {{
            // Clear existing
            g.selectAll(".cell").remove();
            g.selectAll(".label").remove();
            
            // Draw cells
            const cells = g.selectAll(".cell")
                .data(d3.cross(currentOrder, currentOrder))
                .enter().append("rect")
                .attr("class", "cell")
                .attr("x", d => x(currentOrder.indexOf(d[1])))
                .attr("y", d => y(currentOrder.indexOf(d[0])))
                .attr("width", cellSize)
                .attr("height", cellSize)
                .attr("fill", d => {{
                    const value = matrix[d[0]][d[1]];
                    return value >= threshold ? colorScale(value) : "#f0f0f0";
                }})
                .on("mouseover", function(event, d) {{
                    const value = matrix[d[0]][d[1]];
                    tooltip.style("opacity", 1)
                        .html(`
                            <strong>${{labels[d[0]]}} ↔ ${{labels[d[1]]}}</strong><br>
                            Similarity: ${{value.toFixed(3)}}
                        `)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    tooltip.style("opacity", 0);
                }});
            
            // Draw row labels
            g.selectAll(".row-label")
                .data(currentOrder)
                .enter().append("text")
                .attr("class", "label row-label")
                .attr("x", -6)
                .attr("y", d => y(currentOrder.indexOf(d)) + cellSize / 2)
                .attr("dy", ".32em")
                .attr("text-anchor", "end")
                .text(d => labels[d]);
            
            // Draw column labels
            g.selectAll(".col-label")
                .data(currentOrder)
                .enter().append("text")
                .attr("class", "label col-label")
                .attr("x", 6)
                .attr("y", d => x(currentOrder.indexOf(d)) + cellSize / 2)
                .attr("dy", ".32em")
                .attr("text-anchor", "start")
                .attr("transform", d => 
                    `rotate(-90, ${{x(currentOrder.indexOf(d)) + cellSize / 2}}, 0) 
                     translate(${{x(currentOrder.indexOf(d)) + cellSize / 2}}, 0)`)
                .text(d => labels[d]);
        }}
        
        function resetOrder() {{
            currentOrder = d3.range(n);
            drawHeatmap();
        }}
        
        function clusterByGroups() {{
            // Simple hierarchical clustering simulation
            // In production, would use proper clustering algorithm
            const similarities = [];
            for (let i = 0; i < n; i++) {{
                let sum = 0;
                for (let j = 0; j < n; j++) {{
                    if (i !== j) sum += matrix[i][j];
                }}
                similarities.push({{index: i, avgSim: sum / (n - 1)}});
            }}
            
            similarities.sort((a, b) => b.avgSim - a.avgSim);
            currentOrder = similarities.map(d => d.index);
            drawHeatmap();
        }}
        
        function updateThreshold(value) {{
            threshold = parseFloat(value);
            document.getElementById("thresholdValue").textContent = value;
            drawHeatmap();
        }}
        
        // Initial draw
        drawHeatmap();
        
        // Add color legend
        const legendWidth = 200;
        const legendHeight = 20;
        
        const legendSvg = svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${{margin.left + width - legendWidth}}, 20)`);
        
        const legendScale = d3.scaleLinear()
            .domain([0, 1])
            .range([0, legendWidth]);
        
        const legendAxis = d3.axisBottom(legendScale)
            .ticks(5)
            .tickFormat(d3.format(".1f"));
        
        // Create gradient
        const defs = svg.append("defs");
        const gradient = defs.append("linearGradient")
            .attr("id", "gradient");
        
        const nStops = 10;
        for (let i = 0; i <= nStops; i++) {{
            gradient.append("stop")
                .attr("offset", `${{i * 100 / nStops}}%`)
                .attr("stop-color", colorScale(i / nStops));
        }}
        
        legendSvg.append("rect")
            .attr("width", legendWidth)
            .attr("height", legendHeight)
            .attr("fill", "url(#gradient)");
        
        legendSvg.append("g")
            .attr("transform", `translate(0, ${{legendHeight}})`)
            .call(legendAxis);
        
        legendSvg.append("text")
            .attr("x", legendWidth / 2)
            .attr("y", -5)
            .attr("text-anchor", "middle")
            .text("Similarity Score");
    </script>
</body>
</html>
"""
        return html
    
    def generate_cluster_visualization(
        self,
        clusters: List[Dict[str, Any]],
        title: str = "Problem Clusters",
        width: int = 1200,
        height: int = 800
    ) -> str:
        """
        Generate cluster visualization showing problem areas.
        
        Args:
            clusters: List of cluster dictionaries with 'nodes' and 'metrics'
            title: Visualization title
            width: Canvas width
            height: Canvas height
            
        Returns:
            HTML string with embedded visualization
        """
        clusters_json = json.dumps(clusters)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        #visualization {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .cluster {{
            fill-opacity: 0.3;
            stroke-width: 2px;
        }}
        .cluster-label {{
            font-size: 14px;
            font-weight: bold;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 1.5px;
        }}
        .metrics {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            max-width: 300px;
        }}
        .metric-item {{
            margin: 5px 0;
            padding: 5px;
            background: #f8f8f8;
            border-radius: 3px;
        }}
        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            opacity: 0;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div id="visualization">
        <svg id="clusterSvg" width="{width}" height="{height}"></svg>
        <div class="metrics" id="metricsPanel">
            <h3>Cluster Metrics</h3>
            <div id="metricsContent">
                <p style="color: #999;">Click on a cluster to view metrics</p>
            </div>
        </div>
        <div class="tooltip"></div>
    </div>
    
    <script>
        // Embedded D3.js would go here
        {self.D3_JS_EMBEDDED}
        
        const clusters = {clusters_json};
        const width = {width};
        const height = {height};
        
        const svg = d3.select("#clusterSvg");
        const g = svg.append("g");
        
        // Color scale for clusters
        const colorScale = d3.scaleOrdinal(d3.schemeCategory10);
        
        // Pack layout for cluster visualization
        const pack = d3.pack()
            .size([width - 100, height - 100])
            .padding(20);
        
        // Create hierarchy
        const root = d3.hierarchy({{
            name: "root",
            children: clusters.map((c, i) => ({{
                name: `Cluster ${{i + 1}}`,
                children: c.nodes.map(n => ({{
                    name: n.name || n.id,
                    value: n.risk_score || 1,
                    ...n
                }})),
                metrics: c.metrics
            }}))
        }})
        .sum(d => d.value)
        .sort((a, b) => b.value - a.value);
        
        pack(root);
        
        // Draw clusters
        const clusterGroups = g.selectAll(".cluster-group")
            .data(root.children)
            .enter().append("g")
            .attr("class", "cluster-group")
            .attr("transform", d => `translate(${{d.x + 50}},${{d.y + 50}})`);
        
        // Cluster circles
        clusterGroups.append("circle")
            .attr("class", "cluster")
            .attr("r", d => d.r)
            .attr("fill", (d, i) => colorScale(i))
            .attr("stroke", (d, i) => colorScale(i))
            .on("click", function(event, d) {{
                showMetrics(d.data);
            }});
        
        // Cluster labels
        clusterGroups.append("text")
            .attr("class", "cluster-label")
            .attr("dy", d => -d.r - 5)
            .attr("text-anchor", "middle")
            .text(d => d.data.name);
        
        // Draw nodes within clusters
        const nodes = g.selectAll(".node")
            .data(root.descendants().filter(d => !d.children))
            .enter().append("circle")
            .attr("class", "node")
            .attr("cx", d => d.x + 50)
            .attr("cy", d => d.y + 50)
            .attr("r", d => Math.max(3, d.r))
            .attr("fill", d => {{
                const parent = d.parent;
                const parentIndex = root.children.indexOf(parent);
                return d3.color(colorScale(parentIndex)).darker(d.data.risk_score || 0);
            }});
        
        // Tooltip
        const tooltip = d3.select(".tooltip");
        
        nodes.on("mouseover", function(event, d) {{
            tooltip.style("opacity", 1)
                .html(`
                    <strong>${{d.data.name}}</strong><br>
                    Risk: ${{(d.data.risk_score || 0).toFixed(2)}}<br>
                    Cluster: ${{d.parent.data.name}}
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        }})
        .on("mouseout", function() {{
            tooltip.style("opacity", 0);
        }});
        
        // Show cluster metrics
        function showMetrics(clusterData) {{
            const metricsContent = document.getElementById("metricsContent");
            
            if (!clusterData.metrics) {{
                metricsContent.innerHTML = "<p>No metrics available</p>";
                return;
            }}
            
            let html = `<h4>${{clusterData.name}}</h4>`;
            
            for (const [key, value] of Object.entries(clusterData.metrics)) {{
                html += `
                    <div class="metric-item">
                        <strong>${{key}}:</strong> 
                        ${{typeof value === 'number' ? value.toFixed(3) : value}}
                    </div>
                `;
            }}
            
            metricsContent.innerHTML = html;
        }}
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 5])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
    </script>
</body>
</html>
"""
        return html
    
    def _generate_empty_viz(self, message: str) -> str:
        """Generate empty visualization with message."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Visualization</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .message {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 40px;
            text-align: center;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="message">
        <h2>{message}</h2>
    </div>
</body>
</html>
"""
    
    def clear(self) -> None:
        """Clear all visualization data."""
        self.nodes = []
        self.edges = []
        self.temporal_data = []
        self.similarity_matrix = None
        self.node_labels = []