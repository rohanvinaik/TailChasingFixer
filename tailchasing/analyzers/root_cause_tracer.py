"""
Interactive root cause tracing for tail-chasing chains.
Provides visualization and tracing of how errors propagate through LLM fixes.
"""

import ast
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import os
from ..core.issues import Issue


@dataclass
class TailChaseEvent:
    """Represents a single event in a tail-chasing chain."""
    timestamp: str
    event_type: str  # 'error', 'fix_attempt', 'new_issue'
    file: str
    line: int
    description: str
    code_snapshot: Optional[str] = None
    related_symbols: List[str] = None
    confidence: float = 1.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TailChaseChain:
    """Represents a complete tail-chasing chain."""
    chain_id: str
    root_cause: TailChaseEvent
    events: List[TailChaseEvent]
    total_iterations: int
    files_affected: Set[str]
    risk_score: int
    pattern_type: str
    resolution_suggestions: List[str]
    
    def to_dict(self):
        return {
            'chain_id': self.chain_id,
            'root_cause': self.root_cause.to_dict(),
            'events': [e.to_dict() for e in self.events],
            'total_iterations': self.total_iterations,
            'files_affected': list(self.files_affected),
            'risk_score': self.risk_score,
            'pattern_type': self.pattern_type,
            'resolution_suggestions': self.resolution_suggestions
        }


class RootCauseTracer:
    """Trace and visualize tail-chasing chains."""
    
    def __init__(self):
        self.chains = []
        self.event_index = defaultdict(list)
        
    def analyze_tail_chase_chains(self, issues: List[Issue], 
                                git_history: Optional[Dict] = None) -> List[TailChaseChain]:
        """Analyze issues to identify and trace tail-chasing chains."""
        # Group related issues
        issue_groups = self._group_related_issues(issues)
        
        # Build chains from groups
        for group in issue_groups:
            chain = self._build_chain_from_issues(group, git_history)
            if chain and len(chain.events) > 1:  # Only include actual chains
                self.chains.append(chain)
        
        return self.chains
    
    def generate_visual_report(self, output_path: str = "tail_chase_report.html") -> str:
        """Generate an interactive HTML visualization of tail-chasing chains."""
        html_content = self._generate_html_visualization()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def generate_mermaid_diagram(self, chain: TailChaseChain) -> str:
        """Generate a Mermaid diagram for a specific chain."""
        mermaid = ["graph TD"]
        
        # Add root cause
        root_id = "root"
        mermaid.append(f'    {root_id}["{self._escape_mermaid(chain.root_cause.description)}"]')
        mermaid.append(f'    {root_id}:::rootCause')
        
        # Add events and connections
        prev_id = root_id
        for i, event in enumerate(chain.events):
            event_id = f"event{i}"
            label = f"{event.event_type}: {self._escape_mermaid(event.description)}"
            
            # Style based on event type
            if event.event_type == 'error':
                mermaid.append(f'    {event_id}["{label}"]:::error')
            elif event.event_type == 'fix_attempt':
                mermaid.append(f'    {event_id}["{label}"]:::fix')
            else:
                mermaid.append(f'    {event_id}["{label}"]:::newIssue')
            
            # Add connection
            mermaid.append(f'    {prev_id} --> {event_id}')
            prev_id = event_id
        
        # Add styles
        mermaid.extend([
            "    classDef rootCause fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff",
            "    classDef error fill:#ffa94d,stroke:#fd7e14,stroke-width:2px",
            "    classDef fix fill:#8ce99a,stroke:#51cf66,stroke-width:2px",
            "    classDef newIssue fill:#ffd43b,stroke:#fab005,stroke-width:2px"
        ])
        
        return "\n".join(mermaid)
    
    def _group_related_issues(self, issues: List[Issue]) -> List[List[Issue]]:
        """Group issues that are likely part of the same tail-chasing chain."""
        groups = []
        used = set()
        
        # Sort issues by file and line for better grouping
        sorted_issues = sorted(issues, key=lambda i: (i.file or '', i.line or 0))
        
        for i, issue in enumerate(sorted_issues):
            if i in used:
                continue
            
            group = [issue]
            used.add(i)
            
            # Find related issues
            for j, other in enumerate(sorted_issues[i+1:], i+1):
                if j in used:
                    continue
                
                if self._are_issues_related(issue, other):
                    group.append(other)
                    used.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _are_issues_related(self, issue1: Issue, issue2: Issue) -> bool:
        """Determine if two issues are part of the same tail-chasing chain."""
        # Same file or related files
        if issue1.file == issue2.file:
            return True
        
        # Check if symbols are related
        if issue1.symbol and issue2.symbol:
            # Check for naming patterns (e.g., function and function_v2)
            if issue1.symbol in issue2.symbol or issue2.symbol in issue1.symbol:
                return True
        
        # Check evidence for relationships
        evidence1 = issue1.evidence or {}
        evidence2 = issue2.evidence or {}
        
        # Check for shared symbols in evidence
        symbols1 = set(evidence1.get('related_symbols', []))
        symbols2 = set(evidence2.get('related_symbols', []))
        
        if symbols1 & symbols2:  # Intersection
            return True
        
        # Check for specific patterns that indicate chaining
        patterns = ['phantom', 'hallucination', 'circular', 'duplicate']
        
        kind1_patterns = any(p in issue1.kind for p in patterns)
        kind2_patterns = any(p in issue2.kind for p in patterns)
        
        return kind1_patterns and kind2_patterns
    
    def _build_chain_from_issues(self, issues: List[Issue], 
                               git_history: Optional[Dict]) -> Optional[TailChaseChain]:
        """Build a tail-chasing chain from a group of related issues."""
        if not issues:
            return None
        
        # Sort by line number to approximate temporal order
        sorted_issues = sorted(issues, key=lambda i: (i.file or '', i.line or 0))
        
        # Identify root cause (usually the first issue)
        root_issue = sorted_issues[0]
        root_cause = TailChaseEvent(
            timestamp=datetime.now().isoformat(),
            event_type='error',
            file=root_issue.file or 'unknown',
            line=root_issue.line or 0,
            description=root_issue.message,
            related_symbols=[root_issue.symbol] if root_issue.symbol else []
        )
        
        # Build event chain
        events = []
        files_affected = set()
        
        for issue in sorted_issues[1:]:
            event_type = self._determine_event_type(issue)
            
            event = TailChaseEvent(
                timestamp=datetime.now().isoformat(),
                event_type=event_type,
                file=issue.file or 'unknown',
                line=issue.line or 0,
                description=issue.message,
                related_symbols=[issue.symbol] if issue.symbol else [],
                confidence=0.8  # Could be enhanced with more analysis
            )
            
            events.append(event)
            if issue.file:
                files_affected.add(issue.file)
        
        # Determine pattern type
        pattern_type = self._identify_pattern_type(issues)
        
        # Generate resolution suggestions
        suggestions = self._generate_resolution_suggestions(pattern_type, issues)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(issues, events)
        
        return TailChaseChain(
            chain_id=f"chain_{hash(tuple(sorted(files_affected)))}",
            root_cause=root_cause,
            events=events,
            total_iterations=len(events) + 1,
            files_affected=files_affected,
            risk_score=risk_score,
            pattern_type=pattern_type,
            resolution_suggestions=suggestions
        )
    
    def _determine_event_type(self, issue: Issue) -> str:
        """Determine the type of event based on issue characteristics."""
        if 'fix' in issue.kind or 'attempt' in issue.kind:
            return 'fix_attempt'
        elif 'error' in issue.kind or 'missing' in issue.kind:
            return 'error'
        else:
            return 'new_issue'
    
    def _identify_pattern_type(self, issues: List[Issue]) -> str:
        """Identify the predominant pattern type in the chain."""
        pattern_counts = defaultdict(int)
        
        for issue in issues:
            if 'phantom' in issue.kind:
                pattern_counts['phantom_implementation'] += 1
            elif 'circular' in issue.kind:
                pattern_counts['circular_dependency'] += 1
            elif 'duplicate' in issue.kind:
                pattern_counts['duplication_cascade'] += 1
            elif 'hallucination' in issue.kind:
                pattern_counts['hallucination_chain'] += 1
            else:
                pattern_counts['general_tail_chase'] += 1
        
        return max(pattern_counts, key=pattern_counts.get)
    
    def _generate_resolution_suggestions(self, pattern_type: str, 
                                       issues: List[Issue]) -> List[str]:
        """Generate specific resolution suggestions based on pattern type."""
        base_suggestions = [
            "Review the original requirements to understand the actual need",
            "Consider refactoring instead of patching",
            "Run comprehensive tests before making changes"
        ]
        
        pattern_suggestions = {
            'phantom_implementation': [
                "Remove all phantom implementations and start fresh",
                "Verify which functions are actually needed",
                "Check imports against actual module contents"
            ],
            'circular_dependency': [
                "Create a dependency graph to visualize the cycle",
                "Extract shared functionality to a separate module",
                "Consider using dependency injection"
            ],
            'duplication_cascade': [
                "Identify the canonical implementation",
                "Remove all duplicates and create proper imports",
                "Use a shared utility module"
            ],
            'hallucination_chain': [
                "Stop the LLM session and review manually",
                "Delete all hallucinated code",
                "Provide more context in future prompts"
            ]
        }
        
        suggestions = base_suggestions + pattern_suggestions.get(pattern_type, [])
        
        # Add issue-specific suggestions
        for issue in issues[:3]:  # First few issues
            if issue.suggestions:
                suggestions.extend(issue.suggestions[:1])  # Add first suggestion
        
        return list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order
    
    def _calculate_risk_score(self, issues: List[Issue], events: List[TailChaseEvent]) -> int:
        """Calculate risk score based on chain characteristics."""
        base_score = len(events) * 2  # Each event adds risk
        
        # Add severity scores
        severity_score = sum(issue.severity for issue in issues)
        
        # File spread multiplier
        files_affected = len(set(issue.file for issue in issues if issue.file))
        spread_multiplier = 1 + (files_affected - 1) * 0.5
        
        # Pattern severity multiplier
        pattern_multipliers = {
            'hallucination_chain': 2.0,
            'circular_dependency': 1.8,
            'phantom_implementation': 1.5,
            'duplication_cascade': 1.3
        }
        
        pattern_type = self._identify_pattern_type(issues)
        pattern_multiplier = pattern_multipliers.get(pattern_type, 1.0)
        
        total_score = int((base_score + severity_score) * spread_multiplier * pattern_multiplier)
        
        return min(total_score, 100)  # Cap at 100
    
    def _escape_mermaid(self, text: str) -> str:
        """Escape text for Mermaid diagram."""
        return text.replace('"', "'").replace('\n', ' ')[:50]  # Truncate long text
    
    def _generate_html_visualization(self) -> str:
        """Generate complete HTML visualization."""
        chains_json = json.dumps([chain.to_dict() for chain in self.chains], indent=2)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tail-Chasing Analysis Report</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .chain-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chain-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .risk-score {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }}
        .risk-high {{ background: #ff6b6b; color: white; }}
        .risk-medium {{ background: #ffd43b; color: #333; }}
        .risk-low {{ background: #8ce99a; color: #333; }}
        .mermaid-container {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 4px;
            margin: 15px 0;
            overflow-x: auto;
        }}
        .event-list {{
            margin: 15px 0;
        }}
        .event {{
            padding: 10px;
            margin: 5px 0;
            border-left: 3px solid #3498db;
            background: #f8f9fa;
        }}
        .suggestions {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 4px;
            margin-top: 15px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔄 Tail-Chasing Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{len(self.chains)}</div>
            <div>Total Chains Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{sum(chain.total_iterations for chain in self.chains)}</div>
            <div>Total Iterations</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(set().union(*[chain.files_affected for chain in self.chains]))}</div>
            <div>Files Affected</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{max([chain.risk_score for chain in self.chains]) if self.chains else 0}</div>
            <div>Highest Risk Score</div>
        </div>
    </div>
    
    <div id="chains"></div>
    
    <script>
        mermaid.initialize({{ startOnLoad: true }});
        
        const chains = {chains_json};
        const container = document.getElementById('chains');
        
        chains.forEach((chain, index) => {{
            const riskClass = chain.risk_score > 70 ? 'risk-high' : 
                             chain.risk_score > 40 ? 'risk-medium' : 'risk-low';
            
            const chainHtml = `
                <div class="chain-container">
                    <div class="chain-header">
                        <h2>Chain #${{index + 1}}: ${{chain.pattern_type.replace(/_/g, ' ').toUpperCase()}}</h2>
                        <span class="risk-score ${{riskClass}}">Risk Score: ${{chain.risk_score}}</span>
                    </div>
                    
                    <div class="mermaid-container">
                        <div class="mermaid" id="diagram-${{index}}"></div>
                    </div>
                    
                    <h3>📍 Root Cause</h3>
                    <div class="event">
                        <strong>${{chain.root_cause.file}}:${{chain.root_cause.line}}</strong><br>
                        ${{chain.root_cause.description}}
                    </div>
                    
                    <h3>🔗 Event Chain (${{chain.total_iterations}} iterations)</h3>
                    <div class="event-list">
                        ${{chain.events.map(event => `
                            <div class="event">
                                <strong>${{event.event_type}}</strong> at ${{event.file}}:${{event.line}}<br>
                                ${{event.description}}
                            </div>
                        `).join('')}}
                    </div>
                    
                    <div class="suggestions">
                        <h3>💡 Resolution Suggestions</h3>
                        <ul>
                            ${{chain.resolution_suggestions.map(s => `<li>${{s}}</li>`).join('')}}
                        </ul>
                    </div>
                </div>
            `;
            
            container.innerHTML += chainHtml;
            
            // Generate Mermaid diagram
            const diagram = generateMermaidDiagram(chain);
            mermaid.render(`mermaid-${{index}}`, diagram, (svgCode) => {{
                document.getElementById(`diagram-${{index}}`).innerHTML = svgCode;
            }});
        }});
        
        function generateMermaidDiagram(chain) {{
            let diagram = 'graph TD\\n';
            diagram += `    root["${{chain.root_cause.description.substring(0, 50)}}..."]\\n`;
            diagram += '    root:::rootCause\\n';
            
            let prevId = 'root';
            chain.events.forEach((event, i) => {{
                const eventId = `event${{i}}`;
                const label = `${{event.event_type}}: ${{event.description.substring(0, 40)}}...`;
                
                if (event.event_type === 'error') {{
                    diagram += `    ${{eventId}}["${{label}}"]:::error\\n`;
                }} else if (event.event_type === 'fix_attempt') {{
                    diagram += `    ${{eventId}}["${{label}}"]:::fix\\n`;
                }} else {{
                    diagram += `    ${{eventId}}["${{label}}"]:::newIssue\\n`;
                }}
                
                diagram += `    ${{prevId}} --> ${{eventId}}\\n`;
                prevId = eventId;
            }});
            
            diagram += '    classDef rootCause fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff\\n';
            diagram += '    classDef error fill:#ffa94d,stroke:#fd7e14,stroke-width:2px\\n';
            diagram += '    classDef fix fill:#8ce99a,stroke:#51cf66,stroke-width:2px\\n';
            diagram += '    classDef newIssue fill:#ffd43b,stroke:#fab005,stroke-width:2px\\n';
            
            return diagram;
        }}
    </script>
</body>
</html>
"""
        return html
