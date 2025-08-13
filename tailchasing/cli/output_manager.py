"""
Output manager for CLI with multiple formats and verbosity levels.
"""

import json
import yaml
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, TextIO
from enum import Enum
from datetime import datetime
from contextlib import contextmanager
import threading

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.rule import Rule
from rich import box

from ..core.issues import Issue, IssueCollection
from ..core.reporting import PathSanitizer


class VerbosityLevel(Enum):
    """Verbosity levels for output."""
    QUIET = 0    # Only errors and critical info
    NORMAL = 1   # Standard output
    VERBOSE = 2  # Detailed output
    DEBUG = 3    # Debug information


class OutputFormat(Enum):
    """Supported output formats."""
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    HTML = "html"
    SARIF = "sarif"  # Static Analysis Results Interchange Format


class OutputManager:
    """Manages CLI output with different formats and verbosity levels."""
    
    def __init__(
        self,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
        output_format: OutputFormat = OutputFormat.TEXT,
        output_file: Optional[Path] = None,
        use_color: bool = True,
        watch_mode: bool = False,
        project_root: Optional[Path] = None
    ):
        """
        Initialize output manager.
        
        Args:
            verbosity: Verbosity level
            output_format: Output format
            output_file: Optional file to write output to
            use_color: Whether to use colored output
            watch_mode: Whether to show live updates
            project_root: Project root for path sanitization
        """
        self.verbosity = verbosity
        self.output_format = output_format
        self.output_file = output_file
        self.use_color = use_color
        self.watch_mode = watch_mode
        self.path_sanitizer = PathSanitizer(project_root)
        
        # Setup console
        # Use stderr for progress/spinners to avoid conflicts with stdout output
        self.console = Console(
            force_terminal=use_color,
            force_jupyter=False,
            file=sys.stderr if not output_file else open(output_file, 'w'),
            legacy_windows=False  # Ensure proper ANSI support
        )
        
        # Progress tracking
        self.progress: Optional[Progress] = None
        self.live: Optional[Live] = None
        self.current_tasks: Dict[str, Any] = {}
        
        # Statistics
        self.start_time = time.time()
        self.stats: Dict[str, Any] = {}
        
    def __enter__(self):
        """Context manager entry."""
        if self.watch_mode and self.output_format == OutputFormat.TEXT:
            self._start_live_display()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.live:
            self.live.stop()
        if self.output_file and hasattr(self.console.file, 'close'):
            self.console.file.close()
            
    def _start_live_display(self):
        """Start live display for watch mode."""
        layout = self._create_watch_layout()
        self.live = Live(layout, console=self.console, refresh_per_second=2)
        self.live.start()
        
    def _create_watch_layout(self) -> Layout:
        """Create layout for watch mode."""
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        # Header
        layout["header"].update(
            Panel(
                Text("TailChasing Analysis - Live Mode", style="bold cyan"),
                box=box.ROUNDED
            )
        )
        
        # Body split
        layout["body"].split_row(
            Layout(name="progress"),
            Layout(name="stats")
        )
        
        return layout
        
    def log(self, message: str, level: VerbosityLevel = VerbosityLevel.NORMAL):
        """
        Log a message at the specified verbosity level.
        
        Args:
            message: Message to log
            level: Verbosity level required to show this message
        """
        if self.verbosity.value >= level.value:
            if self.output_format == OutputFormat.TEXT:
                if level == VerbosityLevel.DEBUG:
                    self.console.print(f"[dim cyan][DEBUG][/dim cyan] {message}")
                elif level == VerbosityLevel.VERBOSE:
                    self.console.print(f"[cyan][INFO][/cyan] {message}")
                else:
                    self.console.print(message)
                    
    def error(self, message: str):
        """Log an error message (always shown)."""
        if self.output_format == OutputFormat.TEXT:
            self.console.print(f"[red][ERROR][/red] {message}", style="bold red")
        elif self.output_format == OutputFormat.JSON:
            self.stats.setdefault("errors", []).append(message)
            
    def warning(self, message: str):
        """Log a warning message."""
        if self.verbosity.value >= VerbosityLevel.NORMAL.value:
            if self.output_format == OutputFormat.TEXT:
                self.console.print(f"[yellow][WARNING][/yellow] {message}")
                
    def success(self, message: str):
        """Log a success message."""
        if self.output_format == OutputFormat.TEXT:
            self.console.print(f"[green]✓[/green] {message}")
            
    @contextmanager
    def progress_context(self, description: str, total: Optional[int] = None):
        """
        Context manager for showing progress.
        
        Args:
            description: Description of the operation
            total: Total number of items (if known)
        """
        if self.output_format != OutputFormat.TEXT or self.verbosity == VerbosityLevel.QUIET:
            yield None
            return
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn() if total else TextColumn(""),
            TaskProgressColumn() if total else TextColumn(""),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
            refresh_per_second=10,  # Lower refresh rate to reduce artifacts
            disable=not self.console.is_terminal  # Disable in non-terminal environments
        ) as progress:
            task_id = progress.add_task(description, total=total or 100)
            
            def update(advance: int = 1):
                progress.update(task_id, advance=advance)
                
            progress.update_func = update
            yield progress
            
    def start_spinner(self, message: str) -> Optional[Any]:
        """Start a spinner for long-running operations."""
        if self.output_format != OutputFormat.TEXT or self.verbosity == VerbosityLevel.QUIET:
            return None
        
        # Disable spinner in CI/CD environments to avoid artifacts
        import os
        if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
            # Just log the message without spinner in CI
            if self.verbosity.value >= VerbosityLevel.NORMAL.value:
                sys.stderr.write(f"{message}\n")
                sys.stderr.flush()
            return None
            
        if not self.progress:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console,
                transient=True,
                refresh_per_second=10,  # Lower refresh rate to reduce artifacts
                disable=not self.console.is_terminal  # Disable in non-terminal environments
            )
            self.progress.start()
            
        task_id = self.progress.add_task(message, total=None)
        self.current_tasks[message] = task_id
        return task_id
        
    def stop_spinner(self, task_id: Optional[Any] = None, message: Optional[str] = None):
        """Stop a spinner."""
        if not self.progress:
            return
            
        try:
            if task_id is not None:
                self.progress.update(task_id, visible=False)  # Hide the task
                self.progress.stop_task(task_id)
                self.progress.remove_task(task_id)  # Remove the task completely
            elif message and message in self.current_tasks:
                task_id = self.current_tasks[message]
                self.progress.update(task_id, visible=False)  # Hide the task
                self.progress.stop_task(task_id)
                self.progress.remove_task(task_id)  # Remove the task completely
                del self.current_tasks[message]
        except KeyError:
            # Task might already be removed
            pass
            
        if not self.current_tasks and self.progress:
            # Refresh one last time to clear the display
            self.progress.refresh()
            self.progress.stop()
            self.progress = None
            # Force clear any remaining artifacts
            self.console.clear_live()
            # Additional cleanup for terminal
            if self.console.is_terminal:
                sys.stderr.write('\r\033[K\033[?25h')  # Clear line and show cursor
                sys.stderr.flush()
            
    def output_issues(self, issues: List[Issue], title: str = "Issues Found"):
        """
        Output issues in the specified format.
        
        Args:
            issues: List of issues to output
            title: Title for the output
        """
        if self.output_format == OutputFormat.TEXT:
            self._output_issues_text(issues, title)
        elif self.output_format == OutputFormat.JSON:
            self._output_issues_json(issues)
        elif self.output_format == OutputFormat.YAML:
            self._output_issues_yaml(issues)
        elif self.output_format == OutputFormat.HTML:
            self._output_issues_html(issues, title)
        elif self.output_format == OutputFormat.SARIF:
            self._output_issues_sarif(issues)
            
    def _output_issues_text(self, issues: List[Issue], title: str):
        """Output issues in text format with rich formatting."""
        if not issues:
            self.console.print(f"[green]No issues found![/green]")
            return
            
        # Group issues by kind
        issues_by_kind: Dict[str, List[Issue]] = {}
        for issue in issues:
            issues_by_kind.setdefault(issue.kind, []).append(issue)
            
        # Create table
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("File", style="yellow")
        table.add_column("Line", justify="right", style="green")
        table.add_column("Message", style="white")
        table.add_column("Severity", justify="center")
        
        # Add rows
        for kind, kind_issues in sorted(issues_by_kind.items()):
            for issue in kind_issues[:5]:  # Show first 5 of each kind
                severity_color = {
                    1: "green",
                    2: "yellow", 
                    3: "red",
                    4: "bold red"
                }.get(issue.severity, "white")
                
                # Sanitize file path for privacy
                sanitized_file = self.path_sanitizer.sanitize(issue.file) if issue.file else ""
                
                table.add_row(
                    kind,
                    sanitized_file,
                    str(issue.line or ""),
                    issue.message[:80] + "..." if len(issue.message) > 80 else issue.message,
                    f"[{severity_color}]{issue.severity}[/{severity_color}]"
                )
                
            if len(kind_issues) > 5:
                table.add_row(
                    "",
                    f"[dim]... and {len(kind_issues) - 5} more[/dim]",
                    "",
                    "",
                    ""
                )
                
        self.console.print(table)
        
        # Summary
        self.console.print()
        self.console.print(Panel(
            f"Total: [bold]{len(issues)}[/bold] issues in [bold]{len(issues_by_kind)}[/bold] categories",
            title="Summary",
            border_style="blue"
        ))
        
    def _output_issues_json(self, issues: List[Issue]):
        """Output issues in JSON format."""
        # Sanitize issue paths before serialization
        sanitized_issues = []
        for issue in issues:
            sanitized_issue = self.path_sanitizer.sanitize_issue(issue)
            sanitized_issues.append(sanitized_issue.to_dict())
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(issues),
            "issues": sanitized_issues,
            "statistics": self.stats
        }
        
        json_str = json.dumps(output, indent=2, default=str)
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(json_str)
        else:
            print(json_str)
            
    def _output_issues_yaml(self, issues: List[Issue]):
        """Output issues in YAML format."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_issues": len(issues),
            "issues": [issue.to_dict() for issue in issues],
            "statistics": self.stats
        }
        
        yaml_str = yaml.dump(output, default_flow_style=False, sort_keys=False)
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(yaml_str)
        else:
            print(yaml_str)
            
    def _output_issues_html(self, issues: List[Issue], title: str):
        """Output issues in HTML format."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .severity-1 {{ color: green; }}
        .severity-2 {{ color: orange; }}
        .severity-3 {{ color: red; }}
        .severity-4 {{ color: darkred; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total Issues: {len(issues)}</p>
    
    <table>
        <tr>
            <th>Type</th>
            <th>File</th>
            <th>Line</th>
            <th>Message</th>
            <th>Severity</th>
        </tr>
"""
        
        for issue in issues:
            # Sanitize file path for privacy
            sanitized_file = self.path_sanitizer.sanitize(issue.file) if issue.file else ''
            html += f"""
        <tr>
            <td>{issue.kind}</td>
            <td>{sanitized_file}</td>
            <td>{issue.line or ''}</td>
            <td>{issue.message}</td>
            <td class="severity-{issue.severity}">{issue.severity}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(html)
        else:
            print(html)
            
    def _output_issues_sarif(self, issues: List[Issue]):
        """Output issues in SARIF format for GitHub/VS Code integration."""
        sarif = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "TailChasingFixer",
                        "version": "1.0.0",
                        "informationUri": "https://github.com/rohanvinaik/TailChasingFixer"
                    }
                },
                "results": []
            }]
        }
        
        for issue in issues:
            result = {
                "message": {
                    "text": issue.message
                },
                "level": {
                    1: "note",
                    2: "warning",
                    3: "error",
                    4: "error"
                }.get(issue.severity, "warning"),
                "ruleId": issue.kind
            }
            
            if issue.file and issue.line:
                result["locations"] = [{
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": issue.file
                        },
                        "region": {
                            "startLine": issue.line,
                            "endLine": issue.end_line or issue.line
                        }
                    }
                }]
                
            sarif["runs"][0]["results"].append(result)
            
        sarif_str = json.dumps(sarif, indent=2)
        
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(sarif_str)
        else:
            print(sarif_str)
            
    def show_performance_profile(self, profile_data: Dict[str, Any]):
        """
        Show performance profiling data.
        
        Args:
            profile_data: Dictionary containing profiling information
        """
        if self.output_format != OutputFormat.TEXT:
            self.stats["profile"] = profile_data
            return
            
        # Create performance table
        table = Table(title="Performance Profile", show_header=True, header_style="bold cyan")
        table.add_column("Component", style="yellow")
        table.add_column("Time (s)", justify="right", style="green")
        table.add_column("Percentage", justify="right")
        table.add_column("Calls", justify="right")
        table.add_column("Avg (ms)", justify="right")
        
        total_time = sum(data.get("time", 0) for data in profile_data.values())
        
        for component, data in sorted(profile_data.items(), key=lambda x: x[1].get("time", 0), reverse=True):
            time_val = data.get("time", 0)
            calls = data.get("calls", 1)
            percentage = (time_val / total_time * 100) if total_time > 0 else 0
            avg_ms = (time_val / calls * 1000) if calls > 0 else 0
            
            table.add_row(
                component,
                f"{time_val:.3f}",
                f"{percentage:.1f}%",
                str(calls),
                f"{avg_ms:.2f}"
            )
            
        self.console.print(table)
        
        # Memory usage if available
        if "memory" in profile_data:
            mem_table = Table(title="Memory Usage", show_header=True, header_style="bold cyan")
            mem_table.add_column("Metric", style="yellow")
            mem_table.add_column("Value", justify="right", style="green")
            
            for metric, value in profile_data["memory"].items():
                mem_table.add_row(metric, str(value))
                
            self.console.print(mem_table)
            
    def show_dry_run_summary(self, files: List[str], analyzers: List[str], config: Dict[str, Any]):
        """
        Show what would be analyzed in dry-run mode.
        
        Args:
            files: List of files that would be analyzed
            analyzers: List of analyzers that would run
            config: Configuration that would be used
        """
        if self.output_format == OutputFormat.JSON:
            output = {
                "dry_run": True,
                "files_count": len(files),
                "files": files[:10],  # First 10 files
                "analyzers": analyzers,
                "config": config
            }
            print(json.dumps(output, indent=2))
            return
            
        # Create tree view
        tree = Tree("[bold cyan]Dry Run Summary[/bold cyan]")
        
        # Files
        files_branch = tree.add(f"[yellow]Files to analyze:[/yellow] {len(files)}")
        for file in files[:10]:
            files_branch.add(f"[dim]{file}[/dim]")
        if len(files) > 10:
            files_branch.add(f"[dim]... and {len(files) - 10} more[/dim]")
            
        # Analyzers
        analyzers_branch = tree.add(f"[yellow]Analyzers to run:[/yellow] {len(analyzers)}")
        for analyzer in analyzers:
            analyzers_branch.add(f"[green]• {analyzer}[/green]")
            
        # Config highlights
        config_branch = tree.add("[yellow]Configuration:[/yellow]")
        important_configs = ["excluded_paths", "resource_limits", "risk_thresholds"]
        for key in important_configs:
            if key in config:
                config_branch.add(f"[cyan]{key}:[/cyan] {config[key]}")
                
        self.console.print(tree)
        
        # Estimated time
        estimated_time = len(files) * 0.1 * len(analyzers)  # Rough estimate
        self.console.print()
        self.console.print(Panel(
            f"Estimated analysis time: [bold]{estimated_time:.1f}[/bold] seconds\n"
            f"Use [bold green]--no-dry-run[/bold green] to actually run the analysis",
            title="Ready to Analyze",
            border_style="green"
        ))
        
    def update_watch_display(self, current_file: str, progress: float, issues_found: int):
        """
        Update watch mode display.
        
        Args:
            current_file: Currently analyzing file
            progress: Progress percentage (0-100)
            issues_found: Number of issues found so far
        """
        if not self.live or not self.watch_mode:
            return
            
        layout = self.live.renderable
        
        # Update progress
        progress_text = f"""
[cyan]Current File:[/cyan] {current_file}
[cyan]Progress:[/cyan] {progress:.1f}%
[cyan]Issues Found:[/cyan] {issues_found}
[cyan]Elapsed Time:[/cyan] {time.time() - self.start_time:.1f}s
"""
        layout["body"]["progress"].update(Panel(progress_text, title="Progress"))
        
        # Update stats
        stats_text = self._format_stats()
        layout["body"]["stats"].update(Panel(stats_text, title="Statistics"))
        
    def _format_stats(self) -> str:
        """Format statistics for display."""
        lines = []
        for key, value in self.stats.items():
            if isinstance(value, (int, float)):
                lines.append(f"[yellow]{key}:[/yellow] {value}")
            elif isinstance(value, list):
                lines.append(f"[yellow]{key}:[/yellow] {len(value)} items")
        return "\n".join(lines)
        
    def finish(self):
        """Finish output and close resources."""
        if self.progress:
            # Make all tasks invisible before stopping
            for task_id in list(self.current_tasks.values()):
                try:
                    self.progress.update(task_id, visible=False)
                except KeyError:
                    pass
            self.progress.stop()
            self.progress = None
            # Clear any remaining artifacts
            if self.console.is_terminal:
                sys.stderr.write('\r\033[K\033[?25h')  # Clear line and show cursor
                sys.stderr.flush()
        if self.live:
            self.live.stop()
        if self.output_file and hasattr(self.console.file, 'close'):
            self.console.file.close()