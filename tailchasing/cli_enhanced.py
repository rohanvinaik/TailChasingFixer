"""
Enhanced CLI with advanced features for tail-chasing detection.

Provides a rich command-line interface with progress bars, colored output,
and advanced analysis capabilities.
"""

import click
import sys
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree
from rich import print as rprint

from .utils.logging_setup import get_logger
from .loader import collect_files, parse_files
from .symbols import SymbolTable
from .issues import Issue
from .detector import TailChasingDetector
from .analyzers.base import AnalysisContext
from .analyzers.explainer import TailChasingExplainer
from .fixers.suggestion_generator import SuggestionGenerator
from .fixers.fix_applier import FixApplier
from .visualization.report_generator import ReportGenerator
from .visualization.tail_chase_visualizer import TailChaseVisualizer
from .orchestration.orchestrator import TailChasingOrchestrator
from .llm_integration.feedback_generator import FeedbackGenerator
from .semantic.encoder import SemanticEncoder
from .semantic.index import SemanticIndex
from .performance.cache import get_cache_manager
from .performance.parallel import ParallelExecutor
from .performance.monitor import get_monitor, track_performance
from .config import Config


console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """
    TailChasingFixer - Detect and fix LLM-induced anti-patterns in your codebase.
    
    Use 'tailchasing COMMAND --help' for more information on a command.
    """
    if ctx.invoked_subcommand is None:
        # Default to analyze command
        ctx.invoke(analyze)


@cli.command()
@click.argument('path', type=click.Path(exists=True), default='.')
@click.option('--deep', is_flag=True, help='Run all enhanced analyzers including semantic analysis')
@click.option('--ml-enhanced', is_flag=True, help='Use ML-powered detection algorithms')
@click.option('--semantic-analysis', is_flag=True, help='Enable hypervector-based semantic analysis')
@click.option('--confidence-threshold', type=float, default=0.7, help='Set detection confidence threshold (0.0-1.0)')
@click.option('--output-format', type=click.Choice(['text', 'json', 'html', 'markdown']), default='text', help='Choose output format')
@click.option('--severity', type=click.IntRange(1, 5), help='Only show issues with this severity or higher')
@click.option('--parallel', is_flag=True, help='Enable parallel processing for large codebases')
@click.option('--cache', is_flag=True, help='Enable caching for incremental analysis')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@track_performance("cli_analyze")
def analyze(path, deep, ml_enhanced, semantic_analysis, confidence_threshold, 
           output_format, severity, parallel, cache, verbose):
    """
    Analyze codebase for tail-chasing patterns.
    
    Examples:
    
        tailchasing analyze --deep
        
        tailchasing analyze src/ --semantic-analysis --confidence-threshold 0.8
        
        tailchasing analyze --ml-enhanced --output-format json > report.json
    """
    console.print(Panel.fit(
        "[bold cyan]TailChasingFixer Analysis[/bold cyan]\n"
        f"Analyzing: [yellow]{path}[/yellow]",
        border_style="cyan"
    ))
    
    # Initialize components
    detector = TailChasingDetector()
    orchestrator = TailChasingOrchestrator()
    monitor = get_monitor(enable_profiling=verbose)
    
    if cache:
        cache_manager = get_cache_manager()
        console.print("[dim]Cache enabled[/dim]")
    
    # Collect files
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # File collection
        task = progress.add_task("Collecting files...", total=None)
        root_path = Path(path).resolve()
        
        config = Config.find_and_load(root_path).to_dict()
        config['confidence_threshold'] = confidence_threshold
        
        if parallel:
            config['max_workers'] = None  # Use all CPUs
        
        if semantic_analysis or deep:
            config['enable_semantic'] = True
        
        if ml_enhanced or deep:
            config['enable_ml'] = True
        
        # Run analysis
        progress.update(task, description="Running analysis...")
        
        with monitor.track("full_analysis"):
            result = orchestrator.orchestrate(
                path=root_path,
                auto_fix=False,
                config=config
            )
        
        progress.update(task, completed=True)
    
    issues = result.get('issues', [])
    
    # Filter by severity if requested
    if severity:
        issues = [i for i in issues if i.severity >= severity]
    
    # Display results based on format
    if output_format == 'json':
        output = {
            'issues': [i.to_dict() for i in issues],
            'summary': {
                'total_issues': len(issues),
                'by_severity': _count_by_severity(issues),
                'by_type': _count_by_type(issues)
            },
            'performance': monitor.get_summary() if verbose else None
        }
        console.print_json(data=output)
    
    elif output_format == 'html':
        report_gen = ReportGenerator()
        report_gen.add_issues(issues)
        html = report_gen.generate_html_report(include_visualizations=True)
        
        output_file = Path('tailchasing_report.html')
        output_file.write_text(html)
        console.print(f"[green]HTML report saved to: {output_file}[/green]")
    
    elif output_format == 'markdown':
        report_gen = ReportGenerator()
        report_gen.add_issues(issues)
        markdown = report_gen.generate_markdown_summary()
        console.print(markdown)
    
    else:  # text format
        _display_text_results(issues, verbose, monitor)
    
    # Show performance summary if verbose
    if verbose:
        _display_performance_summary(monitor)
    
    # Exit code based on issues found
    if issues:
        sys.exit(1)
    else:
        console.print("[green]✅ No issues found![/green]")
        sys.exit(0)


@cli.command()
@click.argument('path', type=click.Path(exists=True), default='.')
@click.option('--auto', is_flag=True, help='Apply automatic fixes without confirmation')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without making changes')
@click.option('--severity', type=click.IntRange(1, 5), help='Only fix issues with this severity or higher')
@click.option('--backup', is_flag=True, default=True, help='Create backups before applying fixes')
@click.option('--validate', is_flag=True, help='Validate fixes before applying')
@track_performance("cli_fix")
def fix(path, auto, dry_run, severity, backup, validate):
    """
    Apply automatic fixes for detected issues.
    
    Examples:
    
        tailchasing fix --dry-run
        
        tailchasing fix src/ --auto --severity 3
        
        tailchasing fix --validate --backup
    """
    console.print(Panel.fit(
        "[bold yellow]TailChasingFixer Auto-Fix[/bold yellow]\n"
        f"Target: [cyan]{path}[/cyan]\n"
        f"Mode: [{'green' if not dry_run else 'yellow'}]{'Dry Run' if dry_run else 'Apply Fixes'}[/]",
        border_style="yellow"
    ))
    
    # Run detection first
    orchestrator = TailChasingOrchestrator({
        'auto_fix': True,
        'dry_run': dry_run,
        'validate_fixes': validate,
        'create_backups': backup
    })
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Detecting issues...", total=None)
        
        result = orchestrator.orchestrate(
            path=Path(path).resolve(),
            auto_fix=True,
            dry_run=dry_run
        )
        
        progress.update(task, description="Generating fixes...", completed=50)
    
    issues = result.get('issues', [])
    fixes_applied = result.get('fixes_applied', 0)
    
    if severity:
        issues = [i for i in issues if i.severity >= severity]
    
    if not issues:
        console.print("[yellow]No issues found to fix.[/yellow]")
        return
    
    # Display fix plan
    table = Table(title="Fix Plan", show_header=True, header_style="bold magenta")
    table.add_column("Issue Type", style="cyan")
    table.add_column("File", style="yellow")
    table.add_column("Severity", justify="center")
    table.add_column("Fix Available", justify="center")
    
    for issue in issues[:20]:  # Show first 20
        severity_color = _get_severity_color(issue.severity)
        table.add_row(
            issue.kind,
            str(issue.file)[:50],
            f"[{severity_color}]{issue.severity}[/]",
            "[green]✓[/green]" if issue.suggestions else "[red]✗[/red]"
        )
    
    console.print(table)
    
    if not auto and not dry_run:
        if not click.confirm(f"Apply {fixes_applied} fixes?"):
            console.print("[yellow]Fix application cancelled.[/yellow]")
            return
    
    if not dry_run:
        console.print(f"[green]✅ Applied {fixes_applied} fixes successfully![/green]")
        console.print("[dim]Run your tests to ensure everything works correctly.[/dim]")
    else:
        console.print(f"[yellow]Dry run complete. {fixes_applied} fixes would be applied.[/yellow]")


@cli.command()
@click.argument('issue_id', required=False)
@click.option('--file', type=click.Path(exists=True), help='Explain issues in specific file')
@click.option('--type', 'issue_type', help='Explain specific issue type')
@click.option('--examples', is_flag=True, help='Include examples in explanation')
def explain(issue_id, file, issue_type, examples):
    """
    Get detailed explanations for issues.
    
    Examples:
    
        tailchasing explain duplicate_function
        
        tailchasing explain --file src/utils.py
        
        tailchasing explain --type circular_import --examples
    """
    explainer = TailChasingExplainer()
    
    console.print(Panel.fit(
        "[bold blue]TailChasingFixer Explainer[/bold blue]",
        border_style="blue"
    ))
    
    if issue_type:
        # Explain a specific pattern type
        explanation = _get_pattern_explanation(issue_type, examples)
        console.print(explanation)
    
    elif file:
        # Explain issues in a specific file
        detector = TailChasingDetector()
        issues = detector.detect(Path(file).parent)
        file_issues = [i for i in issues if str(i.file) == file]
        
        if not file_issues:
            console.print(f"[yellow]No issues found in {file}[/yellow]")
            return
        
        for issue in file_issues:
            explanation = explainer.explain_issue_enhanced(issue)
            _display_explanation(explanation)
    
    elif issue_id:
        # Explain specific issue by ID (would need issue tracking)
        console.print("[yellow]Issue ID lookup not yet implemented[/yellow]")
    
    else:
        # Show general explanation of all pattern types
        _display_pattern_catalog()


@cli.command()
@click.argument('path', type=click.Path(exists=True), default='.')
@click.option('--output', type=click.Path(), default='report.html', help='Output file for visualization')
@click.option('--open', 'open_browser', is_flag=True, help='Open visualization in browser')
@click.option('--include-graphs', is_flag=True, default=True, help='Include dependency graphs')
@click.option('--include-heatmaps', is_flag=True, default=True, help='Include similarity heatmaps')
@track_performance("cli_visualize")
def visualize(path, output, open_browser, include_graphs, include_heatmaps):
    """
    Generate interactive visualization reports.
    
    Examples:
    
        tailchasing visualize --open
        
        tailchasing visualize src/ --output analysis.html
        
        tailchasing visualize --include-heatmaps
    """
    console.print(Panel.fit(
        "[bold magenta]TailChasingFixer Visualizer[/bold magenta]\n"
        f"Analyzing: [cyan]{path}[/cyan]",
        border_style="magenta"
    ))
    
    # Run analysis
    detector = TailChasingDetector()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Detecting issues...", total=None)
        issues = detector.detect(Path(path).resolve())
        
        progress.update(task, description="Generating visualizations...")
        
        # Generate visualizations
        visualizer = TailChaseVisualizer()
        visualizer.add_issues(issues)
        
        if include_graphs:
            dep_graph = visualizer.generate_dependency_graph()
        
        if include_heatmaps:
            heatmap = visualizer.generate_similarity_heatmap()
        
        # Generate report
        report_gen = ReportGenerator()
        report_gen.add_issues(issues)
        
        html = report_gen.generate_html_report(
            output_path=Path(output),
            include_visualizations=True,
            embed_data=True
        )
        
        progress.update(task, completed=True)
    
    console.print(f"[green]✅ Visualization saved to: {output}[/green]")
    
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{Path(output).resolve()}")


# Helper functions

def _count_by_severity(issues: List[Issue]) -> Dict[int, int]:
    """Count issues by severity."""
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for issue in issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
    return counts


def _count_by_type(issues: List[Issue]) -> Dict[str, int]:
    """Count issues by type."""
    counts = {}
    for issue in issues:
        counts[issue.kind] = counts.get(issue.kind, 0) + 1
    return counts


def _get_severity_color(severity: int) -> str:
    """Get color for severity level."""
    colors = {
        1: "green",
        2: "yellow", 
        3: "yellow",
        4: "red",
        5: "bold red"
    }
    return colors.get(severity, "white")


def _display_text_results(issues: List[Issue], verbose: bool, monitor):
    """Display results in text format."""
    if not issues:
        console.print("[green]✅ No tail-chasing patterns detected![/green]")
        return
    
    # Summary
    severity_counts = _count_by_severity(issues)
    type_counts = _count_by_type(issues)
    
    console.print(f"\n[bold]Found {len(issues)} issue(s)[/bold]\n")
    
    # Severity breakdown
    severity_table = Table(show_header=True, header_style="bold")
    severity_table.add_column("Severity", justify="center")
    severity_table.add_column("Count", justify="center")
    severity_table.add_column("Level")
    
    severity_names = {
        1: "Info", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"
    }
    
    for sev in range(5, 0, -1):
        if severity_counts[sev] > 0:
            color = _get_severity_color(sev)
            severity_table.add_row(
                f"[{color}]{sev}[/]",
                str(severity_counts[sev]),
                f"[{color}]{severity_names[sev]}[/]"
            )
    
    console.print(severity_table)
    
    # Type breakdown
    console.print("\n[bold]Issues by Type:[/bold]")
    type_tree = Tree("Issue Types")
    for issue_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        type_tree.add(f"{issue_type}: {count}")
    console.print(type_tree)
    
    # Detailed issues (first 10)
    console.print("\n[bold]Top Issues:[/bold]\n")
    
    for i, issue in enumerate(issues[:10], 1):
        color = _get_severity_color(issue.severity)
        console.print(Panel(
            f"[{color}]{issue.kind}[/]\n"
            f"File: [cyan]{issue.file}:{issue.line}[/cyan]\n"
            f"Message: {issue.message[:200]}",
            title=f"Issue #{i} (Severity: {issue.severity})",
            border_style=color
        ))
    
    if len(issues) > 10:
        console.print(f"\n[dim]... and {len(issues) - 10} more issues[/dim]")


def _display_performance_summary(monitor):
    """Display performance monitoring summary."""
    summary = monitor.get_summary()
    
    if summary.get('status') == 'no_metrics':
        return
    
    console.print("\n[bold]Performance Summary:[/bold]")
    
    perf_table = Table(show_header=True, header_style="bold cyan")
    perf_table.add_column("Metric")
    perf_table.add_column("Value", justify="right")
    
    perf_table.add_row("Total Duration", f"{summary['total_duration']:.2f}s")
    perf_table.add_row("Items Processed", str(summary['total_items_processed']))
    perf_table.add_row("Throughput", f"{summary['overall_throughput']:.1f} items/s")
    
    memory = summary.get('memory', {})
    perf_table.add_row("Memory Usage", f"{memory.get('current_mb', 0):.1f} MB")
    perf_table.add_row("Memory Increase", f"{memory.get('increase_mb', 0):.1f} MB")
    
    console.print(perf_table)
    
    # Bottlenecks
    bottlenecks = summary.get('bottlenecks', [])
    if bottlenecks:
        console.print("\n[bold yellow]Performance Bottlenecks:[/bold yellow]")
        for b in bottlenecks:
            console.print(f"  • {b['operation']}: {b['duration']:.2f}s ({b['percentage']:.1f}%)")


def _get_pattern_explanation(pattern_type: str, include_examples: bool) -> str:
    """Get explanation for a specific pattern type."""
    explanations = {
        'duplicate_function': """
[bold]Duplicate Functions[/bold]

Functions that are structurally or semantically identical but have different names.
This often occurs when LLMs recreate functionality that already exists.

[yellow]Causes:[/yellow]
• Limited context window preventing the LLM from seeing existing functions
• Incremental development where each request starts fresh
• Different naming conventions used in different sessions

[green]How to Fix:[/green]
1. Search for existing functions before creating new ones
2. Use consistent naming conventions
3. Consolidate duplicate functions into a single implementation
4. Add parameters to handle variations instead of duplicating
""",
        
        'circular_import': """
[bold]Circular Imports[/bold]

Modules that import from each other, creating a dependency cycle.
This pattern emerges when LLMs add imports reactively without considering architecture.

[yellow]Causes:[/yellow]
• Adding imports to fix undefined symbol errors
• Lack of clear dependency hierarchy
• Mixing concerns between modules

[green]How to Fix:[/green]
1. Move shared code to a separate utility module
2. Use local imports (inside functions) when necessary
3. Restructure code to follow a clear dependency hierarchy
4. Consider dependency injection or event patterns
""",
        
        'phantom_function': """
[bold]Phantom Functions[/bold]

Empty stub functions that were created but never implemented.
These appear when LLMs create placeholders to satisfy imports or API expectations.

[yellow]Causes:[/yellow]
• Creating functions to fix import errors
• Placeholder creation without follow-through
• Incomplete implementation during incremental development

[green]How to Fix:[/green]
1. Implement the function properly or remove it
2. If a stub is needed, raise NotImplementedError with a message
3. Add TODO comments with clear requirements
4. Review and implement all stubs before considering code complete
"""
    }
    
    explanation = explanations.get(pattern_type, f"No explanation available for '{pattern_type}'")
    
    if include_examples and pattern_type in explanations:
        explanation += """

[bold]Example:[/bold]
""" + _get_pattern_example(pattern_type)
    
    return explanation


def _get_pattern_example(pattern_type: str) -> str:
    """Get code example for a pattern type."""
    examples = {
        'duplicate_function': """[red]# Bad:[/red]
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def compute_total(values):  # Duplicate!
    result = 0
    for val in values:
        result += val
    return result

[green]# Good:[/green]
def calculate_sum(numbers, initial=0):
    total = initial
    for num in numbers:
        total += num
    return total
""",
        
        'circular_import': """[red]# Bad:[/red]
# module_a.py
from module_b import func_b

# module_b.py
from module_a import func_a  # Circular!

[green]# Good:[/green]
# shared.py
def shared_function(): ...

# module_a.py
from shared import shared_function

# module_b.py
from shared import shared_function
"""
    }
    
    return examples.get(pattern_type, "")


def _display_explanation(explanation):
    """Display an explanation object."""
    console.print(Panel(
        f"[bold]{explanation.summary}[/bold]\n\n"
        f"{explanation.narrative}\n\n"
        f"[yellow]Root Causes:[/yellow]\n" +
        "\n".join(f"  • {cause}" for cause in explanation.root_causes) + "\n\n"
        f"[green]Remediation Steps:[/green]\n" +
        "\n".join(f"  {i}. {step}" for i, step in enumerate(explanation.remediation_steps, 1)),
        title=explanation.pattern_type.replace('_', ' ').title(),
        border_style="blue"
    ))


def _display_pattern_catalog():
    """Display catalog of all pattern types."""
    patterns = [
        ("duplicate_function", "Functions that do the same thing with different names"),
        ("semantic_duplicate", "Functions that are semantically equivalent despite surface differences"),
        ("circular_import", "Modules that import from each other creating cycles"),
        ("phantom_function", "Empty stub functions that were never implemented"),
        ("hallucination_cascade", "Over-engineered abstractions solving non-existent problems"),
        ("context_window_thrashing", "Reimplemented functions due to limited context"),
        ("import_anxiety", "Excessive or defensive importing patterns")
    ]
    
    console.print("[bold]Tail-Chasing Pattern Catalog[/bold]\n")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Pattern", style="yellow")
    table.add_column("Description")
    
    for pattern, description in patterns:
        table.add_row(pattern, description)
    
    console.print(table)
    console.print("\n[dim]Use 'tailchasing explain PATTERN --examples' for detailed information[/dim]")


def main():
    """Entry point for the enhanced CLI."""
    cli()


if __name__ == '__main__':
    main()
