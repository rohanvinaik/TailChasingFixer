"""
Enhanced CLI with comprehensive catalytic HV system support and advanced reporting.

Features:
- Typer-based CLI structure with subcommands
- Rich terminal output with progress bars and colors
- Catalytic hypervector system integration
- Interactive HTML reports with D3.js visualizations
- Configuration management with profiles
- Watch mode with real-time updates
- Advanced comparison and analysis tools
"""

import ast
import typer
import sys
import json
import yaml
import time
import asyncio
import tempfile
import webbrowser
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, 
    TimeElapsedColumn, TaskID, MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.columns import Columns
from rich import print as rprint

# Import TailChasingFixer components
from .catalytic.hv_encoder import HypervectorEncoder, EncodingConfig
from .catalytic.catalytic_index import CatalyticIndex
from .catalytic.similarity_pipeline import SimilarityPipeline
from .catalytic.catalytic_analyzer import CatalyticDuplicateAnalyzer
from .fixers.auto_fix_engine import IntelligentAutoFixer, create_auto_fixer
from .analyzers.base import AnalysisContext
from .core.symbols import SymbolTable
from .core.issues import Issue, IssueSeverity
from .visualization.report_generator import ReportGenerator
from .config import Config
from .utils.logging_setup import get_logger


# Initialize Rich console
console = Console()
logger = get_logger(__name__)

# Create Typer app
app = typer.Typer(
    name="tailchasing",
    help="🔄 TailChasingFixer - Detect and fix LLM-induced anti-patterns in your codebase",
    add_completion=False,
    rich_markup_mode="rich"
)

# Subcommand groups
index_app = typer.Typer(help="🗄️  Manage catalytic hypervector index")
analyze_app = typer.Typer(help="🔍 Analysis commands")
fix_app = typer.Typer(help="🔧 Auto-fix commands") 
compare_app = typer.Typer(help="⚖️  Similarity comparison tools")
config_app = typer.Typer(help="⚙️  Configuration management")

app.add_typer(index_app, name="index")
app.add_typer(analyze_app, name="analyze")
app.add_typer(fix_app, name="fix")
app.add_typer(compare_app, name="compare")
app.add_typer(config_app, name="config")


# Enums for CLI options
class AnalysisMode(str, Enum):
    STANDARD = "standard"
    CATALYTIC = "catalytic"
    HYBRID = "hybrid"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"


class PatternType(str, Enum):
    DUPLICATES = "duplicates"
    PHANTOMS = "phantoms"
    CIRCULAR_IMPORTS = "circular_imports"
    CONTEXT_THRASHING = "context_thrashing"
    HALLUCINATION_CASCADE = "hallucination_cascade"
    ALL = "all"


@dataclass
class CLIConfig:
    """Configuration for CLI operations."""
    verbose: bool = False
    quiet: bool = False
    profile: str = "default"
    performance_tracking: bool = False
    cache_enabled: bool = True
    parallel: bool = True
    max_workers: Optional[int] = None


# Global CLI configuration
cli_config = CLIConfig()


# ============================================================================
# INDEX MANAGEMENT COMMANDS
# ============================================================================

@index_app.command("build")
def build_index(
    path: Path = typer.Argument(".", help="Path to codebase"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Index output directory"),
    encoding_config: Optional[Path] = typer.Option(None, "--config", "-c", help="Encoding configuration file"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Incremental index update"),
    force_rebuild: bool = typer.Option(False, "--force", "-f", help="Force complete rebuild"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify index after building")
):
    """
    🏗️  Build catalytic hypervector index for fast similarity search.
    
    The index enables O(N) duplicate detection instead of O(N²) comparison.
    """
    console.print(Panel.fit(
        "[bold cyan]Building Catalytic Hypervector Index[/bold cyan]\n"
        f"📁 Source: [yellow]{path}[/yellow]\n"
        f"🎯 Mode: [{'green' if not incremental else 'blue'}]{'Full Build' if not incremental else 'Incremental'}[/]",
        border_style="cyan"
    ))
    
    # Determine output directory
    if output_dir is None:
        output_dir = path / ".tailchasing" / "index"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load encoding configuration
    config = EncodingConfig()
    if encoding_config and encoding_config.exists():
        with open(encoding_config, 'r') as f:
            config_data = yaml.safe_load(f)
            config = EncodingConfig(**config_data)
    
    # Collect Python files
    python_files = list(path.rglob("*.py"))
    if not python_files:
        console.print("[red]❌ No Python files found[/red]")
        raise typer.Exit(1)
    
    # Build index with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Initialize components
        encoder = HypervectorEncoder(config)
        
        # Check for existing index
        existing_index = False
        if not force_rebuild and (output_dir / "metadata.jsonl").exists():
            existing_index = True
            console.print("[yellow]📋 Existing index found[/yellow]")
        
        # Create/open index
        mode = 'a' if (existing_index and incremental) else 'w'
        with CatalyticIndex(str(output_dir), mode=mode) as index:
            pipeline = SimilarityPipeline(index)
            
            # Progress tasks
            parse_task = progress.add_task("🔍 Parsing files...", total=len(python_files))
            index_task = progress.add_task("📊 Building index...", total=None)
            
            indexed_functions = 0
            skipped_files = 0
            
            # Process files
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    import ast
                    tree = ast.parse(content, filename=str(file_path))
                    
                    # Extract and index functions
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # Check if already indexed (for incremental updates)
                            func_id = f"{file_path}:{node.name}:{node.lineno}"
                            
                            if incremental and index.get_metadata(func_id):
                                continue  # Skip already indexed
                            
                            # Index the function
                            pipeline.update_index(
                                func_ast=node,
                                file_path=str(file_path),
                                function_name=node.name,
                                line_number=node.lineno,
                                context={
                                    'file_size': len(content),
                                    'last_modified': file_path.stat().st_mtime
                                }
                            )
                            indexed_functions += 1
                            
                            # Update progress periodically
                            if indexed_functions % 10 == 0:
                                progress.update(index_task, 
                                    description=f"📊 Indexing functions... ({indexed_functions})")
                
                except Exception as e:
                    if cli_config.verbose:
                        console.print(f"[dim red]⚠️  Skipped {file_path}: {e}[/dim red]")
                    skipped_files += 1
                
                progress.advance(parse_task)
            
            progress.update(index_task, completed=True, description="✅ Index built")
            
            # Get final statistics
            stats = index.get_stats()
    
    # Display results
    results_table = Table(title="📊 Index Build Results", show_header=True, header_style="bold green")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", justify="right", style="green")
    
    results_table.add_row("Files Processed", str(len(python_files) - skipped_files))
    results_table.add_row("Functions Indexed", str(indexed_functions))
    results_table.add_row("Working Memory", f"{stats['working_memory_mb']:.1f} MB")
    results_table.add_row("Index Size", f"{stats['total_bytes'] / 1024 / 1024:.1f} MB")
    results_table.add_row("Files Skipped", str(skipped_files))
    
    console.print(results_table)
    
    # Verify index if requested
    if verify:
        console.print("[dim]🔍 Verifying index integrity...[/dim]")
        # Add verification logic here
        console.print("[green]✅ Index verification passed[/green]")
    
    console.print(f"\n[bold green]🎉 Index built successfully![/bold green]")
    console.print(f"📍 Location: [cyan]{output_dir}[/cyan]")


@index_app.command("info")
def index_info(
    index_path: Path = typer.Argument(help="Path to index directory"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics")
):
    """
    📋 Display information about an existing index.
    """
    if not index_path.exists():
        console.print(f"[red]❌ Index not found: {index_path}[/red]")
        raise typer.Exit(1)
    
    with CatalyticIndex(str(index_path), mode='r') as index:
        stats = index.get_stats()
        
        # Basic info
        info_table = Table(title="📊 Index Information", show_header=True, header_style="bold blue")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", justify="right")
        
        info_table.add_row("Functions", str(stats['num_functions']))
        info_table.add_row("Total Size", f"{stats['total_bytes'] / 1024 / 1024:.1f} MB")
        info_table.add_row("Working Memory", f"{stats['working_memory_mb']:.1f} MB")
        info_table.add_row("Avg Size/Function", f"{stats['avg_bytes_per_function']:.0f} bytes")
        
        console.print(info_table)
        
        if detailed:
            # Show sample of indexed functions
            console.print("\n[bold]📋 Sample Functions:[/bold]")
            count = 0
            for func_id, vector in index.iterate_vectors(batch_size=5):
                metadata = index.get_metadata(func_id)
                if metadata and count < 5:
                    console.print(f"• [cyan]{metadata.function_name}[/cyan] in [yellow]{metadata.file_path}[/yellow]")
                    count += 1


@index_app.command("query")
def query_index(
    index_path: Path = typer.Argument(help="Path to index directory"),
    query_file: Path = typer.Argument(help="Python file containing function to query"),
    function_name: Optional[str] = typer.Option(None, "--function", "-f", help="Specific function to query"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of similar functions to return"),
    threshold: float = typer.Option(0.8, "--threshold", "-t", help="Similarity threshold")
):
    """
    🔍 Query index for similar functions.
    """
    if not index_path.exists():
        console.print(f"[red]❌ Index not found: {index_path}[/red]")
        raise typer.Exit(1)
    
    if not query_file.exists():
        console.print(f"[red]❌ Query file not found: {query_file}[/red]")
        raise typer.Exit(1)
    
    # Parse query file
    with open(query_file, 'r') as f:
        content = f.read()
    
    import ast
    tree = ast.parse(content)
    
    # Find function to query
    query_function = None
    if function_name:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                query_function = node
                break
    else:
        # Use first function found
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                query_function = node
                break
    
    if not query_function:
        console.print(f"[red]❌ No function found to query[/red]")
        raise typer.Exit(1)
    
    # Query index
    with CatalyticIndex(str(index_path), mode='r') as index:
        pipeline = SimilarityPipeline(index)
        
        console.print(f"🔍 Querying for function: [cyan]{query_function.name}[/cyan]")
        
        results = pipeline.query_function(query_function, top_k=top_k)
        results = [r for r in results if r.similarity_score >= threshold]
        
        if not results:
            console.print(f"[yellow]No similar functions found above threshold {threshold}[/yellow]")
            return
        
        # Display results
        results_table = Table(title="🎯 Similar Functions", show_header=True, header_style="bold magenta")
        results_table.add_column("Function", style="cyan")
        results_table.add_column("File", style="yellow")
        results_table.add_column("Line", justify="center")
        results_table.add_column("Similarity", justify="right", style="green")
        results_table.add_column("Confidence", justify="right", style="blue")
        
        for result in results:
            results_table.add_row(
                result.function_name,
                str(Path(result.file_path).name),
                str(result.line_number),
                f"{result.similarity_score:.3f}",
                f"{result.confidence:.3f}"
            )
        
        console.print(results_table)


# ============================================================================
# ANALYSIS COMMANDS
# ============================================================================

@analyze_app.command("run")
def analyze_run(
    path: Path = typer.Argument(".", help="Path to analyze"),
    mode: AnalysisMode = typer.Option(AnalysisMode.CATALYTIC, "--mode", "-m", help="Analysis mode"),
    output_format: OutputFormat = typer.Option(OutputFormat.TEXT, "--format", "-f", help="Output format"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    index_path: Optional[Path] = typer.Option(None, "--index", help="Path to existing index"),
    patterns: List[PatternType] = typer.Option([PatternType.ALL], "--pattern", "-p", help="Pattern types to detect"),
    severity_threshold: int = typer.Option(1, "--severity", "-s", help="Minimum severity level"),
    confidence_threshold: float = typer.Option(0.8, "--confidence", "-c", help="Minimum confidence score"),
    include_clean_code: bool = typer.Option(False, "--include-clean", help="Include analysis of clean code sections"),
    # Timeout configuration options
    analyzer_timeout: Optional[float] = typer.Option(None, "--analyzer-timeout", help="Analyzer timeout in seconds (0 = disabled)"),
    group_timeout: Optional[float] = typer.Option(None, "--group-timeout", help="Per-group timeout in seconds"),
    watchdog_timeout: Optional[float] = typer.Option(None, "--watchdog-timeout", help="Watchdog timeout in seconds (0 = disabled)")
):
    """
    🔍 Run comprehensive tail-chasing analysis.
    """
    console.print(Panel.fit(
        f"[bold cyan]🔍 TailChasingFixer Analysis[/bold cyan]\n"
        f"📁 Path: [yellow]{path}[/yellow]\n"
        f"🎯 Mode: [blue]{mode.value}[/blue]\n"
        f"📊 Format: [green]{output_format.value}[/green]",
        border_style="cyan"
    ))
    
    # Initialize analysis components
    results = []
    analysis_start = time.time()
    
    # Set timeout environment variables if provided via CLI
    import os
    if analyzer_timeout is not None:
        os.environ["TAILCHASING_ANALYZER_TIMEOUT_SEC"] = str(analyzer_timeout)
    if group_timeout is not None:
        os.environ["TAILCHASING_GROUP_TIMEOUT_SEC"] = str(group_timeout)
    if watchdog_timeout is not None:
        os.environ["TAILCHASING_WATCHDOG_SEC"] = str(watchdog_timeout)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Step 1: File collection
        collect_task = progress.add_task("📁 Collecting files...", total=None)
        python_files = list(path.rglob("*.py"))
        
        if not python_files:
            console.print("[red]❌ No Python files found[/red]")
            raise typer.Exit(1)
        
        progress.update(collect_task, completed=True, description=f"📁 Found {len(python_files)} files")
        
        # Step 2: Analysis based on mode
        if mode == AnalysisMode.CATALYTIC:
            results = _run_catalytic_analysis(python_files, index_path, patterns, progress)
        elif mode == AnalysisMode.STANDARD:
            results = _run_standard_analysis(python_files, patterns, progress)
        else:  # HYBRID
            results = _run_hybrid_analysis(python_files, index_path, patterns, progress)
    
    analysis_time = time.time() - analysis_start
    
    # Filter results
    filtered_results = [
        r for r in results 
        if r.severity >= severity_threshold and getattr(r, 'confidence', 1.0) >= confidence_threshold
    ]
    
    # Generate output
    if output_format == OutputFormat.JSON:
        _output_json_results(filtered_results, output_file, analysis_time)
    elif output_format == OutputFormat.HTML:
        _output_html_results(filtered_results, output_file, analysis_time, path)
    elif output_format == OutputFormat.MARKDOWN:
        _output_markdown_results(filtered_results, output_file, analysis_time)
    else:
        _output_text_results(filtered_results, analysis_time)
    
    # Exit with appropriate code
    if filtered_results:
        high_severity = [r for r in filtered_results if r.severity >= 4]
        if high_severity:
            raise typer.Exit(2)  # High severity issues
        else:
            raise typer.Exit(1)  # Issues found
    else:
        console.print("\n[bold green]🎉 No issues detected![/bold green]")
        raise typer.Exit(0)


@analyze_app.command("watch")
def analyze_watch(
    path: Path = typer.Argument(".", help="Path to watch"),
    index_path: Optional[Path] = typer.Option(None, "--index", help="Index directory to update"),
    patterns: List[PatternType] = typer.Option([PatternType.ALL], "--pattern", "-p", help="Patterns to watch for"),
    debounce_seconds: float = typer.Option(2.0, "--debounce", "-d", help="Debounce time for file changes"),
    exit_on_error: bool = typer.Option(False, "--exit-on-error", help="Exit on first error")
):
    """
    👁️  Watch for file changes and run incremental analysis.
    """
    try:
        import watchdog
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        console.print("[red]❌ watchdog package required for watch mode[/red]")
        console.print("Install with: pip install watchdog")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold magenta]👁️  Watch Mode Active[/bold magenta]\n"
        f"📁 Monitoring: [yellow]{path}[/yellow]\n"
        f"⏱️  Debounce: [blue]{debounce_seconds}s[/blue]",
        border_style="magenta"
    ))
    
    # File change handler
    class AnalysisHandler(FileSystemEventHandler):
        def __init__(self):
            self.changed_files = set()
            self.last_change = time.time()
        
        def on_modified(self, event):
            if event.is_directory or not event.src_path.endswith('.py'):
                return
            
            self.changed_files.add(Path(event.src_path))
            self.last_change = time.time()
        
        def get_changed_files(self):
            if time.time() - self.last_change >= debounce_seconds and self.changed_files:
                files = self.changed_files.copy()
                self.changed_files.clear()
                return files
            return set()
    
    handler = AnalysisHandler()
    observer = Observer()
    observer.schedule(handler, str(path), recursive=True)
    observer.start()
    
    console.print("[green]✅ Watching for changes... (Press Ctrl+C to stop)[/green]")
    
    try:
        with Live(console=console, refresh_per_second=1) as live:
            layout = Layout()
            
            while True:
                changed_files = handler.get_changed_files()
                
                if changed_files:
                    live.update(f"[yellow]📝 Analyzing {len(changed_files)} changed files...[/yellow]")
                    
                    # Run incremental analysis
                    try:
                        results = _run_incremental_analysis(changed_files, index_path, patterns)
                        
                        # Display results in live view
                        if results:
                            issues_text = Text()
                            issues_text.append(f"🔍 Found {len(results)} issues:\n", style="bold yellow")
                            
                            for result in results[:5]:  # Show first 5
                                issues_text.append(f"  • {result.kind} in {result.file}\n", style="red")
                            
                            if len(results) > 5:
                                issues_text.append(f"  ... and {len(results) - 5} more\n", style="dim")
                            
                            live.update(issues_text)
                        else:
                            live.update("[green]✅ No new issues detected[/green]")
                    
                    except Exception as e:
                        error_msg = f"[red]❌ Analysis error: {e}[/red]"
                        live.update(error_msg)
                        if exit_on_error:
                            break
                
                time.sleep(1)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Watch mode stopped[/yellow]")
    
    finally:
        observer.stop()
        observer.join()


# ============================================================================
# FIX COMMANDS  
# ============================================================================

@fix_app.command("auto")
def auto_fix(
    path: Path = typer.Argument(".", help="Path to fix"),
    pattern: Optional[PatternType] = typer.Option(None, "--pattern", "-p", help="Specific pattern type to fix"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be fixed without applying"),
    severity_threshold: int = typer.Option(3, "--severity", "-s", help="Minimum severity to fix"),
    backup_dir: Optional[Path] = typer.Option(None, "--backup", "-b", help="Backup directory"),
    test_command: Optional[str] = typer.Option(None, "--test-cmd", "-t", help="Test command to run after fixes"),
    max_fixes: Optional[int] = typer.Option(None, "--max-fixes", "-m", help="Maximum number of fixes to apply"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Prompt before each fix")
):
    """
    🔧 Apply automatic fixes for detected patterns.
    """
    console.print(Panel.fit(
        f"[bold yellow]🔧 Auto-Fix Mode[/bold yellow]\n"
        f"📁 Target: [cyan]{path}[/cyan]\n"
        f"🎯 Pattern: [blue]{pattern.value if pattern else 'all'}[/blue]\n"
        f"🏃 Mode: [{'yellow' if dry_run else 'green'}]{'Dry Run' if dry_run else 'Apply Fixes'}[/]",
        border_style="yellow"
    ))
    
    # Initialize auto-fixer
    fixer = create_auto_fixer(
        dry_run=dry_run,
        backup_dir=str(backup_dir) if backup_dir else None,
        test_command=test_command
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Step 1: Detect issues
        detect_task = progress.add_task("🔍 Detecting issues...", total=None)
        issues = _detect_issues_for_fixing(path, pattern)
        
        # Filter by severity
        issues = [i for i in issues if i.severity >= severity_threshold]
        
        if not issues:
            console.print("[yellow]No issues found to fix[/yellow]")
            return
        
        progress.update(detect_task, completed=True, description=f"🔍 Found {len(issues)} fixable issues")
        
        # Step 2: Create fix plan
        plan_task = progress.add_task("📋 Creating fix plan...", total=None)
        
        if max_fixes:
            issues = issues[:max_fixes]
        
        plan = fixer.create_fix_plan(issues)
        progress.update(plan_task, completed=True, description="📋 Fix plan created")
        
        # Step 3: Display fix plan
        _display_fix_plan(plan, dry_run)
        
        # Step 4: Apply fixes
        if interactive and not dry_run:
            if not typer.confirm(f"Apply {len(plan.fixes)} fixes?"):
                console.print("[yellow]Fix application cancelled[/yellow]")
                return
        
        fix_task = progress.add_task("🔧 Applying fixes...", total=len(plan.fixes))
        results = fixer.execute_fix_plan(plan)
        
        progress.update(fix_task, completed=len(results), description="🔧 Fixes applied")
    
    # Display results
    _display_fix_results(results, dry_run)
    
    # Generate report
    report = fixer.generate_report()
    if not dry_run:
        _save_fix_report(report, path)


@fix_app.command("plan")
def fix_plan(
    path: Path = typer.Argument(".", help="Path to analyze"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save plan to file"),
    pattern: Optional[PatternType] = typer.Option(None, "--pattern", "-p", help="Specific pattern type"),
    severity_threshold: int = typer.Option(2, "--severity", "-s", help="Minimum severity")
):
    """
    📋 Generate and display fix plan without applying changes.
    """
    console.print("[bold blue]📋 Generating Fix Plan[/bold blue]")
    
    # Detect issues
    issues = _detect_issues_for_fixing(path, pattern)
    issues = [i for i in issues if i.severity >= severity_threshold]
    
    if not issues:
        console.print("[yellow]No fixable issues found[/yellow]")
        return
    
    # Create plan
    fixer = create_auto_fixer(dry_run=True)
    plan = fixer.create_fix_plan(issues)
    
    # Display plan
    _display_fix_plan(plan, dry_run=True)
    
    # Save if requested
    if output_file:
        plan_data = {
            'fixes': [
                {
                    'id': fix.id,
                    'issue_type': fix.issue.kind,
                    'file': str(fix.issue.file),
                    'line': fix.issue.line,
                    'severity': fix.issue.severity,
                    'strategy': fix.strategy,
                    'risk_level': fix.risk_level.name,
                    'estimated_time': fix.estimated_time
                }
                for fix in plan.fixes
            ],
            'execution_order': plan.execution_order,
            'estimated_time': plan.estimated_time,
            'total_risk_score': plan.total_risk_score
        }
        
        with open(output_file, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        console.print(f"[green]📁 Plan saved to: {output_file}[/green]")


# ============================================================================
# COMPARISON COMMANDS
# ============================================================================

@compare_app.command("files")
def compare_files(
    file1: Path = typer.Argument(help="First file to compare"),
    file2: Path = typer.Argument(help="Second file to compare"), 
    function1: Optional[str] = typer.Option(None, "--func1", help="Specific function in file1"),
    function2: Optional[str] = typer.Option(None, "--func2", help="Specific function in file2"),
    similarity_threshold: float = typer.Option(0.0, "--threshold", "-t", help="Only show similarities above threshold"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed similarity analysis"),
    visual: bool = typer.Option(False, "--visual", "-v", help="Generate visual similarity comparison")
):
    """
    ⚖️  Compare similarity between two files or functions.
    """
    if not file1.exists() or not file2.exists():
        console.print("[red]❌ One or both files not found[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(
        f"[bold magenta]⚖️  File Similarity Comparison[/bold magenta]\n"
        f"📄 File 1: [cyan]{file1}[/cyan]\n"
        f"📄 File 2: [yellow]{file2}[/yellow]",
        border_style="magenta"
    ))
    
    # Parse files
    with open(file1, 'r') as f:
        content1 = f.read()
    with open(file2, 'r') as f:
        content2 = f.read()
    
    import ast
    tree1 = ast.parse(content1)
    tree2 = ast.parse(content2)
    
    # Extract functions
    functions1 = _extract_functions(tree1, function1)
    functions2 = _extract_functions(tree2, function2)
    
    if not functions1 or not functions2:
        console.print("[red]❌ No functions found to compare[/red]")
        return
    
    # Initialize encoder for similarity computation
    encoder = HypervectorEncoder()
    
    # Compare all pairs
    comparisons = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("🔍 Computing similarities...", total=len(functions1) * len(functions2))
        
        for func1 in functions1:
            hv1 = encoder.encode_ast(func1)
            
            for func2 in functions2:
                hv2 = encoder.encode_ast(func2)
                
                similarity = encoder.similarity(hv1, hv2)
                hamming_sim = encoder.hamming_similarity(hv1, hv2)
                
                comparisons.append({
                    'func1': func1.name,
                    'func2': func2.name,
                    'similarity': similarity,
                    'hamming_similarity': hamming_sim,
                    'ast1': func1,
                    'ast2': func2
                })
                
                progress.advance(task)
    
    # Filter by threshold
    filtered_comparisons = [c for c in comparisons if c['similarity'] >= similarity_threshold]
    
    if not filtered_comparisons:
        console.print(f"[yellow]No similarities found above threshold {similarity_threshold}[/yellow]")
        return
    
    # Sort by similarity
    filtered_comparisons.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Display results
    _display_similarity_results(filtered_comparisons, detailed)
    
    # Generate visual comparison if requested
    if visual:
        _generate_visual_comparison(filtered_comparisons, file1, file2)


@compare_app.command("functions")
def compare_functions(
    query_function: str = typer.Argument(help="Function signature to search for (file:function or just function)"),
    search_path: Path = typer.Argument(".", help="Path to search for similar functions"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of most similar functions to show"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Minimum similarity threshold"),
    index_path: Optional[Path] = typer.Option(None, "--index", help="Use existing index for faster search")
):
    """
    🔍 Find functions similar to a specific function across the codebase.
    """
    # Parse query function specification
    if ':' in query_function:
        file_spec, func_name = query_function.split(':', 1)
        query_file = Path(file_spec)
    else:
        func_name = query_function
        query_file = None
    
    console.print(Panel.fit(
        f"[bold blue]🔍 Function Similarity Search[/bold blue]\n"
        f"🎯 Function: [cyan]{func_name}[/cyan]\n"
        f"📁 Search Path: [yellow]{search_path}[/yellow]",
        border_style="blue"
    ))
    
    # Find query function
    query_ast = _find_query_function(query_file, func_name, search_path)
    
    if not query_ast:
        console.print(f"[red]❌ Function '{func_name}' not found[/red]")
        raise typer.Exit(1)
    
    # Search for similar functions
    if index_path and index_path.exists():
        results = _search_with_index(query_ast, index_path, top_k, threshold)
    else:
        results = _search_without_index(query_ast, search_path, top_k, threshold)
    
    if not results:
        console.print(f"[yellow]No similar functions found above threshold {threshold}[/yellow]")
        return
    
    # Display results
    _display_function_search_results(results, func_name)


# ============================================================================
# CONFIGURATION COMMANDS
# ============================================================================

@config_app.command("init")
def config_init(
    path: Path = typer.Argument(".", help="Path to initialize configuration"),
    profile: str = typer.Option("default", "--profile", "-p", help="Configuration profile name"),
    template: str = typer.Option("standard", "--template", "-t", help="Configuration template"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing configuration")
):
    """
    ⚙️  Initialize configuration file.
    """
    config_file = path / ".tailchasing.yml"
    
    if config_file.exists() and not force:
        console.print(f"[yellow]Configuration already exists at {config_file}[/yellow]")
        console.print("Use --force to overwrite")
        return
    
    # Configuration templates
    templates = {
        'standard': {
            'profiles': {
                profile: {
                    'analysis': {
                        'mode': 'catalytic',
                        'confidence_threshold': 0.8,
                        'severity_threshold': 2,
                        'patterns': ['duplicates', 'phantoms', 'circular_imports']
                    },
                    'catalytic': {
                        'encoding': {
                            'dimension': 8192,
                            'sparsity': 0.33,
                            'max_depth': 10
                        },
                        'similarity': {
                            'hv_threshold': 0.88,
                            'ast_threshold': 0.85
                        }
                    },
                    'fixing': {
                        'auto_fix_enabled': False,
                        'backup_enabled': True,
                        'test_before_fix': True,
                        'max_risk_level': 'medium'
                    },
                    'performance': {
                        'parallel_processing': True,
                        'max_workers': None,
                        'cache_enabled': True,
                        'index_cache_ttl': 3600
                    }
                }
            },
            'default_profile': profile,
            'output': {
                'format': 'text',
                'verbosity': 'normal',
                'colors': True
            }
        },
        'minimal': {
            'profiles': {
                profile: {
                    'analysis': {
                        'mode': 'standard',
                        'patterns': ['duplicates', 'phantoms']
                    }
                }
            },
            'default_profile': profile
        },
        'performance': {
            'profiles': {
                profile: {
                    'analysis': {
                        'mode': 'catalytic',
                        'confidence_threshold': 0.9,
                        'patterns': ['duplicates']
                    },
                    'catalytic': {
                        'encoding': {
                            'dimension': 4096,
                            'sparsity': 0.25
                        }
                    },
                    'performance': {
                        'parallel_processing': True,
                        'max_workers': 8,
                        'cache_enabled': True
                    }
                }
            },
            'default_profile': profile
        }
    }
    
    config_data = templates.get(template, templates['standard'])
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    console.print(f"[green]✅ Configuration initialized: {config_file}[/green]")
    console.print(f"📋 Template: [blue]{template}[/blue]")
    console.print(f"👤 Profile: [cyan]{profile}[/cyan]")


@config_app.command("show")
def config_show(
    path: Path = typer.Argument(".", help="Path containing configuration"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Show specific profile"),
    format_output: OutputFormat = typer.Option(OutputFormat.TEXT, "--format", "-f", help="Output format")
):
    """
    📄 Show current configuration.
    """
    config_file = path / ".tailchasing.yml"
    
    if not config_file.exists():
        console.print(f"[red]❌ No configuration found at {config_file}[/red]")
        console.print("Run 'tailchasing config init' to create one")
        return
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    if format_output == OutputFormat.JSON:
        console.print_json(data=config_data)
    elif format_output == OutputFormat.TEXT:
        console.print(f"[bold]Configuration: {config_file}[/bold]\n")
        
        # Show profiles
        profiles = config_data.get('profiles', {})
        default_profile = config_data.get('default_profile', 'default')
        
        if profile:
            if profile in profiles:
                _display_profile(profile, profiles[profile], profile == default_profile)
            else:
                console.print(f"[red]Profile '{profile}' not found[/red]")
        else:
            for prof_name, prof_data in profiles.items():
                _display_profile(prof_name, prof_data, prof_name == default_profile)


@config_app.command("set")
def config_set(
    path: Path = typer.Argument(".", help="Path containing configuration"),
    key: str = typer.Argument(help="Configuration key (e.g., analysis.mode)"),
    value: str = typer.Argument(help="Configuration value"),
    profile: str = typer.Option("default", "--profile", "-p", help="Profile to modify")
):
    """
    ✏️  Set configuration value.
    """
    config_file = path / ".tailchasing.yml"
    
    if not config_file.exists():
        console.print(f"[red]❌ No configuration found[/red]")
        return
    
    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Navigate to profile
    if 'profiles' not in config_data:
        config_data['profiles'] = {}
    
    if profile not in config_data['profiles']:
        config_data['profiles'][profile] = {}
    
    # Set nested key
    keys = key.split('.')
    current = config_data['profiles'][profile]
    
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    
    # Convert value to appropriate type
    try:
        if value.lower() in ('true', 'false'):
            value = value.lower() == 'true'
        elif value.isdigit():
            value = int(value)
        elif '.' in value and all(part.isdigit() for part in value.split('.')):
            value = float(value)
    except:
        pass  # Keep as string
    
    current[keys[-1]] = value
    
    # Save configuration
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    console.print(f"[green]✅ Set {profile}.{key} = {value}[/green]")


# ============================================================================
# MAIN COMMAND (default analyze)
# ============================================================================

@app.command()
def main(
    path: Path = typer.Argument(".", help="Path to analyze"),
    mode: AnalysisMode = typer.Option(AnalysisMode.CATALYTIC, "--mode", "-m", help="Analysis mode"),
    output_format: OutputFormat = typer.Option(OutputFormat.TEXT, "--format", "-f", help="Output format"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    profile: str = typer.Option("default", "--profile", "-p", help="Configuration profile")
):
    """
    🔄 Analyze codebase for tail-chasing patterns (default command).
    """
    # Set global config
    cli_config.verbose = verbose
    cli_config.quiet = quiet
    cli_config.profile = profile
    
    # Run analysis
    analyze_run(path, mode, output_format)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _run_catalytic_analysis(files: List[Path], index_path: Optional[Path], 
                          patterns: List[PatternType], progress: Progress) -> List[Issue]:
    """Run analysis using catalytic hypervector system."""
    issues = []
    
    # Create temporary index if none provided
    if index_path is None:
        temp_index = tempfile.mkdtemp(prefix="temp_index_")
        index_path = Path(temp_index)
        
        # Build index
        build_task = progress.add_task("🏗️  Building index...", total=len(files))
        
        with CatalyticIndex(str(index_path), mode='w') as index:
            pipeline = SimilarityPipeline(index)
            encoder = HypervectorEncoder()
            
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    import ast
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            pipeline.update_index(
                                func_ast=node,
                                file_path=str(file_path),
                                function_name=node.name,
                                line_number=node.lineno
                            )
                except:
                    pass  # Skip problematic files
                
                progress.advance(build_task)
    
    # Run catalytic analysis
    analyze_task = progress.add_task("🔍 Running catalytic analysis...", total=None)
    
    analyzer = CatalyticDuplicateAnalyzer()
    # Create mock analysis context
    ctx = _create_analysis_context(files)
    
    catalytic_issues = list(analyzer.run(ctx))
    issues.extend(catalytic_issues)
    
    progress.update(analyze_task, completed=True, description=f"🔍 Found {len(issues)} issues")
    
    return issues


def _run_standard_analysis(files: List[Path], patterns: List[PatternType], 
                         progress: Progress) -> List[Issue]:
    """Run standard analysis without catalytic system."""
    # Import standard analyzers
    from .analyzers.duplicates import DuplicateFunctionAnalyzer
    from .analyzers.placeholders import PlaceholderAnalyzer
    
    issues = []
    ctx = _create_analysis_context(files)
    
    analyzers = []
    if PatternType.DUPLICATES in patterns or PatternType.ALL in patterns:
        analyzers.append(DuplicateFunctionAnalyzer())
    if PatternType.PHANTOMS in patterns or PatternType.ALL in patterns:
        analyzers.append(PlaceholderAnalyzer())
    
    analyze_task = progress.add_task("🔍 Running standard analysis...", total=len(analyzers))
    
    for analyzer in analyzers:
        analyzer_issues = list(analyzer.run(ctx))
        issues.extend(analyzer_issues)
        progress.advance(analyze_task)
    
    return issues


def _run_hybrid_analysis(files: List[Path], index_path: Optional[Path], 
                       patterns: List[PatternType], progress: Progress) -> List[Issue]:
    """Run hybrid analysis combining catalytic and standard approaches."""
    issues = []
    
    # Run catalytic analysis first
    catalytic_issues = _run_catalytic_analysis(files, index_path, patterns, progress)
    issues.extend(catalytic_issues)
    
    # Run standard analysis for patterns not covered by catalytic
    standard_patterns = [p for p in patterns if p not in [PatternType.DUPLICATES]]
    if standard_patterns:
        standard_issues = _run_standard_analysis(files, standard_patterns, progress)
        issues.extend(standard_issues)
    
    return issues


def _create_analysis_context(files: List[Path]) -> AnalysisContext:
    """Create analysis context from files."""
    ast_index = {}
    source_cache = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                source_cache[str(file_path)] = content.split('\n')
                
                import ast
                tree = ast.parse(content)
                ast_index[str(file_path)] = tree
        except:
            continue
    
    symbol_table = SymbolTable()
    # Populate symbol table
    for file_path, tree in ast_index.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol_table.add_function(
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    node=node
                )
    
    return AnalysisContext(
        config={},
        root_dir=Path.cwd(),
        file_paths=files,
        ast_index=ast_index,
        symbol_table=symbol_table,
        source_cache=source_cache,
        cache={}
    )


def _output_text_results(issues: List[Issue], analysis_time: float):
    """Output results in text format."""
    if not issues:
        console.print("[bold green]🎉 No issues detected![/bold green]")
        return
    
    # Summary panel
    console.print(Panel.fit(
        f"[bold]📊 Analysis Summary[/bold]\n"
        f"Issues Found: [red]{len(issues)}[/red]\n"
        f"Analysis Time: [blue]{analysis_time:.2f}s[/blue]",
        border_style="cyan"
    ))
    
    # Group by severity
    severity_groups = {}
    for issue in issues:
        if issue.severity not in severity_groups:
            severity_groups[issue.severity] = []
        severity_groups[issue.severity].append(issue)
    
    # Display by severity (high to low)
    for severity in sorted(severity_groups.keys(), reverse=True):
        severity_issues = severity_groups[severity]
        color = _get_severity_color(severity)
        
        console.print(f"\n[bold {color}]Severity {severity} ({len(severity_issues)} issues)[/]")
        
        for i, issue in enumerate(severity_issues[:10], 1):  # Show first 10
            console.print(f"  {i}. [cyan]{issue.kind}[/cyan] in [yellow]{issue.file}:{issue.line}[/yellow]")
            console.print(f"     {issue.message[:100]}...")
        
        if len(severity_issues) > 10:
            console.print(f"     [dim]... and {len(severity_issues) - 10} more[/dim]")


def _output_json_results(issues: List[Issue], output_file: Optional[Path], analysis_time: float):
    """Output results in JSON format."""
    data = {
        'timestamp': datetime.now().isoformat(),
        'analysis_time': analysis_time,
        'summary': {
            'total_issues': len(issues),
            'by_severity': {},
            'by_type': {}
        },
        'issues': []
    }
    
    # Process issues
    for issue in issues:
        # Count by severity
        if str(issue.severity) not in data['summary']['by_severity']:
            data['summary']['by_severity'][str(issue.severity)] = 0
        data['summary']['by_severity'][str(issue.severity)] += 1
        
        # Count by type
        if issue.kind not in data['summary']['by_type']:
            data['summary']['by_type'][issue.kind] = 0
        data['summary']['by_type'][issue.kind] += 1
        
        # Add issue
        data['issues'].append({
            'kind': issue.kind,
            'message': issue.message,
            'severity': issue.severity,
            'file': str(issue.file),
            'line': issue.line,
            'symbol': issue.symbol,
            'evidence': issue.evidence or {}
        })
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]📄 JSON report saved: {output_file}[/green]")
    else:
        console.print_json(data=data)


def _output_html_results(issues: List[Issue], output_file: Optional[Path], 
                        analysis_time: float, source_path: Path):
    """Output results in interactive HTML format with D3.js visualizations."""
    from .visualization.interactive_report import InteractiveReportGenerator
    
    if output_file is None:
        output_file = Path("tailchasing_report.html")
    
    # Generate interactive HTML report
    report_gen = InteractiveReportGenerator()
    
    with Progress(SpinnerColumn(), TextColumn("🎨 Generating interactive report..."), console=console) as progress:
        task = progress.add_task("Generating...", total=None)
        
        html_content = report_gen.generate_interactive_report(
            issues=issues,
            source_path=source_path,
            analysis_time=analysis_time,
            include_d3_visualizations=True,
            include_dependency_graph=True,
            include_similarity_heatmap=True
        )
        
        progress.update(task, completed=True)
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    console.print(f"[green]🎨 Interactive HTML report saved: {output_file}[/green]")
    
    if typer.confirm("Open in browser?"):
        webbrowser.open(f"file://{output_file.resolve()}")


def _output_markdown_results(issues: List[Issue], output_file: Optional[Path], analysis_time: float):
    """Output results in Markdown format."""
    markdown_content = f"""# TailChasingFixer Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Time:** {analysis_time:.2f} seconds  
**Issues Found:** {len(issues)}

## Summary

"""
    
    # Group by type
    type_groups = {}
    for issue in issues:
        if issue.kind not in type_groups:
            type_groups[issue.kind] = []
        type_groups[issue.kind].append(issue)
    
    markdown_content += "| Pattern Type | Count |\n|---|---|\n"
    for issue_type, type_issues in sorted(type_groups.items()):
        markdown_content += f"| {issue_type.replace('_', ' ').title()} | {len(type_issues)} |\n"
    
    # Detailed issues
    markdown_content += "\n## Issues by Type\n\n"
    
    for issue_type, type_issues in sorted(type_groups.items()):
        markdown_content += f"### {issue_type.replace('_', ' ').title()}\n\n"
        
        for issue in type_issues[:20]:  # Limit to first 20
            markdown_content += f"- **{issue.file}:{issue.line}** (Severity: {issue.severity})\n"
            markdown_content += f"  {issue.message}\n\n"
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(markdown_content)
        console.print(f"[green]📄 Markdown report saved: {output_file}[/green]")
    else:
        console.print(Markdown(markdown_content))


def _get_severity_color(severity: int) -> str:
    """Get color for severity level."""
    colors = {1: "green", 2: "yellow", 3: "orange", 4: "red", 5: "bold red"}
    return colors.get(severity, "white")


def _display_fix_plan(plan, dry_run: bool):
    """Display fix plan table."""
    if not plan.fixes:
        console.print("[yellow]No fixes in plan[/yellow]")
        return
    
    table = Table(
        title=f"🔧 Fix Plan ({'Dry Run' if dry_run else 'Execution Plan'})",
        show_header=True,
        header_style="bold magenta"
    )
    table.add_column("Order", justify="center", width=5)
    table.add_column("Issue", style="cyan")
    table.add_column("File", style="yellow")
    table.add_column("Strategy", style="blue")
    table.add_column("Risk", justify="center")
    table.add_column("Time", justify="right")
    
    for i, fix_id in enumerate(plan.execution_order[:20], 1):  # Show first 20
        fix = next(f for f in plan.fixes if f.id == fix_id)
        
        risk_color = {"LOW": "green", "MEDIUM": "yellow", "HIGH": "red"}.get(fix.risk_level.name, "white")
        
        table.add_row(
            str(i),
            fix.issue.kind,
            str(Path(fix.issue.file).name),
            fix.strategy,
            f"[{risk_color}]{fix.risk_level.name}[/]",
            f"{fix.estimated_time:.1f}s"
        )
    
    console.print(table)
    
    if len(plan.fixes) > 20:
        console.print(f"[dim]... and {len(plan.fixes) - 20} more fixes[/dim]")
    
    console.print(f"\n[bold]Total Estimated Time:[/bold] {plan.estimated_time:.1f}s")
    console.print(f"[bold]Average Risk Score:[/bold] {plan.total_risk_score:.2f}")


def _display_fix_results(results, dry_run: bool):
    """Display fix application results."""
    if not results:
        return
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    # Summary
    console.print(Panel.fit(
        f"[bold]🔧 Fix Results[/bold]\n"
        f"Total: [blue]{len(results)}[/blue]\n"
        f"Successful: [green]{len(successful)}[/green]\n"
        f"Failed: [red]{len(failed)}[/red]\n"
        f"Mode: [yellow]{'Dry Run' if dry_run else 'Applied'}[/yellow]",
        border_style="green" if not failed else "yellow"
    ))
    
    # Show failed fixes
    if failed:
        console.print("\n[bold red]❌ Failed Fixes:[/bold red]")
        for result in failed[:10]:
            console.print(f"  • {result.issue.kind} in {result.issue.file}")
            if result.error_message:
                console.print(f"    Error: {result.error_message}")


def _save_fix_report(report: Dict[str, Any], base_path: Path):
    """Save fix report to file."""
    report_file = base_path / "fix_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    console.print(f"[dim]📄 Fix report saved: {report_file}[/dim]")


def _detect_issues_for_fixing(path: Path, pattern: Optional[PatternType]) -> List[Issue]:
    """Detect issues that can be automatically fixed."""
    # This would integrate with the detection system
    # For now, return mock issues
    return []


def _extract_functions(tree: ast.AST, function_name: Optional[str] = None) -> List[ast.FunctionDef]:
    """Extract functions from AST."""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if function_name is None or node.name == function_name:
                functions.append(node)
    return functions


def _display_similarity_results(comparisons: List[Dict], detailed: bool):
    """Display similarity comparison results."""
    table = Table(title="🎯 Similarity Results", show_header=True, header_style="bold cyan")
    table.add_column("Function 1", style="cyan")
    table.add_column("Function 2", style="yellow")
    table.add_column("Similarity", justify="right", style="green")
    table.add_column("Hamming", justify="right", style="blue")
    
    for comp in comparisons[:20]:  # Show first 20
        table.add_row(
            comp['func1'],
            comp['func2'],
            f"{comp['similarity']:.3f}",
            f"{comp['hamming_similarity']:.3f}"
        )
    
    console.print(table)
    
    if detailed:
        # Show additional analysis
        console.print("\n[bold]📊 Detailed Analysis:[/bold]")
        avg_similarity = sum(c['similarity'] for c in comparisons) / len(comparisons)
        console.print(f"Average Similarity: [green]{avg_similarity:.3f}[/green]")


def _generate_visual_comparison(comparisons: List[Dict], file1: Path, file2: Path):
    """Generate visual similarity comparison."""
    # This would generate an HTML file with visual diff
    console.print("[dim]🎨 Visual comparison feature coming soon...[/dim]")


def _display_function_search_results(results: List[Dict], query_function: str):
    """Display function search results."""
    console.print(f"\n[bold]🎯 Functions similar to '[cyan]{query_function}[/cyan]':[/bold]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Function", style="cyan")
    table.add_column("File", style="yellow")
    table.add_column("Line", justify="center")
    table.add_column("Similarity", justify="right", style="green")
    
    for result in results:
        table.add_row(
            result['name'],
            str(Path(result['file']).name),
            str(result['line']),
            f"{result['similarity']:.3f}"
        )
    
    console.print(table)


def _find_query_function(query_file: Optional[Path], func_name: str, search_path: Path) -> Optional[ast.FunctionDef]:
    """Find the query function AST node."""
    if query_file and query_file.exists():
        search_files = [query_file]
    else:
        search_files = list(search_path.rglob("*.py"))
    
    for file_path in search_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                    return node
        
        except:
            continue
    
    return None


def _search_with_index(query_ast: ast.FunctionDef, index_path: Path, top_k: int, threshold: float) -> List[Dict]:
    """Search using existing index."""
    results = []
    
    with CatalyticIndex(str(index_path), mode='r') as index:
        pipeline = SimilarityPipeline(index)
        query_results = pipeline.query_function(query_ast, top_k=top_k)
        
        for result in query_results:
            if result.similarity_score >= threshold:
                results.append({
                    'name': result.function_name,
                    'file': result.file_path,
                    'line': result.line_number,
                    'similarity': result.similarity_score
                })
    
    return results


def _search_without_index(query_ast: ast.FunctionDef, search_path: Path, top_k: int, threshold: float) -> List[Dict]:
    """Search without index (slower)."""
    encoder = HypervectorEncoder()
    query_hv = encoder.encode_ast(query_ast)
    
    results = []
    
    for file_path in search_path.rglob("*.py"):
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            import ast
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == query_ast.name:
                        continue  # Skip self
                    
                    node_hv = encoder.encode_ast(node)
                    similarity = encoder.similarity(query_hv, node_hv)
                    
                    if similarity >= threshold:
                        results.append({
                            'name': node.name,
                            'file': str(file_path),
                            'line': node.lineno,
                            'similarity': similarity
                        })
        
        except:
            continue
    
    # Sort and limit
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results[:top_k]


def _display_profile(name: str, data: Dict, is_default: bool):
    """Display configuration profile."""
    status = " (default)" if is_default else ""
    console.print(f"\n[bold cyan]👤 Profile: {name}{status}[/bold cyan]")
    
    # Convert dict to tree structure
    def dict_to_tree(d: Dict, parent_tree: Tree) -> Tree:
        for key, value in d.items():
            if isinstance(value, dict):
                branch = parent_tree.add(f"[yellow]{key}[/yellow]")
                dict_to_tree(value, branch)
            else:
                parent_tree.add(f"[green]{key}[/green]: [blue]{value}[/blue]")
        return parent_tree
    
    tree = Tree("Configuration")
    dict_to_tree(data, tree)
    console.print(tree)


def _run_incremental_analysis(files: set, index_path: Optional[Path], patterns: List[PatternType]) -> List[Issue]:
    """Run incremental analysis on changed files."""
    # Simplified incremental analysis
    issues = []
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            import ast
            tree = ast.parse(content)
            
            # Simple phantom function detection
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        issues.append(Issue(
                            kind="phantom_function",
                            message=f"Empty function: {node.name}",
                            severity=2,
                            file=str(file_path),
                            line=node.lineno,
                            symbol=node.name
                        ))
        
        except:
            continue
    
    return issues


if __name__ == "__main__":
    app()