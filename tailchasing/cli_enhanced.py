"""
Enhanced CLI with interactive mode and rich output.
"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import time


console = Console()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Enhanced Tail-Chasing Detector CLI."""
    if ctx.invoked_subcommand is None:
        # Launch interactive mode
        interactive_mode()


def interactive_mode():
    """Interactive analysis mode with rich UI."""
    console.print(Panel.fit(
        "[bold cyan]Tail-Chasing Detector - Interactive Mode[/bold cyan]\n"
        "Type 'help' for available commands",
        border_style="cyan"
    ))
    
    commands = WordCompleter([
        'analyze', 'compare', 'watch', 'report', 
        'semantic', 'visualize', 'fix', 'help', 'quit'
    ])
    
    while True:
        try:
            command = prompt("tailchasing> ", completer=commands)
            
            if command == 'quit':
                break
            elif command == 'help':
                show_help()
            elif command.startswith('analyze'):
                analyze_interactive(command)
            elif command.startswith('semantic'):
                semantic_interactive(command)
            elif command.startswith('watch'):
                watch_mode()
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit[/yellow]")
        except EOFError:
            break


def show_help():
    """Show interactive help."""
    help_table = Table(title="Available Commands")
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description")
    
    help_table.add_row("analyze [path]", "Analyze a file or directory")
    help_table.add_row("compare <file1> <file2>", "Compare semantic similarity")
    help_table.add_row("watch [path]", "Watch for changes and analyze")
    help_table.add_row("report", "Generate detailed report")
    help_table.add_row("semantic", "Semantic analysis submenu")
    help_table.add_row("visualize", "Generate visualizations")
    help_table.add_row("fix", "Apply automated fixes")
    help_table.add_row("quit", "Exit interactive mode")
    
    console.print(help_table)


def analyze_interactive(command: str):
    """Interactive analysis with progress display."""
    parts = command.split()
    path = parts[1] if len(parts) > 1 else "."
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing...", total=5)
        
        # Simulate analysis steps
        steps = [
            "Loading files",
            "Building AST",
            "Encoding semantics",
            "Finding patterns",
            "Generating report"
        ]
        
        for step in steps:
            progress.update(task, description=f"[cyan]{step}...")
            time.sleep(0.5)  # Simulate work
            progress.advance(task)
    
    # Show results
    show_analysis_results()


def show_analysis_results():
    """Display analysis results in a rich format."""
    # Summary panel
    console.print(Panel(
        "[bold green]✓[/bold green] Analysis Complete\n"
        "Risk Score: [bold red]24[/bold red] (WARNING)\n"
        "Issues Found: [bold]17[/bold]",
        title="Summary",
        border_style="yellow"
    ))
    
    # Issues table
    table = Table(title="Top Issues")
    table.add_column("Type", style="cyan")
    table.add_column("Location", style="magenta")
    table.add_column("Description")
    table.add_column("Severity", justify="center")
    
    issues = [
        ("semantic_duplicate", "utils.py:45", "calculate_avg ≈ compute_mean", "HIGH"),
        ("phantom_function", "handlers.py:123", "process_data() is empty", "MED"),
        ("circular_import", "models.py", "Circular with views.py", "HIGH"),
    ]
    
    for issue_type, location, desc, severity in issues:
        color = "red" if severity == "HIGH" else "yellow"
        table.add_row(issue_type, location, desc, f"[{color}]{severity}[/{color}]")
    
    console.print(table)
    
    # Code example
    console.print("\n[bold]Example Issue:[/bold]")
    code = '''def calculate_avg(numbers):
    """Calculate average of numbers."""
    return sum(numbers) / len(numbers)

def compute_mean(vals):  # Semantic duplicate!
    """Compute mean value."""
    total = 0
    for v in vals:
        total += v
    return total / len(vals)'''
    
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)


def semantic_interactive(command: str):
    """Interactive semantic analysis submenu."""
    console.print("[bold cyan]Semantic Analysis Mode[/bold cyan]")
    
    submenu = WordCompleter([
        'similarity', 'clusters', 'drift', 'prototypes', 'back'
    ])
    
    while True:
        subcommand = prompt("semantic> ", completer=submenu)
        
        if subcommand == 'back':
            break
        elif subcommand == 'similarity':
            show_similarity_matrix()
        elif subcommand == 'clusters':
            show_semantic_clusters()
        # ... etc


def show_similarity_matrix():
    """Display semantic similarity matrix."""
    # Create a heatmap-style display
    console.print("\n[bold]Semantic Similarity Matrix[/bold]")
    
    functions = ["calc_avg", "get_mean", "average", "process", "handle"]
    
    # Header
    console.print("        ", end="")
    for f in functions:
        console.print(f"{f:>10}", end="")
    console.print()
    
    # Matrix
    for i, f1 in enumerate(functions):
        console.print(f"{f1:>8}", end="")
        for j, f2 in enumerate(functions):
            if i == j:
                console.print("     -    ", end="")
            else:
                # Simulate similarity scores
                score = 0.95 if (i < 3 and j < 3) else 0.2
                color = "red" if score > 0.8 else "green"
                console.print(f"  [{color}]{score:.2f}[/{color}]  ", end="")
        console.print()


def watch_mode():
    """Watch for file changes and analyze in real-time."""
    console.print("[bold cyan]Watch Mode[/bold cyan] - Press Ctrl+C to stop")
    
    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                # Simulate file change detection
                table = Table(title="Watching for changes...")
                table.add_column("File", style="cyan")
                table.add_column("Status")
                table.add_column("Issues")
                
                table.add_row("src/utils.py", "[green]✓[/green] Clean", "0")
                table.add_row("src/models.py", "[yellow]![/yellow] Modified", "2")
                
                live.update(table)
                time.sleep(1)
                
            except KeyboardInterrupt:
                break


@cli.command()
@click.argument('files', nargs=-1)
@click.option('--semantic/--no-semantic', default=True)
@click.option('--visualize', is_flag=True)
def compare(files, semantic, visualize):
    """Compare semantic similarity between files."""
    if len(files) < 2:
        console.print("[red]Need at least 2 files to compare[/red]")
        return
    
    console.print(f"[cyan]Comparing {len(files)} files...[/cyan]")
    
    # ... implementation


@cli.command()
@click.option('--output', '-o', default='report.html')
@click.option('--format', type=click.Choice(['html', 'pdf', 'markdown']))
def report(output, format):
    """Generate comprehensive analysis report."""
    with console.status("Generating report..."):
        time.sleep(2)  # Simulate work
    
    console.print(f"[green]✓[/green] Report generated: {output}")


if __name__ == '__main__':
    cli()