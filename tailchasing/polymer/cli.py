"""
CLI commands for polymer physics module.

This module provides command-line interface for the polymer physics
calibration and configuration tools.
"""

import click
from pathlib import Path
from rich.console import Console

from .calibrate import CalibrationTool
from .config import (
    ConfigManager,
    PolymerConfig,
    create_default_config_file,
    get_config,
    save_config
)


console = Console()


@click.group(name="polymer")
def polymer_cli():
    """Polymer physics commands for chromatin-inspired analysis."""
    pass


@polymer_cli.group(name="config")
def config_group():
    """Manage polymer physics configuration."""
    pass


@config_group.command(name="init")
@click.option(
    "--path",
    type=click.Path(),
    default=".tailchasing_polymer.yml",
    help="Path for config file"
)
def config_init(path):
    """Initialize polymer physics configuration file."""
    config_path = Path(path)
    
    if config_path.exists():
        if not click.confirm(f"Config file {path} already exists. Overwrite?"):
            console.print("[yellow]Aborted[/yellow]")
            return
    
    if create_default_config_file(config_path):
        console.print(f"[green]✓ Created config file at {path}[/green]")
    else:
        console.print("[red]✗ Failed to create config file[/red]")


@config_group.command(name="show")
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Path to config file"
)
def config_show(path):
    """Display current polymer physics configuration."""
    config_path = Path(path) if path else None
    manager = ConfigManager(config_path)
    config = manager.load()
    manager.display(config)


@config_group.command(name="validate")
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Path to config file"
)
def config_validate(path):
    """Validate polymer physics configuration."""
    config_path = Path(path) if path else None
    config = get_config(config_path)
    
    if config.validate():
        console.print("[green]✓ Configuration is valid[/green]")
    else:
        console.print("[red]✗ Configuration has validation errors[/red]")
        raise click.Exit(1)


@config_group.command(name="set")
@click.argument("parameter")
@click.argument("value")
@click.option(
    "--path",
    type=click.Path(),
    help="Path to config file"
)
def config_set(parameter, value, path):
    """Set a configuration parameter."""
    config_path = Path(path) if path else None
    manager = ConfigManager(config_path)
    
    # Parse value based on parameter
    try:
        if parameter in ["alpha", "epsilon", "kappa", "tad_penalty"]:
            parsed_value = float(value)
        elif parameter.startswith("weight_"):
            weight_name = parameter.replace("weight_", "")
            if weight_name not in ["tok", "ast", "mod", "git"]:
                console.print(f"[red]Unknown weight: {weight_name}[/red]")
                raise click.Exit(1)
            config = manager.load()
            config.weights[weight_name] = float(value)
            manager.save(config)
            console.print(f"[green]✓ Set {parameter} = {value}[/green]")
            return
        elif parameter in ["min_tad_size", "max_tad_size", "contact_matrix_resolution"]:
            parsed_value = int(value)
        elif parameter in ["show_tad_boundaries", "show_loop_anchors"]:
            parsed_value = value.lower() in ["true", "yes", "1"]
        else:
            parsed_value = value
        
        config = manager.update(**{parameter: parsed_value})
        manager.save(config)
        console.print(f"[green]✓ Set {parameter} = {parsed_value}[/green]")
        
    except ValueError as e:
        console.print(f"[red]Invalid value for {parameter}: {e}[/red]")
        raise click.Exit(1)


@polymer_cli.group(name="calibrate")
def calibrate_group():
    """Calibration tools for polymer physics parameters."""
    pass


@calibrate_group.command(name="demo")
@click.option(
    "--n-events",
    type=int,
    default=100,
    help="Number of demo events to generate"
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility"
)
def calibrate_demo(n_events, seed):
    """Run calibration demo with synthetic data."""
    console.print("[bold cyan]Running Polymer Physics Calibration Demo[/bold cyan]\n")
    
    tool = CalibrationTool()
    
    # Generate demo data
    console.print(f"[yellow]Generating {n_events} demo events...[/yellow]")
    events, codebase = tool.generate_demo_data(n_events=n_events, seed=seed)
    
    # Split data
    split_idx = int(len(events) * 0.8)
    train_events = events[:split_idx]
    test_events = events[split_idx:]
    
    console.print(f"Training events: {len(train_events)}")
    console.print(f"Test events: {len(test_events)}\n")
    
    # Fit parameters
    console.print("[yellow]Fitting parameters...[/yellow]")
    result = tool.fit_parameters(train_events, codebase)
    
    # Display results
    tool.display_results(result)
    
    # Validate
    if test_events:
        console.print("\n[yellow]Validation Results:[/yellow]")
        validation = tool.validate_parameters(result, test_events)
        
        for metric, value in validation.items():
            console.print(f"  {metric}: {value:.3f}")
    
    # Save calibrated config if desired
    if click.confirm("\nSave calibrated parameters to config?"):
        config = PolymerConfig(
            alpha=result.alpha,
            weights=result.weights,
            epsilon=result.epsilon,
            kappa=result.kappa
        )
        
        if save_config(config):
            console.print("[green]✓ Saved calibrated parameters[/green]")


@calibrate_group.command(name="grid")
@click.option(
    "--n-events",
    type=int,
    default=100,
    help="Number of events for calibration"
)
@click.option(
    "--grid-points",
    type=int,
    default=3,
    help="Number of grid points per dimension"
)
@click.option(
    "--alpha-min",
    type=float,
    default=0.8,
    help="Minimum alpha value"
)
@click.option(
    "--alpha-max",
    type=float,
    default=2.0,
    help="Maximum alpha value"
)
def calibrate_grid(n_events, grid_points, alpha_min, alpha_max):
    """Perform grid search for optimal parameters."""
    console.print("[bold cyan]Running Grid Search Calibration[/bold cyan]\n")
    
    tool = CalibrationTool()
    
    # Generate data
    events, codebase = tool.generate_demo_data(n_events=n_events)
    
    # Run grid search
    result = tool.grid_search(
        events,
        codebase,
        alpha_range=(alpha_min, alpha_max),
        n_grid_points=grid_points
    )
    
    # Display results
    console.print("\n[green]Best Parameters Found:[/green]")
    tool.display_results(result)


@polymer_cli.command(name="analyze")
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to polymer config file"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file for analysis results"
)
def analyze(path, config, output):
    """
    Analyze codebase using polymer physics model.
    
    This command integrates with the existing ChromatinContactAnalyzer
    to perform polymer physics-based analysis.
    """
    from pathlib import Path
    import json
    
    console.print(f"[cyan]Analyzing {path} with polymer physics model...[/cyan]")
    
    # Load configuration
    config_path = Path(config) if config else None
    polymer_config = get_config(config_path)
    
    # Validate config
    if not polymer_config.validate():
        console.print("[red]Invalid configuration, aborting[/red]")
        raise click.Exit(1)
    
    # Import and run analyzer
    try:
        from ..analyzers.chromatin_contact import ChromatinContactAnalyzer
        from ..core.loader import collect_files, parse_files
        from ..analyzers.base import AnalysisContext
        
        # Collect and parse files
        target_path = Path(path)
        files = collect_files([target_path], exclude_patterns=[])
        parsed_files = parse_files(files)
        
        # Create context
        context = AnalysisContext(
            files=parsed_files,
            config=polymer_config.to_dict()
        )
        
        # Run analyzer
        analyzer = ChromatinContactAnalyzer()
        issues = analyzer.analyze(context)
        
        # Display results
        console.print(f"\n[green]Found {len(issues)} issues[/green]")
        
        if output:
            # Save to file
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(
                    [issue.to_dict() for issue in issues],
                    f,
                    indent=2
                )
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            # Display inline
            for issue in issues[:10]:  # Show first 10
                console.print(f"\n[yellow]{issue.type}[/yellow]: {issue.message}")
                console.print(f"  File: {issue.file_path}:{issue.line}")
                console.print(f"  Risk: {issue.risk_score:.2f}")
        
    except ImportError as e:
        console.print(f"[red]Error importing analyzer: {e}[/red]")
        console.print("[yellow]Make sure ChromatinContactAnalyzer is available[/yellow]")
        raise click.Exit(1)
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise click.Exit(1)


# Register with main CLI if available
def register_polymer_cli(cli):
    """Register polymer CLI with main tailchasing CLI."""
    cli.add_command(polymer_cli)


if __name__ == "__main__":
    polymer_cli()