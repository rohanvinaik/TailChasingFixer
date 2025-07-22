#!/usr/bin/env python3
"""
Enhanced CLI commands for tail-chasing analysis with visualization and explanations.
"""

import click
import sys
from pathlib import Path
from typing import Optional
import webbrowser

from .cli import main as base_main
from .core.loader import collect_files, parse_files
from .core.symbols import SymbolTable
from .analyzers.base import AnalysisContext
from .plugins import load_analyzers
from .config import load_config
from .analyzers.root_cause_tracer import RootCauseTracer
from .analyzers.explainer import TailChasingExplainer
from .core.reporting import render_json


@click.group()
def cli():
    """Enhanced tail-chasing detector with advanced features."""
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True), default='.')
@click.option('--explain', is_flag=True, help='Generate natural language explanations')
@click.option('--visualize', is_flag=True, help='Generate interactive visualization')
@click.option('--output', '-o', help='Output file for visualization (default: tail_chase_report.html)')
@click.option('--open-browser', is_flag=True, help='Open visualization in browser')
@click.option('--json', is_flag=True, help='Output JSON format')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(path: str, explain: bool, visualize: bool, output: Optional[str], 
           open_browser: bool, json: bool, config: Optional[str], verbose: bool):
    """
    Analyze codebase for tail-chasing patterns with advanced features.
    
    Examples:
        tailchasing analyze . --explain
        tailchasing analyze src/ --visualize --open-browser
        tailchasing analyze . --explain --visualize --output report.html
    """
    # Load configuration
    root = Path(path).resolve()
    config_data = load_config(root, config)
    
    if verbose:
        click.echo(f"Analyzing {root}...")
    
    # Run analysis
    files = collect_files(root, 
                         config_data.get('paths', {}).get('include'),
                         config_data.get('paths', {}).get('exclude'))
    
    if verbose:
        click.echo(f"Found {len(files)} Python files")
    
    ast_index = parse_files(files)
    symbol_table = SymbolTable()
    
    for f, tree in ast_index.items():
        try:
            symbol_table.ingest(f, tree, "")
        except Exception as e:
            if verbose:
                click.echo(f"Warning: Failed to parse {f}: {e}", err=True)
    
    # Create analysis context
    ctx = AnalysisContext(config_data, files, ast_index, symbol_table, {})
    
    # Load and run analyzers
    analyzers = load_analyzers(config_data)
    issues = []
    
    for analyzer in analyzers:
        if verbose:
            click.echo(f"Running {analyzer.name}...")
        try:
            analyzer_issues = list(analyzer.run(ctx))
            issues.extend(analyzer_issues)
            if verbose and analyzer_issues:
                click.echo(f"  Found {len(analyzer_issues)} issues")
        except Exception as e:
            click.echo(f"Error in {analyzer.name}: {e}", err=True)
    
    if not issues:
        click.echo("✅ No tail-chasing patterns detected!")
        return
    
    # Output results
    if json:
        click.echo(render_json(issues))
        return
    
    click.echo(f"\n🔍 Found {len(issues)} tail-chasing patterns\n")
    
    # Generate explanations if requested
    if explain:
        explainer = TailChasingExplainer()
        
        # Summary explanation
        summary = explainer.generate_summary_explanation(issues)
        click.echo(summary)
        
        # Individual explanations for top issues
        if verbose:
            click.echo("\n--- Detailed Explanations ---\n")
            for issue in issues[:5]:  # Top 5 issues
                explanation = explainer.explain_issue(issue)
                click.echo(explanation)
                click.echo("\n" + "-" * 80 + "\n")
    
    # Generate visualization if requested
    if visualize:
        tracer = RootCauseTracer()
        chains = tracer.analyze_tail_chase_chains(issues)
        
        if chains:
            output_path = output or "tail_chase_report.html"
            report_path = tracer.generate_visual_report(output_path)
            click.echo(f"\n📊 Visual report generated: {report_path}")
            
            if open_browser:
                webbrowser.open(f"file://{Path(report_path).absolute()}")
        else:
            click.echo("No tail-chasing chains detected for visualization")
    
    # Show summary statistics
    if not json and not explain:
        severity_counts = {}
        kind_counts = {}
        
        for issue in issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            base_kind = issue.kind.split('_')[0]
            kind_counts[base_kind] = kind_counts.get(base_kind, 0) + 1
        
        click.echo("\n📊 Summary:")
        click.echo("By severity:")
        for sev in sorted(severity_counts.keys(), reverse=True):
            click.echo(f"  Severity {sev}: {severity_counts[sev]} issues")
        
        click.echo("\nBy type:")
        for kind, count in sorted(kind_counts.items(), key=lambda x: x[1], reverse=True):
            click.echo(f"  {kind}: {count} issues")
    
    # Exit with error code if issues found
    sys.exit(1 if issues else 0)


@cli.command()
@click.argument('issue_type')
def explain_pattern(issue_type: str):
    """
    Explain a specific tail-chasing pattern type.
    
    Examples:
        tailchasing explain-pattern phantom_function
        tailchasing explain-pattern circular_import
    """
    from .core.issues import Issue
    
    explainer = TailChasingExplainer()
    
    # Create a dummy issue for explanation
    dummy_issue = Issue(
        kind=issue_type,
        message=f"Example {issue_type} pattern",
        severity=2,
        file="example.py",
        line=10,
        symbol="example_function"
    )
    
    explanation = explainer.explain_issue(dummy_issue)
    click.echo(explanation)


@cli.command()
def list_patterns():
    """List all detectable tail-chasing patterns."""
    patterns = {
        'phantom_function': 'Empty stub functions with no implementation',
        'circular_import': 'Modules importing each other in a cycle',
        'duplicate_function': 'Identical functions in multiple places',
        'semantic_duplicate_function': 'Functionally equivalent but syntactically different functions',
        'missing_symbol': 'Imports of non-existent symbols',
        'wrapper_abstraction': 'Trivial wrappers that add no value',
        'hallucination_cascade': 'Entire fictional subsystems created by LLM',
        'context_window_thrashing': 'Repeated implementations due to lost context',
        'import_anxiety': 'Excessive defensive importing',
        'mirror_test': 'Tests that duplicate implementation logic',
        'brittle_test_assertions': 'Tests with overly specific assertions',
        'cargo_cult_*': 'Various patterns of copying without understanding',
        'cross_file_duplication': 'Semantic duplication across different files',
    }
    
    click.echo("🔍 Detectable Tail-Chasing Patterns:\n")
    
    for pattern, description in patterns.items():
        click.echo(f"  {pattern:<30} - {description}")
    
    click.echo("\nUse 'tailchasing explain-pattern <pattern>' for detailed explanation")


@cli.command()
@click.argument('path', type=click.Path(exists=True), default='.')
@click.option('--fix', is_flag=True, help='Apply automatic fixes (use with caution)')
@click.option('--dry-run', is_flag=True, help='Show what would be fixed without applying')
def fix(path: str, fix: bool, dry_run: bool):
    """
    Analyze and optionally fix tail-chasing patterns.
    
    Examples:
        tailchasing fix . --dry-run
        tailchasing fix src/ --fix
    """
    click.echo("🔧 Fix command - Coming soon!")
    click.echo("\nThis will integrate with the intelligent auto-fix system to:")
    click.echo("  - Merge semantic duplicates")
    click.echo("  - Remove phantom implementations")
    click.echo("  - Break circular imports")
    click.echo("  - Clean up cargo cult patterns")
    
    if fix and not dry_run:
        click.echo("\n⚠️  Warning: Auto-fix can modify your code. Always review changes!")


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
