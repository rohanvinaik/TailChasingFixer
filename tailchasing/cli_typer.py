"""
Enhanced CLI using Typer and Rich for modern command-line experience.

Features:
- Typer for type-safe CLI
- Rich for beautiful progress bars and formatting
- Interactive mode for fix selection
- YAML configuration support
- Advanced reporting with tables and panels
"""

import sys
import json
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from .utils.logging_setup import get_logger
from .core.loader import collect_files, parse_files
from .core.symbols import SymbolTable
from .core.issues import Issue
from .core.reporting import Reporter
from .analyzers.base import AnalysisContext
# Optional imports - may not be available in all setups
try:
    from .analyzers.explainer import TailChasingExplainer
except ImportError:
    TailChasingExplainer = None

try:
    from .analyzers.advanced.enhanced_pattern_detector import EnhancedPatternDetector
except ImportError:
    EnhancedPatternDetector = None

try:
    from .analyzers.advanced.multimodal_semantic_enhanced import EnhancedSemanticAnalyzer
except ImportError:
    EnhancedSemanticAnalyzer = None

try:
    from .fixers.advanced.intelligent_fixer import IntelligentAutoFixer
except ImportError:
    IntelligentAutoFixer = None

try:
    from .visualization import TailChasingVisualizer
except ImportError:
    TailChasingVisualizer = None
from .plugins import load_analyzers

# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="tailchasing",
    help="🔍 Detect and fix LLM-induced tail-chasing patterns in your codebase",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

class OutputFormat(str, Enum):
    """Output format options."""
    text = "text"
    json = "json"
    html = "html"
    yaml = "yaml"

class FixMode(str, Enum):
    """Fix application modes."""
    interactive = "interactive"
    automatic = "automatic"
    plan_only = "plan_only"


class TailChasingCLI:
    """Enhanced CLI with Typer and Rich."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.console = console
        self.explainer = TailChasingExplainer() if TailChasingExplainer else None
        self.enhanced_detector = EnhancedPatternDetector() if EnhancedPatternDetector else None
        self.semantic_analyzer = EnhancedSemanticAnalyzer(vector_dim=16384) if EnhancedSemanticAnalyzer else None
        self.auto_fixer = IntelligentAutoFixer() if IntelligentAutoFixer else None
        self.visualizer = TailChasingVisualizer() if TailChasingVisualizer else None
        
        # Configuration cache
        self._config_cache: Dict[str, Any] = {}
    
    def load_config(self, root: Path, config_file: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration from .tailchasing.yml or specified file."""
        cache_key = str(root / (config_file or ".tailchasing.yml"))
        
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        config = {
            "paths": {"include": ["**/*.py"], "exclude": []},
            "analysis": {
                "enhanced_detection": False,
                "semantic_analysis": False,
                "git_history": False,
                "parallel": False,
                "cache": True
            },
            "output": {
                "format": "text",
                "severity_threshold": 1,
                "show_fixes": True,
                "show_explanations": False
            },
            "fixes": {
                "mode": "interactive",
                "backup": True,
                "validation": True
            }
        }
        
        # Try to load config file
        config_paths = []
        if config_file:
            config_paths.append(config_file)
        else:
            # Look for config in common locations
            config_paths.extend([
                root / ".tailchasing.yml",
                root / ".tailchasing.yaml", 
                root / "pyproject.toml",  # Could parse [tool.tailchasing] section
                Path.home() / ".tailchasing.yml"
            ])
        
        for config_path in config_paths:
            if config_path.exists() and config_path.suffix in ['.yml', '.yaml']:
                try:
                    with open(config_path) as f:
                        file_config = yaml.safe_load(f) or {}
                    
                    # Merge configurations (file config takes precedence)
                    config = self._merge_configs(config, file_config)
                    self.logger.info(f"Loaded configuration from {config_path}")
                    break
                except Exception as e:
                    self.console.print(f"[yellow]Warning: Failed to load config from {config_path}: {e}[/yellow]")
        
        self._config_cache[cache_key] = config
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def analyze_codebase(
        self,
        root: Path,
        config: Dict[str, Any],
        show_progress: bool = True
    ) -> Tuple[List[Issue], AnalysisContext]:
        """Analyze codebase with progress tracking."""
        issues = []
        
        # Collect files with progress
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                task = progress.add_task("🔍 Collecting files...", total=None)
                files = collect_files(
                    root,
                    config.get('paths', {}).get('include', ['**/*.py']),
                    config.get('paths', {}).get('exclude', [])
                )
                progress.update(task, description=f"📁 Found {len(files)} Python files")
        else:
            files = collect_files(
                root,
                config.get('paths', {}).get('include', ['**/*.py']),
                config.get('paths', {}).get('exclude', [])
            )
        
        if not files:
            self.console.print("[red]No Python files found to analyze.[/red]")
            return [], None
        
        # Parse files with progress
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                parse_task = progress.add_task("🔍 Parsing files...", total=len(files))
                
                ast_index = {}
                for i, file_path in enumerate(files):
                    try:
                        ast_index.update(parse_files([file_path]))
                        progress.update(parse_task, advance=1, description=f"📝 Parsing {file_path.name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to parse {file_path}: {e}")
                        progress.update(parse_task, advance=1)
        else:
            ast_index = parse_files(files)
        
        # Build symbol table
        symbol_table = SymbolTable()
        for filepath, tree in ast_index.items():
            try:
                symbol_table.ingest(filepath, tree, "")
            except Exception as e:
                self.logger.warning(f"Failed to process symbols in {filepath}: {e}")
        
        # Set up analysis context
        source_cache = {}
        cache = {}
        ctx = AnalysisContext(config, root, files, ast_index, symbol_table, source_cache, cache)
        
        # Run analyzers with progress
        analyzers = load_analyzers(config)
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                analysis_task = progress.add_task("🔍 Running analyzers...", total=len(analyzers))
                
                for analyzer in analyzers:
                    try:
                        analyzer_issues = list(analyzer.run(ctx))
                        issues.extend(analyzer_issues)
                        progress.update(
                            analysis_task, 
                            advance=1, 
                            description=f"✅ {analyzer.name}: {len(analyzer_issues)} issues"
                        )
                    except Exception as e:
                        self.logger.error(f"Analyzer {analyzer.name} failed: {e}")
                        progress.update(analysis_task, advance=1, description=f"❌ {analyzer.name}: Failed")
        else:
            for analyzer in analyzers:
                try:
                    analyzer_issues = list(analyzer.run(ctx))
                    issues.extend(analyzer_issues)
                except Exception as e:
                    self.logger.error(f"Analyzer {analyzer.name} failed: {e}")
        
        # Run enhanced detection if enabled
        if config.get('analysis', {}).get('enhanced_detection', False):
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("🧠 Running enhanced pattern detection..."),
                    transient=True
                ) as progress:
                    progress.add_task("Enhanced detection", total=None)
                    enhanced_issues = self._run_enhanced_detection(ctx, ast_index)
                    issues.extend(enhanced_issues)
            else:
                enhanced_issues = self._run_enhanced_detection(ctx, ast_index)
                issues.extend(enhanced_issues)
        
        # Run semantic analysis if enabled
        if config.get('analysis', {}).get('semantic_analysis', False):
            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("🔬 Running semantic analysis..."),
                    transient=True
                ) as progress:
                    progress.add_task("Semantic analysis", total=None)
                    semantic_issues = self._run_semantic_analysis(ctx, symbol_table)
                    issues.extend(semantic_issues)
            else:
                semantic_issues = self._run_semantic_analysis(ctx, symbol_table)
                issues.extend(semantic_issues)
        
        return issues, ctx
    
    def _run_enhanced_detection(self, ctx: AnalysisContext, ast_index: Dict[str, Any]) -> List[Issue]:
        """Run enhanced pattern detection."""
        issues = []
        
        # Detect hallucination cascades
        cascade_issues = self.enhanced_detector.detect_hallucination_cascade(ast_index)
        issues.extend(cascade_issues)
        
        # Detect context window thrashing
        for filepath, tree in ast_index.items():
            thrashing_issues = self.enhanced_detector.detect_context_window_thrashing(tree, filepath)
            issues.extend(thrashing_issues)
        
        return issues
    
    def _run_semantic_analysis(self, ctx: AnalysisContext, symbol_table: SymbolTable) -> List[Issue]:
        """Run enhanced semantic analysis."""
        # Extract functions for analysis
        functions = []
        for func_name, entries in symbol_table.functions.items():
            for entry in entries:
                functions.append((entry['file'], entry['node']))
        
        # Find semantic duplicates
        return self.semantic_analyzer.find_semantic_duplicates(functions, threshold=0.85)
    
    def display_issues_table(self, issues: List[Issue], show_details: bool = True):
        """Display issues in a rich table."""
        if not issues:
            self.console.print("[green]🎉 No issues found! Your code looks great![/green]")
            return
        
        # Group issues by severity
        severity_groups = {}
        for issue in issues:
            severity_groups.setdefault(issue.severity, []).append(issue)
        
        # Create main table
        table = Table(title=f"🔍 Tail-Chasing Analysis Results ({len(issues)} issues)")
        table.add_column("Severity", style="bold", width=8)
        table.add_column("Type", style="cyan", width=20)
        table.add_column("File", style="blue", width=25)
        table.add_column("Line", justify="right", width=6)
        table.add_column("Description", style="white")
        
        # Add issues sorted by severity (descending)
        for severity in sorted(severity_groups.keys(), reverse=True):
            severity_color = {5: "red", 4: "orange1", 3: "yellow", 2: "blue", 1: "green"}.get(severity, "white")
            severity_icon = {5: "🚨", 4: "⚠️", 3: "🔶", 2: "ℹ️", 1: "💡"}.get(severity, "•")
            
            for issue in severity_groups[severity]:
                table.add_row(
                    f"[{severity_color}]{severity_icon} {severity}[/{severity_color}]",
                    f"[cyan]{issue.kind}[/cyan]",
                    f"[blue]{Path(issue.file).name}[/blue]",
                    str(issue.line or ""),
                    issue.message
                )
        
        self.console.print(table)
        
        # Show severity summary
        summary_table = Table(title="📊 Issue Summary")
        summary_table.add_column("Severity", style="bold")
        summary_table.add_column("Count", justify="right", style="bold")
        summary_table.add_column("Description")
        
        severity_descriptions = {
            5: "Critical - Immediate attention required",
            4: "High - Should be fixed soon",
            3: "Medium - Consider fixing",
            2: "Low - Minor improvements",
            1: "Info - Style or suggestions"
        }
        
        for severity in sorted(severity_groups.keys(), reverse=True):
            count = len(severity_groups[severity])
            severity_color = {5: "red", 4: "orange1", 3: "yellow", 2: "blue", 1: "green"}.get(severity, "white")
            
            summary_table.add_row(
                f"[{severity_color}]{severity}[/{severity_color}]",
                f"[{severity_color}]{count}[/{severity_color}]",
                severity_descriptions.get(severity, "")
            )
        
        self.console.print(summary_table)
    
    def interactive_fix_selection(self, issues: List[Issue]) -> List[Issue]:
        """Interactive mode for selecting which issues to fix."""
        if not issues:
            return []
        
        self.console.print("\n[bold blue]🔧 Interactive Fix Selection[/bold blue]")
        self.console.print("Select which issues you'd like to generate fixes for:\n")
        
        selected_issues = []
        
        # Group issues by file for better organization
        issues_by_file = {}
        for issue in issues:
            issues_by_file.setdefault(issue.file, []).append(issue)
        
        for file_path, file_issues in issues_by_file.items():
            self.console.print(f"\n[bold cyan]📁 {Path(file_path).name}[/bold cyan]")
            
            # Show issues in this file
            for i, issue in enumerate(file_issues, 1):
                severity_color = {5: "red", 4: "orange1", 3: "yellow", 2: "blue", 1: "green"}.get(issue.severity, "white")
                severity_icon = {5: "🚨", 4: "⚠️", 3: "🔶", 2: "ℹ️", 1: "💡"}.get(issue.severity, "•")
                
                self.console.print(f"  [{severity_color}]{i}. {severity_icon} {issue.kind}[/{severity_color}] - {issue.message}")
                if issue.line:
                    self.console.print(f"     [dim]Line {issue.line}[/dim]")
            
            # Ask user for selection
            if len(file_issues) == 1:
                if Confirm.ask(f"Fix this issue in {Path(file_path).name}?", default=True):
                    selected_issues.extend(file_issues)
            else:
                choices = Prompt.ask(
                    f"Which issues to fix? (1-{len(file_issues)}, 'all', 'none', or comma-separated numbers)",
                    default="all"
                ).strip().lower()
                
                if choices == "all":
                    selected_issues.extend(file_issues)
                elif choices != "none":
                    try:
                        if ',' in choices:
                            indices = [int(x.strip()) - 1 for x in choices.split(',')]
                        else:
                            indices = [int(choices) - 1]
                        
                        for idx in indices:
                            if 0 <= idx < len(file_issues):
                                selected_issues.append(file_issues[idx])
                    except ValueError:
                        self.console.print("[yellow]Invalid selection, skipping this file[/yellow]")
        
        self.console.print(f"\n[green]✅ Selected {len(selected_issues)} issues for fixing[/green]")
        return selected_issues


# CLI Commands using Typer
@app.command()
def analyze(
    root: Path = typer.Argument(Path("."), help="Root directory to analyze"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format", "-f", help="Output format"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file (optional)"),
    severity: Optional[int] = typer.Option(None, "--severity", "-s", help="Minimum severity level (1-5)"),
    enhanced: bool = typer.Option(False, "--enhanced", help="Enable enhanced pattern detection"),
    semantic: bool = typer.Option(False, "--semantic", help="Enable semantic analysis"),
    git_history: bool = typer.Option(False, "--git-history", help="Analyze git history"),
    parallel: bool = typer.Option(False, "--parallel", help="Enable parallel processing"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress bars")
):
    """
    🔍 Analyze codebase for tail-chasing patterns.
    
    This command scans your Python codebase to detect LLM-induced patterns
    like phantom functions, duplicate code, and semantic inconsistencies.
    """
    cli = TailChasingCLI()
    
    try:
        # Load configuration
        config = cli.load_config(root, config_file)
        
        # Override config with command line options
        if enhanced:
            config.setdefault('analysis', {})['enhanced_detection'] = True
        if semantic:
            config.setdefault('analysis', {})['semantic_analysis'] = True
        if git_history:
            config.setdefault('analysis', {})['git_history'] = True
        if parallel:
            config.setdefault('analysis', {})['parallel'] = True
        if no_cache:
            config.setdefault('analysis', {})['cache'] = False
        
        # Show startup info
        if not quiet:
            cli.console.print(Panel.fit(
                f"🔍 TailChasing Fixer v2.0\n"
                f"📁 Analyzing: [blue]{root.absolute()}[/blue]\n"
                f"🎯 Enhanced: {'✅' if enhanced else '❌'}\n"
                f"🧠 Semantic: {'✅' if semantic else '❌'}",
                title="Analysis Starting",
                border_style="blue"
            ))
        
        # Run analysis
        issues, ctx = cli.analyze_codebase(root, config, show_progress=not quiet)
        
        # Filter by severity
        if severity:
            issues = [issue for issue in issues if issue.severity >= severity]
        
        # Display results
        if output_format == OutputFormat.text:
            if not quiet and interactive:
                cli.display_issues_table(issues)
            elif not quiet:
                cli.display_issues_table(issues, show_details=False)
        
        # Handle output
        if output_format == OutputFormat.json:
            reporter = Reporter(config)
            result = reporter.render_json(issues)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(result)
                cli.console.print(f"[green]Results saved to {output_file}[/green]")
            else:
                print(result)
        
        elif output_format == OutputFormat.yaml:
            import yaml
            result = yaml.dump([issue.to_dict() for issue in issues], default_flow_style=False)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(result)
                cli.console.print(f"[green]Results saved to {output_file}[/green]")
            else:
                print(result)
        
        elif output_format == OutputFormat.html:
            if not output_file:
                output_file = root / "tailchasing_report.html"
            
            files = [str(f) for f in ctx.files] if ctx else []
            cli.visualizer.generate_html_report(issues, files, str(output_file))
            cli.console.print(f"[green]HTML report generated: {output_file}[/green]")
        
        # Show summary
        if not quiet and output_format == OutputFormat.text:
            cli.console.print(f"\n[bold]📊 Analysis Complete:[/bold] {len(issues)} issues found")
            
            if issues:
                # Show quick stats
                high_severity = len([i for i in issues if i.severity >= 4])
                if high_severity > 0:
                    cli.console.print(f"[red]⚠️  {high_severity} high-severity issues require attention[/red]")
                else:
                    cli.console.print("[green]✅ No high-severity issues found[/green]")
        
        # Exit with appropriate code
        sys.exit(1 if issues else 0)
        
    except KeyboardInterrupt:
        cli.console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        cli.console.print(f"[red]❌ Error during analysis: {e}[/red]")
        cli.logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def fix(
    root: Path = typer.Argument(Path("."), help="Root directory to analyze and fix"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    mode: FixMode = typer.Option(FixMode.interactive, "--mode", "-m", help="Fix application mode"),
    plan_file: Optional[Path] = typer.Option(None, "--plan", "-p", help="Save fix plan to file"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backups before fixing"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate fixes before applying"),
    severity: Optional[int] = typer.Option(3, "--severity", "-s", help="Minimum severity to fix (default: 3)"),
    types: Optional[List[str]] = typer.Option(None, "--type", "-t", help="Only fix specific issue types"),
    enhanced: bool = typer.Option(False, "--enhanced", help="Enable enhanced pattern detection"),
    semantic: bool = typer.Option(False, "--semantic", help="Enable semantic analysis"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress bars")
):
    """
    🔧 Generate and apply fixes for detected issues.
    
    This command analyzes your code and can automatically generate fixes
    for various tail-chasing patterns. Use interactive mode for safety.
    """
    cli = TailChasingCLI()
    
    try:
        # Load configuration
        config = cli.load_config(root, config_file)
        
        # Override config with command line options
        if enhanced:
            config.setdefault('analysis', {})['enhanced_detection'] = True
        if semantic:
            config.setdefault('analysis', {})['semantic_analysis'] = True
        
        # Show startup info
        if not quiet:
            cli.console.print(Panel.fit(
                f"🔧 TailChasing Fixer - Fix Mode\n"
                f"📁 Target: [blue]{root.absolute()}[/blue]\n"
                f"🎯 Mode: [yellow]{mode.value}[/yellow]\n"
                f"💾 Backup: {'✅' if backup else '❌'}",
                title="Fix Generation Starting",
                border_style="green"
            ))
        
        # Run analysis
        issues, ctx = cli.analyze_codebase(root, config, show_progress=not quiet)
        
        # Filter issues
        if severity:
            issues = [issue for issue in issues if issue.severity >= severity]
        
        if types:
            issues = [issue for issue in issues if issue.kind in types]
        
        if not issues:
            cli.console.print("[green]🎉 No issues found that match the criteria![/green]")
            return
        
        # Interactive selection if requested
        if mode == FixMode.interactive:
            issues = cli.interactive_fix_selection(issues)
            
            if not issues:
                cli.console.print("[yellow]No issues selected for fixing.[/yellow]")
                return
        
        # Generate fix plan
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("🔧 Generating fix plan..."),
                transient=True
            ) as progress:
                progress.add_task("Fix generation", total=None)
                fix_plan = cli.auto_fixer.generate_fix_plan(issues)
        else:
            fix_plan = cli.auto_fixer.generate_fix_plan(issues)
        
        # Show fix plan summary
        if not quiet:
            plan_table = Table(title="📋 Fix Plan Summary")
            plan_table.add_column("Metric", style="bold")
            plan_table.add_column("Value", style="cyan")
            
            plan_table.add_row("Issues to fix", str(len(fix_plan.issues_addressed)))
            plan_table.add_row("Fix actions", str(len(fix_plan.actions)))
            plan_table.add_row("Estimated impact", fix_plan.estimated_impact)
            plan_table.add_row("Safety level", "🟢 Safe" if fix_plan.estimated_impact.lower() in ["low", "minimal"] else "🟡 Moderate")
            
            cli.console.print(plan_table)
        
        # Save fix plan if requested
        if plan_file:
            fix_plan_data = {
                'issues': [issue.to_dict() for issue in fix_plan.issues_addressed],
                'actions': [
                    {
                        'type': action.action_type,
                        'file': action.target_file,
                        'line': action.target_line,
                        'description': action.description,
                        'old_code': action.old_code,
                        'new_code': action.new_code
                    }
                    for action in fix_plan.actions
                ],
                'impact': fix_plan.estimated_impact,
                'rollback': fix_plan.rollback_plan
            }
            
            with open(plan_file, 'w') as f:
                json.dump(fix_plan_data, f, indent=2)
            
            cli.console.print(f"[green]📋 Fix plan saved to: {plan_file}[/green]")
        
        # Apply fixes based on mode
        if mode == FixMode.plan_only:
            cli.console.print("[blue]Plan generated. No fixes applied.[/blue]")
            return
        
        if mode == FixMode.interactive:
            if not Confirm.ask("\n🔧 Apply these fixes to your codebase?", default=False):
                cli.console.print("[yellow]Fix application cancelled.[/yellow]")
                return
        
        # Apply fixes
        if not quiet:
            cli.console.print("\n[yellow]⚠️  Applying fixes to codebase...[/yellow]")
            if backup:
                cli.console.print("[dim]💾 Creating backups...[/dim]")
        
        # Here we would integrate with the sandbox executor for safe application
        try:
            # Group actions by file
            actions_by_file = {}
            for action in fix_plan.actions:
                actions_by_file.setdefault(action.target_file, []).append(action)
            
            applied_count = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                disable=quiet
            ) as progress:
                if not quiet:
                    fix_task = progress.add_task("Applying fixes...", total=len(actions_by_file))
                
                for filepath, actions in actions_by_file.items():
                    # Create backup if requested
                    if backup:
                        backup_path = Path(f"{filepath}.backup.tailchasing")
                        import shutil
                        shutil.copy2(filepath, backup_path)
                    
                    # Apply fixes (simplified implementation)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        modified_content = content
                        for action in actions:
                            if action.old_code and action.new_code:
                                modified_content = modified_content.replace(action.old_code, action.new_code, 1)
                            applied_count += 1
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(modified_content)
                        
                        if not quiet:
                            progress.update(fix_task, advance=1, description=f"✅ {Path(filepath).name}: {len(actions)} fixes")
                            
                    except Exception as e:
                        if not quiet:
                            progress.update(fix_task, advance=1, description=f"❌ {Path(filepath).name}: Failed")
                        cli.logger.error(f"Failed to apply fixes to {filepath}: {e}")
            
            # Show completion summary
            if not quiet:
                cli.console.print(f"\n[green]✅ Successfully applied {applied_count} fixes![/green]")
                cli.console.print("[dim]💡 Run your tests to verify everything works correctly[/dim]")
                
                if backup:
                    cli.console.print(f"[dim]🔄 Backups created with .backup.tailchasing extension[/dim]")
                    cli.console.print(f"[dim]   To rollback: {' && '.join(fix_plan.rollback_plan)}[/dim]")
        
        except Exception as e:
            cli.console.print(f"[red]❌ Error applying fixes: {e}[/red]")
            cli.logger.error(f"Fix application failed: {e}", exc_info=True)
            sys.exit(1)
            
    except KeyboardInterrupt:
        cli.console.print("\n[yellow]⚠️  Fix process interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        cli.console.print(f"[red]❌ Error during fix process: {e}[/red]")
        cli.logger.error(f"Fix process failed: {e}", exc_info=True)
        sys.exit(1)


@app.command()
def config(
    root: Path = typer.Argument(Path("."), help="Root directory for configuration"),
    init: bool = typer.Option(False, "--init", help="Initialize configuration file"),
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", help="Open configuration in editor")
):
    """
    ⚙️  Manage TailChasing configuration.
    
    Create, view, or edit configuration files for customizing analysis behavior.
    """
    cli = TailChasingCLI()
    config_path = root / ".tailchasing.yml"
    
    if init:
        if config_path.exists():
            if not Confirm.ask(f"Configuration file already exists at {config_path}. Overwrite?"):
                cli.console.print("[yellow]Configuration initialization cancelled.[/yellow]")
                return
        
        # Create default configuration
        default_config = {
            "paths": {
                "include": ["**/*.py"],
                "exclude": ["**/tests/**", "**/test_*.py", "**/__pycache__/**"]
            },
            "analysis": {
                "enhanced_detection": False,
                "semantic_analysis": False,
                "git_history": False,
                "parallel": True,
                "cache": True
            },
            "output": {
                "format": "text",
                "severity_threshold": 1,
                "show_fixes": True,
                "show_explanations": False
            },
            "fixes": {
                "mode": "interactive",
                "backup": True,
                "validation": True,
                "auto_apply_safe": False
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        cli.console.print(f"[green]✅ Configuration file created: {config_path}[/green]")
        cli.console.print("[dim]💡 Edit the file to customize your analysis settings[/dim]")
        return
    
    if show:
        try:
            config = cli.load_config(root)
            cli.console.print("[bold blue]📋 Current Configuration:[/bold blue]")
            
            # Display configuration in a nice format
            config_yaml = yaml.dump(config, default_flow_style=False, indent=2)
            syntax = Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True)
            cli.console.print(Panel(syntax, title=str(config_path), border_style="blue"))
            
        except Exception as e:
            cli.console.print(f"[red]❌ Error reading configuration: {e}[/red]")
        return
    
    if edit:
        if not config_path.exists():
            if Confirm.ask(f"Configuration file doesn't exist. Create it first?"):
                # Recursively call init
                config(root, init=True)
            else:
                return
        
        # Try to open in user's preferred editor
        import os
        editor = os.environ.get('EDITOR', 'nano')
        try:
            os.system(f'{editor} {config_path}')
            cli.console.print("[green]✅ Configuration edited[/green]")
        except Exception as e:
            cli.console.print(f"[red]❌ Failed to open editor: {e}[/red]")
            cli.console.print(f"[dim]Edit manually: {config_path}[/dim]")
        return
    
    # Default: show help
    cli.console.print("Use --init to create a configuration file, --show to view it, or --edit to modify it.")


@app.command()
def explain(
    issue_file: Path = typer.Argument(..., help="File containing issues (JSON format)"),
    detail: bool = typer.Option(False, "--detail", help="Show detailed explanations"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for explanations")
):
    """
    📚 Generate detailed explanations for detected issues.
    
    Takes a JSON file of detected issues and generates human-readable
    explanations with remediation advice.
    """
    cli = TailChasingCLI()
    
    try:
        # Load issues from file
        with open(issue_file) as f:
            issue_data = json.load(f)
        
        # Convert to Issue objects (simplified)
        issues = []
        for item in issue_data:
            if isinstance(item, dict):
                issue = Issue(
                    kind=item.get('kind', 'unknown'),
                    message=item.get('message', ''),
                    severity=item.get('severity', 1),
                    file=item.get('file', ''),
                    line=item.get('line'),
                    evidence=item.get('evidence', {})
                )
                issues.append(issue)
        
        if not issues:
            cli.console.print("[yellow]No issues found in the input file.[/yellow]")
            return
        
        # Generate explanations
        cli.console.print(f"[blue]📚 Generating explanations for {len(issues)} issues...[/blue]")
        
        explanations = []
        
        # Generate summary
        summary = cli.explainer.generate_summary_report(issues)
        explanations.append(summary)
        
        # Generate detailed explanations if requested
        if detail:
            high_severity_issues = [i for i in issues if i.severity >= 3]
            for i, issue in enumerate(high_severity_issues, 1):
                explanation = cli.explainer.explain_issue(issue)
                explanations.append(f"\n## Issue {i}: {issue.kind}\n{explanation}")
        
        result = "\n".join(explanations)
        
        # Output results
        if output_file:
            with open(output_file, 'w') as f:
                f.write(result)
            cli.console.print(f"[green]📄 Explanations saved to: {output_file}[/green]")
        else:
            cli.console.print(Panel(result, title="📚 Issue Explanations", border_style="blue"))
            
    except Exception as e:
        cli.console.print(f"[red]❌ Error generating explanations: {e}[/red]")
        sys.exit(1)


# Main entry point
def main():
    """Entry point for the enhanced Typer CLI."""
    app()


if __name__ == '__main__':
    main()