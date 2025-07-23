"""Command-line interface for tail-chasing detector."""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from . import __version__
from .config import Config
from .core.loader import collect_files, parse_files
from .core.symbols import SymbolTable
from .core.issues import IssueCollection
from .core.reporting import Reporter
from .core.scoring import RiskScorer
from .analyzers.base import AnalysisContext
from .plugins import load_analyzers


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tailchasing",
        description="Detect LLM-assisted tail-chasing anti-patterns in Python code"
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to analyze (default: current directory)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--fail-on",
        type=int,
        metavar="SCORE",
        help="Exit with error if risk score exceeds threshold"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--exclude",
        action="append",
        help="Additional paths to exclude (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--include",
        action="append",
        help="Paths to include (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--disable",
        action="append",
        help="Disable specific analyzers (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--show-suggestions",
        action="store_true",
        help="Show fix suggestions in terminal output"
    )
    
    parser.add_argument(
        "--generate-fixes",
        action="store_true",
        help="Generate an interactive fix script"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Find project root
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}", file=sys.stderr)
        sys.exit(1)
        
    # Load configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config.find_and_load(root_path)
        
    # Override config with CLI arguments
    if args.exclude:
        excludes = config.get("paths.exclude", [])
        excludes.extend(args.exclude)
        config.set("paths.exclude", excludes)
        
    if args.include:
        config.set("paths.include", args.include)
        
    if args.disable:
        config.set("disabled_analyzers", args.disable)
        
    if args.json:
        formats = config.get("report.formats", [])
        if "json" not in formats:
            formats.append("json")
        config.set("report.formats", formats)
        
    if args.html:
        formats = config.get("report.formats", [])
        if "html" not in formats:
            formats.append("html")
        config.set("report.formats", formats)
        
    if args.output:
        config.set("report.output_dir", str(args.output))
        
    # Collect files
    logging.info(f"Collecting Python files from {root_path}")
    files = collect_files(
        root_path,
        include=config.get("paths.include"),
        exclude=config.get("paths.exclude")
    )
    
    if not files:
        print("No Python files found to analyze", file=sys.stderr)
        sys.exit(0)
        
    logging.info(f"Found {len(files)} Python files")
    
    # Parse files
    logging.info("Parsing Python files")
    ast_index = parse_files(files)
    
    if not ast_index:
        print("No valid Python files could be parsed", file=sys.stderr)
        sys.exit(1)
        
    logging.info(f"Successfully parsed {len(ast_index)} files")
    
    # Build symbol table
    logging.info("Building symbol table")
    symbol_table = SymbolTable()
    source_cache = {}
    
    for file_path, tree in ast_index.items():
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            source_cache[file_path] = source.splitlines()
            symbol_table.ingest_file(file_path, tree, source)
        except Exception as e:
            logging.warning(f"Failed to process {file_path}: {e}")
            
    # Create analysis context
    ctx = AnalysisContext(
        config=config.to_dict(),
        root_dir=root_path,
        file_paths=files,
        ast_index=ast_index,
        symbol_table=symbol_table,
        source_cache=source_cache,
        cache={}  # Initialize empty cache for analyzers
    )
    
    # Load and run analyzers
    logging.info("Running analyzers")
    analyzers = load_analyzers(config.to_dict())
    
    issue_collection = IssueCollection()
    
    for analyzer in analyzers:
        logging.debug(f"Running {analyzer.name} analyzer")
        try:
            for issue in analyzer.run(ctx):
                if not ctx.should_ignore_issue(issue.kind):
                    issue_collection.add(issue)
        except Exception as e:
            logging.error(f"Analyzer {analyzer.name} failed: {e}")
            
    # Deduplicate issues
    issue_collection.deduplicate()
    
    # Generate reports
    reporter = Reporter(config.to_dict())
    scorer = RiskScorer(config.get("scoring_weights"))
    
    # Calculate scores
    module_scores, global_score = scorer.calculate_scores(issue_collection.issues)
    risk_level = scorer.get_risk_level(global_score, config.get("risk_thresholds", {}))
    
    # Print summary to console
    if not args.json or len(config.get("report.formats", [])) > 1:
        print(f"\nTail-Chasing Analysis Complete")
        print(f"{'=' * 40}")
        print(f"Total Issues: {len(issue_collection)}")
        print(f"Global Risk Score: {global_score} ({risk_level})")
        print(f"Affected Modules: {len(module_scores)}")
        
        # Show top issues
        if issue_collection.issues:
            print(f"\nTop Issues:")
            top_issues = sorted(issue_collection.issues, key=lambda i: i.severity, reverse=True)[:5]
            for issue in top_issues:
                location = f"{issue.file}:{issue.line}" if issue.file else "global"
                print(f"  [{issue.kind}] {location} - {issue.message}")
                
                # Show suggestions if requested
                if args.show_suggestions and issue.suggestions:
                    print("    Suggestions:")
                    for suggestion in issue.suggestions[:2]:
                        # Indent multi-line suggestions
                        for line in suggestion.split('\n'):
                            print(f"      {line}")
                    if len(issue.suggestions) > 2:
                        print(f"      ... and {len(issue.suggestions) - 2} more suggestions")
                
    # Generate full reports
    output_dir = Path(config.get("report.output_dir", "."))
    results = reporter.generate_reports(issue_collection.issues, output_dir)
    
    # Handle JSON output to stdout
    if args.json and "json" in results:
        print(results["json"])
        
    # Print file locations for saved reports
    for fmt, path in results.items():
        if path.endswith("_file"):
            print(f"\n{fmt.replace('_file', '').upper()} report saved to: {path}")
            
    # Generate fix script if requested
    if args.generate_fixes and issue_collection.issues:
        try:
            from .core.suggestions import InteractiveFixGenerator
            fix_generator = InteractiveFixGenerator()
            
            # Generate fix script
            fix_path = output_dir / "tailchasing_fixes.py"
            fix_script = fix_generator.generate_fix_script(issue_collection.issues, fix_path)
            fix_path.write_text(fix_script)
            fix_path.chmod(0o755)  # Make executable
            
            print(f"\nFix script generated: {fix_path}")
            print(f"Run it with: python {fix_path}")
            print("\nThe fix script will:")
            print("  - Show each issue interactively")
            print("  - Let you review and apply fixes")
            print("  - Track which fixes were applied")
            
            # Also generate a detailed suggestions file
            from .core.suggestions import FixSuggestionGenerator
            suggestion_gen = FixSuggestionGenerator()
            
            suggestions_path = output_dir / "tailchasing_suggestions.md"
            with open(suggestions_path, 'w') as f:
                f.write("# Tail-Chasing Fix Suggestions\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Group by issue type
                from collections import defaultdict
                by_type = defaultdict(list)
                for issue in issue_collection.issues:
                    by_type[issue.kind].append(issue)
                
                for issue_type, issues in sorted(by_type.items()):
                    f.write(f"\n## {issue_type.replace('_', ' ').title()} ({len(issues)} issues)\n\n")
                    
                    # Show first few examples
                    for issue in issues[:3]:
                        location = f"{issue.file}:{issue.line}" if issue.file else "global"
                        f.write(f"### {location}\n")
                        f.write(f"{issue.message}\n\n")
                        
                        # Get enhanced suggestions
                        enhanced_suggestions = suggestion_gen.generate_suggestions(issue)
                        for suggestion in enhanced_suggestions:
                            f.write(f"{suggestion}\n\n")
                        f.write("---\n\n")
                    
                    if len(issues) > 3:
                        f.write(f"*... and {len(issues) - 3} more {issue_type} issues*\n\n")
            
            print(f"\nDetailed suggestions saved to: {suggestions_path}")
            
        except ImportError:
            print("\nNote: Fix generation requires the enhanced suggestions module.")
        except Exception as e:
            logging.error(f"Failed to generate fix script: {e}")
            
    # Check fail threshold
    if args.fail_on is not None:
        if global_score >= args.fail_on:
            print(f"\nERROR: Risk score {global_score} exceeds threshold {args.fail_on}", file=sys.stderr)
            sys.exit(2)
    else:
        # Use config thresholds
        fail_threshold = config.get("risk_thresholds.fail")
        if fail_threshold and global_score >= fail_threshold:
            print(f"\nERROR: Risk score {global_score} exceeds configured fail threshold {fail_threshold}", file=sys.stderr)
            sys.exit(2)
            
    sys.exit(0)


if __name__ == "__main__":
    main()
