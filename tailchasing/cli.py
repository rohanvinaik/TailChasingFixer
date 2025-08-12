"""Command-line interface for tail-chasing detector."""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from . import __version__
from .config import Config
from .utils.logging_setup import get_logger, log_operation
from .core.loader import collect_files, parse_files
from .core.symbols import SymbolTable
from .core.issues import IssueCollection
from .core.reporting import Reporter
from .core.scoring import RiskScorer
from .analyzers.base import AnalysisContext
from .analyzers.root_cause_clustering import RootCauseClusterer
from .plugins import load_analyzers


# Setup module logger
logger = get_logger(__name__)

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )


def main():
    """Main CLI entry point."""
    log_operation(logger, "cli_main")
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
        "--cluster-root-causes",
        action="store_true",
        help="Cluster issues by root cause using AST analysis"
    )
    
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.7,
        metavar="THRESHOLD",
        help="Similarity threshold for clustering (0.0-1.0, default: 0.7)"
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
        logger.error(f"Path does not exist: {root_path}")
        sys.stderr.write(f"Error: Path does not exist: {root_path}\n")
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
    logger.info(f"Collecting Python files from {root_path}")
    files = collect_files(
        root_path,
        include=config.get("paths.include"),
        exclude=config.get("paths.exclude")
    )
    
    if not files:
        logger.warning("No Python files found to analyze")
        sys.stderr.write("No Python files found to analyze\n")
        sys.exit(0)
        
    logger.info(f"Found {len(files)} Python files")
    
    # Parse files
    logger.info("Parsing Python files")
    ast_index = parse_files(files)
    
    if not ast_index:
        logger.error("No valid Python files could be parsed")
        sys.stderr.write("No valid Python files could be parsed\n")
        sys.exit(1)
        
    logger.info(f"Successfully parsed {len(ast_index)} files")
    
    # Build symbol table
    logger.info("Building symbol table")
    symbol_table = SymbolTable()
    source_cache = {}
    
    for file_path, tree in ast_index.items():
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            source_cache[file_path] = source.splitlines()
            symbol_table.ingest_file(file_path, tree, source)
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            
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
    logger.info("Running analyzers")
    analyzers = load_analyzers(config.to_dict())
    
    issue_collection = IssueCollection()
    
    for analyzer in analyzers:
        logger.info(f"Running {analyzer.name} analyzer")
        try:
            import time
            start_time = time.time()
            for issue in analyzer.run(ctx):
                if not ctx.should_ignore_issue(issue.kind):
                    issue_collection.add(issue)
            elapsed = time.time() - start_time
            logger.info(f"Analyzer {analyzer.name} completed in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Analyzer {analyzer.name} failed: {e}")
            
    # Deduplicate issues
    issue_collection.deduplicate()
    
    # Perform root cause clustering if requested
    if args.cluster_root_causes and issue_collection.issues:
        logger.info("Performing root cause clustering analysis")
        clusterer = RootCauseClusterer(
            similarity_threshold=args.cluster_threshold,
            min_cluster_size=2
        )
        clusters = clusterer.cluster(issue_collection.issues)
        
        if clusters:
            sys.stdout.write(f"\n{'=' * 60}\n")
            sys.stdout.write("ROOT CAUSE CLUSTERING ANALYSIS\n")
            sys.stdout.write(f"{'=' * 60}\n")
            sys.stdout.write(f"Found {len(clusters)} root cause clusters\n")
            sys.stdout.write(f"Clustered {sum(c.size for c in clusters)} of {len(issue_collection.issues)} issues\n\n")
            
            # Show top clusters
            sorted_clusters = sorted(clusters, key=lambda c: (c.severity, c.size), reverse=True)
            for cluster in sorted_clusters[:5]:
                sys.stdout.write(f"Cluster {cluster.cluster_id}: {cluster.size} issues (severity {cluster.severity})\n")
                sys.stdout.write(f"  Root Cause: {cluster.root_cause_guess}\n")
                sys.stdout.write(f"  Confidence: {cluster.confidence:.1%}\n")
                sys.stdout.write(f"  Fix Playbook: {cluster.fix_playbook_id}\n")
                sys.stdout.write(f"  Locations: {', '.join(f'{f}:{l}' for f, l in cluster.locations[:3])}")
                if len(cluster.locations) > 3:
                    sys.stdout.write(f" ... +{len(cluster.locations) - 3} more")
                sys.stdout.write("\n\n")
            
            if len(clusters) > 5:
                sys.stdout.write(f"... and {len(clusters) - 5} more clusters\n")
            
            # Generate detailed cluster report if output directory specified
            if args.output:
                cluster_report = clusterer.generate_report(clusters)
                cluster_path = args.output / "root_cause_clusters.txt"
                cluster_path.write_text(cluster_report)
                sys.stdout.write(f"\nDetailed cluster report saved to: {cluster_path}\n")
    
    # Generate reports
    reporter = Reporter(config.to_dict())
    scorer = RiskScorer(config.get("scoring_weights"))
    
    # Calculate scores
    module_scores, global_score = scorer.calculate_scores(issue_collection.issues)
    risk_level = scorer.get_risk_level(global_score, config.get("risk_thresholds", {}))
    
    # Print summary to console
    if not args.json or len(config.get("report.formats", [])) > 1:
        logger.info(f"Analysis complete: {len(issue_collection)} issues, risk score {global_score}")
        sys.stdout.write(f"\nTail-Chasing Analysis Complete\n")
        sys.stdout.write(f"{'=' * 40}\n")
        sys.stdout.write(f"Total Issues: {len(issue_collection)}\n")
        sys.stdout.write(f"Global Risk Score: {global_score} ({risk_level})\n")
        sys.stdout.write(f"Affected Modules: {len(module_scores)}\n")
        
        # Show top issues
        if issue_collection.issues:
            sys.stdout.write(f"\nTop Issues:\n")
            top_issues = sorted(issue_collection.issues, key=lambda i: i.severity, reverse=True)[:5]
            for issue in top_issues:
                location = f"{issue.file}:{issue.line}" if issue.file else "global"
                sys.stdout.write(f"  [{issue.kind}] {location} - {issue.message}\n")
                
                # Show suggestions if requested
                if args.show_suggestions and issue.suggestions:
                    sys.stdout.write("    Suggestions:\n")
                    for suggestion in issue.suggestions[:2]:
                        # Indent multi-line suggestions
                        for line in suggestion.split('\n'):
                            sys.stdout.write(f"      {line}\n")
                    if len(issue.suggestions) > 2:
                        sys.stdout.write(f"      ... and {len(issue.suggestions) - 2} more suggestions\n")
                
    # Generate full reports
    output_dir = Path(config.get("report.output_dir", "."))
    results = reporter.generate_reports(issue_collection.issues, output_dir)
    
    # Handle JSON output to stdout
    if args.json and "json" in results:
        sys.stdout.write(results["json"] + "\n")
        
    # Always show paths to generated reports in terminal (unless pure JSON mode)
    if not (args.json and len(config.get("report.formats", [])) == 1):
        sys.stdout.write("\nGenerated Reports:\n")
        sys.stdout.write("-" * 40 + "\n")
        
        # Show all generated report files
        report_files = []
        for fmt, content_or_path in results.items():
            if fmt.endswith("_file"):
                report_type = fmt.replace("_file", "").upper()
                report_files.append((report_type, content_or_path))
                logger.info(f"Generated {report_type} report: {content_or_path}")
                sys.stdout.write(f"{report_type} report: {content_or_path}\n")
        
        # If no files were saved, show inline report info
        if not report_files:
            if "text" in results:
                sys.stdout.write("Text report: (displayed above)\n")
            if "json" in results and not args.json:
                sys.stdout.write("JSON report: (use --json flag to output)\n")
            if "html" in results:
                sys.stdout.write("HTML report: (use --output to save to file)\n")
            
    # Always check if fix generation would be helpful (unless pure JSON mode)
    if not (args.json and len(config.get("report.formats", [])) == 1) and issue_collection.issues:
        # Calculate how many issues have fixable patterns
        fixable_types = {
            'semantic_duplicate_function', 'duplicate_function', 'phantom_function',
            'missing_symbol', 'circular_import', 'prototype_fragmentation',
            'hallucination_cascade', 'import_anxiety', 'wrapper_abstraction',
            'context_window_thrashing'
        }
        fixable_issues = [i for i in issue_collection.issues if i.kind in fixable_types]
        
        if fixable_issues:
            sys.stdout.write("\nFix Suggestions:\n")
            sys.stdout.write("-" * 40 + "\n")
            
            if args.generate_fixes:
                # Fix generation code (already exists below)
                pass
            else:
                logger.info(f"Found {len(fixable_issues)} fixable issues")
                sys.stdout.write(f"Found {len(fixable_issues)} fixable issues out of {len(issue_collection.issues)} total\n")
                sys.stdout.write("Run with --generate-fixes to create:\n")
                sys.stdout.write("  • Interactive fix script (tailchasing_fixes.py)\n")
                sys.stdout.write("  • Detailed suggestions file (tailchasing_suggestions.md)\n")
                sys.stdout.write("\nExample: tailchasing . --generate-fixes\n")
    
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
            
            # Show in the standard location for generated files
            if not args.generate_fixes:
                # This case is handled above in "Fix Suggestions" section
                pass
            else:
                logger.info(f"Generated interactive fix script: {fix_path}")
                sys.stdout.write(f"Interactive fix script: {fix_path}\n")
                sys.stdout.write(f"  Run with: python {fix_path}\n")
            
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
            
            logger.info(f"Generated detailed suggestions: {suggestions_path}")
            sys.stdout.write(f"Detailed suggestions: {suggestions_path}\n")
            
        except ImportError:
            logger.warning("Fix generation requires the enhanced suggestions module")
            sys.stdout.write("\nNote: Fix generation requires the enhanced suggestions module.\n")
        except Exception as e:
            logger.error(f"Failed to generate fix script: {e}")
            
    # Check fail threshold
    if args.fail_on is not None:
        if global_score >= args.fail_on:
            logger.error(f"Risk score {global_score} exceeds threshold {args.fail_on}")
            sys.stderr.write(f"\nERROR: Risk score {global_score} exceeds threshold {args.fail_on}\n")
            sys.exit(2)
    else:
        # Use config thresholds
        fail_threshold = config.get("risk_thresholds.fail")
        if fail_threshold and global_score >= fail_threshold:
            logger.error(f"Risk score {global_score} exceeds configured fail threshold {fail_threshold}")
            sys.stderr.write(f"\nERROR: Risk score {global_score} exceeds configured fail threshold {fail_threshold}\n")
            sys.exit(2)
            
    sys.exit(0)


if __name__ == "__main__":
    main()
