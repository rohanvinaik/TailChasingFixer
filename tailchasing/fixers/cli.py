#!/usr/bin/env python3
"""
Command-line interface for the IntelligentAutoFixer.

Provides easy access to auto-fix functionality from the command line.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from ..core.issues import Issue
from .auto_fix_engine import auto_fix_issues


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_issues_from_file(file_path: str) -> List[Issue]:
    """Load issues from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        issues = []
        if isinstance(data, list):
            # List of issue dictionaries
            for issue_dict in data:
                issues.append(Issue.from_dict(issue_dict))
        elif isinstance(data, dict):
            # Single issue dictionary
            issues.append(Issue.from_dict(data))
        else:
            raise ValueError("Invalid issue file format")
        
        return issues
    
    except Exception as e:
        print(f"Error loading issues from {file_path}: {e}")
        sys.exit(1)


def save_report(report: dict, output_file: str):
    """Save fix report to file."""
    try:
        output_path = Path(output_file)
        
        if output_path.suffix.lower() == '.json':
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            # Save as human-readable text
            with open(output_path, 'w') as f:
                f.write(format_text_report(report))
        
        print(f"Report saved to {output_path}")
    
    except Exception as e:
        print(f"Error saving report to {output_file}: {e}")


def format_text_report(report: dict) -> str:
    """Format report as human-readable text."""
    lines = []
    
    # Header
    lines.append("=" * 60)
    lines.append("INTELLIGENT AUTO-FIXER REPORT")
    lines.append("=" * 60)
    
    # Summary
    summary = report['summary']
    lines.append(f"")
    lines.append(f"SUMMARY:")
    lines.append(f"  Total fixes: {summary['total_fixes']}")
    lines.append(f"  Successful: {summary['successful_fixes']}")
    lines.append(f"  Failed: {summary['failed_fixes']}")
    lines.append(f"  Skipped: {summary['skipped_fixes']}")
    lines.append(f"  Rolled back: {summary['rolled_back_fixes']}")
    lines.append(f"  Success rate: {summary['success_rate']:.1%}")
    lines.append(f"  Total time: {summary['total_execution_time']:.2f}s")
    lines.append(f"  Dry run mode: {summary['dry_run_mode']}")
    
    # Breakdown by type
    lines.append(f"")
    lines.append(f"FIXES BY TYPE:")
    for issue_type, stats in report['breakdown_by_type'].items():
        lines.append(f"  {issue_type}:")
        lines.append(f"    Total: {stats['total']}")
        lines.append(f"    Successful: {stats['successful']}")
        lines.append(f"    Failed: {stats['failed']}")
        lines.append(f"    Avg time: {stats['avg_time']:.2f}s")
    
    # Detailed results
    lines.append(f"")
    lines.append(f"DETAILED RESULTS:")
    for i, result in enumerate(report['detailed_results'], 1):
        status_symbol = {
            'completed': '✅',
            'failed': '❌',
            'skipped': '⏭️',
            'rolled_back': '🔄'
        }.get(result['status'], '❓')
        
        lines.append(f"  {i}. {status_symbol} {result['issue_kind']}")
        lines.append(f"     File: {result['issue_file']}:{result.get('issue_line', '?')}")
        if result['issue_symbol']:
            lines.append(f"     Symbol: {result['issue_symbol']}")
        lines.append(f"     Status: {result['status']}")
        lines.append(f"     Time: {result['execution_time']:.2f}s")
        
        if result['changes']:
            lines.append(f"     Changes:")
            for change in result['changes']:
                lines.append(f"       - {change}")
        
        if result['error']:
            lines.append(f"     Error: {result['error']}")
        
        lines.append("")
    
    # Performance metrics
    if 'plan_metrics' in report:
        metrics = report['plan_metrics']
        lines.append(f"PERFORMANCE METRICS:")
        lines.append(f"  Estimated time: {metrics['estimated_time']:.2f}s")
        lines.append(f"  Actual time: {metrics['actual_time']:.2f}s")
        lines.append(f"  Time accuracy: {(1-metrics['time_accuracy']):.1%}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="IntelligentAutoFixer - Automatically fix tail-chasing patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with issues from file
  python -m tailchasing.fixers.cli --issues issues.json --dry-run
  
  # Apply fixes with custom test command
  python -m tailchasing.fixers.cli --issues issues.json --test-cmd "pytest tests/"
  
  # Apply fixes and save detailed report
  python -m tailchasing.fixers.cli --issues issues.json --output report.txt
  
  # High-verbosity dry run
  python -m tailchasing.fixers.cli --issues issues.json --dry-run --verbose
        """
    )
    
    # Input options
    parser.add_argument(
        '--issues', '-i',
        required=True,
        help='JSON file containing issues to fix'
    )
    
    # Fix options
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Preview changes without applying them'
    )
    
    parser.add_argument(
        '--backup-dir', '-b',
        help='Directory to store backup files (default: temp directory)'
    )
    
    parser.add_argument(
        '--test-cmd', '-t',
        default='python -m py_compile',
        help='Command to run tests for validation (default: syntax check)'
    )
    
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=1,
        help='Maximum number of parallel fixes (default: 1)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        help='Save detailed report to file (.json or .txt)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress normal output (only errors)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Risk management
    parser.add_argument(
        '--max-risk',
        choices=['low', 'medium', 'high', 'critical'],
        default='high',
        help='Maximum risk level to apply automatically (default: high)'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip test validation (not recommended)'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if not args.quiet:
        setup_logging(args.verbose)
    
    # Load issues
    if not args.quiet:
        print(f"Loading issues from {args.issues}...")
    
    issues = load_issues_from_file(args.issues)
    
    if not args.quiet:
        print(f"Loaded {len(issues)} issues")
    
    # Configure auto-fixer
    test_command = None if args.skip_tests else args.test_cmd
    
    if not args.quiet:
        mode = "DRY RUN" if args.dry_run else "LIVE"
        print(f"Running auto-fixer in {mode} mode...")
        
        if args.dry_run:
            print("No actual changes will be made.")
        else:
            print("Changes will be applied to files.")
            if args.backup_dir:
                print(f"Backups will be stored in: {args.backup_dir}")
            else:
                print("Backups will be stored in temp directory.")
    
    # Execute fixes
    try:
        report = auto_fix_issues(
            issues=issues,
            dry_run=args.dry_run,
            backup_dir=args.backup_dir,
            test_command=test_command
        )
        
        # Display results
        if not args.quiet:
            summary = report['summary']
            
            print(f"\n📊 RESULTS:")
            print(f"   Total fixes: {summary['total_fixes']}")
            print(f"   Successful: {summary['successful_fixes']}")
            print(f"   Failed: {summary['failed_fixes']}")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            print(f"   Total time: {summary['total_execution_time']:.2f}s")
            
            if summary['failed_fixes'] > 0:
                print(f"\n❌ {summary['failed_fixes']} fixes failed")
                for result in report['detailed_results']:
                    if result['status'] in ['failed', 'rolled_back']:
                        print(f"   - {result['issue_kind']} in {result['issue_file']}")
                        if result['error']:
                            print(f"     Error: {result['error']}")
            
            if summary['successful_fixes'] > 0:
                print(f"\n✅ {summary['successful_fixes']} fixes applied successfully")
        
        # Save report if requested
        if args.output:
            save_report(report, args.output)
        
        # Exit with appropriate code
        if report['summary']['failed_fixes'] > 0:
            sys.exit(1)  # Some fixes failed
        else:
            sys.exit(0)  # All good
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    
    except Exception as e:
        print(f"Error during auto-fix execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()