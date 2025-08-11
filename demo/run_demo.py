#!/usr/bin/env python3
"""
Demo script for Tail-Chasing Detector

This script demonstrates how the tool detects various tail-chasing
patterns in a sample e-commerce codebase.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

# Add parent directory to path to import tailchasing
sys.path.insert(0, str(Path(__file__).parent.parent))

from tailchasing.utils.logging_setup import get_logger, log_operation

# Setup logger
logger = get_logger(__name__)

def print_header(text):
    """Print a formatted header."""
    logger.info(f"Header: {text}")
    sys.stdout.write("\n" + "="*60 + "\n")
    sys.stdout.write(f"  {text}\n")
    sys.stdout.write("="*60 + "\n\n")

def print_section(title):
    """Print a section title."""
    logger.info(f"Section: {title}")
    sys.stdout.write(f"\n📌 {title}\n")
    sys.stdout.write("-" * 40 + "\n")

def run_demo():
    """Run the tail-chasing detection demo."""
    logger.info("Starting tail-chasing detector demo")
    log_operation(logger, "demo_execution")
    print_header("🎯 Tail-Chasing Detector Demo")
    
    sys.stdout.write("This demo shows how the detector identifies common\n")
    sys.stdout.write("tail-chasing patterns in an e-commerce system.\n\n")
    
    # Change to demo directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    # Show the files we're analyzing
    print_section("Files to Analyze")
    files_to_analyze = []
    for py_file in demo_dir.glob("*.py"):
        if py_file.name != "run_demo.py":
            files_to_analyze.append(py_file.name)
            sys.stdout.write(f"  📄 {py_file.name}\n")
    logger.info(f"Analyzing {len(files_to_analyze)} files: {files_to_analyze}")
    
    # Run the analysis
    print_section("Running Analysis")
    logger.info("Executing tailchasing analysis")
    sys.stdout.write("Executing: tailchasing . --config .tailchasing.yml\n\n")
    
    try:
        # Run tailchasing command
        result = subprocess.run(
            [sys.executable, "-m", "tailchasing", ".", "--config", ".tailchasing.yml"],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            logger.debug(f"Analysis output: {len(result.stdout)} chars")
            sys.stdout.write(result.stdout)
        
        # Also run with JSON output for detailed analysis
        json_result = subprocess.run(
            [sys.executable, "-m", "tailchasing", ".", "--config", ".tailchasing.yml", "--json"],
            capture_output=True,
            text=True
        )
        
        if json_result.stdout:
            data = json.loads(json_result.stdout)
            
            # Show detailed findings
            print_section("Detailed Findings")
            
            # Group issues by type
            issues_by_type = {}
            for issue in data.get('issues', []):
                issue_type = issue['kind']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)
            
            # Show semantic duplicates
            if 'semantic_duplicate_function' in issues_by_type:
                logger.info(f"Found {len(issues_by_type['semantic_duplicate_function'])} semantic duplicates")
                sys.stdout.write("\n🔄 Semantic Duplicates Found:\n")
                for issue in issues_by_type['semantic_duplicate_function']:
                    evidence = issue.get('evidence', {})
                    pair = evidence.get('pair', [])
                    if len(pair) >= 2:
                        sys.stdout.write(f"  • {pair[0]['name']} ≈ {pair[1]['name']}\n")
                        sys.stdout.write(f"    Z-score: {evidence.get('z_score', 0):.2f}\n")
                        sys.stdout.write(f"    Files: {pair[0]['file']} & {pair[1]['file']}\n")
            
            # Show circular imports
            if 'circular_import' in issues_by_type:
                logger.info(f"Found {len(issues_by_type['circular_import'])} circular imports")
                sys.stdout.write("\n🔄 Circular Imports:\n")
                for issue in issues_by_type['circular_import']:
                    sys.stdout.write(f"  • {issue['message']}\n")
            
            # Show phantom functions
            if 'phantom_function' in issues_by_type:
                logger.info(f"Found {len(issues_by_type['phantom_function'])} phantom functions")
                sys.stdout.write("\n👻 Phantom Functions:\n")
                for issue in issues_by_type['phantom_function']:
                    sys.stdout.write(f"  • {issue['symbol']} in {issue['file']}\n")
            
            # Show missing symbols
            if 'missing_symbol' in issues_by_type:
                logger.info(f"Found {len(issues_by_type['missing_symbol'])} missing symbols")
                sys.stdout.write("\n❓ Missing/Hallucinated Symbols:\n")
                for issue in issues_by_type['missing_symbol']:
                    sys.stdout.write(f"  • {issue['symbol']} (referenced but not defined)\n")
            
            # Show risk score
            print_section("Risk Assessment")
            score = data.get('total_score', 0)
            logger.info(f"Total risk score: {score}")
            sys.stdout.write(f"Total Risk Score: {score}\n")
            
            if score > 25:
                logger.warning("CRITICAL: Severe tail-chasing patterns detected")
                sys.stdout.write("⚠️  CRITICAL: Severe tail-chasing patterns detected!\n")
            elif score > 10:
                logger.warning("WARNING: Significant tail-chasing patterns found")
                sys.stdout.write("⚠️  WARNING: Significant tail-chasing patterns found\n")
            else:
                logger.info("GOOD: Minor issues only")
                sys.stdout.write("✅ GOOD: Minor issues only\n")
            
            # Show recommendations
            print_section("Recommendations")
            logger.info("Generating recommendations based on analysis")
            sys.stdout.write("Based on the analysis, here are the key actions:\n")
            sys.stdout.write("1. Consolidate the 3 order total calculation functions\n")
            sys.stdout.write("2. Remove trivial wrapper functions\n")
            sys.stdout.write("3. Fix circular imports between modules\n")
            sys.stdout.write("4. Implement the phantom/placeholder functions\n")
            sys.stdout.write("5. Remove references to non-existent modules\n")
            
    except Exception as e:
        logger.error(f"Error running analysis: {e}", exc_info=True)
        sys.stderr.write(f"Error running analysis: {e}\n")
        sys.stderr.write("\nMake sure tail-chasing-detector is installed:\n")
        sys.stderr.write("  pip install -e ..\n")

if __name__ == "__main__":
    run_demo()