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

def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_section(title):
    """Print a section title."""
    print(f"\n📌 {title}")
    print("-" * 40)

def run_demo():
    """Run the tail-chasing detection demo."""
    print_header("🎯 Tail-Chasing Detector Demo")
    
    print("This demo shows how the detector identifies common")
    print("tail-chasing patterns in an e-commerce system.\n")
    
    # Change to demo directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    # Show the files we're analyzing
    print_section("Files to Analyze")
    for py_file in demo_dir.glob("*.py"):
        if py_file.name != "run_demo.py":
            print(f"  📄 {py_file.name}")
    
    # Run the analysis
    print_section("Running Analysis")
    print("Executing: tailchasing . --config .tailchasing.yml\n")
    
    try:
        # Run tailchasing command
        result = subprocess.run(
            [sys.executable, "-m", "tailchasing", ".", "--config", ".tailchasing.yml"],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print(result.stdout)
        
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
                print("\n🔄 Semantic Duplicates Found:")
                for issue in issues_by_type['semantic_duplicate_function']:
                    evidence = issue.get('evidence', {})
                    pair = evidence.get('pair', [])
                    if len(pair) >= 2:
                        print(f"  • {pair[0]['name']} ≈ {pair[1]['name']}")
                        print(f"    Z-score: {evidence.get('z_score', 0):.2f}")
                        print(f"    Files: {pair[0]['file']} & {pair[1]['file']}")
            
            # Show circular imports
            if 'circular_import' in issues_by_type:
                print("\n🔄 Circular Imports:")
                for issue in issues_by_type['circular_import']:
                    print(f"  • {issue['message']}")
            
            # Show phantom functions
            if 'phantom_function' in issues_by_type:
                print("\n👻 Phantom Functions:")
                for issue in issues_by_type['phantom_function']:
                    print(f"  • {issue['symbol']} in {issue['file']}")
            
            # Show missing symbols
            if 'missing_symbol' in issues_by_type:
                print("\n❓ Missing/Hallucinated Symbols:")
                for issue in issues_by_type['missing_symbol']:
                    print(f"  • {issue['symbol']} (referenced but not defined)")
            
            # Show risk score
            print_section("Risk Assessment")
            score = data.get('total_score', 0)
            print(f"Total Risk Score: {score}")
            
            if score > 25:
                print("⚠️  CRITICAL: Severe tail-chasing patterns detected!")
            elif score > 10:
                print("⚠️  WARNING: Significant tail-chasing patterns found")
            else:
                print("✅ GOOD: Minor issues only")
            
            # Show recommendations
            print_section("Recommendations")
            print("Based on the analysis, here are the key actions:")
            print("1. Consolidate the 3 order total calculation functions")
            print("2. Remove trivial wrapper functions")
            print("3. Fix circular imports between modules")
            print("4. Implement the phantom/placeholder functions")
            print("5. Remove references to non-existent modules")
            
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("\nMake sure tail-chasing-detector is installed:")
        print("  pip install -e ..")

if __name__ == "__main__":
    run_demo()