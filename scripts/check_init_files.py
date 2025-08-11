#!/usr/bin/env python3
"""
Check for missing __init__.py files in Python packages.

This script ensures all Python packages have __init__.py files for proper module discovery.
"""

import sys
from pathlib import Path
from typing import List, Set


def find_python_directories(root_path: Path) -> Set[Path]:
    """Find all directories containing Python files."""
    python_dirs = set()
    
    for py_file in root_path.rglob("*.py"):
        # Skip certain directories
        if any(part in str(py_file) for part in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
            continue
        
        # Add the directory containing the Python file
        python_dirs.add(py_file.parent)
    
    return python_dirs


def check_init_files(root_path: Path) -> List[Path]:
    """Check for missing __init__.py files and return list of missing ones."""
    python_dirs = find_python_directories(root_path)
    missing_init_files = []
    
    for py_dir in python_dirs:
        # Skip root directory and certain special directories
        if py_dir == root_path:
            continue
            
        # Skip non-package directories (e.g., scripts, tests at root level)
        if py_dir.parent == root_path and py_dir.name in ['scripts', 'docs', 'examples', 'demo', 'vscode-extension']:
            continue
            
        init_file = py_dir / "__init__.py"
        
        # Check if __init__.py exists
        if not init_file.exists():
            # Additional check: ensure this looks like a Python package
            # (has Python files that aren't just scripts)
            python_files = list(py_dir.glob("*.py"))
            
            # If directory has Python files and looks like a package, it needs __init__.py
            if python_files and not py_dir.name.startswith('.'):
                # Check if this is likely a package (not just standalone scripts)
                has_modules = any(
                    not f.name.startswith('test_') and 
                    not f.name.startswith('_') and
                    f.name not in ['setup.py', 'conftest.py']
                    for f in python_files
                )
                
                if has_modules:
                    missing_init_files.append(py_dir)
    
    return missing_init_files


def main():
    """Main entry point."""
    root_path = Path(__file__).parent.parent
    
    print("🔍 Checking for missing __init__.py files...")
    
    missing_files = check_init_files(root_path)
    
    if not missing_files:
        print("✅ All Python packages have __init__.py files")
        sys.exit(0)
    
    print(f"❌ Found {len(missing_files)} directories missing __init__.py files:")
    
    for missing_dir in sorted(missing_files):
        rel_path = missing_dir.relative_to(root_path)
        print(f"  - {rel_path}/")
        
        # Show what Python files are in that directory
        py_files = list(missing_dir.glob("*.py"))
        for py_file in sorted(py_files)[:3]:  # Show first 3 files
            print(f"    • {py_file.name}")
        if len(py_files) > 3:
            print(f"    • ... and {len(py_files) - 3} more files")
    
    print("\n💡 To fix this, create empty __init__.py files in the missing directories:")
    for missing_dir in sorted(missing_files):
        rel_path = missing_dir.relative_to(root_path)
        print(f"  touch {rel_path}/__init__.py")
    
    sys.exit(1)


if __name__ == "__main__":
    main()