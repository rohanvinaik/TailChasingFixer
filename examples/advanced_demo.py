#!/usr/bin/env python3
"""
Demo script showcasing advanced tail-chasing detection features.

This demonstrates:
1. Hallucination cascade detection
2. Context window thrashing detection
3. Import anxiety pattern detection
4. Enhanced semantic analysis
5. Intelligent auto-fixing
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tailchasing.cli import main as tailchasing_main
from tailchasing.core.loader import collect_files, parse_files
from tailchasing.core.symbols import SymbolTable
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.plugins import load_analyzers
from tailchasing.fixers.advanced import IntelligentAutoFixer


def create_demo_files():
    """Create demo files that exhibit various tail-chasing patterns."""
    
    # Create temporary directory
    demo_dir = tempfile.mkdtemp(prefix="tailchasing_demo_")
    
    # File 1: Hallucination cascade example
    cascade_file = Path(demo_dir) / "order_system.py"
    cascade_file.write_text('''
"""Order processing system - exhibits hallucination cascade."""

# All these classes were created to fix import errors in a cascade
class OrderValidator:
    """Created because OrderProcessor expected it."""
    def __init__(self):
        self.rules = OrderValidationRules()  # This doesn't exist yet
    
    def validate(self, order):
        return self.rules.check(order)

class OrderValidationRules:
    """Created because OrderValidator needed it."""
    def __init__(self):
        self.rule_engine = RuleEngine()  # This also doesn't exist
    
    def check(self, order):
        return self.rule_engine.evaluate(order)

class RuleEngine:
    """Created because OrderValidationRules needed it."""
    def __init__(self):
        self.rules = []
    
    def evaluate(self, order):
        # Just a stub
        return True

class OrderProcessor:
    """The original class that started the cascade."""
    def __init__(self):
        self.validator = OrderValidator()
    
    def process(self, order):
        if self.validator.validate(order):
            return {"status": "processed"}
        return {"status": "failed"}
''')
    
    # File 2: Context window thrashing example
    thrash_file = Path(demo_dir) / "data_processor.py"
    thrash_file.write_text('''
"""Data processing with context window thrashing."""

def process_customer_data(customer_id, data):
    """Process customer data."""
    # Some processing logic
    validated = validate_data(data)
    if validated:
        return transform_data(validated)
    return None

def validate_data(data):
    """Validate input data."""
    if not data:
        return None
    if "id" not in data:
        return None
    return data

def transform_data(data):
    """Transform validated data."""
    return {
        "id": data.get("id"),
        "processed": True
    }

# ... 500+ lines of other code here ...
# Simulating a large file where context is lost
''' + '\n' * 500 + '''

# LLM forgot about process_customer_data and reimplemented it
def handle_customer_data(cust_id, input_data):
    """Handle customer data processing."""
    # Very similar to process_customer_data but slightly different
    checked = check_data_validity(input_data)
    if checked:
        return transform_customer_data(checked)
    return None

def check_data_validity(input_data):
    """Check if data is valid."""
    if not input_data:
        return None
    if "id" not in input_data:
        return None
    return input_data

def transform_customer_data(input_data):
    """Transform customer data."""
    return {
        "id": input_data.get("id"),
        "processed": True,
        "timestamp": None  # Slight difference
    }
''')
    
    # File 3: Import anxiety example
    anxiety_file = Path(demo_dir) / "utils.py"
    anxiety_file.write_text('''
"""Utilities module with import anxiety."""

# Import anxiety - importing everything "just in case"
from typing import (
    Dict, List, Optional, Union, Any, Tuple, Set, FrozenSet,
    Callable, Iterator, Iterable, Generator, TypeVar, Generic,
    Protocol, Final, Literal, ClassVar, NewType, TypedDict,
    overload, cast, get_type_hints, get_origin, get_args
)

# More defensive imports
from collections import (
    defaultdict, Counter, OrderedDict, deque, namedtuple,
    ChainMap, UserDict, UserList, UserString
)

# Even more...
from itertools import (
    chain, count, cycle, repeat, accumulate, combinations,
    combinations_with_replacement, compress, dropwhile,
    filterfalse, groupby, islice, permutations, product,
    starmap, takewhile, tee, zip_longest
)

# But we only use a few things
def count_items(items: List[Any]) -> Dict[str, int]:
    """Count items in a list."""
    return dict(Counter(items))

def get_pairs(items: List[Any]) -> List[Tuple[Any, Any]]:
    """Get pairs from a list."""
    return list(combinations(items, 2))
''')
    
    # File 4: Enhanced semantic duplicate example
    semantic_file = Path(demo_dir) / "analytics.py"
    semantic_file.write_text('''
"""Analytics module with semantic duplicates."""

def calculate_user_metrics(user_data, time_period):
    """Calculate metrics for a user."""
    if not user_data:
        return {"error": "No data"}
    
    try:
        total = 0
        count = 0
        
        for entry in user_data:
            if entry.get("timestamp") in time_period:
                total += entry.get("value", 0)
                count += 1
        
        if count == 0:
            return {"average": 0, "total": 0, "count": 0}
        
        return {
            "average": total / count,
            "total": total,
            "count": count
        }
    except Exception as e:
        return {"error": str(e)}

def compute_customer_statistics(customer_info, date_range):
    """Compute statistics for a customer."""
    if not customer_info:
        return {"error": "No info provided"}
    
    try:
        sum_value = 0
        num_items = 0
        
        for record in customer_info:
            if record.get("date") in date_range:
                sum_value += record.get("amount", 0)
                num_items += 1
        
        if num_items == 0:
            return {"mean": 0, "sum": 0, "items": 0}
        
        return {
            "mean": sum_value / num_items,
            "sum": sum_value,
            "items": num_items
        }
    except Exception as ex:
        return {"error": str(ex)}

# These are semantically identical despite different names
''')
    
    # Create config file
    config_file = Path(demo_dir) / ".tailchasing.yml"
    config_file.write_text('''
# Enable advanced analyzers
enable_advanced_analyzers: true

# Enable semantic analysis
semantic:
  enable: true
  hv_dim: 8192
  min_functions: 2

# Paths
paths:
  include:
    - "."
  exclude:
    - "__pycache__"
    - ".git"

# Scoring weights
scoring:
  weights:
    hallucination_cascade: 5
    context_window_thrashing: 4
    import_anxiety: 2
    enhanced_semantic_duplicate: 4
''')
    
    return demo_dir


def run_advanced_demo():
    """Run the advanced demo."""
    print("🚀 TailChasingFixer Advanced Features Demo")
    print("=" * 50)
    
    # Create demo files
    demo_dir = create_demo_files()
    print(f"\n📁 Created demo files in: {demo_dir}")
    
    try:
        # Load configuration
        config = {
            "enable_advanced_analyzers": True,
            "semantic": {
                "enable": True,
                "hv_dim": 8192,
                "min_functions": 2
            }
        }
        
        # Collect and parse files
        files = collect_files(Path(demo_dir))
        ast_index = parse_files(files)
        
        # Build symbol table
        symbol_table = SymbolTable()
        for filepath, tree in ast_index.items():
            symbol_table.ingest(filepath, tree, "")
        
        # Create analysis context
        ctx = AnalysisContext(
            config=config,
            files=files,
            ast_index=ast_index,
            symbol_table=symbol_table,
            cache={}
        )
        
        # Load analyzers including advanced ones
        analyzers = load_analyzers(config)
        
        print(f"\n🔍 Running {len(analyzers)} analyzers:")
        for analyzer in analyzers:
            print(f"  - {analyzer.name}")
        
        # Run analysis
        all_issues = []
        for analyzer in analyzers:
            issues = list(analyzer.run(ctx))
            if issues:
                print(f"\n📊 {analyzer.name} found {len(issues)} issues:")
                for issue in issues[:3]:  # Show first 3
                    print(f"  - [{issue.kind}] {issue.message}")
                if len(issues) > 3:
                    print(f"  ... and {len(issues) - 3} more")
            all_issues.extend(issues)
        
        # Test intelligent auto-fixer
        if all_issues:
            print("\n🔧 Testing Intelligent Auto-Fixer:")
            fixer = IntelligentAutoFixer()
            fix_plan = fixer.generate_fix_plan(all_issues)
            
            print(f"\n📋 Fix Plan Summary:")
            print(f"  - Total fixes: {len(fix_plan.fixes)}")
            print(f"  - Execution order: {', '.join(fix_plan.execution_order)}")
            print(f"  - Confidence: {fix_plan.confidence:.0%}")
            print(f"  - Risk level: {fix_plan.estimated_impact.get('risk_level', 'unknown')}")
            
            if fix_plan.fixes:
                print(f"\n🔨 Proposed Fixes:")
                for fix in fix_plan.fixes[:3]:
                    print(f"  - [{fix.type}] {fix.description}")
                    print(f"    Confidence: {fix.confidence:.0%}")
                if len(fix_plan.fixes) > 3:
                    print(f"  ... and {len(fix_plan.fixes) - 3} more")
        
        print("\n✅ Demo completed successfully!")
        
    finally:
        # Cleanup
        shutil.rmtree(demo_dir)
        print(f"\n🧹 Cleaned up demo files")


if __name__ == "__main__":
    run_advanced_demo()
