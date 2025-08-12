"""
Performance benchmarks for the tail-chasing detection and fixing system.

Measures detection speed vs codebase size, memory usage profiling,
comparison with baseline, and generates performance reports.
"""

import ast
import gc
import json
import sys
import tempfile
import shutil
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import multiprocessing

try:
    import psutil
except ImportError:
    psutil = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None

from tailchasing.catalytic.hv_encoder import HypervectorEncoder
from tailchasing.catalytic.catalytic_index import CatalyticIndex
from tailchasing.catalytic.similarity_pipeline import SimilarityPipeline
from tailchasing.catalytic.catalytic_analyzer import CatalyticDuplicateAnalyzer
from tailchasing.analyzers.duplicates import DuplicateFunctionAnalyzer
from tailchasing.analyzers.placeholders import PlaceholderAnalyzer
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.symbols import SymbolTable
from tailchasing.fixers.auto_fix_engine import IntelligentAutoFixer
from tailchasing.core.issues import Issue, IssueSeverity


@dataclass
class BenchmarkResult:
    """Result from a performance benchmark."""
    benchmark_name: str
    codebase_size: int  # Number of functions/lines
    execution_time: float  # Seconds
    memory_usage_mb: float  # Peak memory usage
    detection_rate: float  # Issues detected per second
    accuracy_metrics: Optional[Dict[str, float]] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MemoryProfiler:
    """Profile memory usage during benchmark execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = 0
        self.peak_memory = 0
        self.measurements = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring memory usage."""
        if psutil:
            self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        else:
            self.baseline_memory = 0
        self.peak_memory = self.baseline_memory
        self.measurements = []
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        return self.peak_memory - self.baseline_memory
    
    def _monitor_loop(self):
        """Monitor memory usage in background thread."""
        while self.monitoring:
            try:
                if psutil:
                    current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                    self.measurements.append(current_memory)
                    self.peak_memory = max(self.peak_memory, current_memory)
                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory statistics."""
        if not self.measurements:
            return {"peak": 0, "average": 0, "baseline": self.baseline_memory}
        
        adjusted_measurements = [m - self.baseline_memory for m in self.measurements]
        
        return {
            "peak": max(adjusted_measurements),
            "average": statistics.mean(adjusted_measurements),
            "baseline": self.baseline_memory,
            "final": self.measurements[-1] - self.baseline_memory if self.measurements else 0
        }


class SyntheticCodebaseGenerator:
    """Generate synthetic codebases of varying sizes for benchmarking."""
    
    def __init__(self):
        self.function_templates = [
            self._simple_function_template,
            self._complex_function_template,
            self._duplicate_function_template,
            self._phantom_function_template,
            self._nested_function_template
        ]
        
        # Initialize random state if numpy is available
        if np:
            self.rng = np.random.RandomState(42)
        else:
            import random
            random.seed(42)
            self.rng = random
    
    def generate_codebase(self, num_files: int, functions_per_file: int) -> List[str]:
        """Generate a synthetic codebase."""
        files = []
        
        for file_idx in range(num_files):
            file_content = self._generate_file(file_idx, functions_per_file)
            files.append(file_content)
        
        return files
    
    def _generate_file(self, file_idx: int, num_functions: int) -> str:
        """Generate a single file with multiple functions."""
        imports = [
            "import os",
            "import sys",
            "from typing import List, Dict, Optional",
            "import json",
            "from datetime import datetime"
        ]
        
        functions = []
        
        for func_idx in range(num_functions):
            if np:
                template = np.random.choice(self.function_templates)
            else:
                template = self.rng.choice(self.function_templates)
            func_code = template(file_idx, func_idx)
            functions.append(func_code)
        
        file_content = "\n".join(imports) + "\n\n" + "\n\n".join(functions)
        return file_content
    
    def _simple_function_template(self, file_idx: int, func_idx: int) -> str:
        """Simple function template."""
        return f'''
def function_{file_idx}_{func_idx}(x, y):
    """Simple function {file_idx}_{func_idx}."""
    result = x + y
    if result > 0:
        return result * 2
    return 0
        '''.strip()
    
    def _complex_function_template(self, file_idx: int, func_idx: int) -> str:
        """Complex function template."""
        return f'''
def process_data_{file_idx}_{func_idx}(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process data with complex logic {file_idx}_{func_idx}."""
    results = {{}}
    processed_count = 0
    
    for item in items:
        if not item or not isinstance(item, dict):
            continue
        
        key = item.get('key', f'default_{func_idx}')
        value = item.get('value', 0)
        
        if key not in results:
            results[key] = {{
                'total': 0,
                'count': 0,
                'average': 0.0,
                'items': []
            }}
        
        results[key]['total'] += value
        results[key]['count'] += 1
        results[key]['average'] = results[key]['total'] / results[key]['count']
        results[key]['items'].append(item)
        
        processed_count += 1
        
        if processed_count % 100 == 0:
            # Simulate some processing work
            pass
    
    return {{
        'results': results,
        'processed_count': processed_count,
        'timestamp': datetime.now().isoformat()
    }}
        '''.strip()
    
    def _duplicate_function_template(self, file_idx: int, func_idx: int) -> str:
        """Duplicate function template (creates similar functions)."""
        if func_idx % 2 == 0:
            return f'''
def calculate_total_{file_idx}_{func_idx}(items):
    """Calculate total value."""
    total = 0
    for item in items:
        if item.value > 0:
            total += item.value * item.quantity
    return total
            '''.strip()
        else:
            return f'''
def compute_sum_{file_idx}_{func_idx}(products):
    """Compute sum of products."""
    result = 0
    for product in products:
        if product.value > 0:
            result += product.value * product.quantity
    return result
            '''.strip()
    
    def _phantom_function_template(self, file_idx: int, func_idx: int) -> str:
        """Phantom function template."""
        phantom_types = [
            f'''
def placeholder_{file_idx}_{func_idx}():
    """Placeholder function."""
    pass
            ''',
            f'''
def not_implemented_{file_idx}_{func_idx}():
    """Not implemented function."""
    raise NotImplementedError("This function is not implemented yet")
            ''',
            f'''
def todo_function_{file_idx}_{func_idx}():
    """TODO function."""
    # TODO: Implement this function
    return None
            '''
        ]
        
        if np:
            return np.random.choice(phantom_types).strip()
        else:
            return self.rng.choice(phantom_types).strip()
    
    def _nested_function_template(self, file_idx: int, func_idx: int) -> str:
        """Nested function template."""
        return f'''
def outer_function_{file_idx}_{func_idx}(data):
    """Function with nested helper."""
    
    def inner_helper(item):
        """Inner helper function."""
        if isinstance(item, str):
            return item.upper()
        elif isinstance(item, (int, float)):
            return item * 2
        return str(item)
    
    def another_helper(items):
        """Another helper function."""
        return [inner_helper(item) for item in items if item is not None]
    
    if not data:
        return []
    
    if isinstance(data, list):
        return another_helper(data)
    else:
        return [inner_helper(data)]
        '''.strip()


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.generator = SyntheticCodebaseGenerator()
        self.profiler = MemoryProfiler()
        
        # Benchmark configurations
        self.size_variants = [
            (5, 10),    # 50 functions
            (10, 20),   # 200 functions  
            (25, 40),   # 1,000 functions
            (50, 50),   # 2,500 functions
            (100, 50),  # 5,000 functions
            (200, 50),  # 10,000 functions
        ]
    
    def run_all_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """Run the complete benchmark suite."""
        print("Starting comprehensive performance benchmarks...")
        
        results = {
            "hypervector_encoding": [],
            "catalytic_indexing": [],
            "similarity_search": [],
            "pattern_detection": [],
            "auto_fix_performance": [],
            "baseline_comparison": []
        }
        
        # Run benchmarks for each size variant
        for num_files, functions_per_file in self.size_variants:
            total_functions = num_files * functions_per_file
            print(f"\nBenchmarking with {num_files} files, {functions_per_file} functions/file ({total_functions} total)...")
            
            # Generate test codebase
            codebase = self._prepare_test_codebase(num_files, functions_per_file)
            
            # Run individual benchmarks
            results["hypervector_encoding"].append(
                self._benchmark_hypervector_encoding(codebase, total_functions)
            )
            
            results["catalytic_indexing"].append(
                self._benchmark_catalytic_indexing(codebase, total_functions)
            )
            
            results["similarity_search"].append(
                self._benchmark_similarity_search(codebase, total_functions)
            )
            
            results["pattern_detection"].append(
                self._benchmark_pattern_detection(codebase, total_functions)
            )
            
            results["auto_fix_performance"].append(
                self._benchmark_auto_fix_performance(codebase, total_functions)
            )
            
            results["baseline_comparison"].append(
                self._benchmark_baseline_comparison(codebase, total_functions)
            )
            
            # Clean up between runs
            gc.collect()
        
        # Generate reports
        self._generate_performance_report(results)
        
        return results
    
    def _prepare_test_codebase(self, num_files: int, functions_per_file: int) -> Tuple[Path, AnalysisContext]:
        """Prepare a test codebase for benchmarking."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Generate synthetic code
        files = self.generator.generate_codebase(num_files, functions_per_file)
        
        # Write files to disk
        file_paths = []
        ast_index = {}
        source_cache = {}
        
        for i, content in enumerate(files):
            file_path = temp_dir / f"test_file_{i}.py"
            with open(file_path, 'w') as f:
                f.write(content)
            
            file_paths.append(file_path)
            
            # Parse AST
            try:
                tree = ast.parse(content)
                ast_index[str(file_path)] = tree
                source_cache[str(file_path)] = content.split('\n')
            except SyntaxError:
                continue
        
        # Create symbol table
        symbol_table = SymbolTable()
        for file_path, tree in ast_index.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol_table.add_function(
                        name=node.name,
                        file_path=file_path,
                        line_number=node.lineno,
                        node=node
                    )
        
        # Create analysis context
        ctx = AnalysisContext(
            config={},
            root_dir=temp_dir,
            file_paths=file_paths,
            ast_index=ast_index,
            symbol_table=symbol_table,
            source_cache=source_cache,
            cache={}
        )
        
        return temp_dir, ctx
    
    def _benchmark_hypervector_encoding(self, codebase: Tuple[Path, AnalysisContext], num_functions: int) -> BenchmarkResult:
        """Benchmark hypervector encoding performance."""
        temp_dir, ctx = codebase
        
        encoder = HypervectorEncoder()
        
        # Collect functions to encode
        functions = []
        for tree in ctx.ast_index.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node)
        
        # Start benchmarking
        self.profiler.start_monitoring()
        start_time = time.time()
        
        # Encode all functions
        encodings = []
        for func in functions[:num_functions]:  # Limit to expected number
            hv = encoder.encode_ast(func)
            encodings.append(hv)
        
        end_time = time.time()
        memory_usage = self.profiler.stop_monitoring()
        
        execution_time = end_time - start_time
        detection_rate = len(encodings) / execution_time if execution_time > 0 else 0
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return BenchmarkResult(
            benchmark_name="hypervector_encoding",
            codebase_size=len(encodings),
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            detection_rate=detection_rate,
            additional_metrics={
                "encodings_per_second": detection_rate,
                "avg_encoding_time_ms": (execution_time * 1000) / max(len(encodings), 1)
            }
        )
    
    def _benchmark_catalytic_indexing(self, codebase: Tuple[Path, AnalysisContext], num_functions: int) -> BenchmarkResult:
        """Benchmark catalytic index performance."""
        temp_dir, ctx = codebase
        
        index_temp_dir = tempfile.mkdtemp()
        
        try:
            with CatalyticIndex(index_temp_dir, mode='w') as index:
                pipeline = SimilarityPipeline(index)
                
                # Start benchmarking
                self.profiler.start_monitoring()
                start_time = time.time()
                
                # Index all functions
                indexed_count = 0
                for file_path, tree in ctx.ast_index.items():
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            pipeline.update_index(
                                func_ast=node,
                                file_path=file_path,
                                function_name=node.name,
                                line_number=node.lineno
                            )
                            indexed_count += 1
                            
                            if indexed_count >= num_functions:
                                break
                    
                    if indexed_count >= num_functions:
                        break
                
                end_time = time.time()
                memory_usage = self.profiler.stop_monitoring()
                
                execution_time = end_time - start_time
                detection_rate = indexed_count / execution_time if execution_time > 0 else 0
                
                # Get index statistics
                stats = index.get_stats()
                
        finally:
            shutil.rmtree(temp_dir)
            shutil.rmtree(index_temp_dir)
        
        return BenchmarkResult(
            benchmark_name="catalytic_indexing",
            codebase_size=indexed_count,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            detection_rate=detection_rate,
            additional_metrics={
                "functions_per_second": detection_rate,
                "index_working_memory_mb": stats.get('working_memory_mb', 0),
                "avg_indexing_time_ms": (execution_time * 1000) / max(indexed_count, 1)
            }
        )
    
    def _benchmark_similarity_search(self, codebase: Tuple[Path, AnalysisContext], num_functions: int) -> BenchmarkResult:
        """Benchmark similarity search performance."""
        temp_dir, ctx = codebase
        
        index_temp_dir = tempfile.mkdtemp()
        
        try:
            # Build index first
            with CatalyticIndex(index_temp_dir, mode='w') as index:
                pipeline = SimilarityPipeline(index)
                
                functions = []
                for file_path, tree in ctx.ast_index.items():
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            pipeline.update_index(
                                func_ast=node,
                                file_path=file_path,
                                function_name=node.name,
                                line_number=node.lineno
                            )
                            functions.append(node)
                            
                            if len(functions) >= num_functions:
                                break
                    
                    if len(functions) >= num_functions:
                        break
                
                # Now benchmark queries
                query_functions = functions[:min(50, len(functions))]  # Query subset
                
                self.profiler.start_monitoring()
                start_time = time.time()
                
                # Perform similarity queries
                total_results = 0
                for query_func in query_functions:
                    results = pipeline.query_function(query_func, top_k=10)
                    total_results += len(results)
                
                end_time = time.time()
                memory_usage = self.profiler.stop_monitoring()
                
                execution_time = end_time - start_time
                detection_rate = len(query_functions) / execution_time if execution_time > 0 else 0
        
        finally:
            shutil.rmtree(temp_dir)
            shutil.rmtree(index_temp_dir)
        
        return BenchmarkResult(
            benchmark_name="similarity_search",
            codebase_size=len(functions),
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            detection_rate=detection_rate,
            additional_metrics={
                "queries_per_second": detection_rate,
                "avg_query_time_ms": (execution_time * 1000) / max(len(query_functions), 1),
                "avg_results_per_query": total_results / max(len(query_functions), 1)
            }
        )
    
    def _benchmark_pattern_detection(self, codebase: Tuple[Path, AnalysisContext], num_functions: int) -> BenchmarkResult:
        """Benchmark pattern detection across different analyzers."""
        temp_dir, ctx = codebase
        
        analyzers = [
            DuplicateFunctionAnalyzer(),
            PlaceholderAnalyzer(),
            # Add more analyzers as needed
        ]
        
        self.profiler.start_monitoring()
        start_time = time.time()
        
        # Run all analyzers
        total_issues = 0
        for analyzer in analyzers:
            issues = list(analyzer.run(ctx))
            total_issues += len(issues)
        
        end_time = time.time()
        memory_usage = self.profiler.stop_monitoring()
        
        execution_time = end_time - start_time
        detection_rate = total_issues / execution_time if execution_time > 0 else 0
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return BenchmarkResult(
            benchmark_name="pattern_detection",
            codebase_size=num_functions,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            detection_rate=detection_rate,
            additional_metrics={
                "issues_per_second": detection_rate,
                "total_issues_found": total_issues,
                "avg_detection_time_per_function_ms": (execution_time * 1000) / max(num_functions, 1)
            }
        )
    
    def _benchmark_auto_fix_performance(self, codebase: Tuple[Path, AnalysisContext], num_functions: int) -> BenchmarkResult:
        """Benchmark auto-fix performance."""
        temp_dir, ctx = codebase
        
        # Create some issues to fix
        issues = []
        for file_path, tree in ctx.ast_index.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Create phantom function issues for functions with 'pass' or 'NotImplementedError'
                    if ("pass" in ast.dump(node) or "NotImplementedError" in ast.dump(node) or 
                        "TODO" in ast.dump(node)):
                        issue = Issue(
                            kind="phantom_function",
                            message=f"Phantom function: {node.name}",
                            severity=IssueSeverity.WARNING.value,
                            file=file_path,
                            line=node.lineno,
                            symbol=node.name
                        )
                        issues.append(issue)
        
        # Limit issues for performance testing
        issues = issues[:min(20, len(issues))]
        
        if issues:
            fixer = IntelligentAutoFixer(dry_run=True)  # Use dry-run for benchmarking
            
            self.profiler.start_monitoring()
            start_time = time.time()
            
            # Create and execute fix plan
            plan = fixer.create_fix_plan(issues)
            results = fixer.execute_fix_plan(plan)
            
            end_time = time.time()
            memory_usage = self.profiler.stop_monitoring()
            
            execution_time = end_time - start_time
            detection_rate = len(results) / execution_time if execution_time > 0 else 0
            
            successful_fixes = sum(1 for r in results if r.success)
            success_rate = successful_fixes / len(results) if results else 0
        else:
            execution_time = 0.0
            memory_usage = 0.0
            detection_rate = 0.0
            success_rate = 0.0
            successful_fixes = 0
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return BenchmarkResult(
            benchmark_name="auto_fix_performance",
            codebase_size=len(issues),
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            detection_rate=detection_rate,
            accuracy_metrics={"success_rate": success_rate},
            additional_metrics={
                "fixes_per_second": detection_rate,
                "successful_fixes": successful_fixes,
                "avg_fix_time_ms": (execution_time * 1000) / max(len(issues), 1) if issues else 0
            }
        )
    
    def _benchmark_baseline_comparison(self, codebase: Tuple[Path, AnalysisContext], num_functions: int) -> BenchmarkResult:
        """Benchmark against baseline (non-catalytic) implementation."""
        temp_dir, ctx = codebase
        
        # Baseline: Traditional O(N²) duplicate detection
        self.profiler.start_monitoring()
        start_time = time.time()
        
        # Simple O(N²) duplicate detection for comparison
        functions = []
        for tree in ctx.ast_index.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node)
        
        functions = functions[:num_functions]
        
        # Baseline duplicate detection
        duplicates_found = 0
        for i, func1 in enumerate(functions):
            for func2 in functions[i+1:]:
                # Simple AST comparison
                if ast.dump(func1) == ast.dump(func2):
                    duplicates_found += 1
        
        end_time = time.time()
        memory_usage = self.profiler.stop_monitoring()
        
        execution_time = end_time - start_time
        detection_rate = duplicates_found / execution_time if execution_time > 0 else 0
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return BenchmarkResult(
            benchmark_name="baseline_comparison",
            codebase_size=num_functions,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            detection_rate=detection_rate,
            additional_metrics={
                "duplicates_found": duplicates_found,
                "comparisons_per_second": (num_functions * (num_functions - 1) // 2) / max(execution_time, 0.001)
            }
        )
    
    def _generate_performance_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate comprehensive performance report."""
        print("\nGenerating performance report...")
        
        # Create detailed report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_count": multiprocessing.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3) if psutil else "unknown",
                "python_version": str(sys.version_info[:3])
            },
            "benchmarks": {}
        }
        
        for benchmark_name, benchmark_results in results.items():
            report["benchmarks"][benchmark_name] = [r.to_dict() for r in benchmark_results]
        
        # Save detailed report
        report_file = self.output_dir / "performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate summary statistics
        self._generate_summary_report(results)
        
        # Generate visualizations
        self._generate_performance_plots(results)
        
        print(f"Performance report saved to: {report_file}")
    
    def _generate_summary_report(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate summary performance report."""
        summary_file = self.output_dir / "performance_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("TailChasingFixer Performance Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for benchmark_name, benchmark_results in results.items():
                f.write(f"{benchmark_name.upper()}\n")
                f.write("-" * 30 + "\n")
                
                if benchmark_results:
                    # Calculate statistics
                    execution_times = [r.execution_time for r in benchmark_results]
                    memory_usages = [r.memory_usage_mb for r in benchmark_results]
                    detection_rates = [r.detection_rate for r in benchmark_results]
                    codebase_sizes = [r.codebase_size for r in benchmark_results]
                    
                    f.write(f"Codebase sizes tested: {min(codebase_sizes)} - {max(codebase_sizes)} functions\n")
                    f.write(f"Execution time range: {min(execution_times):.3f} - {max(execution_times):.3f} seconds\n")
                    f.write(f"Average execution time: {statistics.mean(execution_times):.3f} seconds\n")
                    f.write(f"Memory usage range: {min(memory_usages):.1f} - {max(memory_usages):.1f} MB\n")
                    f.write(f"Average memory usage: {statistics.mean(memory_usages):.1f} MB\n")
                    f.write(f"Detection rate range: {min(detection_rates):.1f} - {max(detection_rates):.1f} items/sec\n")
                    f.write(f"Average detection rate: {statistics.mean(detection_rates):.1f} items/sec\n")
                
                f.write("\n")
        
        print(f"Summary report saved to: {summary_file}")
    
    def _generate_performance_plots(self, results: Dict[str, List[BenchmarkResult]]):
        """Generate performance visualization plots."""
        if not plt:
            print("Matplotlib not available - skipping plot generation")
            return
            
        try:
            # Create plots for each benchmark
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('TailChasingFixer Performance Benchmarks')
            
            # Plot 1: Execution time vs codebase size
            ax1 = axes[0, 0]
            for benchmark_name, benchmark_results in results.items():
                if benchmark_results:
                    sizes = [r.codebase_size for r in benchmark_results]
                    times = [r.execution_time for r in benchmark_results]
                    ax1.plot(sizes, times, marker='o', label=benchmark_name)
            
            ax1.set_xlabel('Codebase Size (functions)')
            ax1.set_ylabel('Execution Time (seconds)')
            ax1.set_title('Execution Time vs Codebase Size')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Memory usage vs codebase size
            ax2 = axes[0, 1]
            for benchmark_name, benchmark_results in results.items():
                if benchmark_results:
                    sizes = [r.codebase_size for r in benchmark_results]
                    memory = [r.memory_usage_mb for r in benchmark_results]
                    ax2.plot(sizes, memory, marker='s', label=benchmark_name)
            
            ax2.set_xlabel('Codebase Size (functions)')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Memory Usage vs Codebase Size')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Detection rate vs codebase size
            ax3 = axes[1, 0]
            for benchmark_name, benchmark_results in results.items():
                if benchmark_results:
                    sizes = [r.codebase_size for r in benchmark_results]
                    rates = [r.detection_rate for r in benchmark_results]
                    ax3.plot(sizes, rates, marker='^', label=benchmark_name)
            
            ax3.set_xlabel('Codebase Size (functions)')
            ax3.set_ylabel('Detection Rate (items/sec)')
            ax3.set_title('Detection Rate vs Codebase Size')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Scalability comparison
            ax4 = axes[1, 1]
            
            # Compare catalytic vs baseline
            if "catalytic_indexing" in results and "baseline_comparison" in results:
                catalytic = results["catalytic_indexing"]
                baseline = results["baseline_comparison"]
                
                if catalytic and baseline:
                    sizes = [r.codebase_size for r in catalytic]
                    catalytic_times = [r.execution_time for r in catalytic]
                    baseline_times = [r.execution_time for r in baseline]
                    
                    ax4.plot(sizes, catalytic_times, marker='o', label='Catalytic (O(N))')
                    ax4.plot(sizes, baseline_times, marker='s', label='Baseline (O(N²))')
                    
                    ax4.set_xlabel('Codebase Size (functions)')
                    ax4.set_ylabel('Execution Time (seconds)')
                    ax4.set_title('Scalability Comparison')
                    ax4.legend()
                    ax4.grid(True)
                    ax4.set_yscale('log')
            
            plt.tight_layout()
            
            plot_file = self.output_dir / "performance_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Performance plots saved to: {plot_file}")
            
        except ImportError:
            print("Matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"Error generating plots: {e}")


def run_quick_benchmark():
    """Run a quick benchmark for development testing."""
    print("Running quick performance benchmark...")
    
    suite = PerformanceBenchmarkSuite(Path("quick_benchmark_results"))
    
    # Override with smaller test sizes for quick run
    suite.size_variants = [
        (2, 5),   # 10 functions
        (5, 10),  # 50 functions
        (10, 15), # 150 functions
    ]
    
    results = suite.run_all_benchmarks()
    
    print("\nQuick benchmark completed!")
    print(f"Results saved to: {suite.output_dir}")
    
    return results


def run_full_benchmark():
    """Run the complete benchmark suite."""
    print("Running full performance benchmark suite...")
    
    suite = PerformanceBenchmarkSuite(Path("full_benchmark_results"))
    results = suite.run_all_benchmarks()
    
    print("\nFull benchmark completed!")
    print(f"Results saved to: {suite.output_dir}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_benchmark()
    else:
        run_full_benchmark()