#!/usr/bin/env python3
"""
Benchmark script for Tail-Chasing Detector performance.

Tests semantic analysis speed and accuracy on synthetic codebases.
"""

import time
import random
import tempfile
from pathlib import Path
from typing import List, Tuple
import click


def generate_synthetic_functions(num_functions: int, 
                               num_duplicates: int,
                               complexity: str = 'medium') -> str:
    """Generate synthetic Python code with known patterns."""
    code_parts = []
    
    # Generate base functions
    base_functions = []
    for i in range(num_functions - num_duplicates):
        if complexity == 'simple':
            func = f'''
def function_{i}(x, y):
    """Function {i} documentation."""
    return x + y + {i}
'''
        elif complexity == 'medium':
            func = f'''
def process_{i}(data, config=None):
    """Process data for case {i}."""
    if not data:
        return []
    
    result = []
    for item in data:
        if isinstance(item, str):
            processed = item.strip().lower()
        else:
            processed = str(item)
        result.append(processed + "_{i}")
    
    return result
'''
        else:  # complex
            func = f'''
def analyze_complex_{i}(dataset, params=None, verbose=False):
    """Complex analysis function {i}."""
    if params is None:
        params = {{"threshold": 0.5, "iterations": 100}}
    
    results = {{"scores": [], "metadata": {{}}}}
    
    try:
        for idx, data in enumerate(dataset):
            if verbose:
                print(f"Processing item {{idx}}")
            
            # Simulate complex logic
            score = 0
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    score += value * params["threshold"]
                elif isinstance(value, str):
                    score += len(value) / 10
            
            results["scores"].append(score)
            
            if score > params["threshold"] * 10:
                results["metadata"][f"high_score_{{idx}}"] = True
    
    except Exception as e:
        if verbose:
            print(f"Error in analysis: {{e}}")
        results["error"] = str(e)
    
    return results
'''
        base_functions.append(func)
        code_parts.append(func)
    
    # Generate semantic duplicates
    for i in range(num_duplicates):
        # Pick a random base function to duplicate
        base_idx = random.randint(0, len(base_functions) - 1)
        base = base_functions[base_idx]
        
        # Create semantic duplicate with different names/structure
        if 'function_' in base:
            dup = base.replace('function_', 'calc_').replace('x + y', 'y + x')
        elif 'process_' in base:
            dup = base.replace('process_', 'handle_').replace('result = []', 'output = []').replace('result.append', 'output.append').replace('return result', 'return output')
        else:
            dup = base.replace('analyze_complex_', 'compute_advanced_').replace('results = ', 'analysis = ').replace('return results', 'return analysis')
        
        code_parts.append(dup)
    
    return '\n'.join(code_parts)


@click.command()
@click.option('--num-files', default=10, help='Number of files to generate')
@click.option('--functions-per-file', default=20, help='Functions per file')
@click.option('--duplicate-ratio', default=0.2, help='Ratio of semantic duplicates')
@click.option('--complexity', type=click.Choice(['simple', 'medium', 'complex']), default='medium')
@click.option('--output-dir', type=click.Path(), default=None)
def benchmark(num_files, functions_per_file, duplicate_ratio, complexity, output_dir):
    """Run performance benchmark on synthetic codebase."""
    
    click.echo("🏃 Tail-Chasing Detector Performance Benchmark")
    click.echo("=" * 50)
    
    # Create temporary directory if not specified
    if output_dir:
        test_dir = Path(output_dir)
        test_dir.mkdir(exist_ok=True)
        cleanup = False
    else:
        temp_dir = tempfile.TemporaryDirectory()
        test_dir = Path(temp_dir.name)
        cleanup = True
    
    click.echo(f"📁 Test directory: {test_dir}")
    
    # Generate synthetic codebase
    click.echo(f"\n🔨 Generating synthetic codebase...")
    click.echo(f"  - Files: {num_files}")
    click.echo(f"  - Functions per file: {functions_per_file}")
    click.echo(f"  - Total functions: {num_files * functions_per_file}")
    click.echo(f"  - Semantic duplicates: {int(num_files * functions_per_file * duplicate_ratio)}")
    click.echo(f"  - Complexity: {complexity}")
    
    start_time = time.time()
    
    for i in range(num_files):
        num_dups = int(functions_per_file * duplicate_ratio)
        code = generate_synthetic_functions(
            functions_per_file, 
            num_dups,
            complexity
        )
        
        file_path = test_dir / f"module_{i}.py"
        file_path.write_text(code)
    
    generation_time = time.time() - start_time
    click.echo(f"✅ Generated in {generation_time:.2f}s")
    
    # Create config for semantic analysis
    config_path = test_dir / ".tailchasing.yml"
    config_path.write_text("""
semantic:
  enable: true
  hv_dim: 8192
  min_functions: 10
  z_threshold: 2.0
  
paths:
  include: ["."]
  exclude: []
""")
    
    # Run analysis
    click.echo(f"\n🔍 Running tail-chasing analysis...")
    
    import subprocess
    start_time = time.time()
    
    result = subprocess.run(
        ["tailchasing", str(test_dir), "--json"],
        capture_output=True,
        text=True
    )
    
    analysis_time = time.time() - start_time
    
    if result.returncode == 0:
        import json
        try:
            results = json.loads(result.stdout)
            
            click.echo(f"✅ Analysis completed in {analysis_time:.2f}s")
            click.echo(f"\n📊 Results:")
            click.echo(f"  - Total issues: {len(results.get('issues', []))}")
            click.echo(f"  - Risk score: {results.get('total_score', 0)}")
            
            # Count issue types
            issue_counts = {}
            for issue in results.get('issues', []):
                issue_type = issue.get('kind', 'unknown')
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            
            click.echo(f"\n📈 Issue breakdown:")
            for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
                click.echo(f"  - {issue_type}: {count}")
            
            # Performance metrics
            total_functions = num_files * functions_per_file
            functions_per_second = total_functions / analysis_time
            
            click.echo(f"\n⚡ Performance metrics:")
            click.echo(f"  - Functions analyzed: {total_functions}")
            click.echo(f"  - Analysis time: {analysis_time:.2f}s")
            click.echo(f"  - Functions/second: {functions_per_second:.0f}")
            click.echo(f"  - Time per function: {analysis_time/total_functions*1000:.1f}ms")
            
            # Memory usage (if available)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                click.echo(f"  - Memory usage: {memory_mb:.1f}MB")
            except:
                pass
            
            # Accuracy check
            expected_dups = int(num_files * functions_per_file * duplicate_ratio)
            found_dups = issue_counts.get('semantic_duplicate_function', 0)
            accuracy = min(found_dups / expected_dups, 1.0) * 100 if expected_dups > 0 else 0
            
            click.echo(f"\n🎯 Accuracy:")
            click.echo(f"  - Expected semantic duplicates: {expected_dups}")
            click.echo(f"  - Found semantic duplicates: {found_dups}")
            click.echo(f"  - Detection rate: {accuracy:.1f}%")
            
        except json.JSONDecodeError:
            click.echo(f"❌ Failed to parse JSON output")
            click.echo(result.stdout)
    else:
        click.echo(f"❌ Analysis failed with code {result.returncode}")
        click.echo(result.stderr)
    
    # Cleanup
    if cleanup:
        temp_dir.cleanup()
    
    click.echo(f"\n✨ Benchmark complete!")


if __name__ == '__main__':
    benchmark()
