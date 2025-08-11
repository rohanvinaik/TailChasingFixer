# TailChasing Fixer Benchmark Suite

Comprehensive evaluation harness for measuring convergence performance, token usage, and fix quality across different tail-chasing scenarios and models.

## Features

### 1. **Benchmark Scenarios** 
Four complexity levels with expected convergence steps:

- **Simple Import Errors** (1-2 steps): Missing imports that should be quickly resolved
- **Circular Dependencies** (3-4 steps): Circular import patterns requiring careful refactoring
- **Semantic Duplicates** (2-3 steps): Semantically similar functions needing consolidation
- **Complex Refactoring** (5-8 steps): Multi-file refactoring with various anti-patterns

### 2. **Metrics Collection**
Comprehensive metrics tracking:

- **Convergence Metrics**:
  - Steps to solution
  - Success/failure tracking
  - Regression detection
  - Backtracking and retry counts
  - Efficiency scoring

- **Performance Metrics**:
  - Wall-clock time
  - Memory usage (peak and average)
  - CPU utilization
  - I/O operations
  - Resource efficiency scoring

- **Cost Metrics**:
  - Token usage (input/output)
  - Cost estimation per model
  - Wasted tokens on failed attempts
  - Cost efficiency scoring

### 3. **Multi-Model Comparison**
Compare performance across different LLMs:

- Parallel or sequential execution
- Model-specific pricing
- Ranking by various metrics
- Head-to-head comparisons

### 4. **Performance Tracking**
Track trends over time:

- SQLite database for historical data
- Regression detection
- Problem pattern identification
- Trend visualization

## Usage

### Running Single Model Benchmarks

```bash
# Run all scenarios for a single model
python benchmarks/run_benchmarks.py single --model gpt-4

# Run specific scenario
python benchmarks/run_benchmarks.py single --model claude-3 --scenario simple

# Customize execution parameters
python benchmarks/run_benchmarks.py single \
    --model gpt-3.5 \
    --scenario complex \
    --max-steps 30 \
    --timeout 600
```

### Running Model Comparisons

```bash
# Compare multiple models
python benchmarks/run_benchmarks.py compare \
    --compare-models gpt-4,gpt-3.5,claude-3,llama-70b

# Run comparison in parallel
python benchmarks/run_benchmarks.py compare \
    --compare-models gpt-4,claude-3 \
    --parallel

# Compare on specific scenarios
python benchmarks/run_benchmarks.py compare \
    --compare-models gpt-4,mistral \
    --scenario semantic
```

### Tracking Performance

```bash
# Generate performance report
python benchmarks/run_benchmarks.py track

# Generate report with visualizations
python benchmarks/run_benchmarks.py track --plot

# Custom output directory
python benchmarks/run_benchmarks.py track \
    --output-dir custom_results
```

## Integration with Real Models

The current implementation uses mock fix functions for demonstration. To integrate with real LLMs:

1. **Replace Mock Functions**: In `run_benchmarks.py`, replace `create_mock_fix_function` with actual LLM integrations:

```python
def create_gpt4_fix_function():
    from tailchasing.llm.adapters import OpenAIAdapter
    
    adapter = OpenAIAdapter(api_key=os.getenv("OPENAI_API_KEY"))
    
    def fix_function(temp_dir: Path, current_code: Dict[str, str]) -> Dict[str, Any]:
        # Use adapter to fix code
        result = adapter.fix_code(current_code)
        
        return {
            "success": result.success,
            "fixed_code": result.fixed_code,
            "tokens_used": result.tokens_used,
            "description": result.description,
            "files_changed": result.files_changed,
            "error": result.error
        }
    
    return fix_function
```

2. **Add Model Configurations**:

```python
runner.add_model(ModelConfig(
    name="gpt-4",
    fix_function=create_gpt4_fix_function(),
    api_key=os.getenv("OPENAI_API_KEY")
))
```

## Scenario Details

### Simple Import Scenario
- **Files**: 3 Python files with missing imports
- **Issues**: Missing standard library and third-party imports
- **Expected Fix**: Add appropriate import statements
- **Validation**: Check all required imports are present

### Circular Dependency Scenario
- **Files**: 4 model files with circular imports
- **Issues**: Direct circular imports between models
- **Expected Fix**: Use TYPE_CHECKING and protocols
- **Validation**: No runtime circular dependencies

### Semantic Duplicate Scenario
- **Files**: 3 modules with semantically similar functions
- **Issues**: Multiple functions doing the same thing
- **Expected Fix**: Consolidate to single implementation with aliases
- **Validation**: Single implementation with backward compatibility

### Complex Refactoring Scenario
- **Files**: API handlers, database queries, utilities
- **Issues**: Multiple patterns (SQL injection, bare excepts, duplication)
- **Expected Fix**: Comprehensive refactoring
- **Validation**: All issues resolved without regressions

## Metrics Interpretation

### Efficiency Score (0-100)
- **90-100**: Excellent - Converged within minimum expected steps
- **70-89**: Good - Converged within expected range
- **50-69**: Acceptable - Converged but took extra steps
- **0-49**: Poor - Failed or significantly exceeded expectations

### Cost Efficiency
- Considers total tokens used
- Penalizes wasted tokens on failed attempts
- Accounts for model-specific pricing

### Performance Score
- Based on execution time
- Considers memory and CPU usage
- Penalizes resource spikes

## Output Files

The benchmark suite generates several output files:

- `{model}_summary.json`: Summary for single model runs
- `comparison_report_{timestamp}.json`: Multi-model comparison
- `performance_report.json`: Historical performance analysis
- `performance_trends.png`: Visualization of trends
- `performance.db`: SQLite database with all historical data

## Problem Pattern Detection

The system automatically identifies:

- **Frequently Failing**: Scenarios with <50% success rate
- **Regression Prone**: Scenarios that often introduce regressions
- **Inefficient**: Scenarios taking >1.5x expected steps
- **Performance Degradation**: Sudden drops in success rate or efficiency

## Best Practices

1. **Baseline First**: Run benchmarks on your current implementation to establish baseline
2. **Regular Testing**: Run benchmarks after significant changes
3. **Track Trends**: Use performance tracking to identify degradation early
4. **Compare Models**: Test new models against established ones
5. **Monitor Costs**: Track token usage and costs across models

## Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
psutil>=5.8.0
```

## Troubleshooting

### No Data for Plotting
- Ensure benchmarks have been run at least once
- Check the database file exists in output directory

### High Memory Usage
- Reduce `max_steps` parameter
- Run scenarios individually instead of all at once

### Timeout Issues
- Increase `--timeout` parameter
- Check network connectivity for API-based models

## Contributing

To add new scenarios:

1. Create a class inheriting from `BenchmarkScenario`
2. Implement required methods:
   - `get_initial_code()`
   - `get_expected_solution()`
   - `validate_solution()`
3. Add to scenario list in `run_benchmarks.py`

## License

Same as TailChasing Fixer main project.