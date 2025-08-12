# TailChasingFixer Integration Testing Framework

This comprehensive testing framework validates the tail-chasing detection and fixing system across multiple dimensions: accuracy, performance, scalability, and reliability.

## Overview

The framework includes:

1. **Hypervector System Tests** (`test_catalytic_hv.py`) - Tests the catalytic hypervector implementation
2. **Pattern Detection Tests** (`test_pattern_detection.py`) - Validates detection accuracy across all pattern types  
3. **Auto-Fix Tests** (`test_auto_fix.py`) - Tests the automated fixing system
4. **Performance Benchmarks** (`benchmark.py`) - Comprehensive performance and scalability testing
5. **Test Fixtures** (`fixtures/`) - Curated examples of known tail-chasing patterns
6. **Synthetic Generator** (`synthetic_generator.py`) - Creates large synthetic codebases for stress testing

## Quick Start

### Run All Integration Tests
```bash
# Run the complete test suite
pytest tests/integration/ -v

# Run with performance markers
pytest tests/integration/ -v -m "not slow"

# Run only fast tests
pytest tests/integration/ -v -m "not performance"
```

### Run Individual Test Modules
```bash
# Test hypervector system
pytest tests/integration/test_catalytic_hv.py -v

# Test pattern detection accuracy
pytest tests/integration/test_pattern_detection.py -v

# Test auto-fix functionality
pytest tests/integration/test_auto_fix.py -v
```

### Run Performance Benchmarks
```bash
# Quick benchmark
python tests/integration/benchmark.py quick

# Full benchmark suite
python tests/integration/benchmark.py

# Generate synthetic test data
python tests/integration/synthetic_generator.py small
python tests/integration/synthetic_generator.py medium
python tests/integration/synthetic_generator.py large
```

## Test Modules

### 1. Catalytic Hypervector Tests (`test_catalytic_hv.py`)

Tests the advanced hypervector similarity system:

- **Deterministic encoding** - Ensures consistent hypervector generation
- **Similarity thresholds** - Validates similarity computation accuracy  
- **Performance benchmarks** - Tests encoding and search speed
- **Memory-mapped persistence** - Tests index storage and retrieval
- **LSH approximate search** - Validates locality-sensitive hashing

**Key Test Classes:**
- `TestHypervectorEncoder` - Core encoding functionality
- `TestCatalyticIndex` - Memory-mapped storage system
- `TestLSHIndex` - Locality-sensitive hashing
- `TestSimilarityPipeline` - Complete query pipeline
- `TestCatalyticAnalyzer` - Integration with analyzer framework

### 2. Pattern Detection Tests (`test_pattern_detection.py`)

Validates detection accuracy for all tail-chasing patterns:

- **Synthetic pattern generation** - Creates controlled test cases
- **False positive validation** - Ensures <5% false positive rate
- **Pattern type coverage** - Tests all supported pattern types
- **Real codebase testing** - Validates on actual code examples

**Supported Patterns:**
- Duplicate functions (structural and semantic)
- Phantom/placeholder functions  
- Context-window thrashing
- Hallucination cascades
- Circular import dependencies

**Key Test Classes:**
- `TestDuplicateDetection` - Structural duplicate function detection
- `TestPlaceholderDetection` - Phantom function identification
- `TestContextThrashing` - Context-window thrashing patterns
- `TestIntegratedPatternDetection` - Multi-pattern detection

### 3. Auto-Fix Tests (`test_auto_fix.py`)

Validates the automated fixing system:

- **Fix strategy testing** - Tests each fix approach individually
- **AST validity** - Ensures fixes maintain valid Python syntax
- **Rollback mechanisms** - Tests backup and recovery systems
- **Safety validation** - Prevents introduction of new issues

**Key Test Classes:**
- `TestBackupManager` - File backup and restore functionality
- `TestSafetyValidator` - Pre/post-fix validation
- `TestFixStrategies` - Individual fix strategy testing
- `TestIntelligentAutoFixer` - Complete auto-fix system
- `TestFixValidation` - Fix quality assurance

### 4. Performance Benchmarks (`benchmark.py`)

Comprehensive performance and scalability testing:

- **Detection speed vs codebase size** - Scalability analysis
- **Memory usage profiling** - Resource consumption tracking
- **Baseline comparison** - O(N²) vs O(N) performance
- **Performance reporting** - Detailed metrics and visualizations

**Benchmark Categories:**
- Hypervector encoding performance
- Catalytic indexing throughput  
- Similarity search latency
- Pattern detection speed
- Auto-fix performance
- Memory efficiency

### 5. Test Fixtures (`fixtures/`)

Curated examples of tail-chasing patterns:

- `duplicate_functions.py` - Various types of duplicate functions
- `phantom_functions.py` - Placeholder and unimplemented functions
- `context_thrashing.py` - Similar functions causing context issues
- `hallucination_cascade.py` - Interdependent fictional subsystems
- `circular_imports.py` - Circular import dependency examples

### 6. Synthetic Generator (`synthetic_generator.py`)

Generates large synthetic codebases with controlled pattern injection:

- **Configurable generation** - Control file count, function density, pattern rates
- **Realistic code structure** - Generates believable Python code
- **Pattern injection** - Controlled introduction of specific patterns
- **Scalability testing** - Creates codebases from small to very large

## Configuration and Customization

### Test Configuration

Configure test behavior through environment variables or pytest configuration:

```bash
# Skip slow tests
export SKIP_SLOW_TESTS=1

# Use specific test data directory
export TEST_DATA_DIR=/path/to/test/data

# Configure performance test thresholds
export PERFORMANCE_THRESHOLD_SECONDS=30
export MEMORY_THRESHOLD_MB=100
```

### Synthetic Data Generation

Customize synthetic codebase generation:

```python
from tests.integration.synthetic_generator import GenerationConfig, SyntheticCodebaseGenerator

config = GenerationConfig(
    num_files=50,
    functions_per_file=20,
    duplicate_function_rate=0.15,
    phantom_function_rate=0.20,
    # ... other parameters
)

generator = SyntheticCodebaseGenerator(config)
codebase = generator.generate_codebase()
```

### Custom Pattern Testing

Add new pattern types to the testing framework:

```python
# In test_pattern_detection.py
def test_custom_pattern_detection(self):
    """Test custom pattern detection."""
    # Create test code with custom pattern
    custom_code = '''
    # Your custom pattern code here
    '''
    
    files = self.tester.create_test_files([custom_code], "custom")
    ctx = self.tester.create_analysis_context(files)
    
    # Test with your custom analyzer
    analyzer = YourCustomAnalyzer()
    issues = list(analyzer.run(ctx))
    
    # Validate results
    assert len(issues) >= expected_count
```

## Performance Expectations

### Detection Speed Targets

| Codebase Size | Expected Time | Memory Usage |
|---------------|---------------|--------------|
| 100 functions | < 1 second | < 50 MB |
| 1,000 functions | < 5 seconds | < 100 MB |
| 10,000 functions | < 30 seconds | < 200 MB |
| 100,000 functions | < 300 seconds | < 500 MB |

### Accuracy Targets

| Pattern Type | Detection Rate | False Positive Rate |
|--------------|----------------|---------------------|
| Duplicate Functions | > 95% | < 5% |
| Phantom Functions | > 98% | < 2% |
| Context Thrashing | > 90% | < 8% |
| Hallucination Cascade | > 85% | < 10% |
| Circular Imports | > 99% | < 1% |

## Continuous Integration

### GitHub Actions Integration

Add to your `.github/workflows/test.yml`:

```yaml
- name: Run Integration Tests
  run: |
    pytest tests/integration/ -v --tb=short
    
- name: Run Performance Benchmarks
  run: |
    python tests/integration/benchmark.py quick
    
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: benchmark_results/
```

### Test Result Analysis

The framework generates detailed reports:

- `performance_report.json` - Detailed performance metrics
- `performance_summary.txt` - Human-readable summary
- `performance_plots.png` - Visualization charts
- `generation_summary.json` - Synthetic data statistics

## Troubleshooting

### Common Issues

1. **ImportError for optional dependencies**
   ```bash
   # Install optional dependencies
   pip install numpy matplotlib psutil
   ```

2. **Memory issues with large tests**
   ```bash
   # Run with memory limits
   pytest tests/integration/ --tb=short -x
   ```

3. **Slow performance tests**
   ```bash
   # Skip performance tests
   pytest tests/integration/ -m "not performance"
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Data Cleanup

Clean up test data:

```bash
# Remove temporary test files
find /tmp -name "synthetic_codebase_*" -type d -exec rm -rf {} +
find /tmp -name "tailchasing_backups*" -type d -exec rm -rf {} +
```

## Contributing

### Adding New Tests

1. Create test class following existing patterns
2. Use provided fixtures and utilities from `conftest.py`
3. Add appropriate pytest markers
4. Document expected behavior and performance

### Adding New Patterns

1. Add pattern to `synthetic_generator.py`
2. Create fixtures in `fixtures/`
3. Add detection tests in `test_pattern_detection.py`
4. Add auto-fix tests in `test_auto_fix.py`
5. Update benchmark suite if needed

### Performance Testing

1. Use `@pytest.mark.performance` for slow tests
2. Set realistic performance expectations
3. Include memory usage measurements
4. Generate comparison reports

## License

This testing framework is part of TailChasingFixer and follows the same license terms.