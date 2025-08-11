# TailChasing Fixer Tutorial

A step-by-step guide to detecting and fixing LLM-induced tail-chasing patterns in your codebase.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Your First Analysis](#your-first-analysis)
4. [Understanding the Results](#understanding-the-results)
5. [Fixing Issues](#fixing-issues)
6. [Advanced Analysis](#advanced-analysis)
7. [Integration with CI/CD](#integration-with-cicd)
8. [Best Practices](#best-practices)

## Introduction

### What is Tail-Chasing?

Tail-chasing is a software development anti-pattern that emerges when Large Language Models (LLMs) make repeated, superficial fixes without addressing root causes. Like a dog chasing its tail, the code goes in circles without making real progress.

### Common Patterns

1. **Phantom Functions**: Empty implementations that never get filled
2. **Semantic Duplicates**: Multiple functions doing the same thing
3. **Circular Dependencies**: Modules importing each other
4. **Hallucination Cascades**: Entire fictional subsystems created to satisfy errors

## Installation and Setup

### Step 1: Install TailChasing Fixer

```bash
# Basic installation
pip install tail-chasing-detector

# With all features
pip install tail-chasing-detector[all]
```

### Step 2: Verify Installation

```bash
tailchasing --version
# Output: TailChasing Fixer v0.2.0
```

### Step 3: Create Configuration (Optional)

Create a `.tailchasing.yml` file in your project root:

```yaml
# .tailchasing.yml
paths:
  include:
    - src
  exclude:
    - tests
    - __pycache__

analysis:
  enhanced_detection: true
  semantic_analysis: true

risk_thresholds:
  warn: 15
  fail: 30
```

## Your First Analysis

### Basic Analysis

Let's start with a simple Python file that has tail-chasing issues:

```python
# example.py
class DataProcessor:
    def process(self, data):
        """Process data."""
        pass  # TODO: implement
    
    def validate(self, data):
        """Validate data."""
        raise NotImplementedError()

def calculate_average(numbers):
    return sum(numbers) / len(numbers) if numbers else 0

def compute_mean(values):
    if not values:
        return 0
    return sum(values) / len(values)
```

Run the analysis:

```bash
tailchasing analyze example.py
```

### Expected Output

```
🔍 TailChasing Fixer Analysis
============================

📁 Analyzing: example.py
⚠️  Found 4 issues:

[PHANTOM_FUNCTION] example.py:3
  Empty implementation detected: DataProcessor.process
  Severity: 3/5
  Suggestion: Implement actual processing logic or remove if unused

[PHANTOM_FUNCTION] example.py:7
  NotImplementedError stub: DataProcessor.validate
  Severity: 3/5
  Suggestion: Implement validation logic

[SEMANTIC_DUPLICATE] example.py:10,14
  Semantic duplicate functions detected:
  - calculate_average (line 10)
  - compute_mean (line 14)
  Similarity: 92%
  Suggestion: Consolidate into single function

Summary:
  Total Issues: 4
  Risk Score: 12 (MEDIUM)
```

## Understanding the Results

### Issue Types

Each issue has several components:

1. **Type**: The kind of anti-pattern detected
2. **Location**: File and line number
3. **Severity**: Score from 1 (low) to 5 (critical)
4. **Suggestion**: Recommended fix

### Severity Levels

- **1-2**: Minor issues, style improvements
- **3**: Should be fixed, potential problems
- **4**: Important to fix, likely causing issues
- **5**: Critical, must fix immediately

### Risk Scores

The overall risk score helps prioritize:

- **0-10**: Low risk (OK)
- **11-20**: Medium risk (WARNING)
- **21+**: High risk (CRITICAL)

## Fixing Issues

### Interactive Fix Mode

The best way to fix issues is using interactive mode:

```bash
tailchasing fix example.py --interactive
```

This will:
1. Show each issue with context
2. Propose a fix
3. Ask for confirmation
4. Apply the fix

### Example Fix Session

```
🔧 Fix Mode - Interactive
========================

Issue 1/4: [PHANTOM_FUNCTION] DataProcessor.process

Current code:
    def process(self, data):
        """Process data."""
        pass  # TODO: implement

Proposed fix:
    def process(self, data):
        """Process data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data dictionary
        """
        if not data:
            return {}
        
        processed = {
            'validated': self.validate(data),
            'transformed': self._transform(data),
            'timestamp': datetime.now().isoformat()
        }
        return processed

Apply this fix? (y/n/skip/quit): y
✅ Fix applied

Issue 2/4: [SEMANTIC_DUPLICATE] calculate_average, compute_mean
...
```

### Automatic Fixing

For CI/CD or batch processing:

```bash
# Fix all issues with severity >= 3
tailchasing fix . --mode automatic --severity 3

# Fix only specific issue types
tailchasing fix . --type phantom_function,semantic_duplicate

# Create fix plan without applying
tailchasing fix . --plan-only --output fixes.json
```

## Advanced Analysis

### Semantic Analysis

Enable deep semantic analysis to find subtle duplicates:

```bash
tailchasing analyze . --semantic --vector-dim 16384
```

This uses hypervector computing to find semantically similar code even when implementations differ.

### Git History Analysis

Analyze how patterns evolved over time:

```bash
tailchasing analyze . --git-history --days 30
```

### Enhanced Pattern Detection

Enable all advanced detectors:

```bash
tailchasing analyze . --enhanced
```

This enables:
- Hallucination cascade detection
- Context window thrashing detection
- Import anxiety pattern detection

### Custom Configuration

Create detailed configuration for your project:

```yaml
# .tailchasing.yml
semantic:
  vector_dim: 16384
  similarity_threshold: 0.85
  channel_weights:
    data_flow: 1.5
    control_flow: 1.3
    error_handling: 1.2

enhanced_detection:
  hallucination_cascade:
    min_cascade_size: 3
    external_ref_threshold: 0.2
  
  context_thrashing:
    min_distance: 500
    similarity_threshold: 0.6
```

## Integration with CI/CD

### GitHub Actions

```yaml
# .github/workflows/tailchasing.yml
name: TailChasing Analysis

on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install TailChasing
        run: pip install tail-chasing-detector
      
      - name: Run Analysis
        run: |
          tailchasing analyze . \
            --enhanced \
            --format json \
            --output results.json
      
      - name: Check Results
        run: |
          python -c "
          import json
          with open('results.json') as f:
              data = json.load(f)
          if data['summary']['risk_level'] == 'CRITICAL':
              exit(1)
          "
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: tailchasing-report
          path: results.json
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tailchasing
        name: TailChasing Analysis
        entry: tailchasing analyze
        language: system
        files: '\.py$'
        args: ['--severity', '3']
```

### GitLab CI

```yaml
# .gitlab-ci.yml
tailchasing:
  stage: test
  script:
    - pip install tail-chasing-detector
    - tailchasing analyze . --format json --output report.json
  artifacts:
    reports:
      codequality: report.json
```

## Best Practices

### 1. Start with High-Severity Issues

Focus on severity 4-5 issues first:

```bash
tailchasing analyze . --severity 4
```

### 2. Use Semantic Analysis for Refactoring

Before refactoring, check for semantic duplicates:

```bash
tailchasing analyze . --semantic --type semantic_duplicate
```

### 3. Regular Monitoring

Set up weekly analysis to catch new patterns:

```bash
# In cron or CI
tailchasing analyze . --enhanced --output weekly_report.html
```

### 4. Configure Ignores Carefully

Don't ignore issues without understanding them:

```yaml
# .tailchasing.yml
ignore_issue_types:
  - missing_symbol  # Only if using dynamic imports

# Better: Allow specific placeholders
placeholders:
  allow:
    - AbstractBase.abstract_method
    - Protocol.method_signature
```

### 5. Fix in Logical Groups

Group related fixes together:

```bash
# Fix all semantic duplicates in a module
tailchasing fix src/models --type semantic_duplicate

# Fix all phantoms in a class
tailchasing fix src/processor.py --class DataProcessor
```

### 6. Validate Fixes

Always validate fixes before committing:

```bash
# Generate fix plan
tailchasing fix . --plan-only --output plan.json

# Review plan
cat plan.json | jq '.fixes[] | {file, line, description}'

# Apply with validation
tailchasing fix . --plan plan.json --validate
```

### 7. Track Progress

Monitor improvement over time:

```bash
# Run benchmarks
python benchmarks/run_benchmarks.py track --plot

# Generate trend report
tailchasing analyze . --output report.json
python -c "
import json
with open('report.json') as f:
    data = json.load(f)
print(f'Issues: {data['summary']['total_issues']}')
print(f'Risk: {data['summary']['risk_level']}')
"
```

## Common Scenarios

### Scenario 1: New Codebase Assessment

```bash
# Full analysis
tailchasing analyze . --enhanced --semantic --git-history

# Generate comprehensive report
tailchasing analyze . --format html --output assessment.html

# Create fix plan
tailchasing fix . --plan-only --output fix_plan.json
```

### Scenario 2: CI/CD Integration

```bash
# Fail build on critical issues
tailchasing analyze . --fail-on critical

# Generate report for artifacts
tailchasing analyze . --format json --output tailchasing.json
```

### Scenario 3: Refactoring Support

```bash
# Find all duplicates before refactoring
tailchasing analyze . --type semantic_duplicate --threshold 0.7

# Generate consolidation plan
tailchasing fix . --type semantic_duplicate --consolidate
```

### Scenario 4: LLM Code Review

```bash
# Check code generated by LLM
tailchasing analyze generated_code/ --enhanced

# Fix common LLM issues
tailchasing fix generated_code/ --type phantom_function,hallucination
```

## Troubleshooting

### Issue: Analysis is slow

```bash
# Use caching
tailchasing analyze . --cache

# Limit scope
tailchasing analyze src/ --exclude tests

# Reduce vector dimensions
tailchasing analyze . --semantic --vector-dim 8192
```

### Issue: Too many false positives

```yaml
# Adjust thresholds in .tailchasing.yml
semantic:
  similarity_threshold: 0.9  # Increase threshold

analysis:
  min_severity: 3  # Ignore minor issues
```

### Issue: Fixes break tests

```bash
# Validate fixes before applying
tailchasing fix . --validate --test-command "pytest"

# Use conservative mode
tailchasing fix . --conservative
```

## Next Steps

1. **Explore Advanced Features**: Try semantic analysis and enhanced detection
2. **Customize Configuration**: Tailor settings to your project
3. **Integrate with CI/CD**: Add to your build pipeline
4. **Contribute**: Help improve TailChasing Fixer on GitHub

## Resources

- [GitHub Repository](https://github.com/rohanvinaik/TailChasingFixer)
- [API Documentation](https://tailchasing.dev/api)
- [Issue Tracker](https://github.com/rohanvinaik/TailChasingFixer/issues)
- [Community Discussions](https://github.com/rohanvinaik/TailChasingFixer/discussions)

Happy tail-chasing hunting! 🎯