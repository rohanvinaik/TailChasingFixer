# TailChasing Polymer Physics Module

Chromatin-inspired performance analysis using polymer physics concepts.

## Overview

The TailChasing polymer module applies biological concepts from chromatin organization to software performance analysis:

- **Hi-C Contact Matrices**: Visualize interaction patterns between code regions
- **TAD Boundaries**: Identify functional modules and their interactions
- **Polymer Distance Metrics**: Quantify relationships using physics models
- **Thrash Risk Analysis**: Identify and predict performance bottlenecks

## Quick Start

```python
from tailchasing.polymer import HiCHeatmapGenerator, PolymerMetricsReport
from tailchasing.polymer.calibrate import CalibrationTool

# Generate Hi-C style heatmap
generator = HiCHeatmapGenerator()
heatmap = generator.generate_contact_heatmap(contact_matrix)

# Calculate polymer metrics
reporter = PolymerMetricsReport()
metrics = reporter.calculate_polymer_distances(tads, interactions)

# Calibrate parameters
tool = CalibrationTool()
result = tool.fit_parameters(events, codebase)
```

## CLI Usage

The polymer module integrates with the main tailchasing CLI:

```bash
# Show current configuration
tailchasing polymer config show

# Initialize configuration
tailchasing polymer config init

# Run calibration demo
tailchasing polymer calibrate demo

# Validate configuration
tailchasing polymer config validate

# Set configuration parameters
tailchasing polymer config set alpha 1.5
tailchasing polymer config set weight_tok 2.0

# Run grid search calibration
tailchasing polymer calibrate grid --grid-points 5

# Analyze codebase with polymer model
tailchasing polymer analyze /path/to/code --output results.json
```

## Configuration

Configuration is stored in `.tailchasing_polymer.yml`:

```yaml
# Contact probability parameters
alpha: 1.2          # Contact decay exponent
epsilon: 1.0e-06    # Regularization parameter
kappa: 1.0          # Contact strength constant
tad_penalty: 0.7    # Penalty for cross-TAD contacts

# Distance weights
weights:
  tok: 1.0  # Token-level weight
  ast: 2.0  # AST-level weight
  mod: 3.0  # Module-level weight
  git: 4.0  # Git-level weight

# Analysis parameters  
min_tad_size: 10
max_tad_size: 1000
loop_anchor_threshold: 0.5
contact_matrix_resolution: 100

# Performance thresholds
thrash_risk_threshold: 0.7
optimization_threshold: 0.3

# Visualization settings
heatmap_colormap: viridis
show_tad_boundaries: true
show_loop_anchors: true
```

### Environment Variables

You can override configuration using environment variables:

```bash
export TAILCHASING_POLYMER_ALPHA=1.5
export TAILCHASING_POLYMER_EPSILON=1e-7
export TAILCHASING_POLYMER_KAPPA=2.0
export TAILCHASING_POLYMER_WEIGHT_TOK=1.5
```

## Features

### Hi-C Contact Matrices

Visualize code interaction patterns using Hi-C style heatmaps:

```python
from tailchasing.polymer import HiCHeatmapGenerator
import numpy as np

generator = HiCHeatmapGenerator()

# Create sample contact matrix
matrix = np.random.rand(50, 50)

# Generate heatmap
heatmap = generator.generate_contact_heatmap(
    matrix,
    title="Code Interaction Matrix"
)
print(heatmap)
```

Output:
```
Code Interaction Matrix
======================
░░▒▒▓▓██░░▒▒▓▓
▒▒▓▓██░░▒▒▓▓██
▓▓██░░▒▒▓▓██░░
...
Scale:  =0.00 ░=0.25 ▒=0.50 ▓=0.75 █=1.00
```

### TAD Analysis

Identify and analyze Topologically Associating Domains (code modules):

```python
from tailchasing.polymer import TAD, PolymerMetricsReport

# Define TADs
tads = [
    TAD(start=0, end=30, name="auth_module", activity_level=0.8),
    TAD(start=31, end=60, name="data_module", activity_level=0.6),
    TAD(start=61, end=100, name="ui_module", activity_level=0.9)
]

# Define interactions
interactions = [
    (10, 20, 0.8),  # Within auth_module
    (35, 45, 0.7),  # Within data_module
    (15, 70, 0.3),  # Cross-TAD interaction
]

# Calculate metrics
reporter = PolymerMetricsReport()
metrics = reporter.calculate_polymer_distances(tads, interactions)
```

### Parameter Calibration

Calibrate polymer physics parameters using observed thrashing patterns:

```python
from tailchasing.polymer.calibrate import CalibrationTool, ThrashEvent, CodebaseMetrics

tool = CalibrationTool()

# Define thrashing events
events = [
    ThrashEvent(
        file1="auth.py",
        file2="database.py",
        distance_type="mod",
        observed_frequency=0.7,
        latency_ms=150,
        timestamp=1000
    ),
    # ... more events
]

# Define codebase metrics
codebase = CodebaseMetrics(
    total_files=100,
    total_lines=10000,
    module_count=10,
    avg_file_size=100,
    complexity_score=0.6,
    interaction_density=0.3
)

# Fit parameters
result = tool.fit_parameters(events, codebase)
tool.display_results(result)
```

### Thrash Risk Prediction

Predict performance improvements from different fix strategies:

```python
from tailchasing.polymer import PolymerMetricsReport

reporter = PolymerMetricsReport()

# Define fix strategies
strategies = [
    {
        "name": "refactor_auth",
        "impact_score": 0.8,
        "complexity": 0.3,
        "confidence": 0.9
    },
    {
        "name": "cache_queries",
        "impact_score": 0.6,
        "complexity": 0.2,
        "confidence": 0.95
    }
]

# Get predictions
predictions = reporter.predict_thrash_reduction(strategies)

for name, metrics in predictions.items():
    print(f"{name}:")
    print(f"  Estimated reduction: {metrics['estimated_reduction']:.2%}")
    print(f"  Implementation risk: {metrics['implementation_risk']:.2%}")
    print(f"  ROI score: {metrics['roi_score']:.2f}")
    print(f"  Priority: {metrics['recommended_priority']}")
```

## Integration with ChromatinContactAnalyzer

The polymer module integrates seamlessly with the existing `ChromatinContactAnalyzer`:

```python
from tailchasing.analyzers.chromatin_contact import ChromatinContactAnalyzer
from tailchasing.polymer import integrate_chromatin_analysis

# Run chromatin analysis
analyzer = ChromatinContactAnalyzer()
contact_matrix = analyzer.build_contact_matrix(elements, interactions)
tads = analyzer.detect_tads(contact_matrix)

# Enhance with polymer physics
enhanced_report = integrate_chromatin_analysis(
    existing_report={"basic_metrics": {...}},
    contact_matrix=contact_matrix,
    tads=tads,
    interactions=interactions,
    fix_strategies=strategies,
    timeline_data=timeline
)
```

## Visualization Examples

### Thrash Risk Clusters

```python
risk_scores = {
    (10, 20): 0.9,  # High risk
    (30, 40): 0.6,  # Medium risk
    (50, 60): 0.3,  # Low risk
}

visualization = generator.highlight_thrash_clusters(matrix, risk_scores)
print(visualization)
```

### Replication Timing

```python
timeline_data = [
    {"name": "auth_check", "timestamp": 0, "duration": 50, "impact": 0.8},
    {"name": "db_query", "timestamp": 30, "duration": 100, "impact": 0.6},
    {"name": "render_ui", "timestamp": 120, "duration": 30, "impact": 0.3}
]

timing_viz = reporter.visualize_replication_timing(timeline_data)
print(timing_viz)
```

## API Reference

### Core Classes

- `HiCHeatmapGenerator`: Generate Hi-C style visualizations
- `PolymerMetricsReport`: Calculate polymer physics metrics
- `TAD`: Topologically Associating Domain representation
- `ThrashCluster`: Cluster of thrashing positions
- `CalibrationTool`: Parameter calibration utility
- `PolymerConfig`: Configuration management
- `ConfigManager`: Configuration file handling

### Key Functions

- `integrate_chromatin_analysis()`: Integrate with existing reports
- `generate_comparative_matrices()`: Compare before/after matrices
- `run_calibration_demo()`: Run demo calibration
- `create_default_config_file()`: Initialize configuration

## Testing

Run polymer module tests:

```bash
pytest tests/test_polymer/
```

## Contributing

The polymer physics module is part of the TailChasingFixer project. Contributions are welcome!

## License

MIT License - See main project LICENSE file