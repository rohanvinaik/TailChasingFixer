# API Reference

## Core Components

### TailChasingDetector

Main detection orchestrator that coordinates all analyzers.

```python
from tailchasing.core.detector import TailChasingDetector

detector = TailChasingDetector(config={
    'enable_semantic': True,
    'parallel': True,
    'cache': True
})

issues = detector.detect(path='src/')
```

#### Methods

##### `detect(path: Path, config: Optional[Dict] = None) -> List[Issue]`
Detect tail-chasing patterns in the specified path.

**Parameters:**
- `path`: Directory or file to analyze
- `config`: Optional configuration override

**Returns:**
- List of detected issues

##### `detect_incremental(path: Path, since: datetime) -> List[Issue]`
Detect only in files modified since the specified time.

### Issue

Represents a detected tail-chasing pattern.

```python
from tailchasing.core.issues import Issue

issue = Issue(
    kind='duplicate_function',
    file='src/utils.py',
    line=42,
    symbol='calculate_sum',
    message='Duplicate of compute_total',
    severity=3,
    evidence={'similarity': 0.95},
    suggestions=['Consolidate into single function']
)
```

#### Attributes

- `kind` (str): Type of issue (e.g., 'duplicate_function')
- `file` (str): File path where issue was detected
- `line` (int): Line number
- `symbol` (str): Symbol name (function, class, etc.)
- `message` (str): Human-readable description
- `severity` (int): Severity level (1-5)
- `evidence` (Dict): Supporting evidence
- `suggestions` (List[str]): Fix suggestions

## Semantic Analysis

### SemanticEncoder

Encodes Python code into hypervectors for semantic analysis.

```python
from tailchasing.semantic.encoder import SemanticEncoder

encoder = SemanticEncoder(config={
    'dimensions': 8192,
    'channels': ['structure', 'data_flow', 'control_flow']
})

hypervector, features = encoder.encode_function(code)
```

#### Methods

##### `encode_function(code: str) -> Tuple[np.ndarray, Dict[str, List[str]]]`
Encode a function into a hypervector.

**Parameters:**
- `code`: Python function code as string

**Returns:**
- Tuple of (hypervector, feature dictionary)

### SemanticIndex

Manages and queries semantic representations of functions.

```python
from tailchasing.semantic.index import SemanticIndex

index = SemanticIndex(config={
    'z_threshold': 2.5,
    'fdr_alpha': 0.05
})

# Add function
entry = index.add_function(
    function_id='func_1',
    file_path='utils.py',
    name='calculate',
    line_number=10,
    hypervector=hv,
    features=features
)

# Find similar functions
pairs = index.find_similar_pairs(top_k=10)
```

#### Methods

##### `add_function(...) -> FunctionEntry`
Add a function to the index.

##### `find_similar_pairs(top_k: int = 100) -> List[SimilarityPair]`
Find the most similar function pairs.

##### `compute_background_stats()`
Compute background statistics for significance testing.

### HVSpace

Hypervector space operations.

```python
from tailchasing.semantic.hv_space import HVSpace

space = HVSpace(config={'dimensions': 8192})

# Generate random hypervector
hv = space.random_hv(binary=True, sparse=True)

# Bundle vectors (superposition)
bundled = space.bundle([hv1, hv2, hv3])

# Bind vectors (role-filler binding)
bound = space.bind(role_hv, filler_hv)

# Compute similarity
sim = space.similarity(hv1, hv2)
```

## Pattern Analyzers

### HallucinationCascadeDetector

Detects over-engineered abstraction chains.

```python
from tailchasing.analyzers.hallucination_cascade import HallucinationCascadeDetector

detector = HallucinationCascadeDetector(config={
    'min_chain_length': 3,
    'max_abstraction_depth': 5
})

issues = detector.detect(path='src/')
```

### ContextWindowThrashingDetector

Identifies reimplemented functions due to context limitations.

```python
from tailchasing.analyzers.context_window_thrashing import ContextWindowThrashingDetector

detector = ContextWindowThrashingDetector(config={
    'window_size': 500,
    'similarity_threshold': 0.8
})

issues = detector.detect(path='src/')
```

### ImportAnxietyDetector

Detects import-related anti-patterns.

```python
from tailchasing.analyzers.import_anxiety import ImportAnxietyDetector

detector = ImportAnxietyDetector()
issues = detector.detect(path='src/')
```

## Fix Generation

### SuggestionGenerator

Generates fix suggestions for detected issues.

```python
from tailchasing.fixers.suggestion_generator import SuggestionGenerator

generator = SuggestionGenerator()
suggestions = generator.generate_suggestions(issues)

for suggestion in suggestions:
    print(f"Fix for {suggestion.issue_id}:")
    print(f"  Type: {suggestion.fix_type}")
    print(f"  Confidence: {suggestion.confidence}")
    print(f"  Description: {suggestion.description}")
```

### FixApplier

Applies generated fixes to code.

```python
from tailchasing.fixers.fix_applier import FixApplier

applier = FixApplier(config={'dry_run': False, 'backup': True})

for suggestion in suggestions:
    result = applier.apply_fix(suggestion, Path(suggestion.target_file))
    if result.success:
        print(f"Applied fix to {suggestion.target_file}")
    else:
        print(f"Failed: {result.error}")
```

### FixValidator

Validates fixes before application.

```python
from tailchasing.fixers.fix_validator import FixValidator

validator = FixValidator()

result = validator.validate_fix(suggestion)
if result.is_valid:
    # Safe to apply
    applier.apply_fix(suggestion)
```

## Orchestration

### TailChasingOrchestrator

High-level orchestration of detection and fixing.

```python
from tailchasing.orchestration.orchestrator import TailChasingOrchestrator

orchestrator = TailChasingOrchestrator(config={
    'auto_fix': True,
    'dry_run': False,
    'validate_fixes': True
})

result = orchestrator.orchestrate(
    path='src/',
    auto_fix=True
)

print(f"Issues found: {result['issues_found']}")
print(f"Fixes applied: {result['fixes_applied']}")
```

## Performance

### CacheManager

Manages multi-level caching.

```python
from tailchasing.performance.cache import get_cache_manager

cache = get_cache_manager()

# Get AST from cache or parse
ast = cache.ast_cache.get_ast('file.py')

# Cache hypervector
cache.hypervector_cache.set('func_1', hypervector)

# Get cache statistics
stats = cache.get_stats()
```

### ParallelExecutor

Parallel processing for CPU-bound tasks.

```python
from tailchasing.performance.parallel import ParallelExecutor

executor = ParallelExecutor(max_workers=8, use_processes=True)

# Map function over items in parallel
results = executor.map(process_function, items, chunk_size=10)

# Map-reduce operation
aggregated = executor.map_reduce(
    map_func=extract_features,
    reduce_func=combine_features,
    items=functions
)
```

### PerformanceMonitor

Track and analyze performance metrics.

```python
from tailchasing.performance.monitor import get_monitor, track_performance

monitor = get_monitor(enable_profiling=True)

# Track operation
with monitor.track("analysis") as metric:
    result = analyze_codebase()
    metric.items_processed = len(result)

# Get performance summary
summary = monitor.get_summary()
print(f"Total duration: {summary['total_duration']}s")
print(f"Throughput: {summary['overall_throughput']} items/s")

# Decorator for automatic tracking
@track_performance("custom_operation")
def my_function():
    # Function implementation
    pass
```

## Visualization

### TailChaseVisualizer

Generate interactive visualizations.

```python
from tailchasing.visualization.tail_chase_visualizer import TailChaseVisualizer

visualizer = TailChaseVisualizer()
visualizer.add_issues(issues)

# Generate dependency graph
html = visualizer.generate_dependency_graph(
    title="Code Dependencies",
    width=1200,
    height=800
)

# Generate similarity heatmap
heatmap = visualizer.generate_similarity_heatmap()

# Generate temporal animation
animation = visualizer.generate_temporal_evolution()
```

### ReportGenerator

Generate comprehensive reports.

```python
from tailchasing.visualization.report_generator import ReportGenerator

generator = ReportGenerator()
generator.add_issues(issues)

# Generate HTML report
html = generator.generate_html_report(
    include_visualizations=True,
    embed_data=True
)

# Generate JSON report
json_report = generator.generate_json_report()

# Generate Markdown summary
markdown = generator.generate_markdown_summary()
```

## LLM Integration

### FeedbackGenerator

Generate feedback for LLMs.

```python
from tailchasing.llm_integration.feedback_generator import FeedbackGenerator

generator = FeedbackGenerator()
feedback = generator.generate_feedback(issues)

# Get prevention prompt
prompt = generator.generate_prevention_prompt(
    detected_patterns={'duplicate_function', 'circular_import'}
)

# Get context alert
alert = generator.generate_context_alert(
    current_file='utils.py',
    current_function='calculate',
    recent_issues=issues
)
```

### UniversalAdapter

Format feedback for different LLM APIs.

```python
from tailchasing.llm_integration.llm_adapters import UniversalAdapter

adapter = UniversalAdapter()

# Format for OpenAI
openai_prompt = adapter.format_feedback(
    feedback,
    format_type='openai'
)

# Format for Anthropic
anthropic_prompt = adapter.format_feedback(
    feedback,
    format_type='anthropic'
)

# Generate and format in one step
formatted = adapter.generate_and_format(
    issues,
    format_type='openai'
)
```

## CI/CD Integration

### PipelineAnalyzer

Analyze pull requests in CI/CD pipelines.

```python
from tailchasing.ci.pipeline_analyzer import PipelineAnalyzer

analyzer = PipelineAnalyzer()

# Analyze PR
pr_analysis = analyzer.analyze_pr(
    pr_number=123,
    branch='feature-branch',
    base_branch='main'
)

# Check if should block merge
if pr_analysis.should_block_merge(threshold=10.0):
    print("PR has blocking issues")

# Calculate trend
trend = analyzer.calculate_trend(
    analyses=[pr_analysis1, pr_analysis2, pr_analysis3],
    window=10
)
```

### GitHubIntegration

GitHub Actions integration.

```python
from tailchasing.ci.github_integration import GitHubIntegration, GitHubContext

integration = GitHubIntegration(config={
    'auto_comment': True,
    'block_on_risk': True,
    'risk_threshold': 10.0
})

# Run PR check
context = GitHubContext.from_env()
result = integration.run_pr_check(context)

# Handle webhook
result = integration.handle_webhook(
    event_type='pull_request',
    payload=webhook_payload,
    signature=webhook_signature
)

# Generate workflow YAML
workflow = integration.generate_workflow_yaml()
```

## Extending TailChasingFixer

### Creating Custom Analyzers

```python
from tailchasing.analyzers.base import BaseAnalyzer
from tailchasing.core.issues import Issue

class CustomAnalyzer(BaseAnalyzer):
    """Custom pattern analyzer."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = 'custom_pattern'
    
    def analyze(self, ast_tree, filepath):
        """Analyze AST for custom patterns."""
        issues = []
        
        # Your analysis logic here
        for node in ast.walk(ast_tree):
            if self._is_custom_pattern(node):
                issues.append(Issue(
                    kind='custom_pattern',
                    file=filepath,
                    line=node.lineno,
                    message='Custom pattern detected',
                    severity=2
                ))
        
        return issues
    
    def _is_custom_pattern(self, node):
        """Check if node matches custom pattern."""
        # Pattern detection logic
        return False
```

### Creating Custom Fixers

```python
from tailchasing.fixers.base import BaseFixer
from tailchasing.fixers.suggestion_generator import FixSuggestion

class CustomFixer(BaseFixer):
    """Custom fix generator."""
    
    def generate_fix(self, issue):
        """Generate fix for custom issue type."""
        if issue.kind != 'custom_pattern':
            return None
        
        return FixSuggestion(
            issue_id=issue.id,
            fix_type='custom_fix',
            target_file=issue.file,
            target_line=issue.line,
            original_code=self._get_original_code(issue),
            new_code=self._generate_new_code(issue),
            description='Apply custom fix',
            confidence=0.9,
            priority='medium'
        )
```

### Plugin System

```python
# plugins/my_plugin.py
from tailchasing.plugins import register_analyzer, register_fixer

@register_analyzer('my_analyzer')
class MyAnalyzer(BaseAnalyzer):
    # Implementation
    pass

@register_fixer('my_fixer')
class MyFixer(BaseFixer):
    # Implementation
    pass
```

Configuration:
```yaml
# .tailchasing.yml
plugins:
  - my_plugin
  
analyzers:
  my_analyzer:
    enabled: true
    custom_option: value
```

## Configuration Schema

### Complete Configuration Reference

```yaml
# .tailchasing.yml
# Severity threshold (1-5)
severity_threshold: 3

# Enable automatic fixing
auto_fix: false

# Dry run mode (preview fixes without applying)
dry_run: false

# Analyzers configuration
analyzers:
  # Duplicate function detection
  duplicate_function:
    enabled: true
    min_lines: 3
    ignore_test_files: true
    
  # Semantic duplicate detection
  semantic_duplicate:
    enabled: true
    threshold: 0.85
    use_multimodal: true
    channels:
      - structure
      - data_flow
      - control_flow
    
  # Circular import detection
  circular_import:
    enabled: true
    max_depth: 10
    
  # Phantom function detection
  phantom_function:
    enabled: true
    check_docstrings: true
    
  # Hallucination cascade detection
  hallucination_cascade:
    enabled: true
    min_chain_length: 3
    max_abstraction_depth: 5
    min_external_refs: 2
    
  # Context window thrashing detection
  context_window_thrashing:
    enabled: true
    window_size: 500
    similarity_threshold: 0.8
    min_distance: 100
    
  # Import anxiety detection
  import_anxiety:
    enabled: true
    max_imports: 20
    check_wildcards: true

# Semantic analysis configuration
semantic:
  enabled: false
  dimensions: 8192
  sparsity: 0.01
  channels:
    - structure
    - data_flow
    - control_flow
    - identifiers
    - literals
    - operations
  z_threshold: 2.5
  fdr_alpha: 0.05

# Performance configuration
performance:
  parallel: false
  max_workers: null  # Auto-detect
  cache_enabled: false
  cache_ttl: 3600
  memory_limit_mb: 4096
  batch_size: 100

# Reporting configuration
reporting:
  formats:
    - text
  include_visualizations: false
  max_issues_display: 100
  group_by_file: true

# CI/CD configuration
ci:
  enabled: false
  github_actions: false
  risk_threshold: 10.0
  block_on_critical: true
  auto_comment: true
  
# LLM integration
llm:
  enabled: false
  generate_feedback: true
  format: openai
  include_examples: true
  max_prompt_length: 4000

# Paths configuration
paths:
  include:
    - "**/*.py"
  exclude:
    - "**/test_*.py"
    - "**/*_test.py"
    - "**/tests/**"
    - "**/migrations/**"
    - "**/__pycache__/**"
    - "**/venv/**"
    - "**/env/**"
    - "**/.git/**"
```