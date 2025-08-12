"""
Example: Integrating TailChasingFixer into Your Python Application

This shows how to use TailChasingFixer programmatically in your code.
"""

from pathlib import Path
from tailchasing.core.detector import TailChasingDetector
from tailchasing.orchestration.orchestrator import TailChasingOrchestrator
from tailchasing.semantic.encoder import SemanticEncoder
from tailchasing.semantic.index import SemanticIndex
from tailchasing.visualization.report_generator import ReportGenerator
from tailchasing.llm_integration.feedback_generator import FeedbackGenerator
from tailchasing.performance.monitor import get_monitor, track_performance

# Example 1: Basic Detection
def basic_detection_example():
    """Run basic tail-chasing detection on a codebase."""
    
    # Initialize detector
    detector = TailChasingDetector()
    
    # Detect issues in current directory
    issues = detector.detect(Path("."))
    
    # Process results
    for issue in issues:
        print(f"{issue.kind}: {issue.file}:{issue.line}")
        print(f"  Message: {issue.message}")
        print(f"  Severity: {issue.severity}")
        
        # Apply fixes if available
        if issue.suggestions:
            print(f"  Suggested fix: {issue.suggestions[0]}")
    
    return issues


# Example 2: Semantic Analysis with Hypervectors
def semantic_analysis_example():
    """Perform semantic duplicate detection using hypervectors."""
    
    # Initialize components
    encoder = SemanticEncoder(config={
        'dimensions': 8192,
        'channels': ['structure', 'data_flow', 'control_flow']
    })
    
    index = SemanticIndex(config={
        'z_threshold': 2.5,
        'fdr_alpha': 0.05
    })
    
    # Example functions to analyze
    code1 = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
    
    code2 = """
def compute_total(values):
    result = 0
    for val in values:
        result = result + val
    return result
"""
    
    # Encode functions
    hv1, features1 = encoder.encode_function(code1)
    hv2, features2 = encoder.encode_function(code2)
    
    # Add to index
    index.add_function("func1", "file1.py", "calculate_sum", 1, hv1, features1)
    index.add_function("func2", "file2.py", "compute_total", 10, hv2, features2)
    
    # Find similar pairs
    similar_pairs = index.find_similar_pairs(top_k=10)
    
    for pair in similar_pairs:
        print(f"Similar functions detected:")
        print(f"  {pair.func1_id} <-> {pair.func2_id}")
        print(f"  Similarity: {pair.similarity:.2f}")
        print(f"  Statistical significance: p={pair.p_value:.4f}")
        print(f"  Channel contributions: {pair.channel_similarities}")
    
    return similar_pairs


# Example 3: Orchestrated Detection and Fixing
@track_performance("orchestrated_analysis")
def orchestrated_example():
    """Full orchestration with detection, planning, and fixing."""
    
    # Initialize orchestrator with configuration
    orchestrator = TailChasingOrchestrator(config={
        'auto_fix': True,
        'dry_run': True,  # Don't actually apply fixes
        'validate_fixes': True,
        'create_backups': True
    })
    
    # Run full orchestration
    result = orchestrator.orchestrate(
        path=Path("src/"),
        auto_fix=True
    )
    
    # Process results
    print(f"Issues found: {result['issues_found']}")
    print(f"Fixes generated: {result['fixes_generated']}")
    print(f"Fixes applicable: {result['fixes_applicable']}")
    
    # Show fix plan
    if 'fix_plan' in result:
        plan = result['fix_plan']
        print(f"\nFix Plan:")
        print(f"  Risk score: {plan.risk_score}")
        print(f"  Execution order: {plan.execution_order}")
        
        for fix in plan.fixes:
            print(f"\n  Fix: {fix.description}")
            print(f"    Target: {fix.target_file}:{fix.target_line}")
            print(f"    Confidence: {fix.confidence}")
            print(f"    Priority: {fix.priority}")
    
    return result


# Example 4: Generate Reports and Visualizations
def reporting_example(issues):
    """Generate various report formats from detected issues."""
    
    # Initialize report generator
    report_gen = ReportGenerator()
    report_gen.add_issues(issues)
    
    # Generate HTML report with visualizations
    html_report = report_gen.generate_html_report(
        include_visualizations=True,
        embed_data=True
    )
    
    # Save HTML report
    Path("tailchasing_report.html").write_text(html_report)
    
    # Generate JSON report for programmatic access
    json_report = report_gen.generate_json_report()
    
    # Generate Markdown summary for documentation
    markdown_summary = report_gen.generate_markdown_summary()
    
    print("Reports generated:")
    print(f"  HTML: tailchasing_report.html")
    print(f"  JSON: {len(json_report)} bytes")
    print(f"  Markdown: {len(markdown_summary.split())} words")
    
    return {
        'html': html_report,
        'json': json_report,
        'markdown': markdown_summary
    }


# Example 5: LLM Integration for Prevention
def llm_feedback_example(issues):
    """Generate feedback to prevent future tail-chasing in LLMs."""
    
    # Initialize feedback generator
    feedback_gen = FeedbackGenerator()
    
    # Generate comprehensive feedback
    feedback = feedback_gen.generate_feedback(issues)
    
    # Get prevention prompt for LLM
    prevention_prompt = feedback_gen.generate_prevention_prompt(
        detected_patterns={'duplicate_function', 'circular_import'}
    )
    
    print("LLM Prevention Prompt:")
    print(prevention_prompt.system_prompt)
    
    print("\nPattern-Specific Rules:")
    for rule in prevention_prompt.rules:
        print(f"  - {rule}")
    
    # Generate context alert for current coding session
    context_alert = feedback_gen.generate_context_alert(
        current_file="utils.py",
        current_function="process_data",
        recent_issues=issues[:5]
    )
    
    print("\nContext Alert:")
    print(context_alert.alert_message)
    
    return feedback


# Example 6: Performance Monitoring
def performance_monitoring_example():
    """Monitor and optimize performance of analysis."""
    
    # Get performance monitor
    monitor = get_monitor(enable_profiling=True)
    
    # Track operation
    with monitor.track("analysis") as metric:
        detector = TailChasingDetector()
        issues = detector.detect(Path("src/"))
        
        # Update metrics
        metric.items_processed = len(issues)
    
    # Get performance summary
    summary = monitor.get_summary()
    
    print("Performance Summary:")
    print(f"  Total duration: {summary['total_duration']:.2f}s")
    print(f"  Items processed: {summary['total_items_processed']}")
    print(f"  Throughput: {summary['overall_throughput']:.1f} items/s")
    
    # Check for bottlenecks
    bottlenecks = summary['bottlenecks']
    if bottlenecks:
        print("\nBottlenecks detected:")
        for b in bottlenecks:
            print(f"  {b['operation']}: {b['duration']:.2f}s ({b['percentage']:.1f}%)")
    
    # Get recommendations
    recommendations = summary['recommendations']
    if recommendations:
        print("\nPerformance recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    return summary


# Example 7: Custom Analyzer Integration
from tailchasing.analyzers.base import BaseAnalyzer
from tailchasing.core.issues import Issue

class CustomPatternAnalyzer(BaseAnalyzer):
    """Custom analyzer for project-specific patterns."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.name = 'custom_pattern'
    
    def analyze(self, ast_tree, filepath):
        """Analyze AST for custom patterns."""
        import ast
        
        issues = []
        
        # Example: Detect functions with too many parameters
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                
                if param_count > 5:
                    issues.append(Issue(
                        kind='too_many_parameters',
                        file=filepath,
                        line=node.lineno,
                        symbol=node.name,
                        message=f"Function has {param_count} parameters (max: 5)",
                        severity=2,
                        evidence={'parameter_count': param_count},
                        suggestions=[
                            "Consider using a configuration object",
                            "Break down into smaller functions",
                            "Use **kwargs for optional parameters"
                        ]
                    ))
        
        return issues


# Example 8: CI/CD Integration
def ci_integration_example(pr_number, branch):
    """Integrate with CI/CD pipeline for PR analysis."""
    from tailchasing.ci.pipeline_analyzer import PipelineAnalyzer
    
    # Initialize analyzer
    analyzer = PipelineAnalyzer()
    
    # Analyze pull request
    pr_analysis = analyzer.analyze_pr(
        pr_number=pr_number,
        branch=branch,
        base_branch='main'
    )
    
    print(f"PR #{pr_number} Analysis:")
    print(f"  Risk score: {pr_analysis.risk_score:.1f}")
    print(f"  New issues: {pr_analysis.new_issues}")
    print(f"  Fixed issues: {pr_analysis.fixed_issues}")
    print(f"  Should block: {pr_analysis.should_block_merge()}")
    
    # Get trend over recent PRs
    trend = analyzer.calculate_trend(
        analyses=[pr_analysis],  # Would include historical analyses
        window=10
    )
    
    print(f"  Trend: {trend}")
    
    return pr_analysis


# Main execution
if __name__ == "__main__":
    print("TailChasingFixer Integration Examples\n")
    print("=" * 50)
    
    # Run basic detection
    print("\n1. Basic Detection:")
    issues = basic_detection_example()
    
    # Run semantic analysis
    print("\n2. Semantic Analysis:")
    semantic_analysis_example()
    
    # Run orchestrated analysis
    print("\n3. Orchestrated Analysis:")
    orchestrated_example()
    
    # Generate reports
    if issues:
        print("\n4. Report Generation:")
        reporting_example(issues)
        
        print("\n5. LLM Feedback:")
        llm_feedback_example(issues)
    
    # Monitor performance
    print("\n6. Performance Monitoring:")
    performance_monitoring_example()
    
    print("\n" + "=" * 50)
    print("Examples completed successfully!")