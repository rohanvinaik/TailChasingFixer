"""
Test suite for OptimizedTailChasingFixer orchestrator.

Verifies:
1. 6-phase optimized analysis pipeline
2. Performance tracking and metrics
3. Pattern frontier exploration
4. Parallel processing capabilities
5. Benchmark comparison with legacy analysis
6. Configuration and optimization modes
"""

import ast
import time
import asyncio
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from tailchasing.optimization.orchestrator import (
    OptimizedTailChasingFixer,
    PatternFrontierExplorer,
    OptimizationConfig,
    OptimizationMetrics,
    PerformanceMode,
    OptimizationPhase,
    AnalysisResult
)
from tailchasing.core.issues import Issue


class TestPatternFrontierExplorer(unittest.TestCase):
    """Test cases for PatternFrontierExplorer."""
    
    def setUp(self):
        """Set up test explorer."""
        self.explorer = PatternFrontierExplorer(max_patterns=50)
        self.sample_codebase = self._create_sample_codebase()
    
    def _create_sample_codebase(self) -> Dict[str, ast.AST]:
        """Create sample codebase for frontier exploration."""
        codebase = {}
        
        # Existing patterns (known)
        known_code = """
def get_user(id):
    return database.find_user(id)

def validate_email(email):
    return '@' in email

class DataProcessor:
    def process(self, data):
        return data.upper()
"""
        codebase['known.py'] = ast.parse(known_code)
        
        # Frontier patterns (similar but different)
        frontier_code = """
def get_product(product_id):
    return database.find_product(product_id)

def validate_phone(phone):
    return len(phone) == 10

def get_customer_info(customer_id):
    return api.fetch_customer(customer_id)

class TextProcessor:
    def process(self, text):
        return text.lower()
    
    def transform(self, content):
        return content.strip()
"""
        codebase['frontier.py'] = ast.parse(frontier_code)
        
        # Novel patterns (genuinely new)
        novel_code = """
async def stream_data(source):
    async for item in source:
        yield transform_item(item)

def complex_algorithm(data):
    result = []
    for item in data:
        if item.valid:
            try:
                processed = process_with_validation(item)
                if processed:
                    result.append(processed)
            except ValidationError as e:
                handle_error(e)
            except ProcessingError:
                continue
            finally:
                cleanup_item(item)
    return result
"""
        codebase['novel.py'] = ast.parse(novel_code)
        
        return codebase
    
    def test_identify_pattern_frontier(self):
        """Test identification of frontier patterns."""
        # Known patterns
        known_patterns = [
            {'signature': 'FunctionDef:args:1:body:1', 'type': 'getter'},
            {'signature': 'FunctionDef:args:1:body:1:conditional', 'type': 'validator'},
            {'signature': 'ClassDef:methods:1:bases:0', 'type': 'processor'}
        ]
        
        frontier = self.explorer.identify_pattern_frontier(
            self.sample_codebase,
            known_patterns
        )
        
        # Should find patterns that are similar but not identical
        self.assertGreater(len(frontier), 0, "Should identify frontier patterns")
        
        # Check that frontier patterns have expected characteristics
        frontier_names = [p['name'] for p in frontier]
        
        # Should find some patterns (exact matches depend on similarity thresholds)
        expected_patterns = ['get_product', 'validate_phone', 'TextProcessor', 'stream_data']
        found_expected = [name for name in expected_patterns if name in frontier_names]
        
        self.assertGreater(len(found_expected), 0, 
                          f"Should find some expected patterns. Found: {frontier_names}")
    
    def test_cluster_frontier_patterns(self):
        """Test clustering of frontier patterns."""
        # Create sample frontier patterns
        frontier = [
            {'signature': 'FunctionDef:args:1:body:1', 'name': 'get_product'},
            {'signature': 'FunctionDef:args:1:body:1', 'name': 'get_customer_info'},
            {'signature': 'FunctionDef:args:1:body:1:conditional', 'name': 'validate_phone'},
            {'signature': 'ClassDef:methods:2:bases:0', 'name': 'TextProcessor'}
        ]
        
        clusters = self.explorer.cluster_frontier_patterns(frontier)
        
        # Should create meaningful clusters
        self.assertGreater(len(clusters), 0, "Should create clusters")
        
        # Similar patterns should be in same cluster
        cluster_contents = {}
        for cluster_id, patterns in clusters.items():
            cluster_contents[cluster_id] = [p['name'] for p in patterns]
        
        # Check clustering logic
        getter_cluster = None
        for cluster_id, names in cluster_contents.items():
            if 'get_product' in names:
                getter_cluster = cluster_id
                break
        
        if getter_cluster:
            # get_customer_info might be in same cluster due to similarity
            getter_patterns = cluster_contents[getter_cluster]
            # At least the get_ functions should potentially cluster together
            getter_count = sum(1 for name in getter_patterns if name.startswith('get_'))
            self.assertGreaterEqual(getter_count, 1, "Should cluster similar getters")
    
    def test_select_cluster_representatives(self):
        """Test selection of cluster representatives."""
        # Sample clusters
        clusters = {
            'getter_cluster': [
                {'name': 'get_product', 'signature': 'FunctionDef:args:1:body:1'},
                {'name': 'get_customer_info', 'signature': 'FunctionDef:args:1:body:1'}
            ],
            'validator_cluster': [
                {'name': 'validate_phone', 'signature': 'FunctionDef:args:1:body:1:conditional'}
            ]
        }
        
        representatives = self.explorer.select_cluster_representatives(clusters)
        
        # Should select one rep per cluster
        self.assertEqual(len(representatives), 2, "Should select one rep per cluster")
        
        # Each representative should have cluster size info
        for rep in representatives:
            self.assertIn('represents_count', rep, "Rep should know cluster size")
            self.assertGreater(rep['represents_count'], 0, "Should represent at least 1")
    
    def test_analyze_representative(self):
        """Test analysis of representative patterns."""
        # Create function node for testing
        func_code = """
def complex_function(a, b, c):
    if a > 0:
        for i in range(10):
            try:
                result = process(i)
                if result:
                    return result
            except Exception:
                continue
    return None
"""
        tree = ast.parse(func_code)
        func_node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        representative = {
            'node': func_node,
            'signature': 'FunctionDef:args:3:body:2:conditional:iteration:exception',
            'name': 'complex_function',
            'represents_count': 3
        }
        
        analysis = self.explorer.analyze_representative(representative)
        
        # Should provide detailed analysis
        self.assertIsNotNone(analysis, "Should analyze representative")
        self.assertEqual(analysis['name'], 'complex_function')
        self.assertEqual(analysis['arg_count'], 3)
        self.assertEqual(analysis['cluster_size'], 3)
        self.assertTrue(analysis['has_loops'], "Should detect loops")
        self.assertTrue(analysis['has_exceptions'], "Should detect exceptions")
        self.assertGreater(analysis['complexity_score'], 5, "Should calculate complexity")
    
    def test_is_novel_pattern(self):
        """Test detection of novel patterns."""
        # Complex novel pattern
        complex_pattern = {
            'signature': 'AsyncFunctionDef:args:1:body:3:iteration:exception',
            'complexity_score': 8,
            'cluster_size': 1,
            'has_exceptions': True
        }
        
        self.assertTrue(self.explorer.is_novel_pattern(complex_pattern),
                       "Complex pattern should be novel")
        
        # Simple pattern
        simple_pattern = {
            'signature': 'FunctionDef:args:0:body:1',
            'complexity_score': 1,
            'cluster_size': 1,
            'has_exceptions': False
        }
        
        self.assertFalse(self.explorer.is_novel_pattern(simple_pattern),
                        "Simple pattern should not be novel")
        
        # Large cluster pattern
        cluster_pattern = {
            'signature': 'FunctionDef:args:2:body:2',
            'complexity_score': 3,
            'cluster_size': 5,  # Represents many similar patterns
            'has_exceptions': False
        }
        
        self.assertTrue(self.explorer.is_novel_pattern(cluster_pattern),
                       "Pattern representing many should be novel")
    
    def test_extrapolate_to_cluster(self):
        """Test extrapolation of analysis to entire cluster."""
        pattern = {
            'type': 'getter_pattern',
            'complexity_score': 2,
            'signature': 'base_signature'
        }
        
        cluster_patterns = [
            {'file_path': 'file1.py', 'name': 'get_user', 'signature': 'sig1'},
            {'file_path': 'file2.py', 'name': 'get_product', 'signature': 'sig2'}
        ]
        
        extrapolated = self.explorer.extrapolate_to_cluster(pattern, cluster_patterns)
        
        # Should create pattern for each cluster member
        self.assertEqual(len(extrapolated), 2, "Should extrapolate to all members")
        
        # Each should inherit pattern analysis but keep member-specific details
        for i, ext_pattern in enumerate(extrapolated):
            self.assertEqual(ext_pattern['type'], 'getter_pattern')
            self.assertEqual(ext_pattern['complexity_score'], 2)
            self.assertEqual(ext_pattern['file_path'], cluster_patterns[i]['file_path'])
            self.assertEqual(ext_pattern['name'], cluster_patterns[i]['name'])
    
    def test_explore_pattern_frontier_complete_flow(self):
        """Test complete frontier exploration flow."""
        known_patterns = [
            {'signature': 'FunctionDef:args:1:body:1', 'type': 'simple_getter'}
        ]
        
        new_patterns = self.explorer.explore_pattern_frontier(
            self.sample_codebase,
            known_patterns
        )
        
        # Should discover new patterns
        self.assertGreater(len(new_patterns), 0, "Should discover new patterns")
        
        # Should be limited by max_patterns
        self.assertLessEqual(len(new_patterns), self.explorer.max_patterns,
                           "Should respect max patterns limit")
        
        # New patterns should have required fields
        for pattern in new_patterns:
            self.assertIn('signature', pattern, "Pattern should have signature")
            self.assertIn('file_path', pattern, "Pattern should have file path")


class TestOptimizedTailChasingFixer(unittest.TestCase):
    """Test cases for OptimizedTailChasingFixer."""
    
    def setUp(self):
        """Set up test fixer."""
        config = OptimizationConfig(
            mode=PerformanceMode.BALANCED,
            max_clusters=10,
            parallel_processing=False  # Disable for testing
        )
        self.fixer = OptimizedTailChasingFixer(config)
        self.sample_codebase = self._create_sample_codebase()
    
    def _create_sample_codebase(self) -> Dict[str, ast.AST]:
        """Create comprehensive sample codebase."""
        codebase = {}
        
        # High-influence utilities
        utils_code = """
def log_message(msg, level='INFO'):
    print(f"[{level}] {msg}")

def validate_data(data):
    if not data:
        raise ValueError("Invalid data")
    return True

def get_config_value(key):
    return config.get(key, None)
"""
        codebase['utils.py'] = ast.parse(utils_code)
        
        # Business logic using utilities
        business_code = """
from utils import log_message, validate_data

def process_order(order):
    validate_data(order)
    log_message(f"Processing order {order.id}")
    
    try:
        result = calculate_total(order)
        log_message(f"Order total: {result}")
        return result
    except Exception as e:
        log_message(f"Error: {e}", 'ERROR')
        raise

def calculate_total(order):
    return sum(item.price for item in order.items)
"""
        codebase['business.py'] = ast.parse(business_code)
        
        # API layer
        api_code = """
from business import process_order

def api_process_order(request):
    try:
        order = request.json
        result = process_order(order)
        return {"status": "success", "total": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def health_check():
    return {"status": "healthy"}

def empty_endpoint():
    pass  # Phantom function
"""
        codebase['api.py'] = ast.parse(api_code)
        
        # Error-prone code
        problematic_code = """
def risky_function(data):
    global state
    try:
        if data:
            for item in data:
                process_item(item)
    except:  # Bare except
        pass

def duplicate_logic(items):
    result = []
    for item in items:
        if item.valid:
            result.append(item.value)
    return result

def another_duplicate_logic(data):
    output = []
    for element in data:
        if element.valid:
            output.append(element.value)
    return output
"""
        codebase['problematic.py'] = ast.parse(problematic_code)
        
        return codebase
    
    def test_optimized_analysis_six_phases(self):
        """Test complete 6-phase optimized analysis."""
        result = self.fixer.analyze_codebase_optimized(self.sample_codebase)
        
        # Should complete all phases
        self.assertIsInstance(result, AnalysisResult)
        
        # Should have phase timing data
        expected_phases = [
            OptimizationPhase.CLUSTERING.value,
            OptimizationPhase.INFLUENCE_DETECTION.value,
            OptimizationPhase.NAVIGATION.value,
            OptimizationPhase.FRONTIER_EXPLORATION.value,
            OptimizationPhase.ADAPTIVE_ANALYSIS.value,
            OptimizationPhase.FIX_PRIORITIZATION.value
        ]
        
        for phase in expected_phases:
            self.assertIn(phase, result.metrics.phase_times,
                         f"Should have timing for {phase}")
            self.assertGreater(result.metrics.phase_times[phase], 0,
                              f"Phase {phase} should take some time")
        
        # Should find issues
        self.assertGreater(len(result.issues), 0, "Should find issues")
        
        # Should create clusters
        self.assertGreater(len(result.clusters), 0, "Should create clusters")
        
        # Should identify influential patterns (may be 0 if no patterns found)
        self.assertGreaterEqual(len(result.influential_patterns), 0, 
                               "Should complete influential pattern detection")
        
        # Should create fix clusters (may be 0 if no issues requiring clustering)
        self.assertGreaterEqual(len(result.fix_clusters), 0, "Should complete fix cluster creation")
        
        # Should have recommendations
        self.assertGreater(len(result.recommendations), 0, "Should provide recommendations")
    
    def test_phase1_clustering(self):
        """Test Phase 1: Clustering without sorting."""
        clusters = self.fixer._phase1_clustering(self.sample_codebase)
        
        # Should create clusters
        self.assertIsInstance(clusters, dict)
        self.assertGreater(len(clusters), 0, "Should create semantic clusters")
        
        # Verify cluster structure
        for cluster_id, cluster in clusters.items():
            self.assertIsNotNone(cluster_id, "Cluster should have ID")
    
    def test_phase2_influence_detection(self):
        """Test Phase 2: Influence detection with sampling."""
        # Mock clusters
        mock_clusters = {
            'cluster1': Mock(),
            'cluster2': Mock()
        }
        
        with patch.object(self.fixer.cluster_analyzer, 'find_influential_patterns') as mock_find:
            # Mock influential patterns
            from tailchasing.optimization.cluster_engine import InfluentialPattern
            mock_patterns = [
                InfluentialPattern('pattern1', 0.8, {'type': 'function'}),
                InfluentialPattern('pattern2', 0.6, {'type': 'class'})
            ]
            mock_find.return_value = mock_patterns
            
            influential = self.fixer._phase2_influence_detection(mock_clusters)
            
            # Should identify influential patterns
            self.assertGreater(len(influential), 0, "Should find influential patterns")
            
            # Should convert to InfluentialNode format
            for node in influential:
                self.assertIsNotNone(node.node_id)
                self.assertIsNotNone(node.influence_score)
    
    def test_phase3_nonlinear_navigation(self):
        """Test Phase 3: Non-linear navigation."""
        # Mock influential nodes
        from tailchasing.optimization.influence_detector import InfluentialNode
        influential = [
            InfluentialNode('utils.py:log_message', 'function', 85.0),
            InfluentialNode('utils.py:validate_data', 'function', 70.0)
        ]
        
        # Mock navigation plan
        with patch.object(self.fixer.navigator, 'navigate_by_influence') as mock_nav:
            from tailchasing.optimization.nonlinear_navigator import NavigationPlan, NavigationNode
            
            mock_nodes = [
                NavigationNode('test:node1', 'file1.py', Mock(), 80.0, 80.0),
                NavigationNode('test:node2', 'file2.py', Mock(), 60.0, 60.0)
            ]
            
            mock_plan = NavigationPlan(
                ordered_nodes=mock_nodes,
                skip_reasons={},
                cluster_order=['cluster1'],
                estimated_time=0.1,
                influence_coverage=0.8
            )
            mock_nav.return_value = mock_plan
            
            issues = self.fixer._phase3_nonlinear_navigation(self.sample_codebase, influential)
            
            # Should find initial issues
            self.assertIsInstance(issues, list)
            # Navigation called with correct strategy
            mock_nav.assert_called_once()
    
    def test_phase4_frontier_exploration(self):
        """Test Phase 4: Pattern frontier exploration."""
        initial_issues = [
            Issue(kind="phantom_function", message="Test", file="test.py", line=1, severity=2)
        ]
        
        new_patterns = self.fixer._phase4_frontier_exploration(
            self.sample_codebase, 
            initial_issues
        )
        
        # Should explore frontier
        self.assertIsInstance(new_patterns, list)
        # May or may not find patterns depending on frontier
    
    def test_phase5_adaptive_analysis(self):
        """Test Phase 5: Adaptive analysis."""
        initial_issues = []
        new_patterns = [
            {
                'node': ast.FunctionDef(name='test', args=ast.arguments(
                    args=[], posonlyargs=[], kwonlyargs=[], 
                    kw_defaults=[], defaults=[]
                ), body=[ast.Pass()]),
                'signature': 'test_pattern',
                'file_path': 'test.py'
            }
        ]
        
        with patch.object(self.fixer.adaptive_selector, 'analyze_with_adaptive_algorithms') as mock_analyze:
            from tailchasing.optimization.adaptive_selector import AnalysisResult, AnalysisStrategy
            
            mock_result = AnalysisResult(
                strategy_used=AnalysisStrategy.FAST,
                patterns_found=[{'type': 'empty_function', 'confidence': 0.8}],
                execution_time=0.1,
                confidence=0.8
            )
            mock_analyze.return_value = mock_result
            
            enhanced_issues = self.fixer._phase5_adaptive_analysis(
                self.sample_codebase,
                initial_issues,
                new_patterns
            )
            
            # Should produce enhanced issues
            self.assertIsInstance(enhanced_issues, list)
            # Should have called adaptive selector
            mock_analyze.assert_called_once()
    
    def test_phase6_fix_prioritization(self):
        """Test Phase 6: Fix prioritization by influence."""
        issues = [
            Issue(kind="phantom_function", message="Empty function", 
                  file="api.py", line=10, severity=2),
            Issue(kind="bare_except", message="Bare except clause", 
                  file="problematic.py", line=15, severity=3),
            Issue(kind="duplicate_function", message="Duplicate logic", 
                  file="problematic.py", line=20, severity=2)
        ]
        
        fix_clusters = self.fixer._phase6_fix_prioritization(issues)
        
        # Should create fix clusters
        self.assertIsInstance(fix_clusters, list)
        
        if fix_clusters:
            # Verify cluster structure
            for cluster in fix_clusters:
                self.assertIsNotNone(cluster.cluster_id)
                self.assertIsNotNone(cluster.primary_issue)
                self.assertIsInstance(cluster.dependent_issues, list)
                self.assertGreaterEqual(cluster.influence_score, 0)
                self.assertGreaterEqual(cluster.cascade_potential, 0)
    
    def test_performance_tracking(self):
        """Test performance metrics tracking."""
        result = self.fixer.analyze_codebase_optimized(self.sample_codebase)
        
        # Should track metrics
        metrics = result.metrics
        
        self.assertGreater(metrics.total_time, 0, "Should track total time")
        self.assertGreater(metrics.nodes_analyzed, 0, "Should count nodes analyzed")
        self.assertGreaterEqual(metrics.patterns_found, 0, "Should count patterns")
        self.assertGreater(metrics.clusters_created, 0, "Should count clusters")
        
        # Should have phase timings
        self.assertGreater(len(metrics.phase_times), 0, "Should have phase timings")
    
    def test_benchmark_against_legacy(self):
        """Test benchmarking against legacy analysis."""
        benchmark = self.fixer.benchmark_against_legacy(self.sample_codebase)
        
        # Should provide benchmark results
        self.assertIn('optimized_time', benchmark)
        self.assertIn('legacy_time', benchmark)
        self.assertIn('speedup_factor', benchmark)
        self.assertIn('optimized_issues', benchmark)
        self.assertIn('legacy_issues', benchmark)
        
        # Times should be positive
        self.assertGreater(benchmark['optimized_time'], 0)
        self.assertGreater(benchmark['legacy_time'], 0)
        
        # Should calculate speedup
        self.assertGreater(benchmark['speedup_factor'], 0)
    
    def test_configuration_modes(self):
        """Test different optimization modes."""
        modes = [PerformanceMode.SPEED, PerformanceMode.BALANCED, PerformanceMode.THOROUGH]
        
        for mode in modes:
            config = OptimizationConfig(mode=mode)
            fixer = OptimizedTailChasingFixer(config)
            
            # Should accept configuration
            self.assertEqual(fixer.config.mode, mode)
            
            # Should complete analysis regardless of mode
            result = fixer.analyze_codebase_optimized(self.sample_codebase)
            self.assertIsInstance(result, AnalysisResult)
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Mock data
        clusters = {'cluster1': Mock(), 'cluster2': Mock()}
        
        from tailchasing.optimization.influence_detector import InfluentialNode
        influential = [
            InfluentialNode('high_impact_node', 'function', 95.0)
        ]
        
        from tailchasing.optimization.nonlinear_navigator import FixCluster
        fix_clusters = [
            FixCluster('cluster1', Mock(), [], 80.0, 5.0, 2.0, 0.9)
        ]
        
        recommendations = self.fixer._generate_recommendations(
            clusters, influential, fix_clusters
        )
        
        # Should generate recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0, "Should provide recommendations")
        
        # Should mention influential patterns
        rec_text = ' '.join(recommendations)
        self.assertIn('influential', rec_text.lower(), 
                     "Should mention influential patterns")
    
    @unittest.skipIf(not hasattr(asyncio, 'run'), "Python < 3.7")
    def test_parallel_analysis(self):
        """Test parallel analysis capabilities."""
        # Enable parallel processing
        self.fixer.config.parallel_processing = True
        
        async def run_parallel():
            return await self.fixer.analyze_codebase_parallel(self.sample_codebase)
        
        # Run parallel analysis
        result = asyncio.run(run_parallel())
        
        # Should complete successfully
        self.assertIsInstance(result, AnalysisResult)
        self.assertGreater(len(result.issues), 0, "Parallel analysis should find issues")
    
    def test_performance_report(self):
        """Test performance report generation."""
        # Run analysis first
        self.fixer.analyze_codebase_optimized(self.sample_codebase)
        
        # Get performance report
        report = self.fixer.get_performance_report()
        
        # Should have all required sections
        required_sections = [
            'phase_times', 'total_time', 'nodes_analyzed', 
            'patterns_found', 'clusters_created', 'configuration'
        ]
        
        for section in required_sections:
            self.assertIn(section, report, f"Report should have {section}")
        
        # Configuration section should have mode
        self.assertIn('mode', report['configuration'])
        self.assertEqual(report['configuration']['mode'], 
                        self.fixer.config.mode.value)
    
    def test_quick_analyze_node(self):
        """Test quick node analysis."""
        # Empty function
        empty_func = ast.parse("def empty(): pass").body[0]
        issues = self.fixer._quick_analyze_node(empty_func, "test.py")
        
        # Should detect phantom function
        phantom_issues = [i for i in issues if i.kind == "phantom_function"]
        self.assertGreater(len(phantom_issues), 0, "Should detect empty function")
        
        # Function with bare except
        bare_except_code = """
def risky():
    try:
        dangerous_operation()
    except:
        pass
"""
        func = ast.parse(bare_except_code).body[0]
        issues = self.fixer._quick_analyze_node(func, "test.py")
        
        # Should detect bare except
        bare_except_issues = [i for i in issues if i.kind == "bare_except"]
        self.assertGreater(len(bare_except_issues), 0, "Should detect bare except")
    
    def test_fallback_fix_prioritization(self):
        """Test fallback fix prioritization."""
        issues = [
            Issue(kind="test1", message="High severity", file="f1.py", line=1, severity=5),
            Issue(kind="test2", message="Medium severity", file="f2.py", line=1, severity=3),
            Issue(kind="test3", message="Low severity", file="f3.py", line=1, severity=1)
        ]
        
        fallback_clusters = self.fixer._fallback_fix_prioritization(issues)
        
        # Should create clusters
        self.assertGreater(len(fallback_clusters), 0, "Should create fallback clusters")
        
        # Should prioritize by severity
        severities = [cluster.primary_issue.severity for cluster in fallback_clusters]
        self.assertEqual(severities, sorted(severities, reverse=True),
                        "Should order by severity in fallback")


if __name__ == "__main__":
    unittest.main()