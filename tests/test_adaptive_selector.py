"""
Test suite for the AdaptiveAlgorithmSelector.

Verifies:
1. Strategy selection based on complexity/familiarity/influence
2. FAST pattern matching performance
3. DEEP semantic analysis with limited iterations
4. HYBRID scout-and-guide functionality
5. Performance metrics tracking
"""

import ast
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List, Dict, Any

from tailchasing.optimization.adaptive_selector import (
    AdaptiveAlgorithmSelector,
    AnalysisStrategy,
    AnalysisResult,
    PatternCache
)


class TestAdaptiveAlgorithmSelector(unittest.TestCase):
    """Test cases for AdaptiveAlgorithmSelector."""
    
    def setUp(self):
        """Set up test selector."""
        self.selector = AdaptiveAlgorithmSelector(enable_caching=True)
        self.sample_code = self._create_sample_code()
    
    def _create_sample_code(self) -> List[ast.AST]:
        """Create sample AST nodes for testing."""
        code_samples = [
            # Simple function (low complexity)
            """
def simple_function(x):
    return x * 2
""",
            # Complex function (high complexity)
            """
def complex_function(data):
    if data:
        for item in data:
            if item > 0:
                try:
                    while item > 10:
                        item = process(item)
                except Exception as e:
                    if e.critical:
                        raise
                    else:
                        continue
    return data
""",
            # Getter/setter patterns
            """
def get_user_name(user):
    return user.name

def set_user_name(user, name):
    user.name = name
""",
            # Empty function
            """
def empty_function():
    pass
""",
            # Class definition
            """
class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, item):
        return item * 2
"""
        ]
        
        nodes = []
        for code in code_samples:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    nodes.append(node)
        
        return nodes
    
    def test_strategy_selection_by_complexity(self):
        """Test that high complexity triggers DEEP strategy."""
        # Create a highly complex function with many decision points
        complex_code = """
def very_complex(a, b, c, d, e, f, g):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            if g > 0:
                                while a > 0:
                                    for i in range(10):
                                        try:
                                            if i % 2:
                                                if i > 5:
                                                    a -= 1
                                                else:
                                                    b += 1
                                            else:
                                                if b > 10:
                                                    c += 1
                                                elif c > 5:
                                                    d += 1
                                        except Exception as ex:
                                            if ex.code == 1:
                                                pass
                                            elif ex.code == 2:
                                                break
                                        with open('file') as f:
                                            data = f.read()
                                        assert data is not None
    elif a < 0:
        while b > 0:
            for j in range(5):
                if j % 3:
                    b -= 1
    else:
        assert a == 0
    return a + b
"""
        tree = ast.parse(complex_code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        # Calculate complexity manually to verify it's above threshold
        complexity = self.selector._calculate_complexity([func])
        self.assertGreater(complexity, self.selector.COMPLEXITY_THRESHOLD,
                          "Test function should have high complexity")
        
        strategy = self.selector._select_strategy([func], None)
        
        # Should select DEEP for high complexity
        self.assertEqual(strategy, AnalysisStrategy.DEEP,
                        f"High complexity ({complexity}) should trigger DEEP strategy")
    
    def test_strategy_selection_by_familiarity(self):
        """Test that familiar patterns trigger FAST strategy."""
        # Create a simple function
        simple_code = "def simple(): return 42"
        tree = ast.parse(simple_code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        # Mark pattern as familiar
        pattern_sig = self.selector._generate_pattern_signature(func)
        self.selector.pattern_cache[pattern_sig] = PatternCache(
            pattern_signature=pattern_sig,
            occurrence_count=5,  # Above threshold
            last_seen=time.time()
        )
        
        strategy = self.selector._select_strategy([func], None)
        
        # Should select FAST for familiar patterns
        self.assertEqual(strategy, AnalysisStrategy.FAST,
                        "Familiar patterns should trigger FAST strategy")
    
    def test_strategy_selection_by_influence(self):
        """Test that high influence triggers HYBRID strategy."""
        # Create a function with moderate complexity
        code = """
def important_function(x):
    if x > 0:
        return process(x)
    return None
"""
        tree = ast.parse(code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        # Mock influence detector to return high influence
        with patch.object(self.selector.influence_detector, 
                         'scout_influence_bellman_ford_style',
                         return_value=80):  # High influence
            strategy = self.selector._select_strategy([func], None)
        
        # Should select HYBRID for high influence
        self.assertEqual(strategy, AnalysisStrategy.HYBRID,
                        "High influence should trigger HYBRID strategy")
    
    def test_fast_pattern_matching(self):
        """Test FAST strategy pattern detection."""
        # Create various patterns
        patterns_code = """
def empty_func():
    pass

def get_value():
    return self.value

def set_value(val):
    self.value = val

class EmptyClass:
    pass
"""
        tree = ast.parse(patterns_code)
        nodes = [n for n in ast.walk(tree) 
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        
        result = self.selector._fast_pattern_matching(nodes)
        
        # Should detect patterns
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.strategy_used, AnalysisStrategy.FAST)
        self.assertGreater(len(result.patterns_found), 0,
                          "Should detect at least some patterns")
        
        # Check pattern types detected
        pattern_types = {p['type'] for p in result.patterns_found}
        self.assertIn('empty_function', pattern_types)
        self.assertIn('getter_setter', pattern_types)
        self.assertIn('empty_class', pattern_types)
    
    def test_deep_semantic_analysis_limited_iterations(self):
        """Test DEEP strategy with limited iterations."""
        # Mock semantic encoding
        with patch('tailchasing.optimization.adaptive_selector.encode_function') as mock_encode:
            mock_encode.return_value = np.random.rand(100)
            
            result = self.selector._deep_semantic_analysis(
                self.sample_code,
                max_iterations=3
            )
            
            # Should limit iterations
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(result.strategy_used, AnalysisStrategy.DEEP)
            self.assertLessEqual(result.samples_analyzed, 3 * 3,
                               "Should limit samples analyzed")
            
            # Should have high confidence
            self.assertGreaterEqual(result.confidence, 0.9)
    
    def test_hybrid_scout_and_guide(self):
        """Test HYBRID strategy scout-and-guide functionality."""
        # Mock influence detector
        with patch.object(self.selector.influence_detector,
                         'scout_influence_bellman_ford_style',
                         return_value=75):
            
            # Mock semantic encoding
            with patch('tailchasing.optimization.adaptive_selector.encode_function') as mock_encode:
                mock_encode.return_value = np.array([0.5, 0.5, 0.5])
                
                # Scout with deep analysis
                influential = self.selector.scout_with_deep_analysis(
                    self.sample_code,
                    scout_depth=2
                )
                
                # Should find influential patterns
                self.assertIsInstance(influential, list)
                
                # Apply guided analysis
                result = self.selector._guided_fast_analysis(
                    self.sample_code,
                    guide_patterns=influential
                )
                
                # Should use HYBRID strategy
                self.assertEqual(result.strategy_used, AnalysisStrategy.HYBRID)
                self.assertGreaterEqual(result.confidence, 0.85)
    
    def test_scout_sample_selection(self):
        """Test representative sample selection for scouting."""
        # Create diverse code
        large_code = []
        for i in range(100):
            if i % 3 == 0:
                code = f"def func_{i}(): pass"
            elif i % 3 == 1:
                code = f"class Class_{i}: pass"
            else:
                code = f"async def async_{i}(): pass"
            
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    large_code.append(node)
        
        # Select samples
        samples = self.selector._select_representative_samples(
            large_code,
            sample_size=10
        )
        
        # Should select diverse samples (allow some flexibility)
        self.assertLessEqual(len(samples), 10)
        self.assertGreaterEqual(len(samples), 8)
        
        # Check diversity
        sample_types = set(type(s).__name__ for s in samples)
        self.assertGreater(len(sample_types), 1,
                          "Should select diverse node types")
    
    def test_pattern_caching(self):
        """Test pattern familiarity caching."""
        # Create a function
        code = "def test_func(x): return x * 2"
        tree = ast.parse(code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        # Initial check - not familiar
        is_familiar = self.selector._is_familiar_pattern([func])
        self.assertFalse(is_familiar)
        
        # Update cache
        pattern_sig = self.selector._generate_pattern_signature(func)
        for _ in range(5):  # Make it familiar
            self.selector._update_pattern_cache(pattern_sig, AnalysisStrategy.FAST)
        
        # Should now be familiar
        is_familiar = self.selector._is_familiar_pattern([func])
        self.assertTrue(is_familiar)
    
    def test_complexity_calculation(self):
        """Test cyclomatic complexity calculation."""
        # Simple function
        simple_code = "def simple(): return 1"
        simple_tree = ast.parse(simple_code)
        simple_func = next(n for n in ast.walk(simple_tree) 
                          if isinstance(n, ast.FunctionDef))
        
        simple_complexity = self.selector._calculate_complexity([simple_func])
        self.assertEqual(simple_complexity, 1, "Simple function should have complexity 1")
        
        # Complex function
        complex_code = """
def complex(x):
    if x > 0:
        while x > 10:
            try:
                if x % 2:
                    x -= 1
                else:
                    x -= 2
            except:
                break
    elif x < 0:
        assert x != -1
    return x
"""
        complex_tree = ast.parse(complex_code)
        complex_func = next(n for n in ast.walk(complex_tree)
                           if isinstance(n, ast.FunctionDef))
        
        complex_complexity = self.selector._calculate_complexity([complex_func])
        self.assertGreater(complex_complexity, 5,
                          "Complex function should have high complexity")
    
    def test_influence_calculation(self):
        """Test influence score calculation."""
        func = self.sample_code[0]
        
        # Mock influence detector
        with patch.object(self.selector.influence_detector,
                         'scout_influence_bellman_ford_style',
                         return_value=50):
            
            # Without context
            influence = self.selector._calculate_influence([func], None)
            self.assertGreaterEqual(influence, 0)
            self.assertLessEqual(influence, 1)
            
            # With public API context
            context = {'is_public_api': True}
            public_influence = self.selector._calculate_influence([func], context)
            self.assertGreater(public_influence, influence,
                              "Public API should increase influence")
            
            # With tests context
            context = {'has_tests': True}
            tested_influence = self.selector._calculate_influence([func], context)
            self.assertLess(tested_influence, influence,
                           "Having tests should decrease influence")
    
    def test_performance_metrics_tracking(self):
        """Test that performance metrics are tracked correctly."""
        # Run analyses with different strategies
        result1 = self.selector._fast_pattern_matching(self.sample_code[:2])
        self.selector._update_metrics(AnalysisStrategy.FAST, 0.01, 2)
        
        result2 = self.selector._fast_pattern_matching(self.sample_code[2:4])
        self.selector._update_metrics(AnalysisStrategy.FAST, 0.02, 1)
        
        # Get performance summary
        summary = self.selector.get_performance_summary()
        
        self.assertIn('strategies', summary)
        self.assertIn('fast', summary['strategies'])
        
        fast_metrics = summary['strategies']['fast']
        self.assertEqual(fast_metrics['uses'], 2)
        self.assertEqual(fast_metrics['total_patterns'], 3)
        self.assertAlmostEqual(fast_metrics['avg_time'], 0.015, places=3)
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation based on code statistics."""
        # High complexity
        stats = {'cyclomatic_complexity': 15}
        recommendation = self.selector.recommend_strategy(stats)
        self.assertEqual(recommendation, AnalysisStrategy.DEEP)
        
        # Many duplicates
        stats = {'duplicate_ratio': 0.4, 'cyclomatic_complexity': 5}
        recommendation = self.selector.recommend_strategy(stats)
        self.assertEqual(recommendation, AnalysisStrategy.HYBRID)
        
        # Small codebase
        stats = {'total_functions': 30, 'cyclomatic_complexity': 5}
        recommendation = self.selector.recommend_strategy(stats)
        self.assertEqual(recommendation, AnalysisStrategy.DEEP)
        
        # Large critical codebase
        stats = {'total_functions': 1000, 'is_critical_path': True}
        recommendation = self.selector.recommend_strategy(stats)
        self.assertEqual(recommendation, AnalysisStrategy.HYBRID)
        
        # Large non-critical codebase
        stats = {'total_functions': 1000, 'is_critical_path': False}
        recommendation = self.selector.recommend_strategy(stats)
        self.assertEqual(recommendation, AnalysisStrategy.FAST)
    
    def test_adaptive_analysis_complete_flow(self):
        """Test the complete adaptive analysis flow."""
        # Test with different code sections
        with patch.object(self.selector, '_select_strategy') as mock_select:
            # Test FAST strategy
            mock_select.return_value = AnalysisStrategy.FAST
            result = self.selector.analyze_with_adaptive_algorithms(
                self.sample_code[0]
            )
            
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(result.strategy_used, AnalysisStrategy.FAST)
            self.assertGreater(result.execution_time, 0)
            
            # Test DEEP strategy
            mock_select.return_value = AnalysisStrategy.DEEP
            with patch('tailchasing.optimization.adaptive_selector.encode_function') as mock_encode:
                mock_encode.return_value = np.random.rand(100)
                
                result = self.selector.analyze_with_adaptive_algorithms(
                    self.sample_code[1]
                )
                
                self.assertEqual(result.strategy_used, AnalysisStrategy.DEEP)
            
            # Test HYBRID strategy
            mock_select.return_value = AnalysisStrategy.HYBRID
            with patch.object(self.selector.influence_detector,
                            'scout_influence_bellman_ford_style',
                            return_value=75):
                with patch('tailchasing.optimization.adaptive_selector.encode_function') as mock_encode:
                    # Ensure all vectors have the same dimension
                    mock_encode.return_value = np.random.rand(100)
                    
                    result = self.selector.analyze_with_adaptive_algorithms(
                        self.sample_code
                    )
                    
                    self.assertEqual(result.strategy_used, AnalysisStrategy.HYBRID)
    
    def test_pattern_signature_generation(self):
        """Test pattern signature generation for different node types."""
        # Function with various characteristics
        func_code = """
def example_func(a, b, c):
    if a > 0:
        for i in range(10):
            try:
                return process(i)
            except:
                pass
    return None
"""
        tree = ast.parse(func_code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        signature = self.selector._generate_pattern_signature(func)
        
        # Should include key characteristics
        self.assertIn('FunctionDef', signature)
        self.assertIn('args:3', signature)
        self.assertIn('body:', signature)
        self.assertIn('loop', signature)
        self.assertIn('conditional', signature)
        self.assertIn('exception', signature)
        
        # Getter function
        getter_code = "def get_value(): return self.value"
        getter_tree = ast.parse(getter_code)
        getter = next(n for n in ast.walk(getter_tree) if isinstance(n, ast.FunctionDef))
        
        getter_sig = self.selector._generate_pattern_signature(getter)
        self.assertIn('getter', getter_sig)
    
    def test_semantic_similarity_calculation(self):
        """Test vector similarity calculation."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        v3 = np.array([0.0, 1.0, 0.0])
        
        # Same vectors - similarity 1.0
        sim_same = self.selector._calculate_similarity(v1, v2)
        self.assertAlmostEqual(sim_same, 1.0, places=5)
        
        # Orthogonal vectors - similarity 0.0
        sim_ortho = self.selector._calculate_similarity(v1, v3)
        self.assertAlmostEqual(sim_ortho, 0.0, places=5)
        
        # Zero vector edge case
        v_zero = np.array([0.0, 0.0, 0.0])
        sim_zero = self.selector._calculate_similarity(v1, v_zero)
        self.assertEqual(sim_zero, 0.0)
    
    def test_guided_analysis_with_patterns(self):
        """Test guided analysis using scout patterns."""
        # Create guide patterns
        guide_patterns = [
            {
                'type': 'influential_pattern',
                'influence_score': 90,
                'vector': np.random.rand(100),  # Consistent dimension
                'signature': 'FunctionDef:getter:args:0:body:1',
                'node': self.sample_code[0]
            }
        ]
        
        # Mock vector computation
        with patch.object(self.selector, '_get_or_compute_vector') as mock_compute:
            mock_compute.return_value = np.random.rand(100)  # Same dimension
            
            result = self.selector._guided_fast_analysis(
                self.sample_code,
                guide_patterns
            )
            
            self.assertEqual(result.strategy_used, AnalysisStrategy.HYBRID)
            self.assertIn('guide_patterns', result.metrics)
            self.assertEqual(result.metrics['guide_patterns'], 1)
    
    def test_caching_disabled(self):
        """Test behavior when caching is disabled."""
        selector_no_cache = AdaptiveAlgorithmSelector(enable_caching=False)
        
        func = self.sample_code[0]
        
        # Should not be familiar even after multiple checks
        for _ in range(5):
            is_familiar = selector_no_cache._is_familiar_pattern([func])
            self.assertFalse(is_familiar)
        
        # Cache should remain empty
        self.assertEqual(len(selector_no_cache.pattern_cache), 0)


if __name__ == "__main__":
    unittest.main()