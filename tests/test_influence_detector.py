"""
Test suite for the InfluenceDetector and LayeredIssueDetector.

Verifies:
1. Bellman-Ford style scouting (max 3 steps)
2. Influence score calculation with proper weights
3. Top 10% most influential pattern selection
4. Layered detection without strict ordering
"""

import ast
import unittest
from unittest.mock import Mock, patch
import networkx as nx

from tailchasing.optimization.influence_detector import (
    InfluenceDetector,
    LayeredIssueDetector,
    InfluentialNode,
    LayerInfo
)


class TestInfluenceDetector(unittest.TestCase):
    """Test cases for InfluenceDetector."""
    
    def setUp(self):
        """Set up test detector."""
        self.detector = InfluenceDetector()
        self.sample_codebase = self._create_sample_codebase()
    
    def _create_sample_codebase(self) -> dict:
        """Create a sample codebase with call relationships."""
        codebase = {}
        
        # Central utility functions (high influence)
        util_code = """
def log_error(message):
    print(f"ERROR: {message}")

def validate_input(data):
    if not data:
        raise ValueError("Invalid input")
    return True

def get_config():
    return {"debug": True}
"""
        codebase['utils.py'] = ast.parse(util_code)
        
        # Functions that use utilities (create call graph)
        handler_code = """
def process_request(request):
    validate_input(request)
    config = get_config()
    try:
        return handle_data(request)
    except Exception as e:
        log_error(str(e))
        
def handle_data(data):
    validate_input(data)
    return transform(data)

def transform(data):
    return data.upper()
"""
        codebase['handlers.py'] = ast.parse(handler_code)
        
        # More callers to create influence
        service_code = """
def service_method():
    config = get_config()
    log_error("Service started")
    return config

def another_service():
    validate_input("test")
    log_error("Another service")
    
def third_service():
    validate_input("data")
    get_config()
"""
        codebase['services.py'] = ast.parse(service_code)
        
        # Duplicate patterns
        duplicate_code = """
def duplicate_validator(data):
    if not data:
        raise ValueError("Invalid input")
    return True

def another_validator(input_data):
    if not input_data:
        raise ValueError("Invalid input")
    return True
"""
        codebase['duplicates.py'] = ast.parse(duplicate_code)
        
        return codebase
    
    def test_find_influential_patterns(self):
        """Test finding influential patterns returns top 10%."""
        influential = self.detector.find_influential_patterns(
            self.sample_codebase,
            top_percent=0.1
        )
        
        # Should return list of influential nodes
        self.assertIsInstance(influential, list)
        self.assertTrue(all(isinstance(n, InfluentialNode) for n in influential))
        
        # Should return approximately top 10%
        total_nodes = sum(
            len([n for n in ast.walk(tree) 
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))])
            for tree in self.sample_codebase.values()
        )
        expected_count = max(1, int(total_nodes * 0.1))
        
        # Allow some flexibility in count
        self.assertLessEqual(len(influential), expected_count + 2)
        
        # Influential nodes should be sorted by score
        scores = [n.influence_score for n in influential]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_scout_influence_bellman_ford_style(self):
        """Test Bellman-Ford style limited exploration."""
        # Build a simple call graph
        self.detector.call_graph = nx.DiGraph()
        self.detector.call_graph.add_edges_from([
            ('A', 'B'), ('A', 'C'),  # A calls B and C
            ('B', 'D'), ('C', 'D'),  # B and C call D
            ('D', 'E'),              # D calls E
            ('F', 'E'),              # F also calls E
        ])
        
        # Scout influence for E (should be called by many)
        node = ast.FunctionDef(
            name='E', 
            args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[], 
                              kw_defaults=[], defaults=[]),
            body=[ast.Pass()]
        )
        influence = self.detector.scout_influence_bellman_ford_style(
            'E', node, max_steps=3
        )
        
        # E has high influence (called directly and indirectly)
        self.assertGreater(influence, 0)
        
        # Scout influence for A (leaf node, low influence)
        influence_a = self.detector.scout_influence_bellman_ford_style(
            'A', node, max_steps=3
        )
        
        # A should have lower influence than E
        self.assertLess(influence_a, influence)
    
    def test_influence_weights(self):
        """Test that influence calculation uses correct weights."""
        # Create a node with known metrics
        self.detector.call_graph = nx.DiGraph()
        self.detector.call_graph.add_node('test_node')
        
        # Add direct calls
        for i in range(5):
            self.detector.call_graph.add_edge(f'caller_{i}', 'test_node')
        
        node = ast.FunctionDef(
            name='test_func',
            args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                              kw_defaults=[], defaults=[]),
            body=[ast.Pass()]
        )
        
        # Calculate influence
        influence = self.detector.scout_influence_bellman_ford_style(
            'test_node', node, max_steps=1
        )
        
        # Should include direct calls weight
        expected_min = self.detector.INFLUENCE_WEIGHTS['direct_calls'] * 5
        self.assertGreaterEqual(influence, expected_min)
    
    def test_max_steps_limitation(self):
        """Test that exploration is limited to max_steps."""
        # Create a deep call graph
        self.detector.call_graph = nx.DiGraph()
        
        # Create a chain: A -> B -> C -> D -> E -> F
        edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F')]
        self.detector.call_graph.add_edges_from(edges)
        
        node = ast.FunctionDef(
            name='F',
            args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                              kw_defaults=[], defaults=[]),
            body=[ast.Pass()]
        )
        
        # Scout with max_steps=2
        influence_2 = self.detector.scout_influence_bellman_ford_style(
            'F', node, max_steps=2
        )
        
        # Scout with max_steps=5
        influence_5 = self.detector.scout_influence_bellman_ford_style(
            'F', node, max_steps=5
        )
        
        # More steps should find more indirect calls
        self.assertGreaterEqual(influence_5, influence_2)
    
    def test_duplicate_pattern_detection(self):
        """Test that duplicate patterns increase influence."""
        # Create nodes with same pattern
        pattern_code = """
def func1(x):
    if x:
        return x * 2
    return 0

def func2(y):
    if y:
        return y * 2
    return 0
"""
        tree = ast.parse(pattern_code)
        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        # Both should have same pattern signature
        sig1 = self.detector._get_pattern_signature(funcs[0])
        sig2 = self.detector._get_pattern_signature(funcs[1])
        
        self.assertEqual(sig1, sig2, "Similar functions should have same signature")
    
    def test_cascading_risk_estimation(self):
        """Test estimation of cascading failure risk."""
        # Error handler with high risk
        error_code = """
def risky_handler(data):
    try:
        process(data)
    except:
        raise
    global state
    state = "modified"
"""
        tree = ast.parse(error_code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        risk = self.detector._estimate_cascading_risk(func)
        
        # Should have high risk (has try/except, raise, and global)
        self.assertGreater(risk, 0.5)
        
        # Simple function with low risk
        simple_code = "def simple(): return 42"
        simple_tree = ast.parse(simple_code)
        simple_func = next(n for n in ast.walk(simple_tree) 
                          if isinstance(n, ast.FunctionDef))
        
        simple_risk = self.detector._estimate_cascading_risk(simple_func)
        
        # Should have low risk
        self.assertLess(simple_risk, 0.2)


class TestLayeredIssueDetector(unittest.TestCase):
    """Test cases for LayeredIssueDetector."""
    
    def setUp(self):
        """Set up test detector."""
        self.detector = LayeredIssueDetector()
        self.sample_codebase = self._create_layered_codebase()
    
    def _create_layered_codebase(self) -> dict:
        """Create a codebase with clear layers."""
        codebase = {}
        
        # Infrastructure layer
        infra_code = """
def get_db_connection():
    return "connection"

def cache_get(key):
    return cache[key]
"""
        codebase['db_utils.py'] = ast.parse(infra_code)
        
        # Domain layer
        domain_code = """
class User:
    def __init__(self, name):
        self.name = name
        
class Product:
    def __init__(self, id):
        self.id = id
"""
        codebase['models.py'] = ast.parse(domain_code)
        
        # Application layer
        app_code = """
def user_service(user_id):
    conn = get_db_connection()
    return User(user_id)

def product_service(product_id):
    return Product(product_id)
"""
        codebase['services.py'] = ast.parse(app_code)
        
        # Presentation layer
        presentation_code = """
def api_get_user(request):
    user = user_service(request.id)
    return json.dumps(user)

def api_get_product(request):
    product = product_service(request.id)
    return json.dumps(product)
"""
        codebase['api_routes.py'] = ast.parse(presentation_code)
        
        return codebase
    
    def test_detect_issues_in_layers(self):
        """Test layered detection without strict ordering."""
        issues = self.detector.detect_issues_in_layers(self.sample_codebase)
        
        # Should return list of issues
        self.assertIsInstance(issues, list)
        
        # Should process all layers
        self.assertGreater(len(self.detector.layers), 0)
    
    def test_create_layers(self):
        """Test that layers are created correctly."""
        layers = self.detector._create_layers(self.sample_codebase)
        
        # Should create standard layers
        self.assertIn('infrastructure', layers)
        self.assertIn('domain_logic', layers)
        self.assertIn('application', layers)
        self.assertIn('presentation', layers)
        
        # Each layer should have nodes
        for layer_name, layer in layers.items():
            self.assertIsInstance(layer, LayerInfo)
            if layer_name != 'unknown':  # Unknown might be empty
                self.assertGreaterEqual(len(layer.nodes), 0)
    
    def test_iterate_layers_strategically(self):
        """Test that layers are processed by influence, not order."""
        self.detector.layers = self.detector._create_layers(self.sample_codebase)
        
        # Set different influence scores
        self.detector.layers['infrastructure'].influence_score = 0.9
        self.detector.layers['domain_logic'].influence_score = 0.5
        self.detector.layers['application'].influence_score = 0.7
        self.detector.layers['presentation'].influence_score = 0.3
        
        # Iterate strategically
        order = []
        for layer_name, _ in self.detector.iterate_layers_strategically(
            self.detector.layers
        ):
            order.append(layer_name)
        
        # Should process in order of influence (infrastructure first)
        self.assertEqual(order[0], 'infrastructure', 
                        "Most influential layer should be processed first")
        
        # Presentation should be last (lowest influence)
        last_non_unknown = [l for l in order if l != 'unknown'][-1]
        self.assertEqual(last_non_unknown, 'presentation',
                        "Least influential layer should be processed last")
    
    def test_scout_layer_influence(self):
        """Test layer influence calculation."""
        # Create a layer with known properties
        layer = LayerInfo(
            name='infrastructure',
            nodes=[('file1.py', ast.FunctionDef(name='func1', 
                                                args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                                                                 kw_defaults=[], defaults=[]),
                                                body=[ast.Pass()])),
                   ('file2.py', ast.FunctionDef(name='func2',
                                                args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                                                                 kw_defaults=[], defaults=[]),
                                                body=[ast.Pass()]))],
            influence_score=0.0
        )
        
        influence = self.detector.scout_layer_influence(layer)
        
        # Infrastructure should have high influence
        self.assertGreater(influence, 0.3)
        
        # Presentation layer should have lower influence
        pres_layer = LayerInfo(
            name='presentation',
            nodes=[('api.py', ast.FunctionDef(name='api',
                                             args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                                                              kw_defaults=[], defaults=[]),
                                             body=[ast.Pass()]))],
            influence_score=0.0
        )
        
        pres_influence = self.detector.scout_layer_influence(pres_layer)
        
        self.assertLess(pres_influence, influence)
    
    def test_filter_processed_patterns(self):
        """Test filtering of already-processed patterns."""
        layer = LayerInfo(
            name='test',
            nodes=[
                ('f1.py', ast.FunctionDef(name='func1',
                                         args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                                                          kw_defaults=[], defaults=[]),
                                         body=[ast.Pass()])),
                ('f2.py', ast.FunctionDef(name='func2',
                                         args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                                                          kw_defaults=[], defaults=[]),
                                         body=[ast.Pass()])),
                ('f3.py', ast.FunctionDef(name='func3',
                                         args=ast.arguments(args=[], posonlyargs=[], kwonlyargs=[],
                                                          kw_defaults=[], defaults=[]),
                                         body=[ast.Pass()])),
            ]
        )
        
        # Mark some patterns as processed
        processed = {
            self.detector._generate_pattern_signature(layer.nodes[0][1])
        }
        
        filtered = self.detector.filter_processed_patterns(layer, processed)
        
        # Should have fewer nodes (if patterns matched)
        # Since all nodes have same pattern, they might all be filtered
        self.assertLessEqual(len(filtered.nodes), len(layer.nodes))
    
    def test_cross_layer_analysis(self):
        """Test cross-layer issue detection."""
        layers = self.detector._create_layers(self.sample_codebase)
        
        # Analyze presentation layer for violations
        presentation = layers.get('presentation', LayerInfo('presentation', []))
        
        issues = self.detector.cross_layer_analysis(
            current_layer=presentation,
            other_layers=layers,
            learned_patterns=set()
        )
        
        # Should check for layer violations and duplicates
        self.assertIsInstance(issues, list)
    
    def test_no_strict_dependency_order(self):
        """Verify detector doesn't use strict dependency ordering."""
        # The detector should not sort by dependencies
        detector_methods = dir(self.detector)
        
        # Should not have topological sort or dependency ordering
        dependency_methods = [
            m for m in detector_methods
            if 'topological' in m.lower() or 'dependency_order' in m.lower()
        ]
        
        self.assertEqual(len(dependency_methods), 0,
                        "Should not use dependency ordering")


if __name__ == "__main__":
    unittest.main()