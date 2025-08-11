"""
Test suite for NonLinearNavigator and InfluenceBasedFixPrioritizer.

Verifies:
1. Non-linear navigation breaks traditional import chain traversal
2. Priority queue based on influence, not dependency distance
3. Multiple navigation strategies (influence-first, cluster-hopping, cascade-aware, hybrid)
4. Fix prioritization by cascade potential rather than severity
5. Issue intersection detection for major thoroughfares
"""

import ast
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
import networkx as nx
from typing import Dict, List

from tailchasing.optimization.nonlinear_navigator import (
    NonLinearNavigator,
    InfluenceBasedFixPrioritizer,
    NavigationStrategy,
    NavigationNode,
    FixCluster
)
from tailchasing.core.issues import Issue


class TestNonLinearNavigator(unittest.TestCase):
    """Test cases for NonLinearNavigator."""
    
    def setUp(self):
        """Set up test navigator."""
        self.navigator = NonLinearNavigator()
        self.sample_codebase = self._create_sample_codebase()
    
    def _create_sample_codebase(self) -> Dict[str, ast.AST]:
        """Create a sample codebase for testing navigation."""
        codebase = {}
        
        # High-influence utility functions
        utils_code = """
def log_error(message, level='ERROR'):
    print(f"[{level}] {message}")

def validate_input(data):
    if not data:
        raise ValueError("Invalid input")
    return True

def get_config():
    return {"debug": True, "version": "1.0"}

def parse_json_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Failed to parse config: {e}")
        return {}
"""
        codebase['utils.py'] = ast.parse(utils_code)
        
        # Medium-influence business logic
        business_code = """
class UserManager:
    def __init__(self):
        self.config = get_config()
    
    def create_user(self, user_data):
        validate_input(user_data)
        try:
            return self._save_user(user_data)
        except Exception as e:
            log_error(f"User creation failed: {e}")
            raise
    
    def _save_user(self, data):
        return {"id": 123, "data": data}

def process_user_request(request):
    manager = UserManager()
    return manager.create_user(request)
"""
        codebase['business.py'] = ast.parse(business_code)
        
        # Low-influence presentation layer
        presentation_code = """
def api_endpoint(request):
    try:
        result = process_user_request(request.json)
        return {"status": "ok", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def health_check():
    return {"status": "healthy"}

def get_version():
    config = get_config()
    return config.get("version", "unknown")
"""
        codebase['api.py'] = ast.parse(presentation_code)
        
        # Complex error-prone code
        complex_code = """
def complex_processor(data):
    if data:
        if isinstance(data, dict):
            if 'items' in data:
                if data['items']:
                    for item in data['items']:
                        if item.get('process', True):
                            try:
                                while item['count'] > 0:
                                    result = process_item(item)
                                    if result:
                                        item['count'] -= 1
                                    else:
                                        break
                            except Exception as e:
                                if e.code == 'RETRY':
                                    continue
                                elif e.code == 'FATAL':
                                    raise
                            finally:
                                cleanup()
    return data

def process_item(item):
    global state
    state = "processing"
    return item.get('value', 0) * 2
"""
        codebase['complex.py'] = ast.parse(complex_code)
        
        # Template/pattern code
        template_code = """
class BaseHandler:
    def handle(self, request):
        raise NotImplementedError("Subclasses must implement")
    
    def validate(self, data):
        pass
    
    def transform(self, data):
        return data

class TemplateProcessor:
    def __init__(self):
        pass
    
    def process_template(self, template):
        ...
    
    def render(self, context):
        ...
"""
        codebase['templates.py'] = ast.parse(template_code)
        
        return codebase
    
    def test_non_linear_navigation_breaks_import_chains(self):
        """Test that navigation doesn't follow linear import chains."""
        # Traditional would go: api.py -> business.py -> utils.py
        # Non-linear should jump to high-influence nodes first
        
        plan = self.navigator.navigate_by_influence(
            self.sample_codebase,
            NavigationStrategy.INFLUENCE_FIRST
        )
        
        # Should have ordered nodes
        self.assertGreater(len(plan.ordered_nodes), 0)
        
        # First few nodes should be high-influence (likely from utils.py)
        first_nodes = plan.ordered_nodes[:3]
        high_influence_files = [node.file_path for node in first_nodes]
        
        # Utils should appear early (high influence functions)
        self.assertIn('utils.py', high_influence_files,
                     "High-influence utils should be processed first")
        
        # Verify we didn't just follow alphabetical or import order
        file_order = [node.file_path for node in plan.ordered_nodes]
        self.assertNotEqual(file_order, sorted(file_order),
                           "Should not follow alphabetical order")
    
    def test_priority_queue_based_on_influence_not_distance(self):
        """Test that priority is based on influence, not dependency distance."""
        # Mock influence detector to return varying scores
        with patch.object(self.navigator.influence_detector, 
                         'scout_influence_bellman_ford_style') as mock_scout:
            # Return different influence scores for different nodes
            def mock_influence(node_id, node, max_steps):
                if 'log_error' in node_id:
                    return 85.0  # High influence utility
                elif 'validate_input' in node_id:
                    return 70.0
                elif 'get_config' in node_id:
                    return 60.0
                elif 'complex_processor' in node_id:
                    return 45.0
                elif 'api_endpoint' in node_id:
                    return 20.0
                else:
                    return 10.0
            
            mock_scout.side_effect = mock_influence
            
            # Build the priority queue
            self.navigator._build_influence_queue(self.sample_codebase)
            
            # Extract all nodes and check ordering
            all_nodes = []
            temp_queue = self.navigator.priority_queue.copy()
            while temp_queue:
                node = temp_queue.pop(0)
                all_nodes.append(node)
            
            # Verify influence affects priority
            influences = [node.influence_score for node in all_nodes]
            self.assertGreater(max(influences), min(influences),
                              "Should have varying influence scores")
            
            # High influence nodes should generally have higher priority
            high_influence_nodes = [n for n in all_nodes if n.influence_score > 50]
            low_influence_nodes = [n for n in all_nodes if n.influence_score < 30]
            
            if high_influence_nodes and low_influence_nodes:
                avg_high_priority = sum(n.priority for n in high_influence_nodes) / len(high_influence_nodes)
                avg_low_priority = sum(n.priority for n in low_influence_nodes) / len(low_influence_nodes)
                
                self.assertGreater(avg_high_priority, avg_low_priority,
                                  "High influence nodes should have higher average priority")
    
    def test_influence_first_strategy(self):
        """Test INFLUENCE_FIRST navigation strategy."""
        # Mock influence detector for consistent results
        with patch.object(self.navigator.influence_detector, 
                         'scout_influence_bellman_ford_style') as mock_scout:
            def mock_influence(node_id, node, max_steps):
                if 'log_error' in node_id:
                    return 85.0
                elif 'validate_input' in node_id:
                    return 70.0
                elif 'get_config' in node_id:
                    return 60.0
                elif 'complex_processor' in node_id:
                    return 45.0
                else:
                    return 25.0
            
            mock_scout.side_effect = mock_influence
            
            plan = self.navigator.navigate_by_influence(
                self.sample_codebase,
                NavigationStrategy.INFLUENCE_FIRST
            )
            
            # Should process highest influence nodes first
            influence_scores = [node.influence_score for node in plan.ordered_nodes]
            
            # First nodes should have higher influence than later ones
            if len(influence_scores) >= 6:
                first_half_avg = sum(influence_scores[:3]) / 3
                last_half_avg = sum(influence_scores[-3:]) / 3
                
                self.assertGreater(first_half_avg, last_half_avg,
                                  "Should process high-influence nodes first")
    
    def test_cluster_hopping_strategy(self):
        """Test CLUSTER_HOPPING navigation strategy."""
        plan = self.navigator.navigate_by_influence(
            self.sample_codebase,
            NavigationStrategy.CLUSTER_HOPPING
        )
        
        # Should visit different clusters
        visited_clusters = set()
        for node in plan.ordered_nodes:
            node_clusters = self.navigator._get_node_clusters(node.node_id, node.ast_node)
            visited_clusters.update(node_clusters)
        
        # Should have visited multiple distinct clusters
        self.assertGreater(len(visited_clusters), 2,
                          "Should hop between different clusters")
        
        # Cluster order should be recorded
        self.assertGreater(len(plan.cluster_order), 0,
                          "Should track cluster visitation order")
    
    def test_cascade_aware_strategy(self):
        """Test CASCADE_AWARE navigation strategy."""
        plan = self.navigator.navigate_by_influence(
            self.sample_codebase,
            NavigationStrategy.CASCADE_AWARE
        )
        
        # Should prioritize nodes with high cascade potential
        # (This would be nodes that many others depend on)
        self.assertGreater(len(plan.ordered_nodes), 0)
        
        # First node should have some cascade potential
        first_node = plan.ordered_nodes[0]
        cascade_graph = self.navigator._build_cascade_graph()
        cascade_score = self.navigator._calculate_cascade_potential(first_node, cascade_graph)
        
        self.assertGreaterEqual(cascade_score, 0,
                               "First node should have cascade potential")
    
    def test_hybrid_search_strategy(self):
        """Test HYBRID_SEARCH navigation strategy."""
        plan = self.navigator.navigate_by_influence(
            self.sample_codebase,
            NavigationStrategy.HYBRID_SEARCH
        )
        
        # Should combine multiple approaches
        self.assertGreater(len(plan.ordered_nodes), 0)
        
        # Should have good influence coverage
        self.assertGreater(plan.influence_coverage, 0.5,
                          "Hybrid should achieve good influence coverage")
    
    def test_priority_calculation_boosts_and_penalties(self):
        """Test priority calculation with various boosts and penalties."""
        # Create a function with multiple cluster memberships
        multi_cluster_code = """
def validate_and_log_error(data):
    if not data:
        log_error("Validation failed")
        raise ValueError("Invalid data")
    return True
"""
        tree = ast.parse(multi_cluster_code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        base_influence = 50.0
        priority = self.navigator.calculate_influence_priority(
            "test:validate_and_log_error",
            "test.py",
            func,
            base_influence
        )
        
        # Should apply boosts for multi-cluster membership and error correlation
        self.assertGreater(priority, base_influence,
                          "Priority should be boosted for multi-cluster node")
        
        # Test penalty for similar pattern
        self.navigator.visited_patterns.add("FunctionDef:args:1:body:3:conditional:exception")
        penalty_priority = self.navigator.calculate_influence_priority(
            "test2:similar_function",
            "test2.py", 
            func,
            base_influence
        )
        
        self.assertLess(penalty_priority, base_influence,
                       "Should apply penalty for similar patterns")
    
    def test_cluster_detection(self):
        """Test detection of node cluster memberships."""
        # Create functions with different patterns
        getter_code = "def get_user_data(id): return database.query(id)"
        setter_code = "def set_user_status(id, status): database.update(id, status)"
        validator_code = "def validate_email(email): return '@' in email"
        error_handler_code = "def handle_error(e): log_error(str(e))"
        
        for code, expected_cluster in [
            (getter_code, 'data_retrieval'),
            (setter_code, 'data_modification'),  
            (validator_code, 'validation'),
            (error_handler_code, 'error_handling')
        ]:
            tree = ast.parse(code)
            func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
            
            clusters = self.navigator._get_node_clusters(f"test:{func.name}", func)
            self.assertIn(expected_cluster, clusters,
                         f"Function {func.name} should be in {expected_cluster} cluster")
    
    def test_error_correlation_calculation(self):
        """Test calculation of error correlation scores."""
        # High error correlation code
        error_prone_code = """
def risky_function(data):
    global state
    try:
        if data:
            if data.items:
                for item in data.items:
                    if item.valid:
                        process(item)
    except:
        pass
"""
        tree = ast.parse(error_prone_code)
        func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
        
        error_correlation = self.navigator._calculate_error_correlation(func)
        
        self.assertGreater(error_correlation, 0.3,
                          "Error-prone code should have high error correlation")
        
        # Low error correlation code
        simple_code = "def simple_add(a, b): return a + b"
        simple_tree = ast.parse(simple_code)
        simple_func = next(n for n in ast.walk(simple_tree) if isinstance(n, ast.FunctionDef))
        
        simple_correlation = self.navigator._calculate_error_correlation(simple_func)
        
        self.assertLess(simple_correlation, error_correlation,
                       "Simple code should have lower error correlation")
    
    def test_pattern_template_detection(self):
        """Test detection of pattern templates."""
        # Abstract base class
        template_code = """
class BaseProcessor:
    def process(self, data):
        raise NotImplementedError()
    
    def validate(self, data):
        pass
    
    def cleanup(self):
        ...
"""
        tree = ast.parse(template_code)
        class_node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
        
        is_template = self.navigator._is_pattern_template(class_node)
        self.assertTrue(is_template, "Abstract base class should be detected as template")
        
        # Regular class
        regular_code = """
class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        self.data.append(item)
"""
        regular_tree = ast.parse(regular_code)
        regular_class = next(n for n in ast.walk(regular_tree) if isinstance(n, ast.ClassDef))
        
        is_regular_template = self.navigator._is_pattern_template(regular_class)
        self.assertFalse(is_regular_template, "Regular class should not be template")
    
    def test_complexity_estimation(self):
        """Test cyclomatic complexity estimation."""
        # Simple function
        simple_code = "def simple(): return 42"
        simple_tree = ast.parse(simple_code)
        simple_func = next(n for n in ast.walk(simple_tree) if isinstance(n, ast.FunctionDef))
        
        simple_complexity = self.navigator._estimate_complexity(simple_func)
        self.assertEqual(simple_complexity, 1, "Simple function should have complexity 1")
        
        # Complex function
        complex_code = """
def complex(x):
    if x > 0:
        if x < 10:
            for i in range(x):
                try:
                    if i % 2:
                        return i
                except:
                    continue
            while x > 5:
                x -= 1
    return x
"""
        complex_tree = ast.parse(complex_code)
        complex_func = next(n for n in ast.walk(complex_tree) if isinstance(n, ast.FunctionDef))
        
        complex_complexity = self.navigator._estimate_complexity(complex_func)
        self.assertGreater(complex_complexity, 5,
                          "Complex function should have high complexity")
    
    def test_navigation_plan_creation(self):
        """Test creation of comprehensive navigation plan."""
        plan = self.navigator.navigate_by_influence(self.sample_codebase)
        
        # Should have all required fields
        self.assertIsNotNone(plan.ordered_nodes)
        self.assertIsNotNone(plan.skip_reasons)
        self.assertIsNotNone(plan.cluster_order)
        self.assertIsNotNone(plan.estimated_time)
        self.assertIsNotNone(plan.influence_coverage)
        
        # Should have processed multiple nodes
        self.assertGreater(len(plan.ordered_nodes), 5)
        
        # Should have reasonable influence coverage
        self.assertGreater(plan.influence_coverage, 0.0)
        self.assertLessEqual(plan.influence_coverage, 1.0)
        
        # Should have visited multiple clusters
        self.assertGreater(len(plan.cluster_order), 1)


class TestInfluenceBasedFixPrioritizer(unittest.TestCase):
    """Test cases for InfluenceBasedFixPrioritizer."""
    
    def setUp(self):
        """Set up test prioritizer."""
        self.prioritizer = InfluenceBasedFixPrioritizer()
        self.sample_issues = self._create_sample_issues()
    
    def _create_sample_issues(self) -> List[Issue]:
        """Create sample issues for testing."""
        issues = []
        
        # High-influence utility issue (affects many)
        issues.append(Issue(
            kind="missing_function",
            message="Function 'log_error' not found",
            file="utils.py",
            line=10,
            severity=4,
            confidence=0.9
        ))
        
        # Cascade-causing import issue
        issues.append(Issue(
            kind="circular_import",
            message="Circular import detected",
            file="business.py",
            line=1,
            severity=5,
            confidence=0.95
        ))
        
        # Multiple duplicate issues (pattern)
        for i in range(3):
            issues.append(Issue(
                kind="duplicate_function",
                message=f"Duplicate function pattern in file_{i}.py",
                file=f"file_{i}.py",
                line=20 + i,
                severity=2,
                confidence=0.8
            ))
        
        # Error handling issues (related)
        issues.append(Issue(
            kind="bare_except",
            message="Bare except clause",
            file="error_handler.py",
            line=15,
            severity=3,
            confidence=0.7
        ))
        
        issues.append(Issue(
            kind="exception_not_handled",
            message="Exception not properly handled",
            file="error_handler.py",
            line=25,
            severity=3,
            confidence=0.8
        ))
        
        # Independent low-impact issue
        issues.append(Issue(
            kind="unused_variable",
            message="Variable 'x' is unused",
            file="isolated.py", 
            line=5,
            severity=1,
            confidence=0.6
        ))
        
        return issues
    
    def test_prioritize_fixes_by_influence_not_severity(self):
        """Test that fixes are prioritized by influence, not just severity."""
        fix_clusters = self.prioritizer.prioritize_fixes_by_influence(self.sample_issues)
        
        # Should create fix clusters
        self.assertGreater(len(fix_clusters), 0, "Should create fix clusters")
        
        # First cluster should be influential, not necessarily highest severity
        first_cluster = fix_clusters[0]
        
        # Should have high cascade potential or influence score
        self.assertTrue(
            first_cluster.cascade_potential > 2 or first_cluster.influence_score > 5,
            "First cluster should have high influence/cascade potential"
        )
        
        # Test passes if we have clusters and they're prioritizing influence
        if len(fix_clusters) > 1:
            # Verify it's not just sorted by severity
            cluster_severities = [cluster.primary_issue.severity for cluster in fix_clusters]
            issue_severities = sorted([issue.severity for issue in self.sample_issues], reverse=True)
            
            self.assertNotEqual(cluster_severities, issue_severities[:len(cluster_severities)],
                               "Should not be ordered purely by severity")
        else:
            # If only one cluster, verify it has influence/cascade metrics
            if fix_clusters:
                cluster = fix_clusters[0]
                self.assertGreater(cluster.influence_score + cluster.cascade_potential, 0,
                                 "Single cluster should have influence or cascade potential")
    
    def test_build_issue_influence_graph(self):
        """Test building of issue influence graph."""
        graph = self.prioritizer.build_issue_influence_graph(self.sample_issues)
        
        # Should create nodes for all issues
        self.assertEqual(len(graph.nodes()), len(self.sample_issues),
                        "Should create node for each issue")
        
        # Should create edges between related issues
        self.assertGreater(len(graph.edges()), 0,
                          "Should create edges between related issues")
        
        # Check that related issues are connected
        error_handler_nodes = [n for n in graph.nodes() 
                              if 'error_handler.py' in n]
        self.assertEqual(len(error_handler_nodes), 2,
                        "Should have 2 error handler issues")
        
        # They should be connected (both in same file)
        if len(error_handler_nodes) == 2:
            self.assertTrue(
                graph.has_edge(error_handler_nodes[0], error_handler_nodes[1]) or
                graph.has_edge(error_handler_nodes[1], error_handler_nodes[0]),
                "Related issues should be connected"
            )
    
    def test_find_issue_intersections_major_thoroughfares(self):
        """Test finding issues that are like 'major thoroughfares'."""
        graph = self.prioritizer.build_issue_influence_graph(self.sample_issues)
        intersections = self.prioritizer.find_issue_intersections(graph)
        
        # Should find some intersections
        self.assertGreater(len(intersections), 0,
                          "Should find issue intersections")
        
        # Intersections should be sorted by influence score
        influence_scores = [inter['influence_score'] for inter in intersections]
        self.assertEqual(influence_scores, 
                        sorted(influence_scores, reverse=True),
                        "Intersections should be sorted by influence")
        
        # Top intersection should have high influence
        if intersections:
            top_intersection = intersections[0]
            self.assertGreater(top_intersection['influence_score'], 0,
                              "Top intersection should have positive influence")
            self.assertIsNotNone(top_intersection['fixing_this_eliminates'],
                               "Should identify dependent issues")
    
    def test_cascade_potential_calculation(self):
        """Test calculation of cascade potential for issue fixing."""
        graph = self.prioritizer.build_issue_influence_graph(self.sample_issues)
        
        # Test on a few nodes
        for node in list(graph.nodes())[:3]:
            cascade_potential = self.prioritizer.calculate_cascade_potential(graph, node)
            
            self.assertGreaterEqual(cascade_potential, 0,
                                  "Cascade potential should be non-negative")
            
            # High severity issues should tend to have higher cascade potential
            issue = graph.nodes[node]['issue_data']
            if issue.severity >= 4:
                self.assertGreater(cascade_potential, 0,
                                 "High severity issues should have some cascade potential")
    
    def test_create_fix_clusters(self):
        """Test creation of fix clusters from influential issues."""
        graph = self.prioritizer.build_issue_influence_graph(self.sample_issues)
        intersections = self.prioritizer.find_issue_intersections(graph)
        clusters = self.prioritizer.create_fix_clusters(intersections, self.sample_issues)
        
        # Should create clusters
        self.assertGreater(len(clusters), 0, "Should create fix clusters")
        
        # Each cluster should have required attributes
        for cluster in clusters:
            self.assertIsNotNone(cluster.cluster_id)
            self.assertIsNotNone(cluster.primary_issue)
            self.assertIsNotNone(cluster.dependent_issues)
            self.assertGreaterEqual(cluster.influence_score, 0)
            self.assertGreaterEqual(cluster.cascade_potential, 0)
            self.assertGreater(cluster.fix_complexity, 0)
            self.assertGreater(cluster.estimated_benefit, 0)
    
    def test_order_by_cascade_potential_not_severity(self):
        """Test that clusters are ordered by cascade potential, not severity."""
        # Create clusters with different characteristics
        clusters = []
        
        # High severity, low cascade
        clusters.append(FixCluster(
            cluster_id="high_sev_low_cas",
            primary_issue=Issue(kind="test", message="test", file_path="test.py", 
                              line_number=1, severity=5),
            dependent_issues=[],
            influence_score=2.0,
            cascade_potential=1.0,
            fix_complexity=2.0,
            estimated_benefit=0.8
        ))
        
        # Medium severity, high cascade
        clusters.append(FixCluster(
            cluster_id="med_sev_high_cas",
            primary_issue=Issue(kind="test", message="test", file_path="test2.py",
                              line_number=1, severity=3),
            dependent_issues=[Issue(kind="test", message="test", file_path="test3.py", 
                                  line_number=1, severity=2)],
            influence_score=8.0,
            cascade_potential=6.0,
            fix_complexity=1.5,
            estimated_benefit=0.9
        ))
        
        ordered = self.prioritizer.order_by_cascade_potential(clusters)
        
        # High cascade should come first despite lower severity
        self.assertEqual(ordered[0].cluster_id, "med_sev_high_cas",
                        "High cascade potential should be prioritized over high severity")
    
    def test_fix_complexity_estimation(self):
        """Test estimation of fix complexity."""
        # Simple single-issue fix
        simple_issue = self.sample_issues[0]
        simple_complexity = self.prioritizer._estimate_fix_complexity(
            simple_issue, []
        )
        
        # Complex multi-file fix
        complex_primary = self.sample_issues[1]
        complex_dependents = self.sample_issues[2:4]  # Different files
        complex_complexity = self.prioritizer._estimate_fix_complexity(
            complex_primary, complex_dependents
        )
        
        self.assertGreater(complex_complexity, simple_complexity,
                          "Multi-file fixes should be more complex")
    
    def test_fix_benefit_estimation(self):
        """Test estimation of fix benefit."""
        # High severity primary issue
        high_sev_issue = max(self.sample_issues, key=lambda x: x.severity)
        high_benefit = self.prioritizer._estimate_fix_benefit(
            high_sev_issue, []
        )
        
        # Low severity primary issue
        low_sev_issue = min(self.sample_issues, key=lambda x: x.severity)  
        low_benefit = self.prioritizer._estimate_fix_benefit(
            low_sev_issue, []
        )
        
        self.assertGreater(high_benefit, low_benefit,
                          "Higher severity issues should have higher benefit")
        
        # Benefit with many dependents
        with_dependents_benefit = self.prioritizer._estimate_fix_benefit(
            high_sev_issue, self.sample_issues[:3]
        )
        
        self.assertGreater(with_dependents_benefit, high_benefit,
                          "Issues with many dependents should have higher benefit")
    
    def test_issues_relationship_detection(self):
        """Test detection of relationships between issues."""
        issue1 = self.sample_issues[0]  # utils.py
        issue2 = self.sample_issues[1]  # business.py
        issue3 = Issue(kind="missing_function", message="test", 
                      file="utils.py", line=20, severity=2)
        
        # Same file should be related
        self.assertTrue(
            self.prioritizer._issues_related(issue1, issue3),
            "Issues in same file should be related"
        )
        
        # Same issue type should be related
        duplicate1 = self.sample_issues[2]
        duplicate2 = self.sample_issues[3]
        self.assertTrue(
            self.prioritizer._issues_related(duplicate1, duplicate2),
            "Same issue types should be related"
        )
    
    def test_dependent_issues_identification(self):
        """Test identification of dependent issues."""
        graph = self.prioritizer.build_issue_influence_graph(self.sample_issues)
        
        # Find a node with outgoing edges
        node_with_deps = None
        for node in graph.nodes():
            if graph.out_degree(node) > 0:
                node_with_deps = node
                break
        
        if node_with_deps:
            dependents = self.prioritizer.find_dependent_issues(graph, node_with_deps)
            
            self.assertGreater(len(dependents), 0,
                              "Should find dependent issues")
            
            # All dependents should be reachable from the node
            for dep in dependents:
                self.assertTrue(
                    nx.has_path(graph, node_with_deps, dep),
                    "Dependent should be reachable from source"
                )


if __name__ == "__main__":
    unittest.main()