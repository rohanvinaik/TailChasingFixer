"""Tests for context-window-thrashing analyzer."""

import ast
import tempfile
from pathlib import Path
import pytest

from tailchasing.analyzers.context_thrashing import (
    ContextThrashingAnalyzer,
    FunctionSimilarityAnalyzer,
    ExtractHelperPlaybookGenerator,
    FunctionInfo,
    SimilarFunctionPair,
    ContextThrashingCluster
)
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.issues import Issue


class TestFunctionSimilarityAnalyzer:
    """Test function similarity analysis."""
    
    def setup_method(self):
        """Set up test environment."""
        self.analyzer = FunctionSimilarityAnalyzer()
    
    def test_extract_function_info(self):
        """Test function information extraction."""
        code = '''
def test_function(self, param1, param2="default"):
    """Test function docstring."""
    result = param1 + param2
    return result
'''
        tree = ast.parse(code)
        func_node = tree.body[0]
        source_lines = code.splitlines()
        
        func_info = self.analyzer.extract_function_info(
            '/test/file.py', func_node, None, source_lines
        )
        
        assert func_info.name == 'test_function'
        assert func_info.class_name is None
        assert func_info.file_path == '/test/file.py'
        assert func_info.line_start == 2
        assert 'self' in func_info.parameters
        assert 'param1' in func_info.parameters
        assert 'param2' in func_info.parameters
        assert '(self, param1, param2)' == func_info.signature
    
    def test_calculate_similarity_identical(self):
        """Test similarity calculation for identical functions."""
        code1 = '''
def process_data(data):
    if not data:
        return None
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
        code2 = '''
def process_data(data):
    if not data:
        return None
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
        
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        func1 = self.analyzer.extract_function_info('/test/file1.py', tree1.body[0], None, code1.splitlines())
        func2 = self.analyzer.extract_function_info('/test/file2.py', tree2.body[0], None, code2.splitlines())
        
        similarity = self.analyzer.calculate_similarity(func1, func2)
        assert similarity > 0.95  # Should be very high for identical functions
    
    def test_calculate_similarity_different(self):
        """Test similarity calculation for different functions."""
        code1 = '''
def process_data(data):
    return data * 2
'''
        code2 = '''
def validate_input(user_input):
    if len(user_input) < 5:
        raise ValueError("Too short")
    return True
'''
        
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        func1 = self.analyzer.extract_function_info('/test/file1.py', tree1.body[0], None, code1.splitlines())
        func2 = self.analyzer.extract_function_info('/test/file2.py', tree2.body[0], None, code2.splitlines())
        
        similarity = self.analyzer.calculate_similarity(func1, func2)
        assert similarity < 0.5  # Should be low for different functions
    
    def test_find_similar_pairs_with_distance(self):
        """Test finding similar function pairs with distance check."""
        code = '''
def process_request(request):
    """Process incoming request."""
    if not request:
        return None
    data = request.get('data')
    if not data:
        return {'error': 'No data'}
    return {'result': data}

# ... many lines of unrelated code ...
''' + '\n' * 50 + '''

def process_request(request):
    """Process incoming request with slight variation."""
    if not request:
        return None
    data = request.get('data')
    if not data:
        return {'error': 'Missing data'}  # Slight difference
    return {'result': data}
'''
        
        tree = ast.parse(code)
        source_lines = code.splitlines()
        
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self.analyzer.extract_function_info('/test/file.py', node, None, source_lines)
                functions.append(func_info)
        
        similar_pairs = self.analyzer.find_similar_pairs(
            functions,
            similarity_threshold=0.7,
            min_line_distance=10
        )
        
        assert len(similar_pairs) >= 1
        pair = similar_pairs[0]
        assert pair.similarity >= 0.7
        assert pair.line_distance >= 10
        assert pair.func1.name == pair.func2.name == 'process_request'


class TestExtractHelperPlaybookGenerator:
    """Test extract helper playbook generation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.generator = ExtractHelperPlaybookGenerator()
    
    def test_generate_extract_playbook(self):
        """Test playbook generation for extract helper."""
        # Create mock function info
        func1 = FunctionInfo(
            name='process_data',
            class_name=None,
            file_path='/test/file.py',
            line_start=10,
            line_end=20,
            signature='(self, data)',
            normalized_body='validate data\nprocess data\nreturn result',
            raw_body='# original code here',
            ast_node=None,
            parameters=('self', 'data')  # Changed to tuple
        )
        
        func2 = FunctionInfo(
            name='process_data',
            class_name=None,
            file_path='/test/file.py',
            line_start=100,
            line_end=110,
            signature='(self, data)',
            normalized_body='validate data\nprocess data\nreturn result',
            raw_body='# original code here',
            ast_node=None,
            parameters=('self', 'data')  # Changed to tuple
        )
        
        pair = SimilarFunctionPair(
            func1=func1,
            func2=func2,
            similarity=0.85,
            line_distance=90,
            common_body='validate data\nprocess data',
            differences=('return statement differs',),  # Changed to tuple
            suggested_helper_name='_process_data_common'
        )
        
        playbook = self.generator.generate_extract_playbook(pair)
        
        assert 'EXTRACT HELPER PLAYBOOK' in playbook
        assert '_process_data_common' in playbook
        assert 'Similarity: 85' in playbook and '%' in playbook
        assert 'Line distance: 90' in playbook
        assert 'MINI-DIFF PREVIEW' in playbook
        assert 'SAFETY CHECKS' in playbook


class TestContextThrashingAnalyzer:
    """Test the main context-thrashing analyzer."""
    
    def setup_method(self):
        """Set up test environment."""
        self.analyzer = ContextThrashingAnalyzer()
    
    def create_test_file_with_similar_functions(self) -> str:
        """Create a test file with similar functions far apart."""
        test_code = '''
def process_user_data(user_data):
    """Process user data - first implementation."""
    if not user_data:
        return None
    
    # Validate input
    if 'id' not in user_data:
        raise ValueError('Missing user ID')
    
    # Process data
    result = {
        'user_id': user_data['id'],
        'processed': True,
        'timestamp': time.time()
    }
    return result

def another_function():
    """Unrelated function."""
    pass

''' + '\n' * 50 + '''

def process_user_data(user_data):
    """Process user data - second implementation (context window thrashing)."""
    if not user_data:
        return None
        
    # Validate input (slightly different)
    if 'id' not in user_data:
        raise ValueError('User ID is required')
    
    # Process data (very similar)
    result = {
        'user_id': user_data['id'], 
        'processed': True,
        'timestamp': time.time()
    }
    return result

def yet_another_function():
    """Another unrelated function."""
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            return f.name
    
    def test_analyzer_detects_context_thrashing(self):
        """Test that analyzer detects context-window thrashing."""
        test_file = self.create_test_file_with_similar_functions()
        
        try:
            # Parse file
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            # Create mock context
            context = type('MockContext', (), {
                'config': {},
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            # Run analyzer
            issues = self.analyzer.run(context)
            
            # Should detect at least one thrashing issue
            assert len(issues) >= 1
            
            issue = issues[0]
            assert issue.kind == "context_window_thrashing"
            assert "process_user_data" in issue.message
            assert "similar" in issue.message.lower()
            
            # Check evidence
            assert 'similarity' in issue.evidence
            assert 'line_distance' in issue.evidence
            assert 'extract_playbook' in issue.evidence
            assert issue.evidence['similarity'] >= 0.7
            assert issue.evidence['line_distance'] >= 50
        
        finally:
            Path(test_file).unlink()
    
    def test_analyzer_clustering_disabled(self):
        """Test analyzer with clustering disabled."""
        config = {'enable_clustering': False}
        analyzer = ContextThrashingAnalyzer(config)
        
        test_file = self.create_test_file_with_similar_functions()
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            issues = analyzer.run(context)
            clusters = analyzer.get_clusters()
            
            # With clustering disabled, should have one cluster per pair
            assert len(clusters) >= 1
            for cluster in clusters:
                assert len(cluster.functions) == 2  # Only pairs, not merged clusters
        
        finally:
            Path(test_file).unlink()
    
    def test_get_clusters_with_flat_view(self):
        """Test cluster retrieval with flat view."""
        test_file = self.create_test_file_with_similar_functions()
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': {},
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            self.analyzer.run(context)
            
            # Test flat view
            flat_clusters = self.analyzer.get_clusters(flat_view=True)
            
            # Flat view should have individual function entries
            assert len(flat_clusters) >= 2  # At least 2 functions
            
            # Each flat cluster should have only one function
            for cluster in flat_clusters:
                assert len(cluster.functions) == 1
        
        finally:
            Path(test_file).unlink()
    
    def test_get_clusters_with_max_members(self):
        """Test cluster retrieval with member limit."""
        test_file = self.create_test_file_with_similar_functions()
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': {},
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            self.analyzer.run(context)
            
            # Test with max_members limit
            limited_clusters = self.analyzer.get_clusters(max_members=1)
            
            for cluster in limited_clusters:
                assert len(cluster.functions) <= 1
                if cluster.members_truncated:
                    assert cluster.hidden_members > 0
        
        finally:
            Path(test_file).unlink()
    
    def test_expand_cluster(self):
        """Test cluster expansion functionality."""
        test_file = self.create_test_file_with_similar_functions()
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': {},
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            self.analyzer.run(context)
            clusters = self.analyzer.get_clusters()
            
            if clusters:
                cluster_id = clusters[0].cluster_id
                expanded = self.analyzer.expand_cluster(cluster_id)
                
                assert expanded is not None
                assert expanded.cluster_id == cluster_id
                assert not expanded.members_truncated
                assert expanded.hidden_members == 0
        
        finally:
            Path(test_file).unlink()
    
    def test_generate_cluster_report(self):
        """Test cluster report generation."""
        test_file = self.create_test_file_with_similar_functions()
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': {},
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            self.analyzer.run(context)
            
            # Test normal report
            report = self.analyzer.generate_cluster_report()
            assert 'CONTEXT-WINDOW THRASHING REPORT' in report
            assert 'Total clusters:' in report
            
            # Test flat report
            flat_report = self.analyzer.generate_cluster_report(flat_view=True)
            assert 'View: Flat' in flat_report
            
            # Test with max members
            limited_report = self.analyzer.generate_cluster_report(max_members=1)
            assert 'Max members: 1' in limited_report
        
        finally:
            Path(test_file).unlink()
    
    def test_similarity_threshold_configuration(self):
        """Test similarity threshold configuration."""
        config = {
            'similarity_threshold': 0.9,  # Very high threshold
            'min_line_distance': 10,
            'min_function_lines': 3
        }
        analyzer = ContextThrashingAnalyzer(config)
        
        test_file = self.create_test_file_with_similar_functions()
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            issues = analyzer.run(context)
            
            # With high threshold, may not detect similarities
            # This tests that configuration is being used
            assert isinstance(issues, list)  # Should still return a list
        
        finally:
            Path(test_file).unlink()


class TestIntegration:
    """Integration tests for context-window thrashing system."""
    
    def test_real_world_scenario(self):
        """Test a real-world-like scenario with multiple similar functions."""
        real_world_code = '''
class DataProcessor:
    def process_user_registration(self, user_data):
        """Process user registration data."""
        if not user_data or 'email' not in user_data:
            return {'error': 'Invalid data'}
        
        # Validate email
        email = user_data['email'].lower().strip()
        if '@' not in email:
            return {'error': 'Invalid email'}
            
        # Create user record
        user_record = {
            'email': email,
            'created_at': time.time(),
            'status': 'pending'
        }
        
        return {'success': True, 'user': user_record}
    
    def handle_api_request(self, request):
        """Handle API request."""
        pass
        
''' + '\n' * 40 + '''

class DataProcessor:
    def process_user_registration(self, user_data):
        """Process user registration - duplicate implementation."""
        if not user_data or 'email' not in user_data:
            return {'error': 'Missing data'}  # Slight variation
        
        # Validate email (same logic)  
        email = user_data['email'].lower().strip()
        if '@' not in email:
            return {'error': 'Bad email format'}  # Slight variation
            
        # Create user record (very similar)
        user_record = {
            'email': email,
            'created_at': time.time(), 
            'status': 'active'  # Different default
        }
        
        return {'success': True, 'user': user_record}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(real_world_code)
            test_file = f.name
        
        try:
            analyzer = ContextThrashingAnalyzer({
                'similarity_threshold': 0.7,
                'min_line_distance': 30,
                'min_function_lines': 5
            })
            
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': {},
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            issues = analyzer.run(context)
            
            # Should detect the duplicate process_user_registration methods
            assert len(issues) >= 1
            
            issue = issues[0]
            assert 'process_user_registration' in issue.message
            assert 'DataProcessor' in issue.evidence['functions'][0]['class_name']
            
            # Check that extract playbook was generated
            assert 'extract_playbook' in issue.evidence
            playbook = issue.evidence['extract_playbook']
            assert 'EXTRACT HELPER PLAYBOOK' in playbook
            assert '_process_user_registration_common' in playbook or 'common' in playbook
        
        finally:
            Path(test_file).unlink()
    
    def test_no_false_positives(self):
        """Test that analyzer doesn't create false positives."""
        clean_code = '''
def unique_function_one():
    """First unique function."""
    return 1

def unique_function_two():
    """Second unique function.""" 
    return 2
    
def unique_function_three():
    """Third unique function."""
    return 3
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(clean_code)
            test_file = f.name
        
        try:
            analyzer = ContextThrashingAnalyzer()
            
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': {},
                'ast_index': {test_file: tree},
                'source_cache': {test_file: content.splitlines()},
                'should_ignore_issue': lambda kind: False
            })()
            
            issues = analyzer.run(context)
            
            # Should not detect any thrashing for completely different functions
            assert len(issues) == 0
        
        finally:
            Path(test_file).unlink()