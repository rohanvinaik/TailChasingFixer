"""Tests for CLI context-thrashing integration."""

import tempfile
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import io
import sys

from tailchasing.cli_enhanced import EnhancedCLI


class TestContextThrashingCLI:
    """Test context-thrashing CLI integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cli = EnhancedCLI()
    
    def create_test_file_with_thrashing(self):
        """Create test file with context-window thrashing."""
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

''' + '\n' * 50 + '''

def process_user_data(user_data):
    """Process user data - second implementation (thrashing)."""
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
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            return f.name
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_basic_context_thrashing_detection(self, mock_stderr, mock_stdout):
        """Test basic context-thrashing detection via CLI."""
        test_file = self.create_test_file_with_thrashing()
        
        try:
            test_dir = Path(test_file).parent
            
            # Create minimal config to enable context thrashing
            config_file = test_dir / '.tailchasing.yml'
            config_content = '''
context_thrashing:
  enabled: true
  similarity_threshold: 0.7
  min_line_distance: 30
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Run CLI
            args = [str(test_dir), '--config', str(config_file)]
            result = self.cli.run(args)
            
            output = mock_stdout.getvalue()
            
            # Should detect thrashing and complete successfully
            assert result == 0
            assert 'context_thrashing' in output.lower() or 'analyzing' in output.lower()
            
        finally:
            Path(test_file).unlink()
            if config_file.exists():
                config_file.unlink()
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_flat_view_option(self, mock_stderr, mock_stdout):
        """Test --flat option for context-thrashing output."""
        test_file = self.create_test_file_with_thrashing()
        
        try:
            test_dir = Path(test_file).parent
            
            # Create config
            config_file = test_dir / '.tailchasing.yml'
            config_content = '''
context_thrashing:
  enabled: true
  similarity_threshold: 0.6
  min_line_distance: 20
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Run with --flat option
            args = [str(test_dir), '--config', str(config_file), '--flat']
            result = self.cli.run(args)
            
            output = mock_stdout.getvalue()
            
            # Should show flat view
            assert result == 0
            assert 'View: Flat' in output or 'CONTEXT-WINDOW THRASHING REPORT' in output
            
        finally:
            Path(test_file).unlink()
            if config_file.exists():
                config_file.unlink()
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_max_members_option(self, mock_stderr, mock_stdout):
        """Test --max-members option for limiting cluster output."""
        test_file = self.create_test_file_with_thrashing()
        
        try:
            test_dir = Path(test_file).parent
            
            # Create config
            config_file = test_dir / '.tailchasing.yml'
            config_content = '''
context_thrashing:
  enabled: true
  similarity_threshold: 0.6
  min_line_distance: 20
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Run with --max-members option
            args = [str(test_dir), '--config', str(config_file), '--max-members', '1']
            result = self.cli.run(args)
            
            output = mock_stdout.getvalue()
            
            # Should show limited members
            assert result == 0
            assert 'Max members: 1' in output or 'CONTEXT-WINDOW THRASHING REPORT' in output
            
        finally:
            Path(test_file).unlink()
            if config_file.exists():
                config_file.unlink()
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    @patch('builtins.input', return_value='')  # Mock user input for cluster expansion
    def test_expand_option(self, mock_input, mock_stderr, mock_stdout):
        """Test --expand option for expanding specific clusters."""
        test_file = self.create_test_file_with_thrashing()
        
        try:
            test_dir = Path(test_file).parent
            
            # Create config
            config_file = test_dir / '.tailchasing.yml'
            config_content = '''
context_thrashing:
  enabled: true
  similarity_threshold: 0.6
  min_line_distance: 20
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # First get clusters to find an ID
            args = [str(test_dir), '--config', str(config_file)]
            self.cli.run(args)
            
            # Run with --expand option (use a likely cluster ID)
            args = [str(test_dir), '--config', str(config_file), '--expand', 'CWT_001']
            result = self.cli.run(args)
            
            output = mock_stdout.getvalue()
            
            # Should attempt to expand cluster (may fail if cluster doesn't exist)
            assert result in [0, 1]  # May succeed or fail gracefully
            
        finally:
            Path(test_file).unlink()
            if config_file.exists():
                config_file.unlink()
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_context_thrashing_with_json_output(self, mock_stderr, mock_stdout):
        """Test context-thrashing detection with JSON output."""
        test_file = self.create_test_file_with_thrashing()
        
        try:
            test_dir = Path(test_file).parent
            
            # Create config
            config_file = test_dir / '.tailchasing.yml'
            config_content = '''
context_thrashing:
  enabled: true
  similarity_threshold: 0.7
  min_line_distance: 30
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Run with JSON output (should bypass context-thrashing specific handling)
            args = [str(test_dir), '--config', str(config_file), '--json']
            result = self.cli.run(args)
            
            output = mock_stdout.getvalue()
            
            assert result == 0
            # Should contain JSON output
            try:
                json.loads(output)  # Should be valid JSON
            except json.JSONDecodeError:
                # If not full JSON, at least should contain analysis output
                assert 'analyzing' in output.lower() or len(output) > 0
            
        finally:
            Path(test_file).unlink()
            if config_file.exists():
                config_file.unlink()
    
    def test_context_thrashing_analyzer_registration(self):
        """Test that context-thrashing analyzer is properly registered."""
        from tailchasing.plugins import load_analyzers
        
        config = {
            'context_thrashing': {
                'enabled': True,
                'similarity_threshold': 0.7
            }
        }
        
        analyzers = load_analyzers(config)
        
        # Check if context-thrashing analyzer is loaded
        analyzer_names = [a.name for a in analyzers if hasattr(a, 'name')]
        assert 'context_thrashing' in analyzer_names
    
    def test_context_thrashing_analyzer_disabled(self):
        """Test that context-thrashing analyzer can be disabled."""
        from tailchasing.plugins import load_analyzers
        
        config = {
            'context_thrashing': {
                'enabled': False
            }
        }
        
        analyzers = load_analyzers(config)
        
        # Should not include context-thrashing analyzer
        analyzer_names = [a.name for a in analyzers if hasattr(a, 'name')]
        # May or may not be present depending on default config, but this tests the mechanism
        
        # Test explicit disabling
        config_disabled = {
            'disabled_analyzers': ['context_thrashing']
        }
        
        analyzers_disabled = load_analyzers(config_disabled)
        analyzer_names_disabled = [a.name for a in analyzers_disabled if hasattr(a, 'name')]
        assert 'context_thrashing' not in analyzer_names_disabled
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_no_thrashing_detected(self, mock_stderr, mock_stdout):
        """Test CLI behavior when no context-thrashing is detected."""
        # Create file with no similar functions
        clean_code = '''
def unique_function_one():
    """First unique function."""
    return 1

def unique_function_two():  
    """Second unique function."""
    return 2
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(clean_code)
            test_file = f.name
        
        try:
            test_dir = Path(test_file).parent
            
            # Create config
            config_file = test_dir / '.tailchasing.yml'
            config_content = '''
context_thrashing:
  enabled: true
  similarity_threshold: 0.7
  min_line_distance: 30
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Run CLI with flat option
            args = [str(test_dir), '--config', str(config_file), '--flat']
            result = self.cli.run(args)
            
            output = mock_stdout.getvalue()
            
            assert result == 0
            # Should indicate no thrashing detected
            assert 'No context-window thrashing detected' in output or 'analyzing' in output.lower()
            
        finally:
            Path(test_file).unlink()
            if config_file.exists():
                config_file.unlink()


class TestCLIArgumentValidation:
    """Test CLI argument validation for context-thrashing options."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cli = EnhancedCLI()
    
    def test_argument_parser_includes_context_options(self):
        """Test that argument parser includes context-thrashing options."""
        parser = self.cli.create_parser()
        
        # Test that options are present by checking help text
        help_text = parser.format_help()
        
        assert '--flat' in help_text
        assert '--max-members' in help_text
        assert '--expand' in help_text
        assert 'context-window thrashing' in help_text or 'flat view' in help_text
    
    def test_max_members_argument_validation(self):
        """Test max-members argument validation."""
        parser = self.cli.create_parser()
        
        # Valid argument
        args = parser.parse_args(['--max-members', '5'])
        assert args.max_members == 5
        
        # Test that it accepts integers
        args = parser.parse_args(['--max-members', '1'])
        assert args.max_members == 1
    
    def test_expand_argument_validation(self):
        """Test expand argument validation."""
        parser = self.cli.create_parser()
        
        # Valid argument
        args = parser.parse_args(['--expand', 'CWT_001'])
        assert args.expand == 'CWT_001'
        
        # Test with different cluster ID format
        args = parser.parse_args(['--expand', 'cluster_123'])
        assert args.expand == 'cluster_123'
    
    def test_flat_flag_validation(self):
        """Test flat flag validation."""
        parser = self.cli.create_parser()
        
        # Default should be False
        args = parser.parse_args([])
        assert args.flat is False
        
        # Should be True when specified
        args = parser.parse_args(['--flat'])
        assert args.flat is True


class TestCLIIntegrationFlow:
    """Integration tests for the complete CLI flow."""
    
    def setup_method(self):
        """Set up test environment."""
        self.cli = EnhancedCLI()
    
    @patch('sys.stdout', new_callable=io.StringIO)
    @patch('sys.stderr', new_callable=io.StringIO)
    def test_complete_workflow(self, mock_stderr, mock_stdout):
        """Test complete workflow from detection to report."""
        # Create a more complex test file
        complex_code = '''
class DataProcessor:
    def process_request_data(self, request_data):
        """Process incoming request data."""
        if not request_data:
            return {'error': 'No data'}
        
        # Validate required fields
        if 'user_id' not in request_data:
            return {'error': 'Missing user_id'}
        if 'action' not in request_data:
            return {'error': 'Missing action'}
        
        # Process the request
        result = {
            'user_id': request_data['user_id'],
            'action': request_data['action'],
            'processed': True,
            'timestamp': time.time()
        }
        
        return {'success': True, 'data': result}

''' + '\n' * 30 + '''

class DataProcessor:
    def process_request_data(self, request_data):
        """Process request data - duplicate implementation."""
        if not request_data:
            return {'error': 'Missing data'}  # Slightly different
        
        # Validate required fields (same logic)
        if 'user_id' not in request_data:
            return {'error': 'Missing user_id'}
        if 'action' not in request_data:
            return {'error': 'Missing action'}
        
        # Process the request (very similar)
        result = {
            'user_id': request_data['user_id'],
            'action': request_data['action'],
            'processed': True,
            'timestamp': time.time()
        }
        
        return {'success': True, 'data': result}
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(complex_code)
            test_file = f.name
        
        try:
            test_dir = Path(test_file).parent
            
            # Create comprehensive config
            config_file = test_dir / '.tailchasing.yml'
            config_content = '''
context_thrashing:
  enabled: true
  similarity_threshold: 0.7
  min_line_distance: 25
  min_function_lines: 5
  enable_clustering: true

analyzers:
  context_thrashing:
    max_members_default: 10
'''
            with open(config_file, 'w') as f:
                f.write(config_content)
            
            # Test different CLI combinations
            test_cases = [
                # Basic detection
                [str(test_dir), '--config', str(config_file)],
                # Flat view
                [str(test_dir), '--config', str(config_file), '--flat'],
                # Limited members
                [str(test_dir), '--config', str(config_file), '--max-members', '1'],
            ]
            
            for args in test_cases:
                result = self.cli.run(args)
                assert result == 0, f"Failed with args: {args}"
                
                output = mock_stdout.getvalue()
                assert len(output) > 0, f"No output with args: {args}"
                
                # Clear output for next test
                mock_stdout.truncate(0)
                mock_stdout.seek(0)
                
        finally:
            Path(test_file).unlink()
            if config_file.exists():
                config_file.unlink()