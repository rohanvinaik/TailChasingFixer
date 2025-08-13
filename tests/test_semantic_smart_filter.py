"""Tests for smart filtering of backup and linter files."""

import ast
import unittest
from pathlib import Path

from tailchasing.semantic.smart_filter import SemanticDuplicateFilter


class TestSmartFilter(unittest.TestCase):
    """Test the smart filter for excluding backup/linter duplicates."""
    
    def setUp(self):
        """Set up test fixture."""
        self.filter = SemanticDuplicateFilter()
        
        # Create simple test functions
        self.func1_code = """
def process_data(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result
"""
        
        self.func2_code = """
def process_data(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result
"""
        
        self.func1_ast = ast.parse(self.func1_code)
        self.func2_ast = ast.parse(self.func2_code)
        
        # Get function nodes
        self.func1_node = self.func1_ast.body[0]
        self.func2_node = self.func2_ast.body[0]
    
    def test_backup_file_detection(self):
        """Test detection of backup files."""
        # Test various backup patterns
        self.assertTrue(self.filter._is_backup_file("module.py.bak"))
        self.assertTrue(self.filter._is_backup_file("module.backup.py"))
        self.assertTrue(self.filter._is_backup_file("module.py.orig"))
        self.assertTrue(self.filter._is_backup_file("module.py~"))
        self.assertTrue(self.filter._is_backup_file("module.py.20240101"))
        self.assertTrue(self.filter._is_backup_file("module.py.2024-01-01"))
        self.assertTrue(self.filter._is_backup_file("backup/module.py"))
        self.assertTrue(self.filter._is_backup_file("src/backups/module.py"))
        self.assertTrue(self.filter._is_backup_file("module_backup.py"))
        self.assertTrue(self.filter._is_backup_file("module.v1.py"))
        
        # Test non-backup files
        self.assertFalse(self.filter._is_backup_file("module.py"))
        self.assertFalse(self.filter._is_backup_file("src/module.py"))
        self.assertFalse(self.filter._is_backup_file("test_module.py"))
    
    def test_linter_file_detection(self):
        """Test detection of linter-generated files."""
        # Test linter patterns
        self.assertTrue(self.filter._is_linter_file("module.formatted.py"))
        self.assertTrue(self.filter._is_linter_file("module.linted.py"))
        self.assertTrue(self.filter._is_linter_file("module.black.py"))
        self.assertTrue(self.filter._is_linter_file(".mypy_cache/module.py"))
        self.assertTrue(self.filter._is_linter_file("__pycache__/module.py"))
        
        # Test non-linter files
        self.assertFalse(self.filter._is_linter_file("module.py"))
        self.assertFalse(self.filter._is_linter_file("formatted_module.py"))
    
    def test_generated_file_detection(self):
        """Test detection of generated code files."""
        # Test generated patterns
        self.assertTrue(self.filter._is_generated_file("proto_pb2.py"))
        self.assertTrue(self.filter._is_generated_file("service_pb2_grpc.py"))
        self.assertTrue(self.filter._is_generated_file("module.generated.py"))
        self.assertTrue(self.filter._is_generated_file("module.auto.py"))
        self.assertTrue(self.filter._is_generated_file("generated/module.py"))
        self.assertTrue(self.filter._is_generated_file("autogen/module.py"))
        self.assertTrue(self.filter._is_generated_file("build/module.py"))
        
        # Test non-generated files
        self.assertFalse(self.filter._is_generated_file("module.py"))
        self.assertFalse(self.filter._is_generated_file("generator.py"))
    
    def test_backup_similarity_filtering(self):
        """Test that backup file duplicates are filtered."""
        # Test backup file similarity
        result = self.filter._is_backup_file_similarity(
            self.func1_node,
            self.func2_node,
            "src/module.py",
            "src/module.py.bak"
        )
        self.assertTrue(result)
        
        result = self.filter._is_backup_file_similarity(
            self.func1_node,
            self.func2_node,
            "src/module.py",
            "backup/module.py"
        )
        self.assertTrue(result)
        
        # Should not filter if files are unrelated
        result = self.filter._is_backup_file_similarity(
            self.func1_node,
            self.func2_node,
            "src/module1.py",
            "src/module2.py.bak"
        )
        self.assertFalse(result)
    
    def test_linter_similarity_filtering(self):
        """Test that linter file duplicates are filtered."""
        result = self.filter._is_linter_generated_similarity(
            self.func1_node,
            self.func2_node,
            "src/module.py",
            "src/module.formatted.py"
        )
        self.assertTrue(result)
        
        result = self.filter._is_linter_generated_similarity(
            self.func1_node,
            self.func2_node,
            "src/module.py",
            ".mypy_cache/module.py"
        )
        self.assertTrue(result)
    
    def test_generated_code_filtering(self):
        """Test that generated code duplicates are filtered."""
        result = self.filter._is_generated_code_similarity(
            self.func1_node,
            self.func2_node,
            "proto_pb2.py",
            "service_pb2.py"
        )
        self.assertTrue(result)
        
        result = self.filter._is_generated_code_similarity(
            self.func1_node,
            self.func2_node,
            "generated/module1.py",
            "generated/module2.py"
        )
        self.assertTrue(result)
    
    def test_base_filename_extraction(self):
        """Test extraction of base filename."""
        self.assertEqual(self.filter._get_base_filename("module.py.bak"), "module")
        self.assertEqual(self.filter._get_base_filename("module.py.20240101"), "module")
        self.assertEqual(self.filter._get_base_filename("module_backup.py"), "module")
        self.assertEqual(self.filter._get_base_filename("module.v1.py"), "module")
        self.assertEqual(self.filter._get_base_filename("module.py"), "module")


if __name__ == "__main__":
    unittest.main()