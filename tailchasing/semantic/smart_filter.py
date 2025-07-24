# tailchasing/semantic/smart_filter.py
"""
Smart filtering to prevent false positives in semantic duplicate detection.
Because nobody wants their __init__.py files flagged as tail-chasing! 🙄
"""

import ast
import os
from pathlib import Path
from typing import List, Tuple, Set, Dict
import re

class SemanticDuplicateFilter:
    """
    Filters out legitimate semantic similarities that aren't actually tail-chasing.
    """
    
    def __init__(self):
        # Patterns that are legitimately similar but not tail-chasing
        self.legitimate_patterns = {
            'init_files': self._is_init_file_similarity,
            'test_boilerplate': self._is_test_boilerplate,
            'property_accessors': self._is_property_pattern,
            'protocol_implementations': self._is_protocol_implementation,
            'factory_methods': self._is_factory_pattern,
            'configuration_classes': self._is_config_pattern,
            'exception_classes': self._is_exception_pattern,
            'migration_scripts': self._is_migration_pattern,
        }
        
        # Files/directories to treat specially
        self.special_contexts = {
            '__init__.py',
            'conftest.py',
            'setup.py',
            'migrations/',
            'fixtures/',
            'test_',
            'tests/'
        }
    
    def filter_semantic_duplicates(self, 
                                 similar_pairs: List[Tuple],
                                 ast_index: Dict[str, ast.AST]) -> List[Tuple]:
        """
        Filter out false positive semantic duplicates.
        
        Args:
            similar_pairs: List of (func1_info, func2_info, distance, z_score, channels)
            ast_index: AST index for detailed analysis
            
        Returns:
            Filtered list with legitimate duplicates only
        """
        legitimate_duplicates = []
        filtered_out = []
        
        for pair in similar_pairs:
            func1_info, func2_info, distance, z_score, channels = pair
            
            # Check if this is a legitimate pattern
            is_legitimate, reason = self._is_legitimate_similarity(
                func1_info, func2_info, ast_index
            )
            
            if not is_legitimate:
                legitimate_duplicates.append(pair)
            else:
                filtered_out.append((pair, reason))
        
        # Log what we filtered out for transparency
        if filtered_out:
            self._log_filtered_patterns(filtered_out)
        
        return legitimate_duplicates
    
    def _is_legitimate_similarity(self, 
                                func1_info: Tuple, 
                                func2_info: Tuple,
                                ast_index: Dict[str, ast.AST]) -> Tuple[bool, str]:
        """
        Check if similarity between two functions is legitimate (not tail-chasing).
        
        Returns:
            (is_legitimate, reason)
        """
        name1, file1, line1 = func1_info
        name2, file2, line2 = func2_info
        
        # Get the actual AST nodes
        func1_node = self._find_function_node(ast_index[file1], name1, line1)
        func2_node = self._find_function_node(ast_index[file2], name2, line2)
        
        if not func1_node or not func2_node:
            return False, "Could not find function nodes"
        
        # Run through all legitimate pattern checks
        for pattern_name, checker in self.legitimate_patterns.items():
            if checker(func1_node, func2_node, file1, file2):
                return True, f"Legitimate {pattern_name} pattern"
        
        return False, "No legitimate pattern detected"
    
    def _is_init_file_similarity(self, func1: ast.FunctionDef, func2: ast.FunctionDef, 
                               file1: str, file2: str) -> bool:
        """Check if this is legitimate __init__.py similarity."""
        # Both in __init__.py files
        if not (file1.endswith('__init__.py') and file2.endswith('__init__.py')):
            return False
        
        # Common __init__.py patterns that are legitimately similar
        init_patterns = [
            '__version__',
            '__author__',
            '__all__',
            'setup_logging',
            'configure_',
            'init_',
            '_setup_'
        ]
        
        # If both functions match init patterns
        if any(pattern in func1.name for pattern in init_patterns) and \
           any(pattern in func2.name for pattern in init_patterns):
            return True
        
        # Both are simple import/export functions
        if self._is_simple_import_export(func1) and self._is_simple_import_export(func2):
            return True
        
        return False
    
    def _is_test_boilerplate(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                           file1: str, file2: str) -> bool:
        """Check if this is legitimate test boilerplate similarity."""
        # Both in test files
        if not (self._is_test_file(file1) and self._is_test_file(file2)):
            return False
        
        # Common test patterns
        test_patterns = [
            'setUp', 'tearDown', 'test_', 'fixture_', 'mock_', 'patch_'
        ]
        
        # Both are test setup/teardown
        if any(pattern in func1.name for pattern in test_patterns) and \
           any(pattern in func2.name for pattern in test_patterns):
            return True
        
        # Both are simple assertions or fixtures
        if self._is_simple_test_function(func1) and self._is_simple_test_function(func2):
            return True
        
        return False
    
    def _is_property_pattern(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                           file1: str, file2: str) -> bool:
        """Check if these are legitimate property getter/setter patterns."""
        # Look for property decorators
        func1_decorators = {d.id for d in func1.decorator_list if isinstance(d, ast.Name)}
        func2_decorators = {d.id for d in func2.decorator_list if isinstance(d, ast.Name)}
        
        property_decorators = {'property', 'setter', 'getter', 'cached_property'}
        
        # Both have property decorators
        if (func1_decorators & property_decorators) and (func2_decorators & property_decorators):
            return True
        
        # Simple getter/setter pattern
        if self._is_simple_getter_setter(func1) and self._is_simple_getter_setter(func2):
            return True
        
        return False
    
    def _is_protocol_implementation(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                                  file1: str, file2: str) -> bool:
        """Check if these implement the same protocol/interface."""
        # Look for abstract method implementations
        if func1.name == func2.name:
            # Same method name in different classes - likely protocol implementation
            return True
        
        # Common protocol method patterns
        protocol_methods = [
            '__str__', '__repr__', '__len__', '__iter__', '__next__',
            '__enter__', '__exit__', '__call__', '__getitem__', '__setitem__'
        ]
        
        if func1.name in protocol_methods and func2.name in protocol_methods:
            return True
        
        return False
    
    def _is_factory_pattern(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                          file1: str, file2: str) -> bool:
        """Check if these are legitimate factory method patterns."""
        factory_patterns = ['create_', 'make_', 'build_', 'from_', 'new_']
        
        func1_is_factory = any(func1.name.startswith(p) for p in factory_patterns)
        func2_is_factory = any(func2.name.startswith(p) for p in factory_patterns)
        
        if func1_is_factory and func2_is_factory:
            # Check if they have similar simple structure (factory pattern)
            return self._is_simple_factory_structure(func1) and \
                   self._is_simple_factory_structure(func2)
        
        return False
    
    def _is_config_pattern(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                         file1: str, file2: str) -> bool:
        """Check if these are configuration/settings patterns."""
        config_indicators = ['config', 'settings', 'options', 'preferences']
        
        file1_is_config = any(indicator in file1.lower() for indicator in config_indicators)
        file2_is_config = any(indicator in file2.lower() for indicator in config_indicators)
        
        if file1_is_config and file2_is_config:
            # Configuration methods are often legitimately similar
            return True
        
        return False
    
    def _is_exception_pattern(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                            file1: str, file2: str) -> bool:
        """Check if these are exception class patterns."""
        # Look for exception class __init__ methods
        if func1.name == '__init__' and func2.name == '__init__':
            # Check if both classes inherit from Exception
            return True  # Simplified - real implementation would check inheritance
        
        return False
    
    def _is_migration_pattern(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                            file1: str, file2: str) -> bool:
        """Check if these are database migration patterns."""
        migration_indicators = ['migration', 'migrate', 'schema', 'upgrade', 'downgrade']
        
        file1_is_migration = any(indicator in file1.lower() for indicator in migration_indicators)
        file2_is_migration = any(indicator in file2.lower() for indicator in migration_indicators)
        
        return file1_is_migration and file2_is_migration
    
    # Helper methods
    def _find_function_node(self, tree: ast.AST, name: str, line: int) -> ast.FunctionDef:
        """Find the function node in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name and node.lineno == line:
                return node
        return None
    
    def _is_simple_import_export(self, func: ast.FunctionDef) -> bool:
        """Check if function is just importing/exporting."""
        # Function body is just imports or simple assignments
        if len(func.body) <= 3:
            for stmt in func.body:
                if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Assign)):
                    continue
                elif isinstance(stmt, ast.Return):
                    continue
                else:
                    return False
            return True
        return False
    
    def _is_test_file(self, filepath: str) -> bool:
        """Check if file is a test file."""
        return any(indicator in filepath.lower() for indicator in [
            'test_', 'tests/', '_test.py', 'conftest.py', 'fixtures/'
        ])
    
    def _is_simple_test_function(self, func: ast.FunctionDef) -> bool:
        """Check if function is simple test boilerplate."""
        # Simple test functions often just have asserts
        assert_count = sum(1 for node in ast.walk(func) if isinstance(node, ast.Assert))
        return len(func.body) <= 5 and assert_count >= 1
    
    def _is_simple_getter_setter(self, func: ast.FunctionDef) -> bool:
        """Check if function is simple property getter/setter."""
        # Simple property functions usually just return/set an attribute
        if len(func.body) == 1:
            stmt = func.body[0]
            if isinstance(stmt, ast.Return):
                return True
            elif isinstance(stmt, ast.Assign):
                return True
        return False
    
    def _is_simple_factory_structure(self, func: ast.FunctionDef) -> bool:
        """Check if function has simple factory structure."""
        # Factory methods often just instantiate and return
        has_instantiation = False
        has_return = False
        
        for node in ast.walk(func):
            if isinstance(node, ast.Call):
                has_instantiation = True
            elif isinstance(node, ast.Return):
                has_return = True
        
        return has_instantiation and has_return and len(func.body) <= 5
    
    def _log_filtered_patterns(self, filtered_out: List[Tuple]) -> None:
        """Log what patterns were filtered out for transparency."""
        print("\n🧹 Filtered out legitimate patterns:")
        
        pattern_counts = {}
        for (pair, reason) in filtered_out:
            func1_info, func2_info, distance, z_score, channels = pair
            pattern_counts[reason] = pattern_counts.get(reason, 0) + 1
        
        for pattern, count in pattern_counts.items():
            print(f"  - {pattern}: {count} pairs")
