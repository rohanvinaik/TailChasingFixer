# tailchasing/semantic/smart_filter.py
"""
Smart filtering to prevent false positives in semantic duplicate detection.
Because nobody wants their __init__.py files flagged as tail-chasing! 🙄
"""

import ast
from pathlib import Path
from typing import List, Tuple, Dict

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
            'backup_files': self._is_backup_file_similarity,
            'linter_generated': self._is_linter_generated_similarity,
            'generated_code': self._is_generated_code_similarity,
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
        
        # Backup file patterns
        self.backup_patterns = {
            '.bak', '.backup', '.orig', '.save', '.old', '.prev',
            '~', '.tmp', '.temp', '.swp', '.swo', '.swn',  # Editor temp files
            '_backup', '_bak', '_old', '_orig', '_save',
            '.before', '.rollback', '.snapshot',
        }
        
        # Linter/formatter generated patterns
        self.linter_patterns = {
            '.formatted', '.linted', '.fixed', '.corrected',
            '_formatted', '_linted', '_fixed',
            '.black', '.yapf', '.autopep8', '.isort',
            '.pylint', '.flake8', '.mypy_cache',
        }
        
        # Generated code patterns
        self.generated_patterns = {
            '_pb2.py', '_pb2_grpc.py',  # Protocol buffers
            '.generated.py', '_generated.py',
            '.auto.py', '_auto.py',
            'autogen_', 'generated_',
            '.g.py',  # Often used for generated files
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
    
    def _is_backup_file_similarity(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                                  file1: str, file2: str) -> bool:
        """Check if one or both files are backup files - these naturally have duplicates."""
        # Check if either file is a backup
        file1_is_backup = self._is_backup_file(file1)
        file2_is_backup = self._is_backup_file(file2)
        
        # If at least one is a backup, and they're related (same base name), it's legitimate
        if file1_is_backup or file2_is_backup:
            base1 = self._get_base_filename(file1)
            base2 = self._get_base_filename(file2)
            # If base names are similar, this is a legitimate backup duplicate
            if base1 == base2 or base1 in base2 or base2 in base1:
                return True
        
        return False
    
    def _is_linter_generated_similarity(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                                       file1: str, file2: str) -> bool:
        """Check if files are linter/formatter outputs - these will be similar to originals."""
        file1_is_linted = self._is_linter_file(file1)
        file2_is_linted = self._is_linter_file(file2)
        
        # If at least one is a linter output and they have similar base names
        if file1_is_linted or file2_is_linted:
            base1 = self._get_base_filename(file1)
            base2 = self._get_base_filename(file2)
            if base1 == base2 or base1 in base2 or base2 in base1:
                return True
        
        return False
    
    def _is_generated_code_similarity(self, func1: ast.FunctionDef, func2: ast.FunctionDef,
                                     file1: str, file2: str) -> bool:
        """Check if files are generated code - often have legitimate duplicates."""
        file1_is_generated = self._is_generated_file(file1)
        file2_is_generated = self._is_generated_file(file2)
        
        # Both generated files often have similar patterns
        if file1_is_generated and file2_is_generated:
            return True
        
        # If one is generated and matches the base of the other
        if file1_is_generated or file2_is_generated:
            # Check for common generated code patterns in function
            if self._has_generated_code_markers(func1) or self._has_generated_code_markers(func2):
                return True
        
        return False
    
    def _is_backup_file(self, filepath: str) -> bool:
        """Check if a file is a backup file."""
        path = Path(filepath)
        filename = path.name.lower()
        
        # Check for backup patterns in filename
        for pattern in self.backup_patterns:
            if pattern in filename or filename.endswith(pattern):
                return True
        
        # Check for backup directories
        parts = path.parts
        backup_dirs = {'backup', 'backups', 'bak', 'old', 'archive', 'archives', '.backup'}
        if any(part.lower() in backup_dirs for part in parts):
            return True
        
        # Check for timestamped backups (e.g., file.py.20240101, file.2024-01-01.py)
        import re
        timestamp_patterns = [
            r'\.\d{8}$', r'\.\d{8}\.', r'\.\d{4}-\d{2}-\d{2}$', r'\.\d{4}-\d{2}-\d{2}\.', 
            r'\.\d{10}$', r'\.\d{10}\.', # Date/timestamp patterns
            r'\.v\d+$', r'\.v\d+\.', r'_v\d+$', r'_v\d+\.', # Version patterns
        ]
        for pattern in timestamp_patterns:
            if re.search(pattern, filename):
                return True
        
        return False
    
    def _is_linter_file(self, filepath: str) -> bool:
        """Check if a file is linter/formatter generated."""
        path = Path(filepath)
        filename = path.name.lower()
        
        # Check for linter patterns
        for pattern in self.linter_patterns:
            if pattern in filename:
                return True
        
        # Check for linter cache directories
        parts = path.parts
        linter_dirs = {'.mypy_cache', '.pylint.d', '.ruff_cache', '__pycache__'}
        if any(part in linter_dirs for part in parts):
            return True
        
        return False
    
    def _is_generated_file(self, filepath: str) -> bool:
        """Check if a file is generated code."""
        path = Path(filepath)
        filename = path.name.lower()
        
        # Check for generated patterns
        for pattern in self.generated_patterns:
            if pattern in filename or filename.endswith(pattern):
                return True
        
        # Check for generated directories
        parts = path.parts
        generated_dirs = {'generated', 'autogen', 'gen', 'build', 'dist', 'output'}
        if any(part.lower() in generated_dirs for part in parts):
            return True
        
        return False
    
    def _get_base_filename(self, filepath: str) -> str:
        """Extract base filename without backup/version suffixes."""
        path = Path(filepath)
        name = path.name  # Keep full name with extension
        
        # First remove known backup extensions
        for pattern in ['.bak', '.backup', '.orig', '.save', '.old', '.prev', '.tmp', '.temp']:
            if name.endswith(pattern):
                name = name[:-len(pattern)]
        
        # Remove timestamp patterns
        import re
        name = re.sub(r'\.\d{8}(\.\w+)?$', r'\1', name)  # Remove date suffix, keep extension
        name = re.sub(r'\.\d{4}-\d{2}-\d{2}(\.\w+)?$', r'\1', name)  # Remove date suffix
        name = re.sub(r'\.\d{10}(\.\w+)?$', r'\1', name)  # Remove timestamp suffix
        name = re.sub(r'\.v\d+(\.\w+)?$', r'\1', name)  # Remove version suffix
        name = re.sub(r'_v\d+(\.\w+)?$', r'\1', name)  # Remove version suffix
        
        # Remove common backup suffixes from stem
        stem = Path(name).stem
        for pattern in ['_backup', '_bak', '_old', '_orig', '_save']:
            if stem.endswith(pattern):
                stem = stem[:-len(pattern)]
                ext = Path(name).suffix
                name = stem + ext if ext else stem
        
        # Final cleanup - get stem without extension
        return Path(name).stem
    
    def _has_generated_code_markers(self, func: ast.FunctionDef) -> bool:
        """Check if function has markers indicating it's generated code."""
        # Check docstring for generation markers
        docstring = ast.get_docstring(func)
        if docstring:
            generated_markers = [
                'auto-generated', 'autogenerated', 'generated by',
                'do not edit', 'do not modify', 'automatically generated',
                'THIS CODE IS GENERATED', 'Generated from'
            ]
            docstring_lower = docstring.lower()
            if any(marker in docstring_lower for marker in generated_markers):
                return True
        
        # Check for common generated function patterns
        if func.name.startswith('_pb2_') or func.name.endswith('_pb2'):
            return True
        
        return False
    
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
        import sys
        sys.stdout.write("\n🧹 Filtered out legitimate patterns:\n")
        
        pattern_counts = {}
        for (pair, reason) in filtered_out:
            func1_info, func2_info, distance, z_score, channels = pair
            pattern_counts[reason] = pattern_counts.get(reason, 0) + 1
        
        for pattern, count in pattern_counts.items():
            sys.stdout.write(f"  - {pattern}: {count} pairs\n")
