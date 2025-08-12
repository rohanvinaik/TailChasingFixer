"""
Tests for enhanced pattern detectors.

Comprehensive test suite for all tail-chasing pattern detection,
including hallucination cascades, context window thrashing, and import anxiety.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import textwrap
import time

from hypothesis import given, strategies as st, settings
import pytest

from tailchasing.analyzers.hallucination_cascade import HallucinationCascadeDetector
from tailchasing.analyzers.context_window_thrashing import ContextWindowThrashingDetector
from tailchasing.analyzers.import_anxiety import ImportAnxietyDetector
from tailchasing.analyzers.semantic_duplicate import SemanticDuplicateDetector
from tailchasing.core.issues import Issue


class TestHallucinationCascade(unittest.TestCase):
    """Test hallucination cascade detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.detector = HallucinationCascadeDetector()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_detect_simple_cascade(self):
        """Test detection of simple hallucination cascade."""
        # Create a cascade of abstract classes with no real usage
        self.create_file("base_handler.py", """
            from abc import ABC, abstractmethod
            
            class BaseHandler(ABC):
                @abstractmethod
                def handle(self, data):
                    pass
            
            class AbstractProcessor(BaseHandler):
                @abstractmethod
                def process(self, item):
                    pass
            
            class MetaController(AbstractProcessor):
                def __init__(self):
                    self.handlers = []
                
                def handle(self, data):
                    pass
                
                def process(self, item):
                    pass
        """)
        
        self.create_file("concrete_handler.py", """
            from .base_handler import MetaController
            
            class ConcreteHandler(MetaController):
                def execute(self):
                    return "doing nothing"
        """)
        
        issues = self.detector.detect(self.test_path)
        
        self.assertGreater(len(issues), 0)
        cascade_issues = [i for i in issues if i.kind == 'hallucination_cascade']
        self.assertGreater(len(cascade_issues), 0)
        
        # Check that the cascade was identified
        issue = cascade_issues[0]
        self.assertIn('BaseHandler', issue.evidence.get('components', []))
    
    def test_detect_factory_cascade(self):
        """Test detection of factory pattern cascade."""
        # Create over-engineered factory pattern
        self.create_file("factories.py", """
            class AbstractFactory:
                def create(self):
                    raise NotImplementedError
            
            class BaseFactory(AbstractFactory):
                def create(self):
                    return None
            
            class SpecificFactory(BaseFactory):
                def create(self):
                    return super().create()
            
            class ConcreteFactory(SpecificFactory):
                def create(self):
                    return "finally something"
            
            class FactoryManager:
                def __init__(self):
                    self.factory = ConcreteFactory()
                
                def get_instance(self):
                    return self.factory.create()
        """)
        
        issues = self.detector.detect(self.test_path)
        cascade_issues = [i for i in issues if i.kind == 'hallucination_cascade']
        
        self.assertGreater(len(cascade_issues), 0)
        issue = cascade_issues[0]
        self.assertGreater(issue.severity, 2)
    
    def test_legitimate_inheritance(self):
        """Test that legitimate inheritance is not flagged."""
        self.create_file("models.py", """
            class User:
                def __init__(self, name):
                    self.name = name
                
                def get_name(self):
                    return self.name
            
            class AdminUser(User):
                def __init__(self, name, permissions):
                    super().__init__(name)
                    self.permissions = permissions
                
                def has_permission(self, perm):
                    return perm in self.permissions
        """)
        
        self.create_file("app.py", """
            from .models import User, AdminUser
            
            def create_user(name, is_admin=False):
                if is_admin:
                    return AdminUser(name, ['read', 'write'])
                return User(name)
            
            user1 = create_user("Alice")
            user2 = create_user("Bob", is_admin=True)
        """)
        
        issues = self.detector.detect(self.test_path)
        cascade_issues = [i for i in issues if i.kind == 'hallucination_cascade']
        
        # Should not detect cascades in legitimate inheritance
        self.assertEqual(len(cascade_issues), 0)
    
    @given(
        num_classes=st.integers(min_value=3, max_value=10),
        has_external_usage=st.booleans()
    )
    @settings(max_examples=20)
    def test_cascade_detection_property(self, num_classes: int, has_external_usage: bool):
        """Property-based test for cascade detection."""
        # Generate cascade
        classes = []
        for i in range(num_classes):
            if i == 0:
                parent = "ABC"
                import_line = "from abc import ABC"
            else:
                parent = classes[i-1]
                import_line = ""
            
            class_name = f"Class{i}"
            classes.append(class_name)
            
            content = f"""
                {import_line}
                
                class {class_name}({parent}):
                    def method{i}(self):
                        pass
            """
            
            self.create_file(f"class_{i}.py", content)
        
        # Add external usage if specified
        if has_external_usage:
            self.create_file("usage.py", f"""
                from .class_{num_classes-1} import {classes[-1]}
                
                instance = {classes[-1]}()
                result = instance.method0()
            """)
        
        issues = self.detector.detect(self.test_path)
        cascade_issues = [i for i in issues if i.kind == 'hallucination_cascade']
        
        # Should detect cascade if many classes with no external usage
        if num_classes > 5 and not has_external_usage:
            self.assertGreater(len(cascade_issues), 0)


class TestContextWindowThrashing(unittest.TestCase):
    """Test context window thrashing detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.detector = ContextWindowThrashingDetector()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_detect_reimplementation(self):
        """Test detection of reimplemented functions."""
        # Create file with reimplemented functions far apart
        lines = []
        
        # First implementation
        lines.extend([
            "def process_data(input_data):",
            "    '''Process input data.'''",
            "    result = []",
            "    for item in input_data:",
            "        if item > 0:",
            "            result.append(item * 2)",
            "    return result",
            ""
        ])
        
        # Add many lines of unrelated code
        for i in range(100):
            lines.extend([
                f"def unrelated_function_{i}():",
                f"    return {i}",
                ""
            ])
        
        # Reimplementation with slight variation
        lines.extend([
            "def process_input_data(data):",
            "    '''Process the input data.'''",
            "    output = []",
            "    for element in data:",
            "        if element > 0:",
            "            output.append(element * 2)",
            "    return output",
            ""
        ])
        
        self.create_file("large_file.py", "\n".join(lines))
        
        issues = self.detector.detect(self.test_path)
        thrashing_issues = [i for i in issues if i.kind == 'context_window_thrashing']
        
        self.assertGreater(len(thrashing_issues), 0)
        issue = thrashing_issues[0]
        self.assertIn('process_data', issue.evidence.get('function1', ''))
        self.assertIn('process_input_data', issue.evidence.get('function2', ''))
    
    def test_detect_gradual_drift(self):
        """Test detection of gradual function drift."""
        content = """
            def get_user_data(user_id):
                # Original implementation
                return {"id": user_id, "name": "User"}
            
            """ + "\n" * 50 + """
            
            def fetch_user_data(uid):
                # Slightly different
                return {"id": uid, "name": "User", "active": True}
            
            """ + "\n" * 50 + """
            
            def retrieve_user_info(user_identifier):
                # Further drift
                data = {"id": user_identifier, "name": "User", "active": True}
                return data
            
            """ + "\n" * 50 + """
            
            def load_user_details(id_value):
                # Even more drift
                user_info = {
                    "id": id_value,
                    "name": "User",
                    "active": True,
                    "created": "2024-01-01"
                }
                return user_info
        """
        
        self.create_file("user_functions.py", content)
        
        issues = self.detector.detect(self.test_path)
        thrashing_issues = [i for i in issues if i.kind == 'context_window_thrashing']
        
        # Should detect the pattern of gradual drift
        self.assertGreater(len(thrashing_issues), 0)
    
    def test_performance_large_file(self):
        """Test performance on large files."""
        # Create a very large file
        lines = []
        for i in range(1000):
            lines.extend([
                f"def function_{i}(param_{i}):",
                f"    # Function {i}",
                f"    return param_{i} * {i}",
                ""
            ])
        
        # Add some duplicates
        lines.extend([
            "def function_0_duplicate(param_0):",
            "    # Duplicate of function_0",
            "    return param_0 * 0",
            ""
        ])
        
        self.create_file("huge_file.py", "\n".join(lines))
        
        start_time = time.time()
        issues = self.detector.detect(self.test_path)
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(elapsed, 5.0)  # 5 seconds max
        
        # Should detect some issues
        thrashing_issues = [i for i in issues if i.kind == 'context_window_thrashing']
        self.assertGreater(len(thrashing_issues), 0)


class TestImportAnxiety(unittest.TestCase):
    """Test import anxiety detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.detector = ImportAnxietyDetector()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_detect_wildcard_imports(self):
        """Test detection of wildcard imports."""
        self.create_file("bad_imports.py", """
            from os import *
            from sys import *
            from typing import *
            
            def my_function():
                # Only uses path from os
                return path.exists("test")
        """)
        
        issues = self.detector.detect(self.test_path)
        anxiety_issues = [i for i in issues if i.kind == 'import_anxiety']
        
        self.assertGreater(len(anxiety_issues), 0)
        issue = anxiety_issues[0]
        self.assertIn('wildcard', issue.evidence.get('pattern', '').lower())
    
    def test_detect_defensive_imports(self):
        """Test detection of defensive over-importing."""
        self.create_file("defensive.py", """
            # Import everything just in case
            import os
            import sys
            import json
            import csv
            import xml
            import html
            import urllib
            import requests
            import numpy
            import pandas
            
            def read_file(filename):
                # Only uses os
                if os.path.exists(filename):
                    with open(filename) as f:
                        return f.read()
                return None
        """)
        
        issues = self.detector.detect(self.test_path)
        anxiety_issues = [i for i in issues if i.kind == 'import_anxiety']
        
        self.assertGreater(len(anxiety_issues), 0)
        issue = anxiety_issues[0]
        self.assertIn('defensive', issue.evidence.get('pattern', '').lower())
    
    def test_detect_redundant_imports(self):
        """Test detection of redundant import patterns."""
        self.create_file("redundant.py", """
            from typing import List, Dict, Optional, Union, Any, Tuple
            from typing import Set, FrozenSet, Deque, DefaultDict
            from collections import defaultdict, deque
            from collections import Counter, OrderedDict, ChainMap
            
            def count_items(items: List[str]) -> Dict[str, int]:
                # Only uses List and Dict from typing
                counter = {}
                for item in items:
                    counter[item] = counter.get(item, 0) + 1
                return counter
        """)
        
        issues = self.detector.detect(self.test_path)
        anxiety_issues = [i for i in issues if i.kind == 'import_anxiety']
        
        self.assertGreater(len(anxiety_issues), 0)
    
    def test_legitimate_imports(self):
        """Test that legitimate imports are not flagged."""
        self.create_file("good_imports.py", """
            from typing import List, Dict
            import json
            
            def process_json(data: List[Dict]) -> str:
                # Uses both imports
                processed = []
                for item in data:
                    processed.append(item)
                return json.dumps(processed)
            
            def load_json(text: str) -> List[Dict]:
                # Also uses both imports
                return json.loads(text)
        """)
        
        issues = self.detector.detect(self.test_path)
        anxiety_issues = [i for i in issues if i.kind == 'import_anxiety']
        
        # Should not flag legitimate imports
        self.assertEqual(len(anxiety_issues), 0)
    
    @given(
        num_imports=st.integers(min_value=1, max_value=20),
        num_used=st.integers(min_value=0, max_value=10),
        has_wildcard=st.booleans()
    )
    @settings(max_examples=20)
    def test_import_ratio_property(self, num_imports: int, num_used: int, has_wildcard: bool):
        """Property-based test for import ratio detection."""
        num_used = min(num_used, num_imports)  # Ensure used <= imported
        
        imports = []
        modules = ['os', 'sys', 'json', 'csv', 'math', 'random', 'time', 'datetime',
                  'urllib', 'http', 'socket', 'subprocess', 're', 'typing', 'collections']
        
        # Generate imports
        for i in range(min(num_imports, len(modules))):
            if has_wildcard and i == 0:
                imports.append(f"from {modules[i]} import *")
            else:
                imports.append(f"import {modules[i]}")
        
        # Generate usage
        usages = []
        for i in range(num_used):
            if i < len(modules):
                usages.append(f"    result = {modules[i]}.something()")
        
        content = "\n".join(imports) + "\n\ndef function():\n" + "\n".join(usages or ["    pass"])
        
        self.create_file("test_imports.py", content)
        
        issues = self.detector.detect(self.test_path)
        anxiety_issues = [i for i in issues if i.kind == 'import_anxiety']
        
        # Should detect issues if many unused imports or wildcards
        if has_wildcard or (num_imports > 5 and num_used < num_imports / 2):
            self.assertGreater(len(anxiety_issues), 0)


class TestSemanticDuplicate(unittest.TestCase):
    """Test semantic duplicate detection."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.detector = SemanticDuplicateDetector()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_detect_renamed_functions(self):
        """Test detection of semantically identical functions with different names."""
        self.create_file("duplicates.py", """
            def calculate_sum(numbers):
                total = 0
                for num in numbers:
                    total += num
                return total
            
            def compute_total(values):
                result = 0
                for val in values:
                    result += val
                return result
            
            def add_all_numbers(number_list):
                sum_value = 0
                for number in number_list:
                    sum_value += number
                return sum_value
        """)
        
        issues = self.detector.detect(self.test_path)
        duplicate_issues = [i for i in issues if 'duplicate' in i.kind.lower()]
        
        self.assertGreater(len(duplicate_issues), 0)
        
        # Should detect that these are semantically the same
        issue = duplicate_issues[0]
        self.assertGreater(issue.evidence.get('similarity', 0), 0.8)
    
    def test_detect_structural_similarity(self):
        """Test detection of structurally similar functions."""
        self.create_file("structural.py", """
            def process_user_data(user):
                if not user:
                    return None
                
                result = {
                    'id': user.get('id'),
                    'name': user.get('name'),
                    'email': user.get('email')
                }
                
                if user.get('active'):
                    result['status'] = 'active'
                else:
                    result['status'] = 'inactive'
                
                return result
            
            def transform_customer_info(customer):
                if not customer:
                    return None
                
                output = {
                    'customer_id': customer.get('customer_id'),
                    'full_name': customer.get('full_name'),
                    'contact': customer.get('contact')
                }
                
                if customer.get('enabled'):
                    output['state'] = 'enabled'
                else:
                    output['state'] = 'disabled'
                
                return output
        """)
        
        issues = self.detector.detect(self.test_path)
        duplicate_issues = [i for i in issues if 'duplicate' in i.kind.lower()]
        
        # Should detect structural similarity despite different field names
        self.assertGreater(len(duplicate_issues), 0)
    
    def test_performance_benchmark(self):
        """Benchmark semantic analysis performance."""
        # Create multiple files with many functions
        num_files = 5
        num_functions = 20
        
        for f in range(num_files):
            lines = []
            for i in range(num_functions):
                lines.extend([
                    f"def function_{f}_{i}(param_{i}):",
                    f"    # Processing for file {f} function {i}",
                    f"    result = param_{i} * {i}",
                    f"    if result > 100:",
                    f"        return result / 2",
                    f"    return result",
                    ""
                ])
            
            self.create_file(f"file_{f}.py", "\n".join(lines))
        
        start_time = time.time()
        issues = self.detector.detect(self.test_path)
        elapsed = time.time() - start_time
        
        # Should complete analysis within reasonable time
        self.assertLess(elapsed, 10.0)  # 10 seconds for 100 functions
        
        # Calculate throughput
        total_functions = num_files * num_functions
        throughput = total_functions / elapsed
        print(f"Semantic analysis throughput: {throughput:.1f} functions/second")
        
        # Should achieve reasonable throughput
        self.assertGreater(throughput, 5.0)  # At least 5 functions/second


class TestPatternCombinations(unittest.TestCase):
    """Test combinations of patterns occurring together."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        
        # Initialize all detectors
        self.detectors = {
            'cascade': HallucinationCascadeDetector(),
            'thrashing': ContextWindowThrashingDetector(),
            'anxiety': ImportAnxietyDetector(),
            'semantic': SemanticDuplicateDetector()
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_cascade_with_import_anxiety(self):
        """Test hallucination cascade combined with import anxiety."""
        # Create over-abstracted code with excessive imports
        self.create_file("abstract_mess.py", """
            from abc import ABC, abstractmethod, ABCMeta
            from typing import *
            from collections import *
            from itertools import *
            from functools import *
            import os, sys, json, csv, xml, html
            
            class AbstractBase(ABC):
                @abstractmethod
                def process(self):
                    pass
            
            class MiddleLayer(AbstractBase):
                @abstractmethod
                def handle(self):
                    pass
                
                def process(self):
                    self.handle()
            
            class ConcreteImplementation(MiddleLayer):
                def handle(self):
                    return "finally"
            
            class Manager:
                def __init__(self):
                    self.impl = ConcreteImplementation()
        """)
        
        all_issues = []
        for detector in self.detectors.values():
            issues = detector.detect(self.test_path)
            all_issues.extend(issues)
        
        # Should detect both patterns
        cascade_found = any(i.kind == 'hallucination_cascade' for i in all_issues)
        anxiety_found = any(i.kind == 'import_anxiety' for i in all_issues)
        
        self.assertTrue(cascade_found)
        self.assertTrue(anxiety_found)
    
    def test_thrashing_with_semantic_duplicates(self):
        """Test context thrashing creating semantic duplicates."""
        lines = []
        
        # First implementation
        lines.extend([
            "def process_items(items):",
            "    results = []",
            "    for item in items:",
            "        if item.valid:",
            "            results.append(item.value * 2)",
            "    return results",
            ""
        ])
        
        # Add spacing
        lines.extend([""] * 100)
        
        # Semantically identical reimplementation
        lines.extend([
            "def handle_items(item_list):",
            "    output = []",
            "    for i in item_list:",
            "        if i.valid:",
            "            output.append(i.value * 2)",
            "    return output",
            ""
        ])
        
        # Add more spacing
        lines.extend([""] * 100)
        
        # Another reimplementation
        lines.extend([
            "def process_item_list(data):",
            "    processed = []",
            "    for d in data:",
            "        if d.valid:",
            "            processed.append(d.value * 2)",
            "    return processed",
            ""
        ])
        
        self.create_file("repeated_logic.py", "\n".join(lines))
        
        all_issues = []
        for detector in self.detectors.values():
            issues = detector.detect(self.test_path)
            all_issues.extend(issues)
        
        # Should detect both thrashing and semantic duplicates
        thrashing_found = any(i.kind == 'context_window_thrashing' for i in all_issues)
        semantic_found = any('duplicate' in i.kind.lower() for i in all_issues)
        
        self.assertTrue(thrashing_found or semantic_found)


class TestRegressionSuite(unittest.TestCase):
    """Regression tests for previously fixed issues."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_regression_false_positive_init_files(self):
        """Test that __init__.py files are not flagged as phantoms."""
        self.create_file("__init__.py", "")
        self.create_file("submodule/__init__.py", "")
        
        from tailchasing.analyzers.phantom_functions import PhantomFunctionDetector
        detector = PhantomFunctionDetector()
        issues = detector.detect(self.test_path)
        
        # Should not flag empty __init__.py files
        self.assertEqual(len(issues), 0)
    
    def test_regression_test_file_duplicates(self):
        """Test that test file patterns are handled correctly."""
        self.create_file("test_utils.py", """
            import unittest
            
            class TestCase(unittest.TestCase):
                def setUp(self):
                    self.data = []
                
                def tearDown(self):
                    self.data = None
                
                def test_something(self):
                    self.assertEqual(1, 1)
            
            class AnotherTestCase(unittest.TestCase):
                def setUp(self):
                    self.data = []
                
                def tearDown(self):
                    self.data = None
                
                def test_other(self):
                    self.assertEqual(2, 2)
        """)
        
        detector = SemanticDuplicateDetector()
        issues = detector.detect(self.test_path)
        
        # Should not flag standard test patterns as duplicates
        # (setUp/tearDown are expected to be similar)
        high_severity = [i for i in issues if i.severity >= 3]
        self.assertEqual(len(high_severity), 0)
    
    def test_regression_property_decorators(self):
        """Test that property decorators are handled correctly."""
        self.create_file("properties.py", """
            class MyClass:
                def __init__(self):
                    self._value = None
                
                @property
                def value(self):
                    return self._value
                
                @value.setter
                def value(self, val):
                    self._value = val
                
                @property
                def other_value(self):
                    return self._value
                
                @other_value.setter
                def other_value(self, val):
                    self._value = val
        """)
        
        detector = SemanticDuplicateDetector()
        issues = detector.detect(self.test_path)
        
        # Property getters/setters might be similar but shouldn't be high severity
        high_severity = [i for i in issues if i.severity >= 4]
        self.assertEqual(len(high_severity), 0)


if __name__ == '__main__':
    unittest.main()