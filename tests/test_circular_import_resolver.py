"""Tests for circular import resolver."""

import ast
import pytest
import networkx as nx
from pathlib import Path
import tempfile
import textwrap

from tailchasing.analyzers.circular_import_resolver import (
    ImportGraphBuilder,
    SCCAnalyzer,
    CircularImportFixGenerator,
    CircularImportResolver,
    ImportDependency,
    SCCAnalysis,
    CircularImportFix
)
from tailchasing.core.issues import Issue


class TestImportGraphBuilder:
    """Test import graph building functionality."""
    
    def create_test_ast_index(self) -> dict:
        """Create test AST index with circular imports."""
        # Module A imports B
        code_a = textwrap.dedent("""
            from b_module import some_function
            import c_module
            
            def function_a():
                return some_function()
        """)
        
        # Module B imports A  
        code_b = textwrap.dedent("""
            from a_module import function_a
            
            def some_function():
                return function_a()
        """)
        
        # Module C (no circular deps)
        code_c = textwrap.dedent("""
            def function_c():
                pass
        """)
        
        return {
            'a_module.py': ast.parse(code_a),
            'b_module.py': ast.parse(code_b),
            'c_module.py': ast.parse(code_c)
        }
    
    def test_build_graph(self):
        """Test building import graph from AST."""
        builder = ImportGraphBuilder()
        ast_index = self.create_test_ast_index()
        
        graph = builder.build_graph(ast_index)
        
        # Check nodes
        assert 'a_module' in graph.nodes
        assert 'b_module' in graph.nodes
        assert 'c_module' in graph.nodes
        
        # Check edges (should have circular dependency)
        assert graph.has_edge('a_module', 'b_module')
        assert graph.has_edge('b_module', 'a_module')
        assert graph.has_edge('a_module', 'c_module')
    
    def test_file_to_module(self):
        """Test file path to module conversion."""
        builder = ImportGraphBuilder()
        
        assert builder._file_to_module('path/to/module.py') == 'path.to.module'
        assert builder._file_to_module('package/__init__.py') == 'package'
        assert builder._file_to_module('simple.py') == 'simple'
    
    def test_extract_imports(self):
        """Test import extraction from AST."""
        builder = ImportGraphBuilder()
        
        code = textwrap.dedent("""
            import os
            from sys import path
            from collections import defaultdict
            
            def func():
                import json  # Function-level import
        """)
        
        tree = ast.parse(code)
        imports = builder._extract_imports(tree, 'test_module', 'test.py')
        
        # Check import types
        import_names = {imp.target_module for imp in imports}
        assert 'os' in import_names
        assert 'sys' in import_names
        assert 'collections' in import_names
        assert 'json' in import_names
        
        # Check module-level vs function-level
        module_level_imports = [imp for imp in imports if imp.is_module_level]
        function_level_imports = [imp for imp in imports if not imp.is_module_level]
        
        assert len(module_level_imports) == 3  # os, sys, collections
        assert len(function_level_imports) == 1  # json
    
    def test_calculate_import_weight(self):
        """Test import weight calculation."""
        builder = ImportGraphBuilder()
        
        # Module-level from import
        module_import = ImportDependency(
            source_module='a', target_module='b', import_type='from',
            imported_names=['func1', 'func2'], line_number=1, 
            is_module_level=True, ast_node=None
        )
        
        # Function-level simple import
        func_import = ImportDependency(
            source_module='a', target_module='b', import_type='module',
            imported_names=['b'], line_number=5, 
            is_module_level=False, ast_node=None
        )
        
        # Star import
        star_import = ImportDependency(
            source_module='a', target_module='b', import_type='star',
            imported_names=['*'], line_number=1, 
            is_module_level=True, ast_node=None
        )
        
        module_weight = builder._calculate_import_weight(module_import)
        func_weight = builder._calculate_import_weight(func_import)
        star_weight = builder._calculate_import_weight(star_import)
        
        # Module-level should be heavier than function-level
        assert module_weight > func_weight
        
        # Star imports should be heaviest
        assert star_weight > module_weight
        assert star_weight > func_weight


class TestSCCAnalyzer:
    """Test SCC analysis functionality."""
    
    def create_circular_graph(self) -> ImportGraphBuilder:
        """Create graph builder with circular dependencies."""
        builder = ImportGraphBuilder()
        
        # Create a simple circular dependency: A -> B -> A
        ast_index = {
            'module_a.py': ast.parse('from module_b import func_b'),
            'module_b.py': ast.parse('from module_a import func_a')
        }
        
        builder.build_graph(ast_index)
        return builder
    
    def test_find_sccs(self):
        """Test SCC detection."""
        builder = self.create_circular_graph()
        analyzer = SCCAnalyzer(builder)
        
        analyses = analyzer.find_and_analyze_sccs()
        
        # Should find one SCC with 2 modules
        assert len(analyses) == 1
        assert len(analyses[0].modules) == 2
        assert 'module_a' in analyses[0].modules
        assert 'module_b' in analyses[0].modules
    
    def test_suggest_shared_module_name(self):
        """Test shared module name suggestion."""
        builder = self.create_circular_graph()
        analyzer = SCCAnalyzer(builder)
        
        # Test common prefix detection
        modules = ['package.submodule.a', 'package.submodule.b']
        suggested = analyzer._suggest_shared_module_name(modules)
        assert 'package.submodule.shared' == suggested
        
        # Test fallback for no common prefix
        modules = ['completely.different.a', 'other.path.b']
        suggested = analyzer._suggest_shared_module_name(modules)
        assert suggested.endswith('shared')
    
    def test_identify_break_points(self):
        """Test break point identification."""
        builder = self.create_circular_graph()
        analyzer = SCCAnalyzer(builder)
        
        scc = {'module_a', 'module_b'}
        import_edges = []  # Simplified for test
        
        break_points = analyzer._identify_break_points(scc, import_edges)
        
        # Should identify potential break points
        assert isinstance(break_points, list)


class TestCircularImportFixGenerator:
    """Test fix generation functionality."""
    
    def create_test_scc_analysis(self) -> SCCAnalysis:
        """Create test SCC analysis."""
        from tailchasing.analyzers.circular_import_resolver import SharedSymbol
        
        shared_symbols = [
            SharedSymbol(
                name='CommonClass',
                symbol_type='class',
                used_in_modules={'module_a', 'module_b'},
                usage_count=2
            ),
            SharedSymbol(
                name='CONSTANT',
                symbol_type='constant',
                used_in_modules={'module_a', 'module_b'},
                usage_count=2
            )
        ]
        
        return SCCAnalysis(
            scc_id='TEST_SCC',
            modules=['module_a', 'module_b'],
            import_edges=[],
            shared_symbols=shared_symbols,
            suggested_shared_module='shared_module',
            break_points=[('module_a', 'module_b')],
            complexity_score=10.0
        )
    
    def test_generate_shared_module_fix(self):
        """Test shared module fix generation."""
        generator = CircularImportFixGenerator()
        analysis = self.create_test_scc_analysis()
        
        fix = generator._generate_shared_module_fix(analysis)
        
        assert isinstance(fix, CircularImportFix)
        assert fix.fix_type == 'extract_shared'
        assert 'shared_module' in fix.target_file
        assert 'CommonClass' in fix.codemod
        assert 'CONSTANT' in fix.codemod
        assert fix.risk_level == 'HIGH'  # Shared module extraction is high risk
    
    def test_generate_function_scope_fix(self):
        """Test function scope fix generation."""
        generator = CircularImportFixGenerator()
        
        import_edge = ImportDependency(
            source_module='module_a',
            target_module='module_b',
            import_type='from',
            imported_names=['some_func'],
            line_number=5,
            is_module_level=True,
            ast_node=None
        )
        
        fix = generator._generate_function_scope_fix(import_edge)
        
        assert fix.fix_type == 'function_scope'
        assert 'Move import' in fix.description
        assert 'from module_b import some_func' in fix.codemod
        assert fix.risk_level == 'MEDIUM'
    
    def test_generate_lazy_import_fix(self):
        """Test lazy import fix generation."""
        generator = CircularImportFixGenerator()
        analysis = self.create_test_scc_analysis()
        
        fix = generator._generate_lazy_import_fix('module_a', 'module_b', analysis)
        
        assert fix.fix_type == 'lazy_import'
        assert 'TYPE_CHECKING' in fix.codemod
        assert 'lazy import' in fix.description.lower()
        assert fix.risk_level == 'LOW'
    
    def test_topological_sort_fixes(self):
        """Test topological sorting of fixes."""
        generator = CircularImportFixGenerator()
        
        # Create fixes with dependencies
        fix1 = CircularImportFix('fix1', 'test', 'file1.py', 'Fix 1', '', 'LOW', [])
        fix2 = CircularImportFix('fix2', 'test', 'file2.py', 'Fix 2', '', 'LOW', ['fix1'])
        fix3 = CircularImportFix('fix3', 'test', 'file3.py', 'Fix 3', '', 'LOW', ['fix1', 'fix2'])
        
        fixes = [fix3, fix1, fix2]  # Out of order
        sorted_fixes = generator._topological_sort_fixes(fixes)
        
        # Should be in dependency order
        fix_ids = [fix.fix_id for fix in sorted_fixes]
        assert fix_ids.index('fix1') < fix_ids.index('fix2')
        assert fix_ids.index('fix2') < fix_ids.index('fix3')


class TestCircularImportResolver:
    """Test the main resolver."""
    
    def test_analyzer_instantiation(self):
        """Test resolver can be instantiated."""
        config = {
            'circular_import_resolver': {
                'enabled': True,
                'min_scc_size': 2
            }
        }
        
        resolver = CircularImportResolver(config)
        assert resolver.name == 'circular_import_resolver'
        assert resolver.enabled is True
        assert resolver.min_scc_size == 2
    
    def test_run_with_no_cycles(self):
        """Test resolver with no circular imports."""
        config = {'circular_import_resolver': {'enabled': True}}
        resolver = CircularImportResolver(config)
        
        # Mock context with simple AST
        class MockCtx:
            ast_index = {
                'simple.py': ast.parse('def func(): pass')
            }
        
        issues = resolver.run(MockCtx())
        
        # Should have no circular import issues
        circular_issues = [i for i in issues if 'circular_import' in i.kind]
        assert len(circular_issues) == 0
    
    def test_create_scc_issue(self):
        """Test SCC issue creation."""
        config = {'circular_import_resolver': {'enabled': True}}
        resolver = CircularImportResolver(config)
        
        from tailchasing.analyzers.circular_import_resolver import SharedSymbol
        
        scc = SCCAnalysis(
            scc_id='TEST_001',
            modules=['module_a', 'module_b'],
            import_edges=[],
            shared_symbols=[
                SharedSymbol('TestClass', 'class', {'module_a', 'module_b'})
            ],
            suggested_shared_module='test_shared',
            break_points=[],
            complexity_score=15.0
        )
        
        issue = resolver._create_scc_issue(scc, [])
        
        assert issue.kind == 'circular_import_scc'
        assert issue.severity in [3, 4]  # High severity for circular imports
        assert 'module_a' in issue.message
        assert 'module_b' in issue.message
        assert scc.scc_id in issue.evidence['scc_id']
    
    def test_generate_fix_script(self):
        """Test fix script generation."""
        config = {'circular_import_resolver': {'enabled': True}}
        resolver = CircularImportResolver(config)
        
        # Create mock orchestration issue
        orchestration_issue = Issue(
            kind='circular_import_fix_orchestration',
            message='Test orchestration',
            severity=1,
            file='<orchestration>',
            line=0,
            evidence={
                'fixes': [{
                    'fix_id': 'test_fix',
                    'fix_type': 'extract_shared',
                    'target_file': 'shared.py',
                    'description': 'Test fix',
                    'codemod': 'test codemod',
                    'risk_level': 'MEDIUM',
                    'dependencies': []
                }]
            }
        )
        
        script = resolver.generate_fix_script([orchestration_issue], None)
        
        assert '#!/usr/bin/env python3' in script
        assert 'Test fix' in script
        assert 'extract_shared' in script
        assert 'def apply_circular_import_fixes():' in script


class TestIntegration:
    """Integration tests for circular import resolver."""
    
    def test_full_pipeline_with_circular_imports(self):
        """Test complete pipeline with actual circular imports."""
        # Create test files with circular dependencies
        code_a = textwrap.dedent("""
            from module_b import ClassB
            
            class ClassA:
                def method(self):
                    return ClassB()
        """)
        
        code_b = textwrap.dedent("""
            from module_a import ClassA
            
            class ClassB:
                def method(self):
                    return ClassA()
        """)
        
        ast_index = {
            'module_a.py': ast.parse(code_a),
            'module_b.py': ast.parse(code_b)
        }
        
        # Build import graph
        builder = ImportGraphBuilder()
        graph = builder.build_graph(ast_index)
        
        # Analyze SCCs
        analyzer = SCCAnalyzer(builder)
        analyses = analyzer.find_and_analyze_sccs()
        
        # Should find one circular dependency
        assert len(analyses) == 1
        assert len(analyses[0].modules) == 2
        
        # Generate fixes
        fix_generator = CircularImportFixGenerator()
        fixes = fix_generator.generate_fixes(analyses)
        
        # Should generate some fixes
        assert len(fixes) > 0
        
        # Should have some kind of fixes (function-scope, lazy, or shared)
        assert len(fixes) > 0
        
        # Check that we have reasonable fix types
        fix_types = {f.fix_type for f in fixes}
        expected_types = {'function_scope', 'lazy_import', 'extract_shared'}
        assert fix_types.intersection(expected_types)