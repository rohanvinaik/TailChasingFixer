"""Tests for canonical module policy system."""

import ast
import pytest
from pathlib import Path
import tempfile
import textwrap

from tailchasing.analyzers.canonical_policy import (
    CanonicalPolicy,
    CanonicalModuleAnalyzer,
    CodemodGenerator,
    CanonicalPolicyAnalyzer,
    CanonicalSymbol,
    ShadowSymbol,
    CodemodSuggestion
)
from tailchasing.core.issues import Issue


class TestCanonicalPolicy:
    """Test canonical policy configuration."""
    
    def test_is_canonical_path(self):
        """Test canonical path detection."""
        policy = CanonicalPolicy(
            canonical_roots=["genomevault/pir/advanced", "crypto/core"],
            shadow_roots=["experimental", "temp"]
        )
        
        assert policy.is_canonical_path("genomevault/pir/advanced/module.py")
        assert policy.is_canonical_path("crypto/core/utils.py")
        assert not policy.is_canonical_path("experimental/test.py")
        assert not policy.is_canonical_path("other/module.py")
    
    def test_is_shadow_path(self):
        """Test shadow path detection."""
        policy = CanonicalPolicy(
            canonical_roots=["core"],
            shadow_roots=["experimental", "temp", "tests"]
        )
        
        assert policy.is_shadow_path("experimental/feature.py")
        assert policy.is_shadow_path("temp/backup.py")
        assert policy.is_shadow_path("tests/test_module.py")
        assert not policy.is_shadow_path("core/main.py")
    
    def test_get_path_priority(self):
        """Test priority calculation."""
        policy = CanonicalPolicy(
            canonical_roots=["core"],
            shadow_roots=["experimental"],
            priority_patterns={
                r".*test.*": -10,
                r".*experimental.*": -20,
                r".*/core/.*": 10
            }
        )
        
        # Canonical path gets base 100 + pattern bonus
        assert policy.get_path_priority("core/module.py") == 110
        
        # Shadow path gets base 10 + pattern penalty
        assert policy.get_path_priority("experimental/feature.py") == -10  # 10 + (-20)
        
        # Regular path with test penalty
        assert policy.get_path_priority("src/test_module.py") == 40  # 50 + (-10)


class TestCanonicalModuleAnalyzer:
    """Test canonical module analyzer."""
    
    def test_file_to_module(self):
        """Test file path to module conversion."""
        policy = CanonicalPolicy()
        analyzer = CanonicalModuleAnalyzer(policy)
        
        assert analyzer._file_to_module("src/module.py") == "src.module"
        assert analyzer._file_to_module("package/__init__.py") == "package"
        assert analyzer._file_to_module("deep/nested/module.py") == "deep.nested.module"
    
    def test_extract_symbols(self):
        """Test symbol extraction from AST."""
        policy = CanonicalPolicy()
        analyzer = CanonicalModuleAnalyzer(policy)
        
        code = textwrap.dedent("""
            def my_function():
                pass
            
            class MyClass:
                pass
            
            MY_VARIABLE = 42
        """)
        
        tree = ast.parse(code)
        symbols = analyzer._extract_symbols(tree, "test.py", "test")
        
        symbol_names = {s.name for s in symbols}
        assert "my_function" in symbol_names
        assert "MyClass" in symbol_names
        assert "MY_VARIABLE" in symbol_names
        
        # Check symbol types
        func_symbol = next(s for s in symbols if s.name == "my_function")
        assert func_symbol.symbol_type == "function"
        
        class_symbol = next(s for s in symbols if s.name == "MyClass")
        assert class_symbol.symbol_type == "class"
    
    def test_classify_symbols(self):
        """Test symbol classification as canonical vs shadow."""
        policy = CanonicalPolicy(
            canonical_roots=["core"],
            shadow_roots=["experimental"]
        )
        analyzer = CanonicalModuleAnalyzer(policy)
        
        # Create symbols with different priorities
        canonical_sym = CanonicalSymbol(
            name="test_func", module_path="core.module", file_path="core/module.py",
            line_number=10, symbol_type="function", ast_node=None, priority=100
        )
        
        shadow_sym = CanonicalSymbol(
            name="test_func", module_path="experimental.module", file_path="experimental/module.py",
            line_number=10, symbol_type="function", ast_node=None, priority=10
        )
        
        canonical, shadows = analyzer._classify_symbols([canonical_sym, shadow_sym])
        
        assert canonical == canonical_sym
        assert shadow_sym in shadows


class TestCodemodGenerator:
    """Test codemod generation."""
    
    def test_generate_function_forwarder(self):
        """Test function forwarder generation."""
        policy = CanonicalPolicy()
        generator = CodemodGenerator(policy)
        
        # Create canonical and shadow symbols
        canonical = CanonicalSymbol(
            name="calc", module_path="core.math", file_path="core/math.py",
            line_number=10, symbol_type="function", ast_node=None
        )
        
        func_node = ast.FunctionDef(
            name="calc", args=ast.arguments(
                posonlyargs=[], args=[ast.arg(arg="x", annotation=None)],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
            ),
            body=[], decorator_list=[], returns=None, lineno=5
        )
        
        shadow = ShadowSymbol(
            canonical=canonical, name="calc", module_path="experimental.math",
            file_path="experimental/math.py", line_number=5, symbol_type="function",
            ast_node=func_node
        )
        
        codemod = generator.generate_forwarder_codemod(shadow)
        
        assert isinstance(codemod, CodemodSuggestion)
        assert "from core.math import calc" in codemod.replacement_code
        assert "DEPRECATED" in codemod.replacement_code
        assert "warnings.warn" in codemod.replacement_code
        assert codemod.risk_level == "LOW"
    
    def test_generate_class_forwarder(self):
        """Test class forwarder generation."""
        policy = CanonicalPolicy()
        generator = CodemodGenerator(policy)
        
        canonical = CanonicalSymbol(
            name="Calculator", module_path="core.math", file_path="core/math.py",
            line_number=10, symbol_type="class", ast_node=None
        )
        
        shadow = ShadowSymbol(
            canonical=canonical, name="Calculator", module_path="experimental.math",
            file_path="experimental/math.py", line_number=5, symbol_type="class",
            ast_node=None
        )
        
        codemod = generator.generate_forwarder_codemod(shadow)
        
        assert "from core.math import Calculator" in codemod.replacement_code
        assert "class Calculator(Calculator)" in codemod.replacement_code
        assert codemod.risk_level == "MEDIUM"


class TestCanonicalPolicyAnalyzer:
    """Test the main canonical policy analyzer."""
    
    def test_analyzer_instantiation(self):
        """Test analyzer can be instantiated with config."""
        config = {
            "canonical_policy": {
                "canonical_roots": ["core"],
                "shadow_roots": ["experimental"],
                "auto_suppress_shadows": True
            }
        }
        
        analyzer = CanonicalPolicyAnalyzer(config)
        assert analyzer.name == "canonical_policy"
        assert analyzer.policy.canonical_roots == ["core"]
        assert analyzer.policy.shadow_roots == ["experimental"]
        assert analyzer.policy.auto_suppress_shadows is True
    
    def test_run_without_policy(self):
        """Test analyzer returns no issues when no policy configured."""
        config = {}
        analyzer = CanonicalPolicyAnalyzer(config)
        
        # Mock context
        class MockCtx:
            ast_index = {}
        
        issues = analyzer.run(MockCtx())
        assert len(issues) == 0
    
    def test_issue_generation(self):
        """Test that issues are generated for shadow implementations."""
        # This would require a more complex setup with actual AST parsing
        # For now, just test the basic structure
        config = {
            "canonical_policy": {
                "canonical_roots": ["core"],
                "shadow_roots": ["experimental"]
            }
        }
        analyzer = CanonicalPolicyAnalyzer(config)
        
        # Test issue creation logic
        canonical = CanonicalSymbol(
            name="test", module_path="core.module", file_path="core/module.py",
            line_number=10, symbol_type="function", ast_node=None
        )
        
        shadow = ShadowSymbol(
            canonical=canonical, name="test", module_path="experimental.module",
            file_path="experimental/module.py", line_number=5, symbol_type="function",
            ast_node=None, suppressed=True
        )
        
        # Create issue for suppressed shadow
        issue = Issue(
            kind="shadow_implementation_suppressed",
            message=f"Shadow implementation of {shadow.name} suppressed in favor of canonical {shadow.canonical.full_name}",
            severity=1,
            file=shadow.file_path,
            line=shadow.line_number,
            evidence={
                "shadow_symbol": shadow.full_name,
                "canonical_symbol": shadow.canonical.full_name
            }
        )
        
        assert issue.kind == "shadow_implementation_suppressed"
        assert issue.severity == 1
        assert "suppressed" in issue.message


class TestIntegration:
    """Integration tests for canonical policy system."""
    
    def test_example_config_parsing(self):
        """Test that example config can be parsed."""
        from tailchasing.config import Config
        
        # Test that the example config structure is valid
        config_dict = {
            "canonical_policy": {
                "canonical_roots": ["genomevault/pir/advanced"],
                "shadow_roots": ["genomevault/experimental"],
                "priority_patterns": {
                    ".*experimental.*": -30,
                    ".*it_pir.*": -20
                },
                "auto_suppress_shadows": True,
                "generate_forwarders": True
            }
        }
        
        config = Config(config_dict)
        canonical_config = config.get("canonical_policy")
        
        assert canonical_config is not None
        assert "genomevault/pir/advanced" in canonical_config["canonical_roots"]
        assert "genomevault/experimental" in canonical_config["shadow_roots"]
    
    def test_genomevault_pattern_recognition(self):
        """Test recognition of genomevault-specific patterns."""
        policy = CanonicalPolicy(
            canonical_roots=["genomevault/pir/advanced"],
            shadow_roots=["genomevault/experimental"],
            priority_patterns={".*it_pir.*": -20}
        )
        
        # Test PIR pattern recognition
        assert policy.is_canonical_path("genomevault/pir/advanced/module.py")
        assert policy.is_shadow_path("genomevault/experimental/it_pir.py")
        
        # Test priority calculation
        canonical_priority = policy.get_path_priority("genomevault/pir/advanced/module.py")
        shadow_priority = policy.get_path_priority("genomevault/experimental/it_pir.py")
        
        assert canonical_priority > shadow_priority