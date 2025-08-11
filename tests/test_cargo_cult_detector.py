"""
Tests for the cargo cult detector with intelligent parent __init__ checking.
"""

import ast
import pytest
from pathlib import Path

from tailchasing.analyzers.cargo_cult_detector import CargoCultDetector, EnhancedCargoCultDetector
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.symbols import SymbolTable


def create_context_with_code(code_dict):
    """Create an AnalysisContext with the given code files."""
    ast_index = {}
    for filepath, code in code_dict.items():
        ast_index[filepath] = ast.parse(code)
    
    return AnalysisContext(
        config={},
        root_dir=Path("/test"),
        file_paths=[Path(f) for f in code_dict.keys()],
        ast_index=ast_index,
        symbol_table=SymbolTable(),
        source_cache={},
        cache={}
    )


class TestCargoCultDetector:
    """Test the cargo cult detector's parent __init__ checking."""
    
    def test_unnecessary_super_no_parent_init(self):
        """Test detection of unnecessary super().__init__() when parent has no __init__."""
        code = """
class Parent:
    pass

class Child(Parent):
    def __init__(self):
        super().__init__()  # Unnecessary!
        self.value = 1
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        assert len(issues) == 1
        assert issues[0].kind == "unnecessary_super_init"
        assert "Child" in issues[0].message
    
    def test_necessary_super_parent_has_init(self):
        """Test that super().__init__() is not flagged when parent has __init__."""
        code = """
class Parent:
    def __init__(self):
        self.parent_value = 1

class Child(Parent):
    def __init__(self):
        super().__init__()  # Necessary!
        self.child_value = 2
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        assert len(issues) == 0  # No issues - super() is needed
    
    def test_parent_with_trivial_init(self):
        """Test that trivial parent __init__ (only pass/docstring) is detected."""
        code = """
class Parent:
    def __init__(self):
        '''Initialize parent.'''
        pass

class Child(Parent):
    def __init__(self):
        super().__init__()  # Unnecessary - parent init is trivial
        self.value = 1
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        assert len(issues) == 1
        assert issues[0].kind == "unnecessary_super_init"
    
    def test_parent_with_raise_init(self):
        """Test that parent __init__ with raise is considered meaningful."""
        code = """
class AbstractBase:
    def __init__(self):
        raise NotImplementedError("Must implement in subclass")

class Concrete(AbstractBase):
    def __init__(self):
        super().__init__()  # This would actually raise - likely a bug!
        self.value = 1
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        # No issue - parent has meaningful __init__ (even though it raises)
        assert len(issues) == 0
    
    def test_deep_inheritance_chain(self):
        """Test checking through multiple inheritance levels."""
        code = """
class GrandParent:
    pass

class Parent(GrandParent):
    pass

class Child(Parent):
    def __init__(self):
        super().__init__()  # Unnecessary - no ancestor has __init__
        self.value = 1
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        assert len(issues) == 1
        assert issues[0].kind == "unnecessary_super_init"
    
    def test_deep_inheritance_with_init(self):
        """Test deep inheritance where grandparent has __init__."""
        code = """
class GrandParent:
    def __init__(self):
        self.gp_value = 1

class Parent(GrandParent):
    pass  # No __init__ here

class Child(Parent):
    def __init__(self):
        super().__init__()  # Necessary - grandparent has __init__
        self.child_value = 2
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        assert len(issues) == 0  # No issue - grandparent has __init__
    
    def test_multiple_inheritance(self):
        """Test multiple inheritance scenarios."""
        code = """
class Mixin:
    def __init__(self):
        self.mixin_value = 1

class Base:
    pass

class Child(Base, Mixin):
    def __init__(self):
        super().__init__()  # Necessary - Mixin has __init__
        self.child_value = 2
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        assert len(issues) == 0  # No issue - Mixin has __init__
    
    def test_qualified_base_names(self):
        """Test handling of qualified base class names (module.Class)."""
        code = """
import collections

class MyDict(collections.UserDict):
    def __init__(self):
        super().__init__()  # Necessary - UserDict has __init__
        self.custom = 1
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        # Should be conservative with external classes
        assert len(issues) == 0
    
    def test_protocol_base(self):
        """Test that Protocol base classes are handled correctly."""
        code = """
from typing import Protocol

class MyProtocol(Protocol):
    def method(self) -> int:
        ...

class Implementation(MyProtocol):
    def __init__(self):
        super().__init__()  # Unnecessary - Protocol has no __init__
        self.value = 1
    
    def method(self) -> int:
        return self.value
"""
        ctx = create_context_with_code({"test.py": code})
        detector = CargoCultDetector()
        issues = detector.run(ctx)
        
        assert len(issues) == 1
        assert issues[0].kind == "unnecessary_super_init"


class TestEnhancedCargoCultDetector:
    """Test the enhanced cargo cult detector with additional patterns."""
    
    def test_unnecessary_object_inheritance(self):
        """Test detection of explicit object inheritance."""
        code = """
class MyClass(object):  # Python 2 style
    pass
"""
        ctx = create_context_with_code({"test.py": code})
        detector = EnhancedCargoCultDetector()
        issues = detector.run(ctx)
        
        assert any(i.kind == "unnecessary_object_inheritance" for i in issues)
    
    def test_trivial_accessors(self):
        """Test detection of trivial getters and setters."""
        code = """
class Person:
    def __init__(self):
        self._name = ""
    
    def get_name(self):
        return self._name  # Trivial getter
    
    def set_name(self, value):
        self._name = value  # Trivial setter
"""
        ctx = create_context_with_code({"test.py": code})
        detector = EnhancedCargoCultDetector()
        issues = detector.run(ctx)
        
        assert any(i.kind == "trivial_accessors" for i in issues)
    
    def test_unnecessary_class(self):
        """Test detection of classes that could be functions."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b  # Doesn't use self - could be a function
"""
        ctx = create_context_with_code({"test.py": code})
        detector = EnhancedCargoCultDetector()
        issues = detector.run(ctx)
        
        assert any(i.kind == "unnecessary_class" for i in issues)
    
    def test_necessary_class(self):
        """Test that classes using self are not flagged."""
        code = """
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1  # Uses self - needs to be a class
        return self.count
"""
        ctx = create_context_with_code({"test.py": code})
        detector = EnhancedCargoCultDetector()
        issues = detector.run(ctx)
        
        # Should not flag as unnecessary class
        assert not any(i.kind == "unnecessary_class" for i in issues)


if __name__ == "__main__":
    # Run tests
    test_detector = TestCargoCultDetector()
    test_detector.test_unnecessary_super_no_parent_init()
    test_detector.test_necessary_super_parent_has_init()
    test_detector.test_parent_with_trivial_init()
    test_detector.test_deep_inheritance_chain()
    test_detector.test_deep_inheritance_with_init()
    test_detector.test_multiple_inheritance()
    
    test_enhanced = TestEnhancedCargoCultDetector()
    test_enhanced.test_unnecessary_object_inheritance()
    test_enhanced.test_trivial_accessors()
    test_enhanced.test_unnecessary_class()
    test_enhanced.test_necessary_class()
    
    print("All tests passed!")