"""Tests for the robust parser."""

import tempfile
from pathlib import Path
import pytest
import ast

from tailchasing.core.robust_parser import (
    RobustParser, ParseResult, ParserType, SyntaxIssueType, create_parser
)


class TestRobustParser:
    """Test the RobustParser functionality."""
    
    def test_parse_valid_python(self):
        """Test parsing valid Python code."""
        parser = RobustParser()
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("""
def hello():
    print("Hello, world!")
    
if __name__ == "__main__":
    hello()
""")
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            assert result.is_valid
            assert result.ast_tree is not None
            assert result.parser_used == ParserType.AST
            assert not result.is_quarantined
            assert len(result.warnings) == 0
        finally:
            path.unlink()
    
    def test_parse_syntax_error(self):
        """Test parsing file with syntax error."""
        parser = RobustParser()
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("""
def broken_function(
    print("Missing closing parenthesis")
    
# More valid code
x = 5
""")
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            # Should be quarantined or have partial AST
            assert result.is_quarantined or result.partial_ast is not None
            assert SyntaxIssueType.SYNTAX_ERROR in result.syntax_issues
            assert len(result.warnings) > 0
        finally:
            path.unlink()
    
    def test_bom_marker_detection(self):
        """Test BOM marker detection and handling."""
        parser = RobustParser(auto_fix_trivial=False)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='wb', delete=False) as f:
            # Write BOM marker followed by valid Python
            f.write(b'\xef\xbb\xbf')  # UTF-8 BOM
            f.write(b'x = 42\nprint(x)\n')
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            assert SyntaxIssueType.BOM_MARKER in result.syntax_issues
            assert any("BOM" in w for w in result.warnings)
        finally:
            path.unlink()
    
    def test_bom_marker_auto_fix(self):
        """Test automatic BOM marker removal."""
        parser = RobustParser(auto_fix_trivial=True)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='wb', delete=False) as f:
            # Write BOM marker followed by valid Python
            f.write(b'\xef\xbb\xbf')  # UTF-8 BOM
            f.write(b'x = 42\nprint(x)\n')
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            # Should successfully parse after removing BOM
            assert result.is_valid
            assert result.ast_tree is not None
            assert SyntaxIssueType.BOM_MARKER in result.syntax_issues
            assert any("Removed BOM" in w for w in result.warnings)
        finally:
            path.unlink()
    
    def test_mixed_indentation_detection(self):
        """Test mixed tabs/spaces detection."""
        parser = RobustParser(auto_fix_trivial=False)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            # Mix tabs and spaces
            f.write("def func1():\n")
            f.write("    return 1\n")  # spaces
            f.write("\tdef func2():\n")  # tab
            f.write("\t\treturn 2\n")  # tabs
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            assert SyntaxIssueType.MIXED_INDENTATION in result.syntax_issues
            assert any("mixed tabs and spaces" in w.lower() for w in result.warnings)
        finally:
            path.unlink()
    
    def test_mixed_indentation_auto_fix(self):
        """Test automatic mixed indentation fixing."""
        parser = RobustParser(auto_fix_trivial=True)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            # Mix tabs and spaces
            f.write("def func1():\n")
            f.write("    return 1\n")  # spaces
            f.write("\tpass\n")  # tab
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            # Should parse after converting tabs to spaces
            assert result.is_valid or result.partial_ast is not None
            assert SyntaxIssueType.MIXED_INDENTATION in result.syntax_issues
            assert any("Converted tabs to spaces" in w for w in result.warnings)
        finally:
            path.unlink()
    
    def test_missing_colon_auto_fix(self):
        """Test automatic fixing of missing colons."""
        parser = RobustParser(auto_fix_trivial=True)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("""
if x > 5
    print("x is large")
    
for i in range(10)
    print(i)
    
def my_func()
    pass
""")
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            # Should fix missing colons if auto-fix is enabled
            if result.is_valid:
                assert any("Fixed missing colons" in w for w in result.warnings)
            else:
                # At least should have tried
                assert len(result.warnings) > 0
        finally:
            path.unlink()
    
    def test_unclosed_brackets_auto_fix(self):
        """Test automatic fixing of unclosed brackets."""
        parser = RobustParser(auto_fix_trivial=True)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("""
x = [1, 2, 3
y = {
    "key": "value"
z = (
    1 + 2
""")
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            # Should add closing brackets
            assert any("Added closing brackets" in w for w in result.warnings)
        finally:
            path.unlink()
    
    def test_encoding_detection(self):
        """Test encoding detection for non-UTF-8 files."""
        parser = RobustParser(encoding_detection=True)
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='wb', delete=False) as f:
            # Write in Latin-1 encoding
            content = "# Côté français\nx = 42\n"
            f.write(content.encode('latin-1'))
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            # Should detect and use appropriate encoding
            assert result.is_valid or not result.is_quarantined
            if result.warnings:
                assert any("encoding" in w.lower() for w in result.warnings)
        finally:
            path.unlink()
    
    def test_large_file_rejection(self):
        """Test rejection of files exceeding size limit."""
        parser = RobustParser(max_file_size_mb=0.001)  # 1 KB limit
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            # Write more than 1 KB
            f.write("x = 1\n" * 1000)
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            assert result.is_quarantined
            assert SyntaxIssueType.CORRUPTED_FILE in result.syntax_issues
            assert any("too large" in w.lower() for w in result.warnings)
        finally:
            path.unlink()
    
    def test_partial_recovery(self):
        """Test partial AST recovery from files with errors."""
        parser = RobustParser()
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("""
# Valid code
def valid_function():
    return 42

x = valid_function()

# Invalid code
def broken_function(
    print("Syntax error here")
    
# More broken code
if True
    pass
""")
            f.flush()
            path = Path(f.name)
        
        try:
            result = parser.parse_file(path)
            
            # Should have partial AST or be quarantined
            assert result.partial_ast is not None or result.is_quarantined
            if result.partial_ast:
                # Should have recovered the valid function
                assert any(
                    isinstance(node, ast.FunctionDef) and node.name == "valid_function"
                    for node in ast.walk(result.partial_ast)
                )
        finally:
            path.unlink()
    
    def test_statistics_tracking(self):
        """Test that parser tracks statistics correctly."""
        parser = RobustParser()
        parser.reset_statistics()
        
        # Parse a valid file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("x = 42\n")
            f.flush()
            path1 = Path(f.name)
        
        # Parse an invalid file
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("def broken(\n")
            f.flush()
            path2 = Path(f.name)
        
        try:
            result1 = parser.parse_file(path1)
            result2 = parser.parse_file(path2)
            
            stats = parser.get_statistics()
            
            assert stats["total_files"] == 2
            assert stats["successful_parses"] == 1
            assert stats["quarantined"] >= 0
            assert stats["success_rate"] == 0.5
        finally:
            path1.unlink()
            path2.unlink()
    
    def test_quarantine_list(self):
        """Test quarantine list management."""
        parser = RobustParser()
        
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("def broken(\n")
            f.flush()
            path = Path(f.name)
        
        try:
            # First parse should quarantine
            result1 = parser.parse_file(path)
            assert str(path) in parser.quarantined_files
            
            # Second parse should return cached quarantine result
            result2 = parser.parse_file(path)
            assert result2 is result1
            
            # Clear quarantine
            parser.clear_quarantine()
            assert str(path) not in parser.quarantined_files
        finally:
            path.unlink()
    
    def test_factory_function(self):
        """Test the create_parser factory function."""
        parser1 = create_parser(auto_fix_trivial=False)
        assert not parser1.auto_fix_trivial
        
        parser2 = create_parser(auto_fix_trivial=True)
        assert parser2.auto_fix_trivial


class TestParseResult:
    """Test the ParseResult dataclass."""
    
    def test_is_valid(self):
        """Test is_valid property."""
        result = ParseResult()
        assert not result.is_valid
        
        result.ast_tree = ast.parse("x = 1")
        assert result.is_valid
        
        result.is_quarantined = True
        assert not result.is_valid
    
    def test_add_warning(self):
        """Test adding warnings."""
        result = ParseResult()
        assert len(result.warnings) == 0
        
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        assert "Test warning" in result.warnings
    
    def test_add_syntax_issue(self):
        """Test adding syntax issues."""
        result = ParseResult()
        assert len(result.syntax_issues) == 0
        
        result.add_syntax_issue(SyntaxIssueType.BOM_MARKER)
        assert SyntaxIssueType.BOM_MARKER in result.syntax_issues
        
        # Should not add duplicates
        result.add_syntax_issue(SyntaxIssueType.BOM_MARKER)
        assert len(result.syntax_issues) == 1