"""
Robust parser with fallback strategies for handling syntax errors.

This module provides a resilient parsing system that attempts multiple
parsing strategies and can handle files with various syntax issues.
"""

import ast
import re
import logging
import chardet
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from enum import Enum

# Optional dependencies for fallback parsing
try:
    import libcst
    HAS_LIBCST = True
except ImportError:
    HAS_LIBCST = False
    libcst = None

try:
    import parso
    HAS_PARSO = True
except ImportError:
    HAS_PARSO = False
    parso = None

logger = logging.getLogger(__name__)


class ParserType(Enum):
    """Enum for parser types."""
    AST = "ast"
    LIBCST = "libcst"
    PARSO = "parso"
    NONE = "none"


class SyntaxIssueType(Enum):
    """Types of syntax issues that can be detected."""
    BOM_MARKER = "bom_marker"
    MIXED_INDENTATION = "mixed_indentation"
    ENCODING_ERROR = "encoding_error"
    SYNTAX_ERROR = "syntax_error"
    INCOMPLETE_PARSE = "incomplete_parse"
    CORRUPTED_FILE = "corrupted_file"


@dataclass
class ParseResult:
    """Result from parsing attempt."""
    ast_tree: Optional[ast.AST] = None
    parser_used: ParserType = ParserType.NONE
    warnings: List[str] = field(default_factory=list)
    is_quarantined: bool = False
    file_path: Optional[str] = None
    syntax_issues: List[SyntaxIssueType] = field(default_factory=list)
    partial_ast: Optional[ast.AST] = None  # For partial recovery
    line_count: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if parsing produced a valid AST."""
        return self.ast_tree is not None and not self.is_quarantined
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_syntax_issue(self, issue: SyntaxIssueType) -> None:
        """Add a syntax issue."""
        if issue not in self.syntax_issues:
            self.syntax_issues.append(issue)


class RobustParser:
    """
    Robust parser with multiple fallback strategies.
    
    Attempts to parse Python files using various parsers and strategies,
    handling common syntax issues gracefully.
    """
    
    def __init__(self, auto_fix_trivial: bool = False, 
                 encoding_detection: bool = True,
                 max_file_size_mb: int = 50):
        """
        Initialize the robust parser.
        
        Args:
            auto_fix_trivial: Whether to attempt automatic fixing of trivial syntax issues
            encoding_detection: Whether to detect file encoding automatically
            max_file_size_mb: Maximum file size in MB to attempt parsing
        """
        self.auto_fix_trivial = auto_fix_trivial
        self.encoding_detection = encoding_detection
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        
        # Track quarantined files
        self.quarantined_files: Dict[str, ParseResult] = {}
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "successful_parses": 0,
            "partial_parses": 0,
            "quarantined": 0,
            "parser_usage": {
                ParserType.AST: 0,
                ParserType.LIBCST: 0,
                ParserType.PARSO: 0,
                ParserType.NONE: 0
            }
        }
    
    def parse_file(self, file_path: Union[str, Path]) -> ParseResult:
        """
        Parse a Python file with fallback strategies.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            ParseResult with parsing information
        """
        file_path = Path(file_path)
        self.stats["total_files"] += 1
        
        # Check if already quarantined
        if str(file_path) in self.quarantined_files:
            return self.quarantined_files[str(file_path)]
        
        result = ParseResult(file_path=str(file_path))
        
        # Check file size
        if not self._check_file_size(file_path, result):
            return self._quarantine_file(file_path, result)
        
        # Read file content
        content, encoding = self._read_file_content(file_path, result)
        if content is None:
            return self._quarantine_file(file_path, result)
        
        # Pre-scan for issues
        content = self._pre_scan_and_fix(content, result)
        
        # Count lines for statistics
        result.line_count = len(content.splitlines())
        
        # Try parsing strategies in order
        strategies = [
            (self._parse_with_ast, "Standard AST parser"),
            (self._parse_with_libcst, "LibCST parser"),
            (self._parse_with_parso, "Parso parser"),
            (self._parse_with_recovery, "Recovery parser")
        ]
        
        for strategy, name in strategies:
            try:
                if strategy(content, file_path, result):
                    self.stats["successful_parses"] += 1
                    self.stats["parser_usage"][result.parser_used] += 1
                    return result
            except Exception as e:
                result.add_warning(f"{name} failed: {str(e)}")
                continue
        
        # All strategies failed, quarantine the file
        return self._quarantine_file(file_path, result)
    
    def _check_file_size(self, file_path: Path, result: ParseResult) -> bool:
        """Check if file size is within limits."""
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                result.add_warning(
                    f"File too large ({file_size / 1024 / 1024:.2f} MB), "
                    f"exceeds limit of {self.max_file_size_bytes / 1024 / 1024:.2f} MB"
                )
                result.add_syntax_issue(SyntaxIssueType.CORRUPTED_FILE)
                return False
            return True
        except Exception as e:
            result.add_warning(f"Could not check file size: {e}")
            return False
    
    def _read_file_content(self, file_path: Path, result: ParseResult) -> Tuple[Optional[str], Optional[str]]:
        """
        Read file content with encoding detection.
        
        Returns:
            Tuple of (content, encoding) or (None, None) if failed
        """
        # Try UTF-8 first (most common)
        try:
            content = file_path.read_text(encoding='utf-8')
            return content, 'utf-8'
        except UnicodeDecodeError:
            pass
        
        # Try encoding detection if enabled
        if self.encoding_detection:
            try:
                with open(file_path, 'rb') as f:
                    raw_content = f.read()
                
                # Detect encoding
                detected = chardet.detect(raw_content)
                encoding = detected.get('encoding')
                
                if encoding and detected.get('confidence', 0) > 0.7:
                    try:
                        content = raw_content.decode(encoding)
                        result.add_warning(f"Using detected encoding: {encoding}")
                        return content, encoding
                    except Exception:
                        pass
            except Exception as e:
                result.add_warning(f"Encoding detection failed: {e}")
        
        # Try common encodings
        for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
            try:
                content = file_path.read_text(encoding=encoding)
                result.add_warning(f"Using fallback encoding: {encoding}")
                return content, encoding
            except Exception:
                continue
        
        result.add_warning("Could not read file with any encoding")
        result.add_syntax_issue(SyntaxIssueType.ENCODING_ERROR)
        return None, None
    
    def _pre_scan_and_fix(self, content: str, result: ParseResult) -> str:
        """
        Pre-scan content for common issues and optionally fix them.
        
        Args:
            content: File content
            result: ParseResult to add warnings to
            
        Returns:
            Possibly modified content
        """
        original_content = content
        
        # Check for BOM marker
        if content.startswith('\ufeff'):
            result.add_syntax_issue(SyntaxIssueType.BOM_MARKER)
            if self.auto_fix_trivial:
                content = content[1:]
                result.add_warning("Removed BOM marker")
            else:
                result.add_warning("File contains BOM marker")
        
        # Check for mixed indentation at indent level 0
        lines = content.splitlines(keepends=True)
        has_tabs = False
        has_spaces = False
        
        for line in lines:
            if line.strip():  # Non-empty line
                if line[0] == '\t':
                    has_tabs = True
                elif line[0] == ' ':
                    has_spaces = True
        
        if has_tabs and has_spaces:
            result.add_syntax_issue(SyntaxIssueType.MIXED_INDENTATION)
            if self.auto_fix_trivial:
                # Convert tabs to spaces (4 spaces per tab)
                content = content.expandtabs(4)
                result.add_warning("Converted tabs to spaces")
            else:
                result.add_warning("File has mixed tabs and spaces at indent level 0")
        
        # Fix common syntax issues if auto-fix is enabled
        if self.auto_fix_trivial:
            # Fix trailing backslashes that aren't in strings
            content = self._fix_trailing_backslashes(content, result)
            
            # Fix unclosed brackets/parentheses at EOF
            content = self._fix_unclosed_brackets(content, result)
            
            # Fix missing colons after def/class/if/etc
            content = self._fix_missing_colons(content, result)
        
        return content
    
    def _fix_trailing_backslashes(self, content: str, result: ParseResult) -> str:
        """Fix problematic trailing backslashes."""
        lines = content.splitlines(keepends=True)
        modified = False
        
        for i, line in enumerate(lines):
            # Skip if in string literal (simple check)
            if '"""' in line or "'''" in line or line.count('"') % 2 != 0 or line.count("'") % 2 != 0:
                continue
            
            # Check for trailing backslash not in a string
            if line.rstrip().endswith('\\') and not line.rstrip().endswith('\\\\'):
                lines[i] = line.rstrip()[:-1] + '\n'
                modified = True
        
        if modified:
            result.add_warning("Fixed trailing backslashes")
            return ''.join(lines)
        return content
    
    def _fix_unclosed_brackets(self, content: str, result: ParseResult) -> str:
        """Attempt to fix unclosed brackets at end of file."""
        # Count brackets
        open_parens = content.count('(') - content.count(')')
        open_brackets = content.count('[') - content.count(']')
        open_braces = content.count('{') - content.count('}')
        
        additions = []
        if open_parens > 0:
            additions.append(')' * open_parens)
        if open_brackets > 0:
            additions.append(']' * open_brackets)
        if open_braces > 0:
            additions.append('}' * open_braces)
        
        if additions:
            content = content.rstrip() + '\n' + ''.join(additions) + '\n'
            result.add_warning(f"Added closing brackets: {''.join(additions)}")
        
        return content
    
    def _fix_missing_colons(self, content: str, result: ParseResult) -> str:
        """Fix missing colons after control flow statements."""
        lines = content.splitlines(keepends=True)
        modified = False
        
        patterns = [
            (r'^(\s*)(if\s+.+[^:])$', r'\1\2:'),
            (r'^(\s*)(elif\s+.+[^:])$', r'\1\2:'),
            (r'^(\s*)(else[^:])$', r'\1\2:'),
            (r'^(\s*)(for\s+.+[^:])$', r'\1\2:'),
            (r'^(\s*)(while\s+.+[^:])$', r'\1\2:'),
            (r'^(\s*)(def\s+.+\)[^:])$', r'\1\2:'),
            (r'^(\s*)(class\s+.+[^:])$', r'\1\2:'),
            (r'^(\s*)(try[^:])$', r'\1\2:'),
            (r'^(\s*)(except[^:])$', r'\1\2:'),
            (r'^(\s*)(finally[^:])$', r'\1\2:'),
        ]
        
        for i, line in enumerate(lines):
            for pattern, replacement in patterns:
                if re.match(pattern, line.rstrip()):
                    lines[i] = re.sub(pattern, replacement, line.rstrip()) + '\n'
                    modified = True
                    break
        
        if modified:
            result.add_warning("Fixed missing colons")
            return ''.join(lines)
        return content
    
    def _parse_with_ast(self, content: str, file_path: Path, result: ParseResult) -> bool:
        """
        Try parsing with standard ast module.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
            result.ast_tree = tree
            result.parser_used = ParserType.AST
            return True
        except SyntaxError as e:
            result.add_warning(f"AST parse error: {e}")
            result.add_syntax_issue(SyntaxIssueType.SYNTAX_ERROR)
            # Try to extract partial AST up to error
            if e.lineno:
                try:
                    lines = content.splitlines()
                    partial_content = '\n'.join(lines[:e.lineno-1])
                    if partial_content.strip():
                        result.partial_ast = ast.parse(partial_content, filename=str(file_path))
                except:
                    pass
            return False
        except Exception as e:
            result.add_warning(f"AST parse failed: {e}")
            return False
    
    def _parse_with_libcst(self, content: str, file_path: Path, result: ParseResult) -> bool:
        """
        Try parsing with LibCST.
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_LIBCST:
            return False
        
        try:
            # Parse with LibCST
            cst_tree = libcst.parse_module(content)
            
            # Convert to AST
            # LibCST doesn't directly convert to ast.AST, so we parse again with ast
            # after LibCST validates it
            tree = ast.parse(content, filename=str(file_path))
            result.ast_tree = tree
            result.parser_used = ParserType.LIBCST
            result.add_warning("Parsed with LibCST fallback")
            return True
        except Exception as e:
            result.add_warning(f"LibCST parse failed: {e}")
            return False
    
    def _parse_with_parso(self, content: str, file_path: Path, result: ParseResult) -> bool:
        """
        Try parsing with Parso.
        
        Returns:
            True if successful, False otherwise
        """
        if not HAS_PARSO:
            return False
        
        try:
            # Parse with Parso
            parso_tree = parso.parse(content)
            
            # Check for errors
            errors = list(parso_tree.iter_errors())
            if errors:
                result.add_warning(f"Parso found {len(errors)} syntax errors")
                for error in errors[:5]:  # Show first 5 errors
                    result.add_warning(f"  - {error}")
            
            # Try to convert to AST (Parso validates but we still need AST)
            try:
                tree = ast.parse(content, filename=str(file_path))
                result.ast_tree = tree
                result.parser_used = ParserType.PARSO
                result.add_warning("Parsed with Parso fallback")
                return True
            except:
                # Parso parsed it but AST couldn't - partial success
                result.parser_used = ParserType.PARSO
                result.add_warning("Parso parsed but AST conversion failed")
                result.add_syntax_issue(SyntaxIssueType.INCOMPLETE_PARSE)
                return False
                
        except Exception as e:
            result.add_warning(f"Parso parse failed: {e}")
            return False
    
    def _parse_with_recovery(self, content: str, file_path: Path, result: ParseResult) -> bool:
        """
        Try parsing with error recovery strategies.
        
        This is the last resort - tries to extract as much valid AST as possible.
        
        Returns:
            True if any AST was recovered, False otherwise
        """
        lines = content.splitlines()
        
        # Strategy 1: Parse line by line and collect valid statements
        valid_statements = []
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                try:
                    # Try to parse as a single statement
                    tree = ast.parse(line)
                    if tree.body:
                        valid_statements.extend(tree.body)
                except:
                    # Try to parse as expression
                    try:
                        expr = ast.parse(line, mode='eval')
                        valid_statements.append(ast.Expr(value=expr.body, lineno=i+1, col_offset=0))
                    except:
                        pass
        
        if valid_statements:
            # Create a module with recovered statements
            module = ast.Module(body=valid_statements, type_ignores=[])
            result.partial_ast = module
            result.add_warning(f"Recovered {len(valid_statements)} statements using line-by-line parsing")
            result.add_syntax_issue(SyntaxIssueType.INCOMPLETE_PARSE)
            self.stats["partial_parses"] += 1
            return False
        
        # Strategy 2: Binary search for largest parseable prefix
        left, right = 0, len(lines)
        best_ast = None
        best_line_count = 0
        
        while left < right:
            mid = (left + right + 1) // 2
            partial_content = '\n'.join(lines[:mid])
            try:
                tree = ast.parse(partial_content)
                best_ast = tree
                best_line_count = mid
                left = mid
            except:
                right = mid - 1
        
        if best_ast and best_line_count > 0:
            result.partial_ast = best_ast
            result.add_warning(f"Recovered AST for first {best_line_count} lines using binary search")
            result.add_syntax_issue(SyntaxIssueType.INCOMPLETE_PARSE)
            self.stats["partial_parses"] += 1
            return False
        
        return False
    
    def _quarantine_file(self, file_path: Path, result: ParseResult) -> ParseResult:
        """
        Quarantine a file that cannot be parsed.
        
        Args:
            file_path: Path to the file
            result: ParseResult with accumulated warnings
            
        Returns:
            Updated ParseResult marked as quarantined
        """
        result.is_quarantined = True
        result.parser_used = ParserType.NONE
        result.add_warning(f"File quarantined: {file_path}")
        
        self.quarantined_files[str(file_path)] = result
        self.stats["quarantined"] += 1
        self.stats["parser_usage"][ParserType.NONE] += 1
        
        # Extract just the filename from the full path
        filename = file_path.name if hasattr(file_path, 'name') else str(file_path).split('/')[-1]
        logger.warning(f"⚠️  Partially analyzed: {filename} (syntax errors found)")
        if result.syntax_issues:
            logger.info(f"   Found {len(result.syntax_issues)} syntax issues in file")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        stats = self.stats.copy()
        if stats["total_files"] > 0:
            stats["success_rate"] = stats["successful_parses"] / stats["total_files"]
            stats["quarantine_rate"] = stats["quarantined"] / stats["total_files"]
            stats["partial_parse_rate"] = stats["partial_parses"] / stats["total_files"]
        else:
            stats["success_rate"] = 0
            stats["quarantine_rate"] = 0
            stats["partial_parse_rate"] = 0
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset parsing statistics."""
        self.stats = {
            "total_files": 0,
            "successful_parses": 0,
            "partial_parses": 0,
            "quarantined": 0,
            "parser_usage": {
                ParserType.AST: 0,
                ParserType.LIBCST: 0,
                ParserType.PARSO: 0,
                ParserType.NONE: 0
            }
        }
    
    def clear_quarantine(self) -> None:
        """Clear the quarantine list."""
        self.quarantined_files.clear()


def create_parser(auto_fix_trivial: bool = False) -> RobustParser:
    """
    Factory function to create a RobustParser instance.
    
    Args:
        auto_fix_trivial: Whether to enable automatic fixing of trivial syntax issues
        
    Returns:
        Configured RobustParser instance
    """
    return RobustParser(auto_fix_trivial=auto_fix_trivial)