"""
Ignore Manager - Handles .tcdignore files and exclusion patterns.

This module provides gitignore-style pattern matching for excluding files
and directories from TailChasing analysis.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

try:
    import pathspec
except ImportError:
    raise ImportError(
        "pathspec library is required for ignore functionality. "
        "Install with: pip install pathspec"
    )

logger = logging.getLogger(__name__)


class IgnoreManager:
    """
    Manages file exclusion patterns using gitignore syntax.
    
    Features:
    1. Reads .tcdignore files with gitignore syntax
    2. Supports default patterns for common noise
    3. Merges patterns from multiple sources (file, CLI, defaults)
    4. Efficient pattern matching using pathspec library
    """
    
    # Default patterns to always ignore
    DEFAULT_PATTERNS = [
        # Python cache and compiled files
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".Python",
        "*.so",
        
        # Type checking and linting caches
        ".mypy_cache/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".hypothesis/",
        ".coverage",
        "htmlcov/",
        ".tox/",
        "*.cover",
        
        # Jupyter notebooks
        ".ipynb_checkpoints/",
        "*/.ipynb_checkpoints/*",
        
        # Virtual environments
        "venv/",
        ".venv/",
        "env/",
        ".env/",
        "ENV/",
        "env.bak/",
        "venv.bak/",
        
        # Build and distribution
        "build/",
        "develop-eggs/",
        "dist/",
        "downloads/",
        "eggs/",
        ".eggs/",
        "lib/",
        "lib64/",
        "parts/",
        "sdist/",
        "var/",
        "wheels/",
        "*.egg-info/",
        ".installed.cfg",
        "*.egg",
        "MANIFEST",
        
        # Version control
        ".git/",
        ".svn/",
        ".hg/",
        
        # IDE and editor files
        ".idea/",
        ".vscode/",
        "*.swp",
        "*.swo",
        "*~",
        ".DS_Store",
        "Thumbs.db",
        
        # Documentation builds
        "docs/_build/",
        "site/",
        
        # Backup files
        "*.backup",
        "*.bak",
        "*.tmp",
        "*.temp",
        "*/backup_*/*",
        "*/*backup*/*",
        
        # Log files
        "*.log",
        "logs/",
        
        # Database files
        "*.db",
        "*.sqlite",
        "*.sqlite3",
        
        # Node modules (for mixed projects)
        "node_modules/",
        
        # Temporary directories
        "tmp/",
        "temp/",
        
        # Generated files
        "*.generated.py",
        "*_pb2.py",
        "*_pb2_grpc.py",
    ]
    
    # File to look for in project root
    IGNORE_FILENAME = ".tcdignore"
    
    def __init__(
        self,
        root_path: Union[str, Path],
        additional_patterns: Optional[List[str]] = None,
        use_defaults: bool = True,
        ignore_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the IgnoreManager.
        
        Args:
            root_path: Root directory of the project
            additional_patterns: Extra patterns to add (e.g., from CLI)
            use_defaults: Whether to include default patterns
            ignore_file: Path to ignore file (default: .tcdignore in root)
        """
        self.root_path = Path(root_path).absolute()
        self.patterns: List[str] = []
        self.spec: Optional[pathspec.PathSpec] = None
        
        # Collect patterns from all sources
        all_patterns = []
        
        # 1. Add default patterns if requested
        if use_defaults:
            all_patterns.extend(self.DEFAULT_PATTERNS)
            logger.debug(f"Added {len(self.DEFAULT_PATTERNS)} default ignore patterns")
        
        # 2. Read patterns from ignore file
        ignore_file_path = self._find_ignore_file(ignore_file)
        if ignore_file_path:
            file_patterns = self._read_ignore_file(ignore_file_path)
            all_patterns.extend(file_patterns)
            logger.debug(f"Loaded {len(file_patterns)} patterns from {ignore_file_path}")
        
        # 3. Add additional patterns from CLI or config
        if additional_patterns:
            all_patterns.extend(additional_patterns)
            logger.debug(f"Added {len(additional_patterns)} additional patterns")
        
        # Remove duplicates while preserving order
        seen = set()
        self.patterns = []
        for pattern in all_patterns:
            pattern = pattern.strip()
            if pattern and pattern not in seen and not pattern.startswith('#'):
                seen.add(pattern)
                self.patterns.append(pattern)
        
        # Compile patterns into PathSpec for efficient matching
        self._compile_patterns()
        
        logger.info(f"IgnoreManager initialized with {len(self.patterns)} patterns")
    
    def _find_ignore_file(self, ignore_file: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """
        Find the ignore file to use.
        
        Args:
            ignore_file: Explicit path to ignore file
            
        Returns:
            Path to ignore file if found, None otherwise
        """
        if ignore_file:
            path = Path(ignore_file)
            if path.is_absolute():
                return path if path.exists() else None
            else:
                # Relative to root
                full_path = self.root_path / path
                return full_path if full_path.exists() else None
        
        # Look for default .tcdignore in root and parent directories
        current = self.root_path
        max_depth = 5  # Don't go up too many levels
        
        for _ in range(max_depth):
            tcdignore = current / self.IGNORE_FILENAME
            if tcdignore.exists():
                return tcdignore
            
            # Also check for .gitignore as fallback
            gitignore = current / ".gitignore"
            if gitignore.exists() and not tcdignore.exists():
                logger.debug(f"Using .gitignore as fallback: {gitignore}")
                return gitignore
            
            # Move up one directory
            parent = current.parent
            if parent == current:  # Reached root
                break
            current = parent
        
        return None
    
    def _read_ignore_file(self, file_path: Path) -> List[str]:
        """
        Read patterns from an ignore file.
        
        Args:
            file_path: Path to the ignore file
            
        Returns:
            List of patterns from the file
        """
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        patterns.append(line)
        except Exception as e:
            logger.warning(f"Error reading ignore file {file_path}: {e}")
        
        return patterns
    
    def _compile_patterns(self):
        """Compile patterns into a PathSpec for efficient matching."""
        if self.patterns:
            try:
                self.spec = pathspec.PathSpec.from_lines('gitwildmatch', self.patterns)
            except Exception as e:
                logger.error(f"Error compiling ignore patterns: {e}")
                self.spec = None
    
    def should_ignore(self, path: Union[str, Path]) -> bool:
        """
        Check if a path should be ignored.
        
        Args:
            path: Path to check (absolute or relative to root)
            
        Returns:
            True if the path should be ignored, False otherwise
        """
        if not self.spec:
            return False
        
        # Convert to Path object
        path = Path(path)
        
        # Make path relative to root for matching
        try:
            if path.is_absolute():
                # Handle symlinks and resolve paths properly
                try:
                    rel_path = path.relative_to(self.root_path)
                except ValueError:
                    # Try with resolved paths
                    try:
                        rel_path = path.absolute().relative_to(self.root_path.absolute())
                    except ValueError:
                        # Path is not under root, don't ignore
                        return False
            else:
                rel_path = path
        except Exception:
            # Path is not under root or other error, don't ignore
            return False
        
        # Convert to string with forward slashes for pathspec
        path_str = str(rel_path).replace('\\', '/')
        
        # Check if path matches any ignore pattern
        # Also check all parent directories
        result = self.spec.match_file(path_str)
        
        # If not matched, check if any parent directory is ignored
        if not result:
            parts = path_str.split('/')
            for i in range(1, len(parts)):
                parent_path = '/'.join(parts[:i]) + '/'
                if self.spec.match_file(parent_path):
                    return True
        
        return result
    
    def filter_paths(self, paths: List[Union[str, Path]]) -> List[Path]:
        """
        Filter a list of paths, removing ignored ones.
        
        Args:
            paths: List of paths to filter
            
        Returns:
            List of paths that should not be ignored
        """
        filtered = []
        ignored_count = 0
        
        for path in paths:
            if not self.should_ignore(path):
                filtered.append(Path(path))
            else:
                ignored_count += 1
                logger.debug(f"Ignoring: {path}")
        
        if ignored_count > 0:
            logger.info(f"Filtered out {ignored_count} ignored paths")
        
        return filtered
    
    def add_pattern(self, pattern: str):
        """
        Add a new ignore pattern.
        
        Args:
            pattern: Pattern to add (gitignore syntax)
        """
        pattern = pattern.strip()
        if pattern and pattern not in self.patterns:
            self.patterns.append(pattern)
            self._compile_patterns()
    
    def add_patterns(self, patterns: List[str]):
        """
        Add multiple ignore patterns.
        
        Args:
            patterns: List of patterns to add
        """
        added = False
        for pattern in patterns:
            pattern = pattern.strip()
            if pattern and pattern not in self.patterns:
                self.patterns.append(pattern)
                added = True
        
        if added:
            self._compile_patterns()
    
    def get_patterns(self) -> List[str]:
        """Get the current list of ignore patterns."""
        return self.patterns.copy()
    
    def save_to_file(self, file_path: Optional[Union[str, Path]] = None):
        """
        Save current patterns to an ignore file.
        
        Args:
            file_path: Path to save to (default: .tcdignore in root)
        """
        if file_path is None:
            file_path = self.root_path / self.IGNORE_FILENAME
        else:
            file_path = Path(file_path)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# TailChasing ignore file\n")
                f.write("# Uses gitignore syntax\n\n")
                
                # Write default patterns section
                f.write("# Default patterns (Python cache, build artifacts, etc.)\n")
                for pattern in self.DEFAULT_PATTERNS[:10]:  # Sample of defaults
                    f.write(f"{pattern}\n")
                f.write("\n# Custom patterns\n")
                
                # Write custom patterns
                custom_patterns = [p for p in self.patterns if p not in self.DEFAULT_PATTERNS]
                for pattern in custom_patterns:
                    f.write(f"{pattern}\n")
            
            logger.info(f"Saved ignore patterns to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving ignore file: {e}")
    
    def __repr__(self) -> str:
        """String representation of IgnoreManager."""
        return f"IgnoreManager(root={self.root_path}, patterns={len(self.patterns)})"
    
    def get_statistics(self) -> dict:
        """
        Get statistics about ignore patterns.
        
        Returns:
            Dictionary with pattern statistics
        """
        stats = {
            "total_patterns": len(self.patterns),
            "default_patterns": len([p for p in self.patterns if p in self.DEFAULT_PATTERNS]),
            "custom_patterns": len([p for p in self.patterns if p not in self.DEFAULT_PATTERNS]),
            "directory_patterns": len([p for p in self.patterns if p.endswith('/')]),
            "wildcard_patterns": len([p for p in self.patterns if '*' in p]),
            "negation_patterns": len([p for p in self.patterns if p.startswith('!')]),
        }
        
        return stats


def create_default_tcdignore(root_path: Union[str, Path]):
    """
    Create a default .tcdignore file with common patterns.
    
    Args:
        root_path: Root directory for the .tcdignore file
    """
    root_path = Path(root_path)
    ignore_file = root_path / ".tcdignore"
    
    if ignore_file.exists():
        logger.warning(f"{ignore_file} already exists, not overwriting")
        return
    
    content = """# TailChasing ignore file
# Uses gitignore syntax: https://git-scm.com/docs/gitignore

# Python cache and compiled files
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so

# Type checking and testing
.mypy_cache/
.pytest_cache/
.ruff_cache/
.hypothesis/
.coverage
htmlcov/
.tox/

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Virtual environments
venv/
.venv/
env/
.env/

# Build artifacts
build/
dist/
*.egg-info/
*.egg

# Version control
.git/
.svn/
.hg/

# IDE files
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store

# Backups
*.backup
*.bak
*/backup_*/*
*/*backup*/*

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
*.tmp
*.temp

# Generated files
*.generated.py
*_pb2.py
*_pb2_grpc.py

# Custom patterns (add your own below)
# examples/
# legacy/
# deprecated/
"""
    
    try:
        with open(ignore_file, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Created default .tcdignore file at {ignore_file}")
    except Exception as e:
        logger.error(f"Error creating .tcdignore file: {e}")