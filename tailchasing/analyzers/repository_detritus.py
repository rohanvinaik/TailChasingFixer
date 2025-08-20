"""Repository Detritus Analyzer - Detects and identifies common junk files in repository roots.

This analyzer identifies files and patterns that commonly accumulate in repository roots
during development, including:
- Temporary files and backups
- Build artifacts
- Editor swap files
- OS-generated files
- Debug outputs
- Test artifacts that should be in .gitignore
"""

import re
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Set, Any
from dataclasses import dataclass

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue


@dataclass
class DetritusPattern:
    """Represents a detritus file pattern."""
    
    file_path: str
    category: str  # 'temp', 'backup', 'build', 'editor', 'os', 'debug', 'test'
    size_bytes: int
    pattern_matched: str
    should_gitignore: bool
    recommendation: str


class RepositoryDetritusAnalyzer(BaseAnalyzer):
    """
    Detects common junk files and patterns that accumulate in repository roots.
    
    Categories detected:
    1. Temporary files (.tmp, .temp, ~, .bak)
    2. Build artifacts (dist/, build/, *.egg-info)
    3. Editor files (.swp, .swo, .idea/, .vscode/settings.json with user-specific)
    4. OS files (.DS_Store, Thumbs.db, desktop.ini)
    5. Debug outputs (*.log, *.stackdump, core dumps)
    6. Python artifacts (__pycache__, *.pyc, .pytest_cache)
    7. Package manager artifacts (node_modules/, vendor/)
    8. Test coverage reports (htmlcov/, .coverage)
    """
    
    name = "repository_detritus"
    
    # Pattern definitions
    TEMP_PATTERNS = [
        r'.*\.tmp$',
        r'.*\.temp$',
        r'.*~$',
        r'.*\.orig$',
        r'tmp_.*',
        r'temp_.*',
    ]
    
    BACKUP_PATTERNS = [
        r'.*\.bak$',
        r'.*\.backup\d*$',
        r'.*\.backup$',
        r'.*\.old$',
        r'.*\.\d{8}$',  # Date backups
        r'.*_backup$',
        r'.*_old$',
        r'.*\(copy\).*',
        r'.*\.save$',
    ]
    
    BUILD_PATTERNS = [
        r'^dist/?$',
        r'^build/?$',
        r'^target/?$',
        r'.*\.egg-info/?$',
        r'^out/?$',
        r'^bin/?$',
        r'.*\.exe$',
        r'.*\.dll$',
        r'.*\.so$',
        r'.*\.dylib$',
    ]
    
    EDITOR_PATTERNS = [
        r'.*\.swp$',
        r'.*\.swo$',
        r'.*\.swn$',
        r'\.\#.*',  # Emacs
        r'.*\.sublime-workspace$',
        r'.*\.kate-swp$',
        r'\.idea/workspace\.xml$',
        r'\.vscode/.*\.json$',  # User settings
    ]
    
    OS_PATTERNS = [
        r'^\.DS_Store$',
        r'^Thumbs\.db$',
        r'^desktop\.ini$',
        r'^\$RECYCLE\.BIN/?$',
        r'^\.Spotlight-V100/?$',
        r'^\.Trashes/?$',
        r'^ehthumbs\.db$',
        r'^\.fseventsd/?$',
    ]
    
    DEBUG_PATTERNS = [
        r'.*\.log$',
        r'.*\.stackdump$',
        r'^core\.\d+$',  # Core dumps
        r'^nohup\.out$',
        r'.*\.trace$',
        r'.*\.debug$',
        r'^debug\.txt$',
        r'^error\.txt$',
    ]
    
    PYTHON_PATTERNS = [
        r'^__pycache__/?$',
        r'.*\.pyc$',
        r'.*\.pyo$',
        r'.*\.pyd$',
        r'^\.pytest_cache/?$',
        r'^\.mypy_cache/?$',
        r'^\.tox/?$',
        r'^\.nox/?$',
        r'^\.hypothesis/?$',
        r'^htmlcov/?$',
        r'.*\.cover$',
    ]
    
    PACKAGE_PATTERNS = [
        r'^node_modules/?$',
        r'^vendor/?$',
        r'^bower_components/?$',
        r'^jspm_packages/?$',
        r'^packages/?$',  # .NET
    ]
    
    TEST_PATTERNS = [
        r'^\.coverage.*$',
        r'^coverage\.xml$',
        r'^coverage\.json$',
        r'^\.nyc_output/?$',
        r'^test-results/?$',
        r'^test-reports/?$',
        r'^junit\.xml$',
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(self.name)
        self.config = config or {}
        self.min_size_to_report = self.config.get('min_size_bytes', 1024)  # 1KB minimum
        self.check_subdirs = self.config.get('check_subdirs', False)  # Only root by default
        self.patterns_found: List[DetritusPattern] = []
        
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Scan repository for detritus files."""
        # Get repository root
        root_dir = self._find_repo_root(ctx)
        if not root_dir:
            return
            
        # Check if .gitignore exists
        gitignore_path = root_dir / '.gitignore'
        gitignore_patterns = self._parse_gitignore(gitignore_path) if gitignore_path.exists() else set()
        
        # Scan for detritus
        detritus_files = self._scan_for_detritus(root_dir, gitignore_patterns)
        
        # Group by category
        by_category = self._group_by_category(detritus_files)
        
        # Generate issues
        for pattern in detritus_files:
            yield self._create_detritus_issue(pattern, gitignore_patterns)
            
        # Create summary issue if significant detritus found
        if len(detritus_files) >= 5:
            yield self._create_summary_issue(by_category, root_dir)
            
    def _find_repo_root(self, ctx: AnalysisContext) -> Optional[Path]:
        """Find the repository root directory."""
        if ctx.files:
            # Use the common parent of all files
            file_paths = [Path(f) for f in ctx.files]
            # Find common parent
            common_parent = file_paths[0].parent
            for f in file_paths[1:]:
                while not f.is_relative_to(common_parent):
                    common_parent = common_parent.parent
                    
            # Look for .git directory
            current = common_parent
            while current != current.parent:
                if (current / '.git').exists():
                    return current
                current = current.parent
                
            return common_parent
        return None
        
    def _parse_gitignore(self, gitignore_path: Path) -> Set[str]:
        """Parse .gitignore file to get ignored patterns."""
        patterns = set()
        try:
            with open(gitignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.add(line)
        except (OSError, IOError):
            pass
        return patterns
        
    def _scan_for_detritus(self, root_dir: Path, gitignore_patterns: Set[str]) -> List[DetritusPattern]:
        """Scan directory for detritus files."""
        detritus = []
        
        # Determine scan depth
        if self.check_subdirs:
            pattern = '**/*'
        else:
            pattern = '*'
            
        for item in root_dir.glob(pattern):
            # Skip if path is too deep (avoid scanning entire node_modules, etc.)
            if self.check_subdirs:
                relative_parts = len(item.relative_to(root_dir).parts)
                if relative_parts > 3:
                    continue
                    
            if item.is_file() or item.is_dir():
                relative_path = str(item.relative_to(root_dir))
                
                # Check against patterns
                category, pattern_matched = self._categorize_file(relative_path)
                
                if category:
                    # Get file size
                    size = 0
                    if item.is_file():
                        try:
                            size = item.stat().st_size
                        except (OSError, IOError):
                            size = 0
                    elif item.is_dir():
                        # Estimate directory size (don't recurse deeply)
                        size = self._estimate_dir_size(item)
                        
                    # Check if it's in gitignore
                    should_gitignore = not self._is_in_gitignore(relative_path, gitignore_patterns)
                    
                    # Skip small files unless they're patterns that should always be reported
                    if size < self.min_size_to_report and category not in ['os', 'editor']:
                        continue
                        
                    detritus.append(DetritusPattern(
                        file_path=relative_path,
                        category=category,
                        size_bytes=size,
                        pattern_matched=pattern_matched or "",
                        should_gitignore=should_gitignore,
                        recommendation=self._get_recommendation(category, relative_path)
                    ))
                    
        return detritus
        
    def _categorize_file(self, path: str) -> tuple[Optional[str], Optional[str]]:
        """Categorize a file based on patterns."""
        # Check each category (test before python to prioritize .coverage as test)
        categories = [
            ('temp', self.TEMP_PATTERNS),
            ('backup', self.BACKUP_PATTERNS),
            ('build', self.BUILD_PATTERNS),
            ('editor', self.EDITOR_PATTERNS),
            ('os', self.OS_PATTERNS),
            ('debug', self.DEBUG_PATTERNS),
            ('test', self.TEST_PATTERNS),  # Check test before python
            ('python', self.PYTHON_PATTERNS),
            ('package', self.PACKAGE_PATTERNS),
        ]
        
        for category, patterns in categories:
            for pattern in patterns:
                if re.match(pattern, path, re.IGNORECASE):
                    return category, pattern
                    
        return None, None
        
    def _is_in_gitignore(self, path: str, gitignore_patterns: Set[str]) -> bool:
        """Check if path matches any gitignore pattern."""
        for pattern in gitignore_patterns:
            # Simple pattern matching (not full gitignore semantics)
            if pattern.endswith('/'):
                if path.startswith(pattern) or path + '/' == pattern:
                    return True
            elif '*' in pattern:
                # Convert to regex
                regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
                if re.match(regex_pattern, path):
                    return True
            elif path == pattern or path.startswith(pattern + '/'):
                return True
        return False
        
    def _estimate_dir_size(self, dir_path: Path) -> int:
        """Estimate directory size without deep recursion."""
        total = 0
        try:
            for item in dir_path.iterdir():
                if item.is_file():
                    total += item.stat().st_size
                # Don't recurse into subdirectories for estimation
        except (OSError, IOError):
            pass
        return total
        
    def _get_recommendation(self, category: str, path: str) -> str:
        """Get recommendation for handling detritus."""
        recommendations = {
            'temp': "Remove temporary file and add pattern to .gitignore",
            'backup': "Move to backup location outside repository or remove",
            'build': "Add to .gitignore and clean build artifacts",
            'editor': "Add editor-specific patterns to global gitignore",
            'os': "Add OS-specific patterns to global gitignore",
            'debug': "Remove debug output and add *.log to .gitignore",
            'python': "Add Python artifacts to .gitignore (use GitHub's Python template)",
            'package': "Ensure package directories are in .gitignore",
            'test': "Add test artifacts to .gitignore",
        }
        return recommendations.get(category, "Review and add to .gitignore if needed")
        
    def _create_detritus_issue(self, pattern: DetritusPattern, gitignore_patterns: Set[str]) -> Issue:
        """Create issue for detritus file."""
        severity = "MEDIUM" if pattern.should_gitignore else "LOW"
        
        # Increase severity for large files/directories
        if pattern.size_bytes > 10 * 1024 * 1024:  # 10MB
            severity = "HIGH"
            
        message = f"Repository detritus: {pattern.file_path}"
        if pattern.should_gitignore:
            message += " (not in .gitignore)"
            
        return Issue(
            kind="repository_detritus",
            severity=3 if severity == "HIGH" else (2 if severity == "MEDIUM" else 1),
            message=message,
            file=pattern.file_path,
            line=0,
            evidence={
                "category": pattern.category,
                "size_bytes": pattern.size_bytes,
                "size_human": self._format_size(pattern.size_bytes),
                "pattern_matched": pattern.pattern_matched,
                "should_gitignore": pattern.should_gitignore,
                "recommendation": pattern.recommendation
            },
            suggestions=[pattern.recommendation]
        )
        
    def _create_summary_issue(self, by_category: Dict[str, List[DetritusPattern]], root_dir: Path) -> Issue:
        """Create summary issue for all detritus found."""
        total_files = sum(len(patterns) for patterns in by_category.values())
        total_size = sum(p.size_bytes for patterns in by_category.values() for p in patterns)
        
        categories_summary = {}
        for category, patterns in by_category.items():
            categories_summary[category] = {
                "count": len(patterns),
                "total_size": sum(p.size_bytes for p in patterns),
                "files": [p.file_path for p in patterns[:5]]  # First 5 examples
            }
            
        return Issue(
            kind="repository_detritus_summary",
            severity=3 if total_files > 10 else 2,
            message=f"Found {total_files} detritus files/directories ({self._format_size(total_size)}) in repository",
            file=str(root_dir),
            line=0,
            evidence={
                "total_files": total_files,
                "total_size_bytes": total_size,
                "total_size_human": self._format_size(total_size),
                "categories": categories_summary
            },
            suggestions=["Clean repository and update .gitignore to prevent accumulation"]
        )
        
    def _group_by_category(self, patterns: List[DetritusPattern]) -> Dict[str, List[DetritusPattern]]:
        """Group detritus patterns by category."""
        by_category: Dict[str, List[DetritusPattern]] = {}
        for pattern in patterns:
            if pattern.category not in by_category:
                by_category[pattern.category] = []
            by_category[pattern.category].append(pattern)
        return by_category
        
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes = size_bytes / 1024.0
        return f"{size_bytes:.1f} TB"