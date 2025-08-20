"""
Comprehensive Import Anxiety Analysis for TailChasingFixer.

This module detects LLM-specific import anxiety patterns including defensive
over-importing, unused imports, and common LLM behaviors like error handling
anxiety and class import sprees.
"""

from __future__ import annotations
import ast
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..base import BaseAnalyzer, AnalysisContext
from ...core.issues import Issue

logger = logging.getLogger(__name__)


@dataclass
class ImportInfo:
    """Information about a single import statement."""
    
    import_type: str  # 'import' or 'from_import'
    module_name: str
    imported_name: str
    alias: Optional[str] = None
    line_number: int = 0
    is_wildcard: bool = False
    is_standard_library: bool = False
    is_third_party: bool = False
    
    def get_usage_name(self) -> str:
        """Get the name used in code for this import."""
        if self.alias:
            return self.alias
        elif self.import_type == 'import':
            return self.module_name.split('.')[-1] if '.' in self.module_name else self.module_name
        else:
            return self.imported_name
    
    def get_full_name(self) -> str:
        """Get the full qualified name of the import."""
        if self.import_type == 'import':
            return self.module_name
        else:
            return f"{self.module_name}.{self.imported_name}"


@dataclass
class UsageInfo:
    """Information about usage of an imported item."""
    
    name: str
    line_numbers: List[int] = field(default_factory=list)
    usage_contexts: List[str] = field(default_factory=list)  # 'call', 'attribute', 'name'
    is_in_docstring: bool = False
    is_in_comment: bool = False
    
    def get_usage_count(self) -> int:
        """Get total usage count."""
        return len(self.line_numbers)
    
    def is_actually_used(self) -> bool:
        """Check if this is real usage (not just in comments/docstrings)."""
        return self.get_usage_count() > 0 and not (self.is_in_docstring or self.is_in_comment)


@dataclass
class ImportCluster:
    """A cluster of related imports."""
    
    cluster_id: str
    imports: List[ImportInfo] = field(default_factory=list)
    pattern_type: str = "mixed"
    anxiety_score: float = 0.0
    unused_ratio: float = 0.0
    evidence_indicators: List[str] = field(default_factory=list)
    
    def add_import(self, import_info: ImportInfo) -> None:
        """Add an import to the cluster."""
        self.imports.append(import_info)
    
    def calculate_metrics(self, usage_data: Dict[str, UsageInfo]) -> None:
        """Calculate cluster metrics."""
        if not self.imports:
            return
        
        # Calculate unused ratio
        total_imports = len(self.imports)
        unused_count = 0
        
        for imp in self.imports:
            usage_name = imp.get_usage_name()
            if usage_name not in usage_data or not usage_data[usage_name].is_actually_used():
                unused_count += 1
        
        self.unused_ratio = unused_count / total_imports if total_imports > 0 else 0.0
        
        # Calculate anxiety score based on various factors
        self._calculate_anxiety_score()
    
    def _calculate_anxiety_score(self) -> None:
        """Calculate anxiety score for this cluster."""
        score = 0.0
        indicators = []
        
        # Factor 1: Unused imports (0-0.4)
        unused_penalty = min(0.4, self.unused_ratio * 0.5)
        score += unused_penalty
        if unused_penalty > 0.2:
            indicators.append(f"High unused import ratio: {self.unused_ratio:.1%}")
        
        # Factor 2: Pattern-specific scoring (0-0.3)
        pattern_scores = {
            "error_handling_anxiety": 0.3,
            "class_import_spree": 0.25,
            "import_everything": 0.3,
            "wildcard_import": 0.35,
            "redundant_imports": 0.2
        }
        pattern_score = pattern_scores.get(self.pattern_type, 0.1)
        score += pattern_score
        
        # Factor 3: Cluster size penalty (0-0.2)
        if len(self.imports) > 10:
            size_penalty = min(0.2, (len(self.imports) - 10) * 0.02)
            score += size_penalty
            indicators.append(f"Large import cluster: {len(self.imports)} imports")
        
        # Factor 4: Mixed import types penalty (0-0.1)
        import_types = {imp.import_type for imp in self.imports}
        if len(import_types) > 1:
            score += 0.1
            indicators.append("Mixed import statement types")
        
        self.anxiety_score = min(1.0, score)
        self.evidence_indicators = indicators


class ImportAnxietyDetector(BaseAnalyzer):
    """
    Comprehensive import anxiety detection system.
    
    Detects LLM-specific import patterns including:
    - Error handling anxiety (importing many exception types)
    - Class import spree (importing many classes at once)  
    - Import everything patterns (excessive imports from modules)
    - Unused imports and ratios
    - Related import clusters
    """
    
    name = "import_anxiety"
    
    # Standard library modules (simplified list)
    STDLIB_MODULES = {
        'os', 'sys', 'json', 'csv', 'sqlite3', 'datetime', 'pathlib', 're',
        'collections', 'itertools', 'functools', 'typing', 'logging',
        'urllib', 'http', 'xml', 'html', 'email', 'subprocess', 'threading',
        'multiprocessing', 'concurrent', 'ast', 'inspect', 'importlib',
        'tempfile', 'shutil', 'hashlib', 'pickle', 'copy', 'math', 'random',
        'time', 'calendar', 'statistics', 'decimal', 'fractions'
    }
    
    # Common third-party modules
    COMMON_THIRD_PARTY = {
        'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow',
        'torch', 'requests', 'flask', 'django', 'fastapi', 'sqlalchemy',
        'pytest', 'click', 'rich', 'pydantic', 'networkx'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # Thresholds
        self.min_imports_for_anxiety = self.config.get('min_imports_for_anxiety', 5)
        self.unused_ratio_threshold = self.config.get('unused_ratio_threshold', 0.5)
        self.anxiety_score_threshold = self.config.get('anxiety_score_threshold', 0.6)
        
        # Pattern detection thresholds
        self.error_handling_threshold = self.config.get('error_handling_threshold', 0.4)
        self.class_import_threshold = self.config.get('class_import_threshold', 0.7)
        self.import_everything_threshold = self.config.get('import_everything_threshold', 15)
        
        # Analysis options
        self.track_stdlib_separately = self.config.get('track_stdlib_separately', True)
        self.ignore_test_files = self.config.get('ignore_test_files', True)
        self.detect_cross_file_patterns = self.config.get('detect_cross_file_patterns', True)
        
        logger.debug(f"ImportAnxietyDetector initialized: "
                    f"unused_ratio_threshold={self.unused_ratio_threshold}, "
                    f"min_imports={self.min_imports_for_anxiety}")
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run comprehensive import anxiety detection.
        
        Args:
            ctx: Analysis context containing AST index and configuration
            
        Returns:
            List of Issue objects representing import anxiety patterns
        """
        issues = []
        
        try:
            logger.info(f"Running import anxiety detection on {len(ctx.ast_index)} files")
            
            # Step 1: Extract all imports and usage across the codebase
            logger.debug("Extracting imports and usage from all files")
            all_file_data = {}
            
            for file_path, tree in ctx.ast_index.items():
                if ctx.is_excluded(file_path):
                    continue
                
                # Skip test files if configured
                if self.ignore_test_files and self._is_test_file(file_path):
                    continue
                
                file_imports = self._extract_file_imports(tree, file_path)
                file_usage = self._extract_file_usage(tree, file_imports)
                
                all_file_data[file_path] = {
                    'imports': file_imports,
                    'usage': file_usage,
                    'tree': tree
                }
            
            # Step 2: Analyze each file for import anxiety patterns
            logger.debug("Analyzing files for import anxiety patterns")
            
            for file_path, data in all_file_data.items():
                file_issues = self._analyze_file_import_anxiety(
                    file_path, data['imports'], data['usage'], ctx
                )
                issues.extend(file_issues)
            
            # Step 3: Cross-file pattern analysis (if enabled)
            if self.detect_cross_file_patterns:
                logger.debug("Analyzing cross-file import patterns")
                cross_file_issues = self._analyze_cross_file_patterns(all_file_data, ctx)
                issues.extend(cross_file_issues)
            
            logger.info(f"Import anxiety detection complete: found {len(issues)} issues")
            
        except Exception as e:
            logger.error(f"Error in import anxiety detection: {e}", exc_info=True)
        
        return issues
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file is a test file."""
        path = Path(file_path)
        return (
            'test' in path.name.lower() or 
            path.name.startswith('test_') or
            path.name.endswith('_test.py') or
            'tests' in path.parts
        )
    
    def _extract_file_imports(self, tree: ast.AST, file_path: str) -> List[ImportInfo]:
        """Extract all imports from a file."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_info = ImportInfo(
                        import_type='import',
                        module_name=alias.name,
                        imported_name=alias.name,
                        alias=alias.asname,
                        line_number=node.lineno,
                        is_wildcard=False,
                        is_standard_library=self._is_stdlib_module(alias.name),
                        is_third_party=self._is_third_party_module(alias.name)
                    )
                    imports.append(import_info)
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ''
                for alias in node.names:
                    is_wildcard = alias.name == '*'
                    
                    import_info = ImportInfo(
                        import_type='from_import',
                        module_name=module_name,
                        imported_name=alias.name,
                        alias=alias.asname,
                        line_number=node.lineno,
                        is_wildcard=is_wildcard,
                        is_standard_library=self._is_stdlib_module(module_name),
                        is_third_party=self._is_third_party_module(module_name)
                    )
                    imports.append(import_info)
        
        return imports
    
    def _extract_file_usage(self, tree: ast.AST, imports: List[ImportInfo]) -> Dict[str, UsageInfo]:
        """Extract usage information for imports in a file."""
        usage = defaultdict(lambda: UsageInfo(name=""))
        
        # Create mapping from usage names to import info
        usage_name_to_import = {}
        for imp in imports:
            usage_name = imp.get_usage_name()
            usage_name_to_import[usage_name] = imp
            if usage_name not in usage:
                usage[usage_name] = UsageInfo(name=usage_name)
        
        # Track usage throughout the AST
        for node in ast.walk(tree):
            # Direct name usage
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in usage_name_to_import:
                    usage[node.id].line_numbers.append(node.lineno)
                    usage[node.id].usage_contexts.append('name')
            
            # Attribute access (e.g., module.function)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id in usage_name_to_import:
                    usage[node.value.id].line_numbers.append(node.lineno)
                    usage[node.value.id].usage_contexts.append('attribute')
                
                # Handle from imports (e.g., imported function used directly)
                attr_name = node.attr
                if attr_name in usage_name_to_import:
                    usage[attr_name].line_numbers.append(node.lineno)
                    usage[attr_name].usage_contexts.append('attribute')
            
            # Function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in usage_name_to_import:
                    usage[node.func.id].line_numbers.append(node.lineno)
                    usage[node.func.id].usage_contexts.append('call')
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in usage_name_to_import:
                        usage[node.func.value.id].line_numbers.append(node.lineno)
                        usage[node.func.value.id].usage_contexts.append('call')
        
        return dict(usage)
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        root_module = module_name.split('.')[0]
        return root_module in self.STDLIB_MODULES
    
    def _is_third_party_module(self, module_name: str) -> bool:
        """Check if a module is a common third-party module."""
        root_module = module_name.split('.')[0]
        return root_module in self.COMMON_THIRD_PARTY
    
    def _analyze_file_import_anxiety(
        self, 
        file_path: str, 
        imports: List[ImportInfo], 
        usage: Dict[str, UsageInfo],
        ctx: AnalysisContext
    ) -> List[Issue]:
        """Analyze import anxiety patterns in a single file."""
        issues = []
        
        if len(imports) < self.min_imports_for_anxiety:
            return issues
        
        # Group imports into clusters for analysis
        clusters = self._cluster_imports(imports, usage)
        
        # Analyze each cluster
        for cluster in clusters:
            if cluster.anxiety_score >= self.anxiety_score_threshold:
                issue = self._create_import_anxiety_issue(cluster, file_path, ctx)
                if issue:
                    issues.append(issue)
        
        # Look for specific LLM patterns
        pattern_issues = self._detect_llm_import_patterns(imports, usage, file_path, ctx)
        issues.extend(pattern_issues)
        
        return issues
    
    def _cluster_imports(self, imports: List[ImportInfo], usage: Dict[str, UsageInfo]) -> List[ImportCluster]:
        """Cluster imports based on modules and patterns."""
        clusters = []
        
        # Group by module
        module_groups = defaultdict(list)
        for imp in imports:
            module_groups[imp.module_name].append(imp)
        
        cluster_id = 0
        for module_name, module_imports in module_groups.items():
            if len(module_imports) >= 3:  # Only cluster if significant
                cluster_id += 1
                cluster = ImportCluster(cluster_id=f"module_{cluster_id}")
                
                for imp in module_imports:
                    cluster.add_import(imp)
                
                # Detect pattern type
                cluster.pattern_type = self._detect_cluster_pattern(module_imports)
                
                # Calculate metrics
                cluster.calculate_metrics(usage)
                
                clusters.append(cluster)
        
        # Create additional clusters for specific patterns
        pattern_clusters = self._create_pattern_clusters(imports, usage)
        clusters.extend(pattern_clusters)
        
        return clusters
    
    def _detect_cluster_pattern(self, imports: List[ImportInfo]) -> str:
        """Detect the pattern type for a cluster of imports."""
        if not imports:
            return "empty"
        
        # Check for wildcard imports
        if any(imp.is_wildcard for imp in imports):
            return "wildcard_import"
        
        # Check for error handling anxiety
        error_related = []
        for imp in imports:
            name = imp.imported_name.lower()
            if any(keyword in name for keyword in ['error', 'exception', 'warning', 'fault']):
                error_related.append(imp)
        
        if len(error_related) / len(imports) >= self.error_handling_threshold:
            return "error_handling_anxiety"
        
        # Check for class import spree
        class_imports = []
        for imp in imports:
            # Heuristic: class names are typically capitalized
            if imp.imported_name[0].isupper():
                class_imports.append(imp)
        
        if len(class_imports) / len(imports) >= self.class_import_threshold:
            return "class_import_spree"
        
        # Check for "import everything" pattern
        if len(imports) >= self.import_everything_threshold:
            return "import_everything"
        
        # Check for function/utility spree
        if all(imp.imported_name[0].islower() for imp in imports):
            return "utility_spree"
        
        # Check for redundant imports (same functionality)
        if self._detect_redundant_functionality(imports):
            return "redundant_imports"
        
        return "mixed_imports"
    
    def _detect_redundant_functionality(self, imports: List[ImportInfo]) -> bool:
        """Detect if imports provide redundant functionality."""
        # Simplified heuristic: check for similar names
        names = [imp.imported_name.lower() for imp in imports]
        
        # Look for similar names that might be redundant
        similar_groups = defaultdict(list)
        for name in names:
            # Group by first few characters or common stems
            if len(name) >= 4:
                stem = name[:4]
                similar_groups[stem].append(name)
        
        # If any group has multiple similar names, might be redundant
        for group in similar_groups.values():
            if len(group) >= 3:
                return True
        
        return False
    
    def _create_pattern_clusters(self, imports: List[ImportInfo], usage: Dict[str, UsageInfo]) -> List[ImportCluster]:
        """Create clusters based on specific patterns."""
        clusters = []
        
        # Wildcard cluster
        wildcard_imports = [imp for imp in imports if imp.is_wildcard]
        if wildcard_imports:
            cluster = ImportCluster(cluster_id="wildcard_cluster")
            for imp in wildcard_imports:
                cluster.add_import(imp)
            cluster.pattern_type = "wildcard_import"
            cluster.calculate_metrics(usage)
            clusters.append(cluster)
        
        # Unused imports cluster
        unused_imports = []
        for imp in imports:
            usage_name = imp.get_usage_name()
            if usage_name not in usage or not usage[usage_name].is_actually_used():
                unused_imports.append(imp)
        
        if len(unused_imports) >= 5:
            cluster = ImportCluster(cluster_id="unused_cluster")
            for imp in unused_imports:
                cluster.add_import(imp)
            cluster.pattern_type = "unused_imports"
            cluster.calculate_metrics(usage)
            clusters.append(cluster)
        
        return clusters
    
    def _detect_llm_import_patterns(
        self, 
        imports: List[ImportInfo], 
        usage: Dict[str, UsageInfo],
        file_path: str,
        ctx: AnalysisContext
    ) -> List[Issue]:
        """Detect specific LLM import anxiety patterns."""
        issues = []
        
        # Pattern 1: Defensive exception importing
        exception_imports = self._find_exception_imports(imports)
        if len(exception_imports) >= 5:
            unused_exceptions = []
            for imp in exception_imports:
                usage_name = imp.get_usage_name()
                if usage_name not in usage or not usage[usage_name].is_actually_used():
                    unused_exceptions.append(imp)
            
            if len(unused_exceptions) >= 3:
                issue = self._create_exception_anxiety_issue(
                    exception_imports, unused_exceptions, file_path
                )
                issues.append(issue)
        
        # Pattern 2: Over-importing from popular modules
        popular_module_issues = self._detect_popular_module_overimport(imports, usage, file_path)
        issues.extend(popular_module_issues)
        
        # Pattern 3: Copy-paste import blocks
        duplicate_import_issues = self._detect_duplicate_import_blocks(imports, file_path)
        issues.extend(duplicate_import_issues)
        
        return issues
    
    def _find_exception_imports(self, imports: List[ImportInfo]) -> List[ImportInfo]:
        """Find imports that are likely exceptions."""
        exception_imports = []
        exception_keywords = ['error', 'exception', 'warning', 'fault', 'abort']
        
        for imp in imports:
            name_lower = imp.imported_name.lower()
            if (any(keyword in name_lower for keyword in exception_keywords) or
                imp.imported_name.endswith('Error') or 
                imp.imported_name.endswith('Exception')):
                exception_imports.append(imp)
        
        return exception_imports
    
    def _detect_popular_module_overimport(
        self, 
        imports: List[ImportInfo], 
        usage: Dict[str, UsageInfo],
        file_path: str
    ) -> List[Issue]:
        """Detect over-importing from popular modules."""
        issues = []
        
        # Group imports by module
        module_imports = defaultdict(list)
        for imp in imports:
            if imp.is_third_party or imp.module_name in self.COMMON_THIRD_PARTY:
                module_imports[imp.module_name].append(imp)
        
        for module_name, module_imp_list in module_imports.items():
            if len(module_imp_list) >= 8:  # Threshold for "too many" imports
                unused_count = 0
                unused_items = []
                
                for imp in module_imp_list:
                    usage_name = imp.get_usage_name()
                    if usage_name not in usage or not usage[usage_name].is_actually_used():
                        unused_count += 1
                        unused_items.append(imp.imported_name)
                
                unused_ratio = unused_count / len(module_imp_list)
                if unused_ratio > 0.4:  # More than 40% unused
                    issue = Issue(
                        kind="popular_module_overimport",
                        message=f"Over-importing from {module_name}: {len(module_imp_list)} imports, {unused_count} unused ({unused_ratio:.1%})",
                        severity=2,
                        file=file_path,
                        line=module_imp_list[0].line_number,
                        confidence=min(0.9, unused_ratio + 0.3),
                        evidence={
                            'module': module_name,
                            'total_imports': len(module_imp_list),
                            'unused_count': unused_count,
                            'unused_ratio': unused_ratio,
                            'unused_items': unused_items[:10],
                            'pattern_type': 'popular_module_overimport'
                        },
                        suggestions=self._generate_overimport_suggestions(
                            module_name, unused_items, len(module_imp_list)
                        )
                    )
                    issues.append(issue)
        
        return issues
    
    def _detect_duplicate_import_blocks(self, imports: List[ImportInfo], file_path: str) -> List[Issue]:
        """Detect potential duplicate or redundant import blocks."""
        issues = []
        
        # Look for imports with similar line numbers (likely in blocks)
        import_blocks = []
        current_block = []
        
        sorted_imports = sorted(imports, key=lambda x: x.line_number)
        
        for imp in sorted_imports:
            if not current_block or imp.line_number - current_block[-1].line_number <= 3:
                current_block.append(imp)
            else:
                if len(current_block) >= 5:
                    import_blocks.append(current_block)
                current_block = [imp]
        
        if len(current_block) >= 5:
            import_blocks.append(current_block)
        
        # Look for suspiciously large blocks
        for block in import_blocks:
            if len(block) >= 15:
                module_diversity = len(set(imp.module_name for imp in block))
                if module_diversity / len(block) < 0.3:  # Low diversity suggests copy-paste
                    issue = Issue(
                        kind="large_import_block",
                        message=f"Suspiciously large import block: {len(block)} imports with low module diversity",
                        severity=2,
                        file=file_path,
                        line=block[0].line_number,
                        confidence=0.7,
                        evidence={
                            'block_size': len(block),
                            'module_diversity': module_diversity,
                            'start_line': block[0].line_number,
                            'end_line': block[-1].line_number,
                            'modules': list(set(imp.module_name for imp in block))
                        },
                        suggestions=[
                            "Review if all imports in this block are necessary",
                            "Consider grouping imports by purpose rather than copying entire blocks",
                            "Use import statements closer to where functionality is needed",
                            "Break large import blocks into focused sections"
                        ]
                    )
                    issues.append(issue)
        
        return issues
    
    def _analyze_cross_file_patterns(self, all_file_data: Dict[str, Any], ctx: AnalysisContext) -> List[Issue]:
        """Analyze import patterns across multiple files."""
        issues = []
        
        # Look for repeated import patterns across files
        import_patterns = defaultdict(list)
        
        for file_path, data in all_file_data.items():
            imports = data['imports']
            
            # Create signature for import patterns
            module_counts = Counter(imp.module_name for imp in imports)
            pattern_sig = tuple(sorted(
                (module, count) for module, count in module_counts.items() 
                if count >= 3
            ))
            
            if pattern_sig:
                import_patterns[pattern_sig].append(file_path)
        
        # Find patterns that appear in multiple files
        for pattern_sig, file_list in import_patterns.items():
            if len(file_list) >= 3:  # Pattern appears in 3+ files
                total_imports = sum(count for module, count in pattern_sig)
                if total_imports >= 8:  # Significant import volume
                    issue = Issue(
                        kind="repeated_import_pattern",
                        message=f"Import pattern repeated across {len(file_list)} files: {total_imports} imports from {len(pattern_sig)} modules",
                        severity=2,
                        file=file_list[0],
                        line=1,
                        confidence=0.8,
                        evidence={
                            'pattern_files': file_list,
                            'modules': [module for module, count in pattern_sig],
                            'total_imports': total_imports,
                            'file_count': len(file_list)
                        },
                        suggestions=[
                            "Consider creating a shared utility module for common imports",
                            "Review if all files actually need all these imports",
                            "Create import conventions for the project",
                            "Use __init__.py files to provide cleaner import interfaces"
                        ]
                    )
                    issues.append(issue)
        
        return issues
    
    def _create_import_anxiety_issue(
        self, 
        cluster: ImportCluster, 
        file_path: str, 
        ctx: AnalysisContext
    ) -> Optional[Issue]:
        """Create an Issue object for an import anxiety cluster."""
        if not cluster.imports:
            return None
        
        primary_import = cluster.imports[0]
        
        # Build description based on pattern type
        pattern_descriptions = {
            "error_handling_anxiety": "Defensive over-importing of error/exception types",
            "class_import_spree": "Importing many classes at once",
            "import_everything": "Excessive imports from modules",
            "wildcard_import": "Using wildcard imports",
            "unused_imports": "Large number of unused imports",
            "utility_spree": "Importing many utility functions",
            "redundant_imports": "Importing redundant functionality"
        }
        
        description = pattern_descriptions.get(cluster.pattern_type, "Import anxiety detected")
        description += f": {len(cluster.imports)} imports, {cluster.unused_ratio:.1%} unused"
        
        # Generate evidence
        evidence = {
            'cluster_id': cluster.cluster_id,
            'pattern_type': cluster.pattern_type,
            'import_count': len(cluster.imports),
            'unused_ratio': cluster.unused_ratio,
            'anxiety_score': cluster.anxiety_score,
            'evidence_indicators': cluster.evidence_indicators,
            'modules': list(set(imp.module_name for imp in cluster.imports)),
            'import_details': [
                {
                    'module': imp.module_name,
                    'name': imp.imported_name,
                    'line': imp.line_number,
                    'type': imp.import_type,
                    'is_wildcard': imp.is_wildcard
                }
                for imp in cluster.imports[:10]  # Limit for readability
            ]
        }
        
        # Generate suggestions
        suggestions = self._generate_cluster_suggestions(cluster)
        
        # Determine severity
        severity = 1  # Info
        if cluster.anxiety_score > 0.7:
            severity = 2  # Warning
        if cluster.anxiety_score > 0.8 or cluster.unused_ratio > 0.7:
            severity = 3  # High
        
        return Issue(
            kind=f"import_anxiety_{cluster.pattern_type}",
            message=description,
            severity=severity,
            file=file_path,
            line=primary_import.line_number,
            confidence=cluster.anxiety_score,
            evidence=evidence,
            suggestions=suggestions
        )
    
    def _create_exception_anxiety_issue(
        self, 
        exception_imports: List[ImportInfo], 
        unused_exceptions: List[ImportInfo],
        file_path: str
    ) -> Issue:
        """Create an issue for exception handling anxiety."""
        return Issue(
            kind="error_handling_anxiety",
            message=f"Error handling anxiety: importing {len(exception_imports)} exception types, {len(unused_exceptions)} unused",
            severity=2,
            file=file_path,
            line=exception_imports[0].line_number,
            confidence=min(0.9, len(unused_exceptions) / len(exception_imports) + 0.3),
            evidence={
                'total_exceptions': len(exception_imports),
                'unused_exceptions': len(unused_exceptions),
                'unused_ratio': len(unused_exceptions) / len(exception_imports),
                'exception_names': [imp.imported_name for imp in exception_imports],
                'unused_names': [imp.imported_name for imp in unused_exceptions]
            },
            suggestions=[
                "Import exceptions only when you plan to handle them specifically",
                "Use generic Exception catching for unexpected errors",
                "Remove unused exception imports",
                "Consider using try/except with specific exceptions closer to where errors might occur"
            ]
        )
    
    def _generate_cluster_suggestions(self, cluster: ImportCluster) -> List[str]:
        """Generate suggestions for resolving import anxiety."""
        suggestions = []
        
        # Pattern-specific suggestions
        if cluster.pattern_type == "error_handling_anxiety":
            suggestions.extend([
                "Import only the exception types you actually handle",
                "Use broad exception catching (Exception) for unexpected errors",
                "Consider using contextlib for resource management instead of manual exception handling"
            ])
        
        elif cluster.pattern_type == "class_import_spree":
            suggestions.extend([
                "Import only the classes you actually instantiate",
                "Consider using factory patterns or dependency injection",
                "Group related class imports and review their necessity"
            ])
        
        elif cluster.pattern_type == "import_everything":
            suggestions.extend([
                "Use qualified imports (import module) instead of importing many items",
                "Import items closer to where they're used",
                "Review if all imported functionality is necessary"
            ])
        
        elif cluster.pattern_type == "wildcard_import":
            suggestions.extend([
                "Replace wildcard imports with specific imports",
                "Use qualified imports to avoid namespace pollution",
                "List specific items: from module import item1, item2"
            ])
        
        # General suggestions based on unused ratio
        if cluster.unused_ratio > 0.5:
            unused_names = []
            # This would need the usage data to be passed in
            suggestions.append(f"Remove {cluster.unused_ratio:.1%} unused imports to reduce clutter")
        
        # General suggestions
        suggestions.extend([
            "Import statements should reflect actual usage patterns",
            "Consider using lazy imports for expensive modules",
            "Group related imports together for better organization"
        ])
        
        return suggestions
    
    def _generate_overimport_suggestions(self, module_name: str, unused_items: List[str], total_count: int) -> List[str]:
        """Generate suggestions for over-importing from popular modules."""
        suggestions = [
            f"Remove unused imports from {module_name}: {', '.join(unused_items[:5])}{'...' if len(unused_items) > 5 else ''}",
            f"Consider using qualified imports: import {module_name}",
            "Import only what you actually use to improve code clarity",
            f"Review if all {total_count} imports from {module_name} are necessary"
        ]
        
        # Module-specific suggestions
        if module_name in ['numpy', 'pandas']:
            suggestions.append(f"For {module_name}, consider importing the module and using qualified access")
        elif module_name in ['matplotlib', 'pyplot']:
            suggestions.append("For matplotlib, import submodules as needed rather than many components")
        
        return suggestions


# Alias for backward compatibility  
ImportAnxietyAnalyzer = ImportAnxietyDetector
