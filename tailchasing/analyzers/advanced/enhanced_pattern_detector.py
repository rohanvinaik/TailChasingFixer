"""
Enhanced Pattern Detector for advanced tail-chasing anti-patterns.

This module implements sophisticated detection algorithms for LLM-induced
patterns that go beyond simple structural analysis. It uses AST analysis,
git history when available, and statistical methods to identify complex
anti-patterns.
"""

from __future__ import annotations
import ast
import hashlib
import logging
import os
import re
import subprocess
from collections import Counter
from typing import Dict, List, Optional, Set, Any
import importlib.util

from ..base import BaseAnalyzer, AnalysisContext
from ...core.issues import Issue
from .pattern_types import (
    TailChasingPattern, PatternEvidence, PatternSeverity, 
    PatternCluster
)

logger = logging.getLogger(__name__)


class EnhancedPatternDetector(BaseAnalyzer):
    """
    Advanced pattern detector for sophisticated tail-chasing anti-patterns.
    
    This detector identifies complex patterns that emerge from LLM-assisted
    development, including fictional subsystems, context window limitations,
    and defensive programming patterns.
    """
    
    name = "enhanced_pattern_detector"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.git_available = self._check_git_availability()
        
        # Pattern detection thresholds
        self.hallucination_threshold = self.config.get('hallucination_threshold', 0.7)
        self.context_thrashing_threshold = self.config.get('context_thrashing_threshold', 0.8)
        self.import_anxiety_threshold = self.config.get('import_anxiety_threshold', 3.0)
        
        # Caches for performance
        self._import_cache: Dict[str, Set[str]] = {}
        self._function_cache: Dict[str, List[ast.FunctionDef]] = {}
        self._git_history_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Pattern clusters for cross-pattern analysis
        self.pattern_clusters: List[PatternCluster] = []
        
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run enhanced pattern detection across the entire codebase.
        
        Args:
            ctx: Analysis context containing AST index and configuration
            
        Returns:
            List of Issue objects representing detected patterns
        """
        issues = []
        patterns = []
        
        try:
            logger.info(f"Running enhanced pattern detection on {len(ctx.ast_index)} files")
            
            # Detect hallucination cascades
            logger.debug("Detecting hallucination cascades")
            hallucination_patterns = self.detect_hallucination_cascade(ctx.ast_index, ctx)
            patterns.extend(hallucination_patterns)
            
            # Detect context window thrashing
            logger.debug("Detecting context window thrashing")
            for filepath, tree in ctx.ast_index.items():
                if not ctx.is_excluded(filepath):
                    thrashing_patterns = self.detect_context_window_thrashing(tree, filepath, ctx)
                    patterns.extend(thrashing_patterns)
            
            # Detect import anxiety
            logger.debug("Detecting import anxiety patterns")
            anxiety_patterns = self.detect_import_anxiety(ctx.ast_index, ctx)
            patterns.extend(anxiety_patterns)
            
            # Perform cross-pattern clustering analysis
            if patterns:
                logger.debug("Performing pattern clustering analysis")
                self._cluster_related_patterns(patterns)
            
            # Convert patterns to issues
            for pattern in patterns:
                issue = Issue(**pattern.to_issue_dict())
                issues.append(issue)
                
            logger.info(f"Enhanced pattern detection complete: found {len(issues)} patterns")
            
        except Exception as e:
            logger.error(f"Error in enhanced pattern detection: {e}", exc_info=True)
            # Don't fail the entire analysis, just log the error
            
        return issues
    
    def detect_hallucination_cascade(
        self, 
        ast_index: Dict[str, ast.AST],
        ctx: Optional[AnalysisContext] = None
    ) -> List[TailChasingPattern]:
        """
        Detect hallucination cascades where LLMs create fictional subsystems.
        
        A hallucination cascade occurs when an LLM:
        1. References non-existent modules or functions
        2. Creates placeholder implementations that reference more fictional code
        3. Builds entire subsystems around non-existent foundations
        
        Args:
            ast_index: Dictionary mapping file paths to AST trees
            ctx: Optional analysis context for additional information
            
        Returns:
            List of detected hallucination cascade patterns
        """
        patterns = []
        
        try:
            # Step 1: Identify all imports and function calls
            import_map = self._build_import_map(ast_index)
            call_map = self._build_call_map(ast_index)
            
            # Step 2: Find fictional modules (imports that don't exist)
            fictional_modules = self._find_fictional_modules(import_map, ctx)
            
            # Step 3: Trace cascading references
            for primary_file, fictional_imports in fictional_modules.items():
                if not fictional_imports:
                    continue
                    
                # Build evidence for this cascade
                evidence = PatternEvidence()
                evidence.add_file(primary_file)
                
                # Trace how fictional imports propagate
                cascade_depth = self._trace_cascade_depth(
                    primary_file, fictional_imports, import_map, call_map
                )
                
                if cascade_depth > 1:  # Multi-level cascade detected
                    pattern_id = f"hallucination_{hashlib.md5(primary_file.encode()).hexdigest()[:8]}"
                    
                    # Calculate confidence based on cascade characteristics
                    confidence = min(0.95, 0.6 + (cascade_depth * 0.1) + 
                                   (len(fictional_imports) * 0.05))
                    
                    # Create pattern with rich evidence
                    pattern = TailChasingPattern.create_hallucination_cascade(
                        pattern_id=pattern_id,
                        primary_file=primary_file,
                        fictional_modules=list(fictional_imports),
                        confidence=confidence
                    )
                    
                    # Add cascade-specific evidence
                    evidence.add_complexity_metric("cascade_depth", cascade_depth)
                    evidence.add_complexity_metric("fictional_module_count", len(fictional_imports))
                    
                    # Add git history if available
                    if self.git_available and ctx:
                        git_evidence = self._analyze_git_history(primary_file, ctx)
                        if git_evidence:
                            evidence.creation_timestamps = git_evidence.get('timestamps', [])
                            evidence.commit_messages = git_evidence.get('messages', [])
                    
                    pattern.evidence = evidence
                    
                    # Adjust severity based on cascade scope
                    if cascade_depth > 3 or len(fictional_imports) > 5:
                        pattern.severity = PatternSeverity.CRITICAL
                    
                    patterns.append(pattern)
                    
                    logger.debug(
                        f"Hallucination cascade detected in {primary_file}: "
                        f"{len(fictional_imports)} fictional modules, depth {cascade_depth}"
                    )
        
        except Exception as e:
            logger.error(f"Error detecting hallucination cascades: {e}", exc_info=True)
            
        return patterns
    
    def detect_context_window_thrashing(
        self, 
        tree: ast.AST, 
        filepath: str,
        ctx: Optional[AnalysisContext] = None
    ) -> List[TailChasingPattern]:
        """
        Detect context window thrashing patterns.
        
        Context window thrashing occurs when:
        1. Similar functions are reimplemented instead of reused
        2. Functionality is duplicated due to context limitations
        3. Progressive refinement creates multiple similar implementations
        
        Args:
            tree: AST tree of a single file
            filepath: Path to the file being analyzed
            ctx: Optional analysis context
            
        Returns:
            List of detected context window thrashing patterns
        """
        patterns = []
        
        try:
            # Extract all functions from the file
            functions = self._extract_functions(tree, filepath)
            if len(functions) < 2:
                return patterns
                
            # Find clusters of similar functions
            similarity_clusters = self._find_function_similarity_clusters(
                functions, self.context_thrashing_threshold
            )
            
            for cluster in similarity_clusters:
                if len(cluster['functions']) < 2:
                    continue
                    
                # Build evidence for thrashing pattern
                evidence = PatternEvidence()
                evidence.add_file(filepath)
                
                func_names = [f['name'] for f in cluster['functions']]
                for name in func_names:
                    evidence.add_function(name)
                
                # Calculate metrics
                avg_similarity = cluster['avg_similarity']
                size_variance = cluster['size_variance']
                
                evidence.add_complexity_metric("average_similarity", avg_similarity)
                evidence.add_complexity_metric("size_variance", size_variance)
                evidence.add_complexity_metric("cluster_size", len(cluster['functions']))
                
                # Analyze naming patterns for evidence of iteration
                naming_pattern = self._analyze_naming_patterns(func_names)
                if naming_pattern['has_versioning']:
                    evidence.add_complexity_metric("naming_versions", naming_pattern['version_count'])
                
                # Calculate confidence
                confidence = min(0.95, avg_similarity * 0.8 + 
                               (len(cluster['functions']) - 2) * 0.1)
                
                # Boost confidence if naming suggests iteration
                if naming_pattern['has_versioning']:
                    confidence = min(0.98, confidence + 0.15)
                
                pattern_id = f"context_thrashing_{hashlib.md5((filepath + str(func_names)).encode()).hexdigest()[:8]}"
                
                pattern = TailChasingPattern.create_context_thrashing(
                    pattern_id=pattern_id,
                    primary_file=filepath,
                    duplicate_implementations=func_names,
                    confidence=confidence
                )
                
                # Add specific suggestions based on analysis
                if naming_pattern['has_versioning']:
                    pattern.add_suggestion("Remove older versions and keep only the latest implementation")
                
                if size_variance > 0.5:
                    pattern.add_suggestion("Consider if functions serve different purposes despite similarity")
                
                # Set line number to first function in cluster
                first_func = cluster['functions'][0]
                pattern.primary_line = first_func['line']
                
                pattern.evidence = evidence
                patterns.append(pattern)
                
                logger.debug(
                    f"Context window thrashing detected in {filepath}: "
                    f"{len(func_names)} similar functions (similarity: {avg_similarity:.2f})"
                )
                
        except Exception as e:
            logger.error(f"Error detecting context window thrashing in {filepath}: {e}", exc_info=True)
            
        return patterns
    
    def detect_import_anxiety(
        self, 
        ast_index: Dict[str, ast.AST],
        ctx: Optional[AnalysisContext] = None
    ) -> List[TailChasingPattern]:
        """
        Detect import anxiety patterns where developers over-import defensively.
        
        Import anxiety manifests as:
        1. Importing entire modules when only specific functions are needed
        2. Redundant imports across related files
        3. Importing modules "just in case" without using them
        4. Multiple import styles for the same functionality
        
        Args:
            ast_index: Dictionary mapping file paths to AST trees
            ctx: Optional analysis context
            
        Returns:
            List of detected import anxiety patterns
        """
        patterns = []
        
        try:
            # Analyze import patterns across all files
            import_analysis = self._analyze_import_patterns(ast_index)
            
            for filepath, file_imports in import_analysis.items():
                # Check for various anxiety indicators
                anxiety_indicators = self._check_import_anxiety_indicators(
                    filepath, file_imports, ast_index.get(filepath)
                )
                
                # Calculate anxiety score
                anxiety_score = self._calculate_import_anxiety_score(anxiety_indicators)
                
                if anxiety_score > self.import_anxiety_threshold:
                    # Build evidence
                    evidence = PatternEvidence()
                    evidence.add_file(filepath)
                    
                    excessive_imports = []
                    
                    # Collect specific problematic imports
                    if anxiety_indicators['unused_imports']:
                        excessive_imports.extend(anxiety_indicators['unused_imports'])
                        evidence.add_complexity_metric("unused_import_count", 
                                                     len(anxiety_indicators['unused_imports']))
                    
                    if anxiety_indicators['wildcard_imports']:
                        excessive_imports.extend(anxiety_indicators['wildcard_imports'])
                        evidence.add_complexity_metric("wildcard_import_count",
                                                     len(anxiety_indicators['wildcard_imports']))
                    
                    if anxiety_indicators['redundant_imports']:
                        excessive_imports.extend(anxiety_indicators['redundant_imports'])
                        evidence.add_complexity_metric("redundant_import_count",
                                                     len(anxiety_indicators['redundant_imports']))
                    
                    evidence.add_complexity_metric("anxiety_score", anxiety_score)
                    
                    # Calculate confidence based on evidence strength
                    confidence = min(0.9, 0.5 + (anxiety_score - self.import_anxiety_threshold) * 0.1)
                    
                    pattern_id = f"import_anxiety_{hashlib.md5(filepath.encode()).hexdigest()[:8]}"
                    
                    pattern = TailChasingPattern.create_import_anxiety(
                        pattern_id=pattern_id,
                        primary_file=filepath,
                        excessive_imports=excessive_imports,
                        confidence=confidence
                    )
                    
                    # Add specific suggestions based on findings
                    if anxiety_indicators['unused_imports']:
                        pattern.add_suggestion(f"Remove {len(anxiety_indicators['unused_imports'])} unused imports")
                    
                    if anxiety_indicators['wildcard_imports']:
                        pattern.add_suggestion("Replace wildcard imports with specific imports")
                    
                    if anxiety_indicators['redundant_imports']:
                        pattern.add_suggestion("Consolidate redundant import statements")
                    
                    pattern.evidence = evidence
                    patterns.append(pattern)
                    
                    logger.debug(
                        f"Import anxiety detected in {filepath}: "
                        f"score {anxiety_score:.1f}, {len(excessive_imports)} problematic imports"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting import anxiety: {e}", exc_info=True)
            
        return patterns
    
    # Helper methods for pattern detection
    
    def _check_git_availability(self) -> bool:
        """Check if git is available for history analysis."""
        try:
            subprocess.run(['git', '--version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Git not available for history analysis")
            return False
    
    def _build_import_map(self, ast_index: Dict[str, ast.AST]) -> Dict[str, Set[str]]:
        """Build a map of all imports in each file."""
        import_map = {}
        
        for filepath, tree in ast_index.items():
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
            
            import_map[filepath] = imports
            self._import_cache[filepath] = imports
            
        return import_map
    
    def _build_call_map(self, ast_index: Dict[str, ast.AST]) -> Dict[str, Set[str]]:
        """Build a map of all function calls in each file.""" 
        call_map = {}
        
        for filepath, tree in ast_index.items():
            calls = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            
            call_map[filepath] = calls
            
        return call_map
    
    def _find_fictional_modules(
        self, 
        import_map: Dict[str, Set[str]], 
        ctx: Optional[AnalysisContext] = None
    ) -> Dict[str, Set[str]]:
        """Find imports that reference non-existent modules."""
        fictional_modules = {}
        
        for filepath, imports in import_map.items():
            fictional = set()
            
            for module_name in imports:
                if not self._module_exists(module_name, ctx):
                    fictional.add(module_name)
            
            if fictional:
                fictional_modules[filepath] = fictional
                
        return fictional_modules
    
    def _module_exists(self, module_name: str, ctx: Optional[AnalysisContext] = None) -> bool:
        """Check if a module actually exists."""
        # Skip standard library and common packages
        stdlib_modules = {
            'os', 'sys', 'json', 'csv', 'sqlite3', 'datetime', 'pathlib',
            're', 'collections', 'itertools', 'functools', 'typing',
            'logging', 'urllib', 'http', 'xml', 'html', 'email',
            'subprocess', 'threading', 'multiprocessing', 'concurrent'
        }
        
        common_packages = {
            'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow',
            'torch', 'flask', 'django', 'requests', 'click', 'pyyaml',
            'pytest', 'setuptools', 'wheel', 'pip'
        }
        
        module_base = module_name.split('.')[0]
        
        if module_base in stdlib_modules or module_base in common_packages:
            return True
        
        # Check if it's a local module
        if ctx:
            # Convert module name to potential file paths
            potential_paths = [
                os.path.join(str(ctx.root_dir), *module_name.split('.')) + '.py',
                os.path.join(str(ctx.root_dir), *module_name.split('.'), '__init__.py')
            ]
            
            for path in potential_paths:
                if os.path.exists(path):
                    return True
        
        # Try to import it
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    
    def _trace_cascade_depth(
        self,
        primary_file: str,
        fictional_imports: Set[str], 
        import_map: Dict[str, Set[str]],
        call_map: Dict[str, Set[str]]
    ) -> int:
        """Trace how deep the hallucination cascade goes."""
        depth = 1
        current_level = fictional_imports
        seen = set(fictional_imports)
        
        # Trace cascade through multiple levels
        for level in range(5):  # Limit to prevent infinite loops
            next_level = set()
            
            for module in current_level:
                # Find files that might implement this fictional module
                for filepath, calls in call_map.items():
                    if module in calls:
                        # This file references the fictional module
                        file_imports = import_map.get(filepath, set())
                        for imp in file_imports:
                            if imp not in seen and not self._module_exists(imp):
                                next_level.add(imp)
                                seen.add(imp)
            
            if next_level:
                depth += 1
                current_level = next_level
            else:
                break
        
        return depth
    
    def _extract_functions(self, tree: ast.AST, filepath: str) -> List[Dict[str, Any]]:
        """Extract function information from an AST."""
        if filepath in self._function_cache:
            return self._function_cache[filepath]
        
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'end_line': getattr(node, 'end_lineno', node.lineno),
                    'ast_node': node,
                    'body_size': len(node.body),
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node) or "",
                    'signature_hash': self._compute_function_signature_hash(node)
                }
                functions.append(func_info)
        
        self._function_cache[filepath] = functions
        return functions
    
    def _find_function_similarity_clusters(
        self, 
        functions: List[Dict[str, Any]], 
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Find clusters of similar functions."""
        clusters = []
        processed = set()
        
        for i, func1 in enumerate(functions):
            if i in processed:
                continue
                
            cluster = {
                'functions': [func1],
                'similarities': [],
                'avg_similarity': 0.0,
                'size_variance': 0.0
            }
            
            for j, func2 in enumerate(functions[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self._calculate_function_similarity(func1, func2)
                
                if similarity >= threshold:
                    cluster['functions'].append(func2)
                    cluster['similarities'].append(similarity)
                    processed.add(j)
            
            if len(cluster['functions']) > 1:
                cluster['avg_similarity'] = sum(cluster['similarities']) / len(cluster['similarities'])
                
                # Calculate size variance
                sizes = [f['body_size'] for f in cluster['functions']]
                mean_size = sum(sizes) / len(sizes)
                variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
                cluster['size_variance'] = variance / mean_size if mean_size > 0 else 0
                
                clusters.append(cluster)
                processed.add(i)
        
        return clusters
    
    def _calculate_function_similarity(self, func1: Dict[str, Any], func2: Dict[str, Any]) -> float:
        """Calculate similarity between two functions."""
        # Name similarity
        name_sim = self._string_similarity(func1['name'], func2['name'])
        
        # Signature similarity  
        args1 = set(func1['args'])
        args2 = set(func2['args'])
        if args1 or args2:
            sig_sim = len(args1 & args2) / len(args1 | args2)
        else:
            sig_sim = 1.0
        
        # Size similarity
        size1, size2 = func1['body_size'], func2['body_size']
        if size1 == 0 and size2 == 0:
            size_sim = 1.0
        elif size1 == 0 or size2 == 0:
            size_sim = 0.0
        else:
            size_sim = 1.0 - abs(size1 - size2) / max(size1, size2)
        
        # Docstring similarity
        doc_sim = self._string_similarity(func1['docstring'], func2['docstring'])
        
        # Weighted combination
        return (name_sim * 0.4 + sig_sim * 0.3 + size_sim * 0.2 + doc_sim * 0.1)
    
    def _compute_function_signature_hash(self, node: ast.FunctionDef) -> str:
        """Compute a hash of the function signature."""
        signature_parts = [
            node.name,
            str(len(node.args.args)),
            str(len(node.args.defaults)),
            str(bool(node.args.vararg)),
            str(bool(node.args.kwarg))
        ]
        
        signature = "|".join(signature_parts)
        return hashlib.md5(signature.encode()).hexdigest()[:16]
    
    def _analyze_naming_patterns(self, func_names: List[str]) -> Dict[str, Any]:
        """Analyze function names for patterns suggesting iteration."""
        pattern_info = {
            'has_versioning': False,
            'version_count': 0,
            'common_prefixes': [],
            'numeric_suffixes': []
        }
        
        # Check for versioning patterns (v1, v2, _new, _old, etc.)
        version_patterns = [
            r'.*v\d+$',
            r'.*_v\d+$', 
            r'.*\d+$',
            r'.*_(new|old|updated|revised|fixed)$'
        ]
        
        versioned_names = []
        for name in func_names:
            for pattern in version_patterns:
                if re.match(pattern, name):
                    versioned_names.append(name)
                    break
        
        if versioned_names:
            pattern_info['has_versioning'] = True
            pattern_info['version_count'] = len(versioned_names)
        
        # Find common prefixes
        if len(func_names) > 1:
            prefixes = []
            for i in range(len(func_names)):
                for j in range(i+1, len(func_names)):
                    common = self._common_prefix(func_names[i], func_names[j])
                    if len(common) > 3:  # Meaningful prefix
                        prefixes.append(common)
            
            if prefixes:
                pattern_info['common_prefixes'] = list(set(prefixes))
        
        return pattern_info
    
    def _analyze_import_patterns(self, ast_index: Dict[str, ast.AST]) -> Dict[str, Dict[str, Any]]:
        """Analyze import patterns across all files."""
        import_analysis = {}
        
        for filepath, tree in ast_index.items():
            file_analysis = {
                'imports': [],
                'from_imports': [],
                'wildcard_imports': [],
                'used_names': set(),
                'import_locations': {}
            }
            
            # Collect all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_info = {
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        }
                        file_analysis['imports'].append(import_info)
                        file_analysis['import_locations'][alias.asname or alias.name] = node.lineno
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            if alias.name == '*':
                                file_analysis['wildcard_imports'].append({
                                    'module': node.module,
                                    'line': node.lineno
                                })
                            else:
                                import_info = {
                                    'module': node.module,
                                    'name': alias.name,
                                    'alias': alias.asname,
                                    'line': node.lineno
                                }
                                file_analysis['from_imports'].append(import_info)
                                file_analysis['import_locations'][alias.asname or alias.name] = node.lineno
                
                elif isinstance(node, ast.Name):
                    file_analysis['used_names'].add(node.id)
            
            import_analysis[filepath] = file_analysis
        
        return import_analysis
    
    def _check_import_anxiety_indicators(
        self, 
        filepath: str, 
        file_imports: Dict[str, Any], 
        tree: Optional[ast.AST]
    ) -> Dict[str, List[str]]:
        """Check for specific import anxiety indicators."""
        indicators = {
            'unused_imports': [],
            'wildcard_imports': [],
            'redundant_imports': [],
            'over_imports': []
        }
        
        used_names = file_imports['used_names']
        
        # Check for unused imports
        for imp in file_imports['imports']:
            import_name = imp['alias'] or imp['module'].split('.')[0]
            if import_name not in used_names:
                indicators['unused_imports'].append(imp['module'])
        
        for imp in file_imports['from_imports']:
            import_name = imp['alias'] or imp['name']
            if import_name not in used_names:
                indicators['unused_imports'].append(f"{imp['module']}.{imp['name']}")
        
        # Wildcard imports are automatically problematic
        for imp in file_imports['wildcard_imports']:
            indicators['wildcard_imports'].append(imp['module'])
        
        # Check for redundant imports (same module imported multiple ways)
        module_import_count = Counter()
        for imp in file_imports['imports']:
            module_import_count[imp['module'].split('.')[0]] += 1
        
        for imp in file_imports['from_imports']:
            module_import_count[imp['module'].split('.')[0]] += 1
        
        for module, count in module_import_count.items():
            if count > 1:
                indicators['redundant_imports'].append(module)
        
        return indicators
    
    def _calculate_import_anxiety_score(self, indicators: Dict[str, List[str]]) -> float:
        """Calculate an anxiety score based on import indicators."""
        score = 0.0
        
        # Weight different types of anxiety
        score += len(indicators['unused_imports']) * 0.5
        score += len(indicators['wildcard_imports']) * 1.5
        score += len(indicators['redundant_imports']) * 1.0
        score += len(indicators['over_imports']) * 0.8
        
        return score
    
    def _analyze_git_history(self, filepath: str, ctx: AnalysisContext) -> Optional[Dict[str, Any]]:
        """Analyze git history for a file if available."""
        if not self.git_available or filepath in self._git_history_cache:
            return self._git_history_cache.get(filepath)
        
        try:
            # Get recent commits for the file
            result = subprocess.run([
                'git', 'log', '--oneline', '-n', '10', '--', filepath
            ], capture_output=True, text=True, cwd=str(ctx.root_dir))
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                history = {
                    'timestamps': [],
                    'messages': []
                }
                
                for line in lines:
                    if line.strip():
                        parts = line.split(' ', 1)
                        if len(parts) > 1:
                            history['messages'].append(parts[1])
                
                self._git_history_cache[filepath] = history
                return history
                
        except Exception as e:
            logger.debug(f"Error analyzing git history for {filepath}: {e}")
        
        return None
    
    def _cluster_related_patterns(self, patterns: List[TailChasingPattern]) -> None:
        """Cluster related patterns for cross-pattern analysis."""
        if len(patterns) < 2:
            return
        
        # Simple clustering based on file overlap
        clusters = []
        processed = set()
        
        for i, pattern1 in enumerate(patterns):
            if i in processed:
                continue
            
            cluster = PatternCluster(
                cluster_id=f"cluster_{len(clusters)}",
                patterns=[pattern1]
            )
            
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                if j in processed:
                    continue
                
                # Check if patterns share files
                files1 = set(pattern1.affected_files)
                files2 = set(pattern2.affected_files)
                
                if files1 & files2:  # Patterns overlap
                    cluster.add_pattern(pattern2)
                    processed.add(j)
            
            if len(cluster.patterns) > 1:
                clusters.append(cluster)
                self.pattern_clusters.append(cluster)
                processed.add(i)
                
                logger.debug(f"Pattern cluster formed: {len(cluster.patterns)} related patterns")
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using simple character overlap."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        s1_chars = set(s1.lower())
        s2_chars = set(s2.lower())
        
        intersection = s1_chars & s2_chars
        union = s1_chars | s2_chars
        
        return len(intersection) / len(union) if union else 0.0
    
    def _common_prefix(self, s1: str, s2: str) -> str:
        """Find common prefix between two strings."""
        prefix = ""
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                prefix += c1
            else:
                break
        return prefix