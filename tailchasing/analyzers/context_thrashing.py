"""Context-window-thrashing analyzer for detecting similar functions separated by distance."""

import ast
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import re

from .base import Analyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno, safe_get_end_lineno


@dataclass(frozen=True)
class FunctionInfo:
    """Information about a function for similarity analysis."""
    name: str
    class_name: Optional[str]
    file_path: str
    line_start: int
    line_end: int
    signature: str
    normalized_body: str
    raw_body: str
    ast_node: ast.FunctionDef
    parameters: tuple  # Changed from List[str] to tuple for hashability
    return_type: Optional[str] = None


@dataclass(frozen=True)
class SimilarFunctionPair:
    """Pair of similar functions that may cause context-window thrashing."""
    func1: FunctionInfo
    func2: FunctionInfo
    similarity: float
    line_distance: int
    common_body: str
    differences: tuple  # Changed from List[str] to tuple for hashability
    suggested_helper_name: str
    extract_playbook: Optional[str] = None


@dataclass 
class ContextThrashingCluster:
    """Cluster of functions causing context-window thrashing."""
    cluster_id: str
    functions: List[FunctionInfo]
    primary_pair: SimilarFunctionPair
    file_path: str
    issue_count: int
    severity: float
    suggested_helper: str
    extract_playbook: str
    mini_diff: str = ""
    members_truncated: bool = False
    hidden_members: int = 0


class FunctionSimilarityAnalyzer:
    """Analyzes function similarity using AST and text comparison."""
    
    def __init__(self):
        self.normalization_patterns = [
            (r'\b\d+\b', 'NUM'),  # Numbers
            (r'\b[a-f0-9]{8,}\b', 'HEX'),  # Hex values
            (r'["\'][^"\']*["\']', 'STR'),  # String literals
            (r'\b[a-zA-Z_]\w*_\d+\b', 'VAR_N'),  # Variables with numbers
            (r'\s+', ' '),  # Normalize whitespace
        ]
    
    def extract_function_info(self, file_path: str, node: ast.FunctionDef, 
                             class_name: Optional[str], source_lines: List[str]) -> FunctionInfo:
        """Extract detailed information about a function."""
        # Get function signature
        signature = self._extract_signature(node)
        
        # Get function body
        start_line = safe_get_lineno(node, 1)
        end_line = safe_get_end_lineno(node) or start_line
        
        # Extract raw body text
        if start_line <= len(source_lines):
            raw_body_lines = source_lines[start_line-1:end_line]
            raw_body = '\n'.join(raw_body_lines)
        else:
            raw_body = ast.unparse(node)
        
        # Normalize body for similarity comparison
        normalized_body = self._normalize_code(raw_body)
        
        # Extract parameters
        parameters = tuple(arg.arg for arg in node.args.args)
        
        # Extract return type if annotated
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        return FunctionInfo(
            name=node.name,
            class_name=class_name,
            file_path=file_path,
            line_start=start_line,
            line_end=end_line,
            signature=signature,
            normalized_body=normalized_body,
            raw_body=raw_body,
            ast_node=node,
            parameters=parameters,
            return_type=return_type
        )
    
    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature."""
        try:
            # Use ast.unparse if available (Python 3.9+)
            sig_parts = []
            
            # Regular args
            for arg in node.args.args:
                sig_parts.append(arg.arg)
            
            # Varargs
            if node.args.vararg:
                sig_parts.append(f"*{node.args.vararg.arg}")
            
            # Keyword-only args
            for arg in node.args.kwonlyargs:
                sig_parts.append(arg.arg)
            
            # Kwargs
            if node.args.kwarg:
                sig_parts.append(f"**{node.args.kwarg.arg}")
            
            return f"({', '.join(sig_parts)})"
        except:
            return "()"
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for similarity comparison."""
        normalized = code
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove comments and docstrings
        lines = normalized.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove inline comments
                if '#' in line and not ('"""' in line or "'''" in line):
                    line = line[:line.index('#')].strip()
                if line:
                    clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def calculate_similarity(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Calculate similarity between two functions using multiple metrics."""
        # Text similarity using SequenceMatcher
        text_sim = difflib.SequenceMatcher(
            None, 
            func1.normalized_body, 
            func2.normalized_body
        ).ratio()
        
        # AST structure similarity
        ast_sim = self._calculate_ast_similarity(func1.ast_node, func2.ast_node)
        
        # Parameter similarity
        param_sim = self._calculate_parameter_similarity(func1.parameters, func2.parameters)
        
        # Weighted combination
        return (text_sim * 0.5) + (ast_sim * 0.3) + (param_sim * 0.2)
    
    def _calculate_ast_similarity(self, node1: ast.AST, node2: ast.AST) -> float:
        """Calculate AST structure similarity."""
        try:
            # Convert to normalized AST strings
            ast_str1 = ast.dump(node1, indent=None)
            ast_str2 = ast.dump(node2, indent=None)
            
            # Normalize variable names and literals
            norm_ast1 = re.sub(r"arg='[^']*'", "arg='VAR'", ast_str1)
            norm_ast1 = re.sub(r"value=\d+", "value=NUM", norm_ast1)
            norm_ast1 = re.sub(r"s='[^']*'", "s='STR'", norm_ast1)
            
            norm_ast2 = re.sub(r"arg='[^']*'", "arg='VAR'", ast_str2)
            norm_ast2 = re.sub(r"value=\d+", "value=NUM", norm_ast2)
            norm_ast2 = re.sub(r"s='[^']*'", "s='STR'", norm_ast2)
            
            return difflib.SequenceMatcher(None, norm_ast1, norm_ast2).ratio()
        except:
            return 0.0
    
    def _calculate_parameter_similarity(self, params1: tuple, params2: tuple) -> float:
        """Calculate parameter list similarity."""
        if not params1 and not params2:
            return 1.0
        if not params1 or not params2:
            return 0.0
        
        # Jaccard similarity of parameter sets
        set1 = set(params1)
        set2 = set(params2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def find_similar_pairs(self, functions: List[FunctionInfo], 
                          similarity_threshold: float = 0.7,
                          min_line_distance: int = 50) -> List[SimilarFunctionPair]:
        """Find pairs of similar functions that are far apart."""
        similar_pairs = []
        
        # Group by function name
        by_name = defaultdict(list)
        for func in functions:
            by_name[func.name].append(func)
        
        # Find similar pairs within same-named functions
        for name, func_list in by_name.items():
            if len(func_list) < 2:
                continue
            
            for i, func1 in enumerate(func_list):
                for func2 in func_list[i+1:]:
                    # Check line distance
                    line_distance = abs(func2.line_start - func1.line_start)
                    if line_distance < min_line_distance:
                        continue
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(func1, func2)
                    if similarity >= similarity_threshold:
                        # Find common body patterns
                        common_body, differences = self._analyze_differences(func1, func2)
                        
                        # Suggest helper name
                        helper_name = self._suggest_helper_name(func1, func2)
                        
                        pair = SimilarFunctionPair(
                            func1=func1,
                            func2=func2,
                            similarity=similarity,
                            line_distance=line_distance,
                            common_body=common_body,
                            differences=tuple(differences),  # Convert to tuple
                            suggested_helper_name=helper_name
                        )
                        
                        similar_pairs.append(pair)
        
        return similar_pairs
    
    def _analyze_differences(self, func1: FunctionInfo, func2: FunctionInfo) -> Tuple[str, List[str]]:
        """Analyze differences between two similar functions."""
        # Use difflib to find common subsequences
        lines1 = func1.normalized_body.splitlines()
        lines2 = func2.normalized_body.splitlines()
        
        matcher = difflib.SequenceMatcher(None, lines1, lines2)
        common_lines = []
        differences = []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                common_lines.extend(lines1[i1:i2])
            else:
                differences.append(f"Difference: {tag} at lines {i1}-{i2} vs {j1}-{j2}")
        
        common_body = '\n'.join(common_lines)
        
        return common_body, differences
    
    def _suggest_helper_name(self, func1: FunctionInfo, func2: FunctionInfo) -> str:
        """Suggest a name for the extracted helper function."""
        base_name = func1.name
        
        # Common patterns for helper names
        if base_name == '__init__':
            return '_init_common'
        elif base_name.startswith('get_'):
            return f'_get_{base_name[4:]}_common'
        elif base_name.startswith('set_'):
            return f'_set_{base_name[4:]}_common'
        elif base_name.startswith('process_'):
            return f'_process_{base_name[8:]}_helper'
        else:
            return f'_{base_name}_common'


class ExtractHelperPlaybookGenerator:
    """Generates playbooks for extracting helper functions."""
    
    def generate_extract_playbook(self, pair: SimilarFunctionPair) -> str:
        """Generate a playbook for extracting common functionality."""
        func1, func2 = pair.func1, pair.func2
        helper_name = pair.suggested_helper_name
        
        playbook_lines = [
            f"EXTRACT HELPER PLAYBOOK: {helper_name}",
            "=" * 50,
            "",
            f"Functions: {func1.name} (line {func1.line_start}) and {func2.name} (line {func2.line_start})",
            f"Similarity: {pair.similarity:.1%}",
            f"Line distance: {pair.line_distance} lines",
            "",
            "SUGGESTED REFACTORING:",
            f"1. Extract common functionality to {helper_name}()",
            f"2. Replace duplicate code with calls to {helper_name}()",
            f"3. Merge both {func1.name} paths using the helper",
            "",
            "MINI-DIFF PREVIEW:",
            self._generate_mini_diff(pair),
            "",
            "SAFETY CHECKS:",
            "- Ensure helper function signature covers all use cases",
            "- Verify all parameters are properly passed",
            "- Run tests to confirm behavior is preserved",
        ]
        
        return '\n'.join(playbook_lines)
    
    def _generate_mini_diff(self, pair: SimilarFunctionPair) -> str:
        """Generate a mini-diff showing the proposed changes."""
        helper_name = pair.suggested_helper_name
        func1, func2 = pair.func1, pair.func2
        
        # Extract common parameters from both functions
        common_params = set(func1.parameters).intersection(set(func2.parameters))
        unique_params1 = set(func1.parameters) - common_params
        unique_params2 = set(func2.parameters) - common_params
        
        # Suggest helper signature
        helper_params = list(common_params)
        if unique_params1:
            helper_params.extend([f"{p}=None" for p in unique_params1])
        if unique_params2:
            helper_params.extend([f"{p}=None" for p in unique_params2])
        
        helper_sig = f"def {helper_name}({', '.join(helper_params)}):"
        
        diff_lines = [
            "```diff",
            f"+ {helper_sig}",
            f'+     """Common functionality extracted from {func1.name}."""',
            "+     # TODO: Implement common logic here",
            f"+     pass",
            "",
            f"  def {func1.name}{func1.signature}:",
            f"-     # Original implementation (lines {func1.line_start}-{func1.line_end})",
            f"+     return self.{helper_name}({', '.join(common_params)})",
            "",
            f"  def {func2.name}{func2.signature}:",  
            f"-     # Original implementation (lines {func2.line_start}-{func2.line_end})",
            f"+     return self.{helper_name}({', '.join(common_params)})",
            "```"
        ]
        
        return '\n'.join(diff_lines)


class ContextThrashingAnalyzer(Analyzer):
    """Analyzer for detecting context-window-thrashing patterns."""
    
    name = "context_thrashing"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.similarity_analyzer = FunctionSimilarityAnalyzer()
        self.playbook_generator = ExtractHelperPlaybookGenerator()
        self._clusters: List[ContextThrashingCluster] = []
        
        # Configuration - use the config passed in or fall back to empty dict
        cfg = config or {}
        context_cfg = cfg.get('context_thrashing', {})
        
        self.similarity_threshold = context_cfg.get('similarity_threshold', 0.7)
        self.min_line_distance = context_cfg.get('min_line_distance', 50)
        self.min_function_lines = context_cfg.get('min_function_lines', 5)
        self.max_members_default = context_cfg.get('max_members_default', 5)
        self.enable_clustering = context_cfg.get('enable_clustering', True)
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Run context-window-thrashing analysis."""
        issues = []
        self._clusters.clear()
        
        for file_path, tree in ctx.ast_index.items():
            file_issues = self._analyze_file(file_path, tree, ctx)
            issues.extend(file_issues)
        
        return issues
    
    def _analyze_file(self, file_path: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Analyze a single file for context-window-thrashing."""
        # Extract all functions from the file
        functions = self._extract_functions(file_path, tree, ctx)
        
        # Filter by minimum size
        functions = [f for f in functions if (f.line_end - f.line_start) >= self.min_function_lines]
        
        if len(functions) < 2:
            return []
        
        # Find similar function pairs
        similar_pairs = self.similarity_analyzer.find_similar_pairs(
            functions,
            similarity_threshold=self.similarity_threshold,
            min_line_distance=self.min_line_distance
        )
        
        if not similar_pairs:
            return []
        
        # Group into clusters
        clusters = self._group_into_clusters(similar_pairs, file_path)
        self._clusters.extend(clusters)
        
        # Create issues
        issues = []
        for cluster in clusters:
            issue = self._create_cluster_issue(cluster)
            issues.append(issue)
        
        return issues
    
    def _extract_functions(self, file_path: str, tree: ast.AST, 
                          ctx: AnalysisContext) -> List[FunctionInfo]:
        """Extract all functions from an AST."""
        functions = []
        
        # Get source lines for body extraction
        source_lines = ctx.source_cache.get(file_path, [])
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Determine if it's a method (inside a class)
                class_name = None
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        for child in ast.walk(parent):
                            if child is node:
                                class_name = parent.name
                                break
                        if class_name:
                            break
                
                func_info = self.similarity_analyzer.extract_function_info(
                    file_path, node, class_name, source_lines
                )
                functions.append(func_info)
        
        return functions
    
    def _group_into_clusters(self, pairs: List[SimilarFunctionPair], 
                           file_path: str) -> List[ContextThrashingCluster]:
        """Group similar pairs into clusters."""
        if not self.enable_clustering:
            # Create one cluster per pair
            clusters = []
            for i, pair in enumerate(pairs):
                cluster_id = f"CWT_{i+1:03d}"
                cluster = ContextThrashingCluster(
                    cluster_id=cluster_id,
                    functions=[pair.func1, pair.func2],
                    primary_pair=pair,
                    file_path=file_path,
                    issue_count=1,
                    severity=self._calculate_severity(pair),
                    suggested_helper=pair.suggested_helper_name,
                    extract_playbook=self.playbook_generator.generate_extract_playbook(pair)
                )
                clusters.append(cluster)
            return clusters
        
        # Group pairs that share functions
        cluster_groups = []
        used_pairs = set()
        
        for pair in pairs:
            if pair in used_pairs:
                continue
            
            # Start a new cluster
            cluster_functions = {pair.func1, pair.func2}
            cluster_pairs = [pair]
            used_pairs.add(pair)
            
            # Find related pairs
            for other_pair in pairs:
                if other_pair in used_pairs:
                    continue
                
                # Check if this pair shares functions with the cluster
                other_functions = {other_pair.func1, other_pair.func2}
                if cluster_functions.intersection(other_functions):
                    cluster_functions.update(other_functions)
                    cluster_pairs.append(other_pair)
                    used_pairs.add(other_pair)
            
            cluster_groups.append((list(cluster_functions), cluster_pairs))
        
        # Create cluster objects
        clusters = []
        for i, (functions, cluster_pairs) in enumerate(cluster_groups):
            cluster_id = f"CWT_{i+1:03d}"
            
            # Use the highest similarity pair as primary
            primary_pair = max(cluster_pairs, key=lambda p: p.similarity)
            
            # Calculate average severity
            avg_severity = sum(self._calculate_severity(p) for p in cluster_pairs) / len(cluster_pairs)
            
            cluster = ContextThrashingCluster(
                cluster_id=cluster_id,
                functions=functions,
                primary_pair=primary_pair,
                file_path=file_path,
                issue_count=len(cluster_pairs),
                severity=avg_severity,
                suggested_helper=primary_pair.suggested_helper_name,
                extract_playbook=self.playbook_generator.generate_extract_playbook(primary_pair)
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_severity(self, pair: SimilarFunctionPair) -> float:
        """Calculate severity score for a similar function pair."""
        base_score = 2.0  # Medium severity
        
        # Increase severity based on similarity
        similarity_bonus = (pair.similarity - self.similarity_threshold) * 2
        
        # Increase severity based on line distance (more thrashing)
        distance_bonus = min(pair.line_distance / 100, 1.0)
        
        # Increase severity for common function names that often cause thrashing
        name_bonus = 0
        if pair.func1.name in ['__init__', 'process', 'handle', 'setup', 'configure']:
            name_bonus = 0.5
        
        return min(base_score + similarity_bonus + distance_bonus + name_bonus, 5.0)
    
    def _create_cluster_issue(self, cluster: ContextThrashingCluster) -> Issue:
        """Create an issue for a context-thrashing cluster."""
        primary = cluster.primary_pair
        
        if cluster.issue_count == 1:
            message = f"Context-window thrashing: {primary.func1.name}() at lines {primary.func1.line_start} and {primary.func2.line_start} ({primary.similarity:.1%} similar, {primary.line_distance} lines apart)"
        else:
            message = f"Context-window thrashing cluster: {cluster.issue_count} similar functions (primary: {primary.func1.name})"
        
        evidence = {
            'cluster_id': cluster.cluster_id,
            'similarity': primary.similarity,
            'line_distance': primary.line_distance,
            'suggested_helper': cluster.suggested_helper,
            'function_count': len(cluster.functions),
            'issue_count': cluster.issue_count,
            'extract_playbook': cluster.extract_playbook,
            'functions': [
                {
                    'name': f.name,
                    'class_name': f.class_name,
                    'line_start': f.line_start,
                    'line_end': f.line_end,
                    'signature': f.signature
                }
                for f in cluster.functions
            ],
            'mini_diff': self.playbook_generator._generate_mini_diff(primary)
        }
        
        return Issue(
            kind="context_window_thrashing",
            message=message,
            severity=int(cluster.severity),
            file=cluster.file_path,
            line=primary.func1.line_start,
            evidence=evidence
        )
    
    def get_clusters(self, flat_view: bool = False, max_members: Optional[int] = None) -> List[ContextThrashingCluster]:
        """Get clusters with optional formatting controls."""
        clusters = self._clusters.copy()
        
        if max_members is None:
            max_members = self.max_members_default
        
        # Apply member truncation
        for cluster in clusters:
            if len(cluster.functions) > max_members:
                cluster.members_truncated = True
                cluster.hidden_members = len(cluster.functions) - max_members
                cluster.functions = cluster.functions[:max_members]
        
        if flat_view:
            # Flatten clusters to individual function entries
            flat_clusters = []
            for cluster in clusters:
                for func in cluster.functions:
                    flat_cluster = ContextThrashingCluster(
                        cluster_id=f"{cluster.cluster_id}_{func.name}_{func.line_start}",
                        functions=[func],
                        primary_pair=cluster.primary_pair,
                        file_path=cluster.file_path,
                        issue_count=1,
                        severity=cluster.severity,
                        suggested_helper=cluster.suggested_helper,
                        extract_playbook=cluster.extract_playbook
                    )
                    flat_clusters.append(flat_cluster)
            return flat_clusters
        
        return clusters
    
    def expand_cluster(self, cluster_id: str) -> Optional[ContextThrashingCluster]:
        """Expand a specific cluster to show all members."""
        for cluster in self._clusters:
            if cluster.cluster_id == cluster_id:
                # Return full cluster without truncation
                full_cluster = ContextThrashingCluster(
                    cluster_id=cluster.cluster_id,
                    functions=cluster.functions,
                    primary_pair=cluster.primary_pair,
                    file_path=cluster.file_path,
                    issue_count=cluster.issue_count,
                    severity=cluster.severity,
                    suggested_helper=cluster.suggested_helper,
                    extract_playbook=cluster.extract_playbook,
                    members_truncated=False,
                    hidden_members=0
                )
                return full_cluster
        
        return None
    
    def generate_cluster_report(self, flat_view: bool = False, 
                               max_members: Optional[int] = None) -> str:
        """Generate a formatted report of context-thrashing clusters."""
        clusters = self.get_clusters(flat_view, max_members)
        
        if not clusters:
            return "No context-window thrashing detected."
        
        report_lines = [
            "CONTEXT-WINDOW THRASHING REPORT",
            "=" * 50,
            f"View: {'Flat' if flat_view else 'Clustered'}",
            f"Max members: {max_members or 'No limit'}",
            f"Total clusters: {len(clusters)}",
            ""
        ]
        
        for cluster in clusters:
            report_lines.extend([
                f"Cluster {cluster.cluster_id}: {len(cluster.functions)} functions",
                f"  File: {cluster.file_path}",
                f"  Severity: {cluster.severity:.1f}",
                f"  Suggested helper: {cluster.suggested_helper}",
                ""
            ])
            
            for func in cluster.functions:
                class_prefix = f"{func.class_name}." if func.class_name else ""
                report_lines.append(f"    {class_prefix}{func.name}{func.signature} (lines {func.line_start}-{func.line_end})")
            
            if cluster.members_truncated:
                report_lines.append(f"    ... and {cluster.hidden_members} more (use --expand {cluster.cluster_id})")
            
            report_lines.append("")
        
        return '\n'.join(report_lines)