"""
Root-cause clustering for tail-chasing issues using AST-level semantic analysis.

Groups findings by semantic similarity rather than line-by-line duplicates, enabling
identification of systemic patterns and root causes across the codebase.
"""

import ast
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.spatial.distance')
    from scipy.spatial.distance import jaccard
    from scipy.cluster.hierarchy import fclusterdata

from ..core.issues import Issue


@dataclass
class IssueCluster:
    """Represents a cluster of semantically related issues."""
    cluster_id: str
    representative_issue: Issue
    members: List[Issue]
    root_cause_guess: str
    fix_playbook_id: str
    confidence: float
    ast_signature: str
    locations: List[Tuple[str, int]]  # (file, line) pairs
    
    @property
    def size(self) -> int:
        return len(self.members)
    
    @property
    def severity(self) -> int:
        """Aggregate severity based on member issues."""
        return max(issue.severity for issue in self.members)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary for reporting."""
        return {
            'cluster_id': self.cluster_id,
            'size': self.size,
            'severity': self.severity,
            'root_cause': self.root_cause_guess,
            'fix_playbook': self.fix_playbook_id,
            'confidence': self.confidence,
            'locations': self.locations,
            'representative': {
                'file': self.representative_issue.file,
                'line': self.representative_issue.line,
                'kind': self.representative_issue.kind,
                'message': self.representative_issue.message
            }
        }


class ASTNormalizer(ast.NodeTransformer):
    """Normalizes AST by stripping literals and standardizing identifiers."""
    
    def __init__(self, preserve_structure: bool = True):
        self.preserve_structure = preserve_structure
        self.var_counter = 0
        self.var_mapping = {}
        
    def visit_Name(self, node):
        """Normalize variable names to generic placeholders."""
        if node.id not in self.var_mapping:
            self.var_mapping[node.id] = f"VAR_{self.var_counter}"
            self.var_counter += 1
        node.id = self.var_mapping[node.id]
        return node
    
    def visit_Constant(self, node):
        """Replace constants with type placeholders."""
        # Check bool before int/float since bool is subclass of int
        if isinstance(node.value, bool):
            node.value = "BOOL_LITERAL"
        elif isinstance(node.value, str):
            node.value = "STR_LITERAL"
        elif isinstance(node.value, (int, float)):
            node.value = "NUM_LITERAL"
        else:
            node.value = "LITERAL"
        return node
    
    def visit_Str(self, node):  # For Python < 3.8 compatibility
        """Replace string literals."""
        node.s = "STR_LITERAL"
        return node
    
    def visit_Num(self, node):  # For Python < 3.8 compatibility
        """Replace numeric literals."""
        node.n = "NUM_LITERAL"
        return node
    
    def visit_FunctionDef(self, node):
        """Normalize function definitions."""
        if self.preserve_structure:
            # Keep function structure but normalize name
            node.name = "FUNC"
        self.generic_visit(node)
        return node
    
    def visit_ClassDef(self, node):
        """Normalize class definitions."""
        if self.preserve_structure:
            node.name = "CLASS"
        self.generic_visit(node)
        return node


class ASTHasher:
    """Provides various hashing strategies for AST comparison."""
    
    @staticmethod
    def ast_to_sexpr(node: ast.AST) -> str:
        """Convert AST to S-expression for hashing."""
        if isinstance(node, ast.AST):
            constructor = node.__class__.__name__
            args = []
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    args.extend(ASTHasher.ast_to_sexpr(item) for item in value)
                elif isinstance(value, ast.AST):
                    args.append(ASTHasher.ast_to_sexpr(value))
                elif value is not None:
                    args.append(repr(value))
            return f"({constructor} {' '.join(args)})" if args else f"({constructor})"
        return repr(node)
    
    @staticmethod
    def exact_hash(ast_node: ast.AST) -> str:
        """Generate exact hash using SHA256 for AST."""
        normalizer = ASTNormalizer(preserve_structure=True)
        normalized = normalizer.visit(ast_node)
        sexpr = ASTHasher.ast_to_sexpr(normalized)
        return hashlib.sha256(sexpr.encode()).hexdigest()
    
    @staticmethod
    def minhash(ast_node: ast.AST, num_perm: int = 128) -> np.ndarray:
        """Generate MinHash signature for approximate matching."""
        normalizer = ASTNormalizer(preserve_structure=True)
        normalized = normalizer.visit(ast_node)
        sexpr = ASTHasher.ast_to_sexpr(normalized)
        
        # Generate shingles (n-grams) from S-expression
        shingles = ASTHasher._generate_shingles(sexpr, n=3)
        
        # MinHash implementation
        min_hashes = []
        for i in range(num_perm):
            min_hash = float('inf')
            for shingle in shingles:
                hash_val = hash(shingle + str(i))
                if hash_val < min_hash:
                    min_hash = hash_val
            min_hashes.append(min_hash)
        
        return np.array(min_hashes)
    
    @staticmethod
    def _generate_shingles(text: str, n: int = 3) -> Set[str]:
        """Generate n-gram shingles from text."""
        shingles = set()
        for i in range(len(text) - n + 1):
            shingles.add(text[i:i+n])
        return shingles
    
    @staticmethod
    def structural_hash(ast_node: ast.AST) -> str:
        """Generate hash based only on AST structure, ignoring all values."""
        import copy
        # Deep copy to avoid modifying original
        node_copy = copy.deepcopy(ast_node)
        
        # Normalize to structure only
        normalizer = ASTNormalizer(preserve_structure=True)
        normalized = normalizer.visit(node_copy)
        
        # Generate hash from normalized structure
        sexpr = ASTHasher.ast_to_sexpr(normalized)
        return hashlib.md5(sexpr.encode()).hexdigest()


class FixPlaybook:
    """Manages fix strategies for different root cause patterns."""
    
    PLAYBOOKS = {
        'duplicate_implementation': {
            'id': 'DEDUP_001',
            'name': 'Deduplicate Implementation',
            'description': 'Consolidate duplicate implementations into single canonical version',
            'steps': [
                'Identify canonical implementation',
                'Update all references to use canonical version',
                'Remove duplicate implementations',
                'Add deprecation warnings if needed'
            ],
            'risk': 'MEDIUM'
        },
        'shadow_module': {
            'id': 'SHADOW_001',
            'name': 'Remove Shadow Module',
            'description': 'Experimental or temporary module shadows canonical implementation',
            'steps': [
                'Identify canonical module',
                'Migrate unique features to canonical module',
                'Update imports throughout codebase',
                'Remove shadow module'
            ],
            'risk': 'HIGH'
        },
        'circular_pattern': {
            'id': 'CIRC_001',
            'name': 'Break Circular Dependency',
            'description': 'Refactor to eliminate circular import patterns',
            'steps': [
                'Map dependency cycle',
                'Identify shared abstractions',
                'Extract common interface',
                'Refactor imports to use interface'
            ],
            'risk': 'HIGH'
        },
        'missing_abstraction': {
            'id': 'ABST_001',
            'name': 'Extract Common Abstraction',
            'description': 'Multiple similar implementations indicate missing abstraction',
            'steps': [
                'Identify common patterns',
                'Design abstract base or protocol',
                'Implement base abstraction',
                'Refactor implementations to use abstraction'
            ],
            'risk': 'MEDIUM'
        },
        'boilerplate_proliferation': {
            'id': 'BOIL_001',
            'name': 'Reduce Boilerplate',
            'description': 'Repeated boilerplate code across modules',
            'steps': [
                'Identify boilerplate patterns',
                'Create utilities or decorators',
                'Replace boilerplate with utilities',
                'Document utility usage'
            ],
            'risk': 'LOW'
        },
        'canonical_policy_violation': {
            'id': 'CANON_001',
            'name': 'Enforce Canonical Policy',
            'description': 'Shadow implementations violating canonical module policy',
            'steps': [
                'Identify canonical vs shadow implementations',
                'Generate import forwarders for shadows',
                'Apply codemod to replace shadows',
                'Add deprecation warnings',
                'Update documentation to reference canonical paths'
            ],
            'risk': 'LOW'
        },
        'crypto_scc_break': {
            'id': 'SCC_001',
            'name': 'Break Circular Import SCC',
            'description': 'Break strongly connected component in crypto/zk modules',
            'steps': [
                'Create shared abstraction module (*_shared.py)',
                'Move common interfaces to shared module',
                'Localize imports to break cycles',
                'Update all references to use shared interfaces'
            ],
            'risk': 'HIGH'
        },
        'stub_implementation': {
            'id': 'STUB_001',
            'name': 'Implement Critical Stubs',
            'description': 'Implement missing HSM/PQC/STARK stubs with proper guards',
            'steps': [
                'Implement stub with proper interface',
                'Add comprehensive test coverage',
                'Add runtime availability checks',
                'Document implementation status and requirements'
            ],
            'risk': 'CRITICAL'
        }
    }
    
    @classmethod
    def get_playbook(cls, playbook_id: str) -> Dict[str, Any]:
        """Retrieve a specific playbook by ID."""
        for playbook in cls.PLAYBOOKS.values():
            if playbook['id'] == playbook_id:
                return playbook
        return None
    
    @classmethod
    def suggest_playbook(cls, cluster: 'IssueCluster') -> str:
        """Suggest appropriate playbook based on cluster characteristics."""
        # Analyze cluster to determine best playbook
        issue_kinds = {issue.kind for issue in cluster.members}
        root_cause = cluster.root_cause_guess.lower()
        
        # Check for canonical policy violations
        if 'shadow_implementation' in issue_kinds or 'canonical policy violation' in root_cause:
            return 'canonical_policy_violation'
        
        # Check for specific genomevault patterns
        if 'crypto' in root_cause and 'scc' in root_cause:
            return 'crypto_scc_break'
        
        if 'hsm' in root_cause or 'pqc' in root_cause or 'stark' in root_cause:
            return 'stub_implementation'
        
        if 'duplicate_function' in issue_kinds or 'semantic_duplicate' in issue_kinds:
            if 'experimental' in root_cause or 'shadow' in root_cause:
                return 'shadow_module'
            elif 'it_pir' in root_cause:
                return 'canonical_policy_violation'  # PIR shadows should use canonical policy
            return 'duplicate_implementation'
        
        if 'circular_import' in issue_kinds:
            if 'crypto' in root_cause or 'zk_proofs' in root_cause:
                return 'crypto_scc_break'
            return 'circular_pattern'
        
        if 'phantom_function' in issue_kinds or 'missing_symbol' in issue_kinds:
            if 'cryptographic stubs' in root_cause:
                return 'stub_implementation'
            return 'missing_abstraction'
        
        if len(cluster.members) > 10:  # Many similar issues
            return 'boilerplate_proliferation'
        
        return 'duplicate_implementation'  # Default


class RootCauseClusterer:
    """Clusters issues by root cause using AST-level semantic analysis."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 min_cluster_size: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.hasher = ASTHasher()
        self.normalizer = ASTNormalizer()
        
    def cluster(self, issues: List[Issue]) -> List[IssueCluster]:
        """
        Cluster issues by semantic similarity at AST level.
        
        Args:
            issues: List of issues to cluster
            
        Returns:
            List of issue clusters with root cause analysis
        """
        # Group issues by file and extract AST info
        issue_signatures = self._extract_signatures(issues)
        
        # Perform clustering
        exact_clusters = self._cluster_exact_matches(issue_signatures)
        fuzzy_clusters = self._cluster_fuzzy_matches(issue_signatures)
        
        # Merge and refine clusters
        final_clusters = self._merge_clusters(exact_clusters, fuzzy_clusters)
        
        # Analyze root causes and assign playbooks
        analyzed_clusters = self._analyze_root_causes(final_clusters)
        
        return analyzed_clusters
    
    def _extract_signatures(self, issues: List[Issue]) -> Dict[Issue, Dict[str, Any]]:
        """Extract AST signatures for each issue."""
        signatures = {}
        
        for issue in issues:
            if not issue.file:
                continue
                
            try:
                # Get AST context for the issue
                ast_context = self._get_ast_context(issue)
                if ast_context:
                    exact_hash = self.hasher.exact_hash(ast_context)
                    minhash = self.hasher.minhash(ast_context)
                    structural = self.hasher.structural_hash(ast_context)
                    
                    signatures[issue] = {
                        'ast': ast_context,
                        'exact_hash': exact_hash,
                        'minhash': minhash,
                        'structural_hash': structural,
                        'sexpr': self.hasher.ast_to_sexpr(ast_context)
                    }
            except Exception as e:
                # Skip issues we can't parse
                continue
                
        return signatures
    
    def _get_ast_context(self, issue: Issue) -> Optional[ast.AST]:
        """Extract AST context for an issue."""
        if not issue.file or not Path(issue.file).exists():
            return None
            
        try:
            with open(issue.file, 'r') as f:
                source = f.read()
            tree = ast.parse(source)
            
            # Find the node at the issue location
            for node in ast.walk(tree):
                if hasattr(node, 'lineno') and node.lineno == issue.line:
                    return node
                    
            # If exact line not found, return enclosing function/class
            return self._find_enclosing_context(tree, issue.line)
            
        except Exception:
            return None
    
    def _find_enclosing_context(self, tree: ast.AST, line: int) -> Optional[ast.AST]:
        """Find the enclosing function or class for a given line."""
        best_node = None
        best_distance = float('inf')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if hasattr(node, 'lineno'):
                    distance = abs(node.lineno - line)
                    if distance < best_distance:
                        best_distance = distance
                        best_node = node
                        
        return best_node
    
    def _cluster_exact_matches(self, signatures: Dict[Issue, Dict]) -> List[List[Issue]]:
        """Cluster issues with exact AST matches."""
        hash_groups = defaultdict(list)
        
        for issue, sig in signatures.items():
            hash_groups[sig['exact_hash']].append(issue)
            
        return [group for group in hash_groups.values() if len(group) >= self.min_cluster_size]
    
    def _cluster_fuzzy_matches(self, signatures: Dict[Issue, Dict]) -> List[List[Issue]]:
        """Cluster issues using MinHash for near-duplicates."""
        if len(signatures) < 2:
            return []
            
        issues = list(signatures.keys())
        minhashes = np.array([sig['minhash'] for sig in signatures.values()])
        
        # Compute pairwise Jaccard distances
        n = len(issues)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Estimate Jaccard similarity from MinHash
                similarity = np.mean(minhashes[i] == minhashes[j])
                distances[i, j] = distances[j, i] = 1 - similarity
        
        # Hierarchical clustering
        if n > 1:
            clusters = fclusterdata(distances, 1 - self.similarity_threshold, 
                                  criterion='distance', metric='precomputed')
            
            # Group issues by cluster ID
            cluster_groups = defaultdict(list)
            for issue, cluster_id in zip(issues, clusters):
                cluster_groups[cluster_id].append(issue)
                
            return [group for group in cluster_groups.values() 
                   if len(group) >= self.min_cluster_size]
        
        return []
    
    def _merge_clusters(self, exact: List[List[Issue]], fuzzy: List[List[Issue]]) -> List[List[Issue]]:
        """Merge exact and fuzzy clusters, removing duplicates."""
        all_clusters = exact.copy()
        
        for fuzzy_cluster in fuzzy:
            # Check if this cluster overlaps with existing ones
            is_new = True
            for i, existing in enumerate(all_clusters):
                overlap = set(fuzzy_cluster) & set(existing)
                if len(overlap) > len(fuzzy_cluster) * 0.5:  # >50% overlap
                    # Merge into existing cluster
                    all_clusters[i] = list(set(existing) | set(fuzzy_cluster))
                    is_new = False
                    break
                    
            if is_new:
                all_clusters.append(fuzzy_cluster)
                
        return all_clusters
    
    def _analyze_root_causes(self, clusters: List[List[Issue]]) -> List[IssueCluster]:
        """Analyze clusters to identify root causes and assign fix strategies."""
        analyzed = []
        
        for i, cluster_issues in enumerate(clusters):
            # Select representative issue (most severe or central)
            representative = max(cluster_issues, key=lambda x: x.severity)
            
            # Analyze patterns to guess root cause
            root_cause = self._guess_root_cause(cluster_issues)
            
            # Get appropriate fix playbook
            playbook_type = FixPlaybook.suggest_playbook(None)  # Temporary
            playbook = FixPlaybook.PLAYBOOKS[playbook_type]
            
            # Extract locations
            locations = [(issue.file, issue.line) for issue in cluster_issues 
                        if issue.file]
            
            # Calculate confidence based on cluster coherence
            confidence = self._calculate_confidence(cluster_issues)
            
            # Create cluster object
            cluster = IssueCluster(
                cluster_id=f"RC_{i:04d}",
                representative_issue=representative,
                members=cluster_issues,
                root_cause_guess=root_cause,
                fix_playbook_id=playbook['id'],
                confidence=confidence,
                ast_signature=hashlib.md5(
                    f"{representative.file}:{representative.line}".encode()
                ).hexdigest()[:8],
                locations=locations
            )
            
            # Update playbook suggestion with actual cluster
            cluster.fix_playbook_id = FixPlaybook.PLAYBOOKS[
                FixPlaybook.suggest_playbook(cluster)
            ]['id']
            
            analyzed.append(cluster)
            
        return analyzed
    
    def _guess_root_cause(self, issues: List[Issue]) -> str:
        """Analyze issue patterns to guess root cause."""
        # Collect evidence
        issue_kinds = [issue.kind for issue in issues]
        files = [issue.file for issue in issues if issue.file]
        
        # Check for canonical policy issues
        if any('shadow_implementation' in kind for kind in issue_kinds):
            return "Canonical policy violation: shadow implementations detected"
            
        # Common patterns
        if len(set(issue_kinds)) == 1:
            kind = issue_kinds[0]
            if kind == 'duplicate_function':
                if any('test' in f for f in files):
                    return "Test utilities duplicated across test files"
                elif any('experimental' in f or 'v2' in f for f in files):
                    return "Experimental shadow of canonical module"
                elif any('it_pir' in f for f in files):
                    return "PIR implementation shadows (genomevault hotspot)"
                else:
                    return "Missing shared utility module"
                    
            elif kind == 'circular_import':
                if any('crypto' in f and '__init__' in f for f in files):
                    return "Crypto module circular imports (SCC hotspot)"
                elif any('zk_proofs' in f and 'core' in f for f in files):
                    return "ZK proofs core circular imports (SCC hotspot)"
                else:
                    return "Tight coupling between modules requiring refactoring"
                    
            elif kind == 'phantom_function':
                if any('hsm' in f or 'pqc' in f or 'stark' in f for f in files):
                    return "Missing critical cryptographic stubs (HSM/PQC/STARK)"
                else:
                    return "Incomplete refactoring leaving stub implementations"
                    
        # Mixed issue types in same cluster
        if 'duplicate_function' in issue_kinds and 'semantic_duplicate' in issue_kinds:
            return "Progressive code evolution without cleanup"
            
        # File pattern analysis
        if len(set(files)) == 1:
            return f"Localized code quality issues in {files[0]}"
            
        # Default with domain-specific context
        if any('genomevault' in f for f in files):
            return f"GenomeVault systematic {', '.join(set(issue_kinds))} pattern"
        else:
            return f"Systematic {', '.join(set(issue_kinds))} pattern across {len(set(files))} files"
    
    def _calculate_confidence(self, issues: List[Issue]) -> float:
        """Calculate confidence score for cluster quality."""
        # Factors that increase confidence:
        # - Same issue type
        # - Similar severity
        # - Co-located in same module
        
        base_confidence = 0.5
        
        # Issue type consistency
        issue_types = {issue.kind for issue in issues}
        if len(issue_types) == 1:
            base_confidence += 0.2
            
        # Severity consistency
        severities = {issue.severity for issue in issues}
        if len(severities) == 1:
            base_confidence += 0.1
            
        # File locality
        files = {issue.file for issue in issues if issue.file}
        if files:
            # Check if files are in same directory
            dirs = {str(Path(f).parent) for f in files}
            if len(dirs) == 1:
                base_confidence += 0.2
            elif len(dirs) <= 2:
                base_confidence += 0.1
                
        return min(base_confidence, 1.0)
    
    def generate_report(self, clusters: List[IssueCluster]) -> str:
        """Generate a report of clustered root causes."""
        lines = []
        lines.append("=" * 60)
        lines.append("ROOT CAUSE ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Clusters: {len(clusters)}")
        lines.append(f"Total Issues: {sum(c.size for c in clusters)}")
        lines.append("")
        
        # Sort clusters by size and severity
        sorted_clusters = sorted(clusters, 
                               key=lambda c: (c.severity, c.size), 
                               reverse=True)
        
        for cluster in sorted_clusters[:10]:  # Top 10 clusters
            lines.append(f"Cluster {cluster.cluster_id}")
            lines.append("-" * 40)
            lines.append(f"Size: {cluster.size} issues")
            lines.append(f"Severity: {cluster.severity}")
            lines.append(f"Confidence: {cluster.confidence:.1%}")
            lines.append(f"Root Cause: {cluster.root_cause_guess}")
            
            playbook = FixPlaybook.get_playbook(cluster.fix_playbook_id)
            if playbook:
                lines.append(f"Fix Strategy: {playbook['name']} ({playbook['risk']} risk)")
                
            lines.append(f"Locations:")
            for file, line in cluster.locations[:5]:  # Show first 5
                lines.append(f"  - {file}:{line}")
            if len(cluster.locations) > 5:
                lines.append(f"  ... and {len(cluster.locations) - 5} more")
                
            lines.append("")
            
        return "\n".join(lines)