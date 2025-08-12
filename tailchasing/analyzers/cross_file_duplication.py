"""
Detect semantic duplication across different files and modules.
Extends beyond single-file analysis to find project-wide redundancy.
"""

import ast
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import os
from ..core.issues import Issue
from ..core.utils import safe_get_lineno
from .base import AnalysisContext
from ..semantic.encoder import encode_function
from ..semantic.similarity import compute_similarity


class CrossFileDuplicationAnalyzer:
    """Detect semantic duplication across different files in the project."""
    
    name = "cross_file_duplication"
    
    def __init__(self):
        self.function_index = {}  # Maps function signatures to locations
        self.semantic_index = {}  # Maps semantic vectors to functions
        self.module_boundaries = defaultdict(set)  # Track module relationships
        
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Analyze entire project for cross-file semantic duplication."""
        issues = []
        
        # Skip if semantic analysis is disabled
        if not ctx.config.get('semantic', {}).get('enable', False):
            return issues
        
        # Build comprehensive function index
        self._build_function_index(ctx)
        
        # Find cross-file duplications
        duplications = self._find_cross_file_duplications(ctx)
        
        # Generate issues for significant duplications
        for dup_group in duplications:
            issue = self._create_duplication_issue(dup_group, ctx)
            if issue:
                issues.append(issue)
        
        # Detect module-level duplication patterns
        module_issues = self._analyze_module_patterns(ctx)
        issues.extend(module_issues)
        
        return issues
    
    def _build_function_index(self, ctx: AnalysisContext):
        """Build comprehensive index of all functions in the project."""
        for file, tree in ctx.ast_index.items():
            module_name = self._get_module_name(file)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Create unique identifier
                    func_id = f"{module_name}.{node.name}"
                    
                    # Store function metadata
                    self.function_index[func_id] = {
                        'file': file,
                        'line': safe_get_lineno(node),
                        'name': node.name,
                        'module': module_name,
                        'ast': node,
                        'signature': self._extract_signature(node),
                        'docstring': ast.get_docstring(node)
                    }
                    
                    # Encode semantic vector if enabled
                    if ctx.config.get('semantic', {}).get('enable', False):
                        hv_space = ctx.cache.get('semantic_index', {}).get('space')
                        if hv_space:
                            vector = encode_function(node, file, hv_space, ctx.config.get('semantic', {}))
                            self.semantic_index[func_id] = vector
                    
                    # Track module relationships
                    imports = self._extract_imports(tree)
                    self.module_boundaries[module_name].update(imports)
    
    def _find_cross_file_duplications(self, ctx: AnalysisContext) -> List[List[Dict]]:
        """Find groups of semantically similar functions across files."""
        duplication_groups = []
        processed = set()
        
        # Compare all function pairs
        func_ids = list(self.function_index.keys())
        
        for i, func_id1 in enumerate(func_ids):
            if func_id1 in processed:
                continue
                
            func1 = self.function_index[func_id1]
            similar_group = [func1]
            processed.add(func_id1)
            
            for func_id2 in func_ids[i+1:]:
                if func_id2 in processed:
                    continue
                    
                func2 = self.function_index[func_id2]
                
                # Skip if in same file
                if func1['file'] == func2['file']:
                    continue
                
                # Check semantic similarity
                similarity = self._calculate_similarity(func_id1, func_id2, ctx)
                
                if similarity > 0.85:  # High similarity threshold
                    similar_group.append(func2)
                    processed.add(func_id2)
            
            if len(similar_group) > 1:
                duplication_groups.append(similar_group)
        
        return duplication_groups
    
    def _calculate_similarity(self, func_id1: str, func_id2: str, ctx: AnalysisContext) -> float:
        """Calculate similarity between two functions."""
        func1 = self.function_index[func_id1]
        func2 = self.function_index[func_id2]
        
        # Start with name similarity
        name_sim = self._name_similarity(func1['name'], func2['name'])
        
        # Signature similarity
        sig_sim = self._signature_similarity(func1['signature'], func2['signature'])
        
        # Semantic similarity if available
        sem_sim = 0.0
        if func_id1 in self.semantic_index and func_id2 in self.semantic_index:
            vec1 = self.semantic_index[func_id1]
            vec2 = self.semantic_index[func_id2]
            # Use hypervector similarity
            if hasattr(ctx.cache.get('semantic_index', {}), 'space'):
                space = ctx.cache['semantic_index']['space']
                sem_sim = 1.0 - space.hamming(vec1, vec2)
        
        # AST structure similarity
        ast_sim = self._ast_similarity(func1['ast'], func2['ast'])
        
        # Weighted combination
        weights = {'name': 0.1, 'signature': 0.2, 'semantic': 0.5, 'ast': 0.2}
        
        if sem_sim > 0:  # If semantic similarity is available, use it
            return (weights['name'] * name_sim +
                   weights['signature'] * sig_sim +
                   weights['semantic'] * sem_sim +
                   weights['ast'] * ast_sim)
        else:  # Fall back to structural similarity
            return (0.2 * name_sim + 0.4 * sig_sim + 0.4 * ast_sim)
    
    def _create_duplication_issue(self, dup_group: List[Dict], ctx: AnalysisContext) -> Optional[Issue]:
        """Create an issue for a group of duplicate functions."""
        if len(dup_group) < 2:
            return None
        
        # Find the "primary" implementation (most complete/documented)
        primary = max(dup_group, key=lambda f: (
            len(f.get('docstring', '') or ''),
            len(ast.dump(f['ast']))
        ))
        
        # Build location list
        locations = [(f['file'], f['line']) for f in dup_group]
        modules = list(set(f['module'] for f in dup_group))
        
        # Check if this is likely an intentional pattern
        if self._is_intentional_duplication(dup_group):
            return None
        
        return Issue(
            kind="cross_file_duplication",
            message=f"Function '{primary['name']}' has {len(dup_group)-1} semantic duplicate(s) across {len(modules)} modules",
            severity=3,
            file=primary['file'],
            line=primary['line'],
            symbol=primary['name'],
            evidence={
                'locations': locations,
                'modules': modules,
                'primary_candidate': f"{primary['module']}.{primary['name']}"
            },
            suggestions=[
                f"Consider consolidating duplicates into a shared module",
                f"Primary implementation candidate: {primary['module']}.{primary['name']}",
                "Extract common functionality to a utility module",
                "If duplicates serve different purposes, ensure they're properly differentiated"
            ]
        )
    
    def _analyze_module_patterns(self, ctx: AnalysisContext) -> List[Issue]:
        """Detect module-level duplication patterns."""
        issues = []
        
        # Find modules with high internal duplication
        for module, funcs in self._group_by_module().items():
            if len(funcs) < 5:  # Skip small modules
                continue
            
            # Calculate intra-module duplication ratio
            dup_count = self._count_internal_duplicates(funcs, ctx)
            dup_ratio = dup_count / len(funcs)
            
            if dup_ratio > 0.3:  # More than 30% duplication
                issues.append(Issue(
                    kind="module_duplication_pattern",
                    message=f"Module '{module}' has high internal duplication ({dup_ratio:.1%})",
                    severity=2,
                    file=funcs[0]['file'],
                    line=1,
                    evidence={
                        'duplication_ratio': dup_ratio,
                        'function_count': len(funcs),
                        'duplicate_count': dup_count
                    },
                    suggestions=[
                        "Refactor module to reduce redundancy",
                        "Extract common patterns into base classes or utilities",
                        "Consider splitting into more focused modules"
                    ]
                ))
        
        # Find suspiciously similar modules
        module_pairs = self._find_similar_modules(ctx)
        for mod1, mod2, similarity in module_pairs:
            if similarity > 0.7:  # High module similarity
                issues.append(Issue(
                    kind="duplicate_modules",
                    message=f"Modules '{mod1}' and '{mod2}' are {similarity:.0%} similar",
                    severity=3,
                    evidence={
                        'modules': [mod1, mod2],
                        'similarity': similarity
                    },
                    suggestions=[
                        "Consider merging these modules",
                        "Extract shared functionality to a common module",
                        "Ensure modules have distinct responsibilities"
                    ]
                ))
        
        return issues
    
    def _get_module_name(self, filepath: str) -> str:
        """Extract module name from file path."""
        # Remove file extension and convert path to module notation
        module_path = os.path.splitext(filepath)[0]
        # Convert path separators to dots
        module_name = module_path.replace(os.sep, '.')
        # Remove common prefixes
        for prefix in ['src.', 'lib.', './']:
            if module_name.startswith(prefix):
                module_name = module_name[len(prefix):]
        return module_name
    
    def _extract_signature(self, func: ast.FunctionDef) -> Dict:
        """Extract function signature for comparison."""
        return {
            'args': [arg.arg for arg in func.args.args],
            'defaults': len(func.args.defaults),
            'kwonly': [arg.arg for arg in func.args.kwonlyargs],
            'returns': bool(func.returns)
        }
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from a module."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        return imports
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between function names."""
        if name1 == name2:
            return 1.0
        
        # Check for common patterns (get_X vs fetch_X, etc.)
        patterns = [
            (['get', 'fetch', 'retrieve'], 0.8),
            (['set', 'update', 'save'], 0.8),
            (['create', 'make', 'build'], 0.8),
            (['delete', 'remove', 'destroy'], 0.8),
            (['check', 'verify', 'validate'], 0.8)
        ]
        
        for synonyms, score in patterns:
            for syn in synonyms:
                if syn in name1 and any(s in name2 for s in synonyms):
                    return score
        
        # Levenshtein-style similarity
        return 1.0 - (self._edit_distance(name1, name2) / max(len(name1), len(name2)))
    
    def _signature_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between function signatures."""
        # Compare argument names
        args1 = set(sig1['args'])
        args2 = set(sig2['args'])
        
        if not args1 and not args2:
            return 1.0
        if not args1 or not args2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(args1 & args2)
        union = len(args1 | args2)
        
        return intersection / union if union > 0 else 0.0
    
    def _ast_similarity(self, ast1: ast.AST, ast2: ast.AST) -> float:
        """Calculate AST structure similarity."""
        # Simple approach: compare normalized AST dumps
        dump1 = ast.dump(ast1, annotate_fields=False)
        dump2 = ast.dump(ast2, annotate_fields=False)
        
        # Remove variable names and constants for structural comparison
        norm1 = self._normalize_ast_dump(dump1)
        norm2 = self._normalize_ast_dump(dump2)
        
        if norm1 == norm2:
            return 1.0
        
        # Character-level similarity
        common = sum(1 for c1, c2 in zip(norm1, norm2) if c1 == c2)
        total = max(len(norm1), len(norm2))
        
        return common / total if total > 0 else 0.0
    
    def _normalize_ast_dump(self, dump: str) -> str:
        """Normalize AST dump for comparison."""
        # Replace common patterns
        import re
        dump = re.sub(r"Name\(id='[^']+'\)", "Name(id='VAR')", dump)
        dump = re.sub(r"Constant\(value=[^)]+\)", "Constant(value=CONST)", dump)
        dump = re.sub(r"arg\(arg='[^']+'\)", "arg(arg='ARG')", dump)
        return dump
    
    def _is_intentional_duplication(self, dup_group: List[Dict]) -> bool:
        """Check if duplication might be intentional (e.g., interfaces, adapters)."""
        # Check for common intentional patterns
        names = [f['name'] for f in dup_group]
        
        # Interface implementations
        if all('interface' in f['module'].lower() or 'abstract' in f['module'].lower() 
               for f in dup_group):
            return True
        
        # Adapter pattern
        if all('adapter' in name.lower() or 'wrapper' in name.lower() for name in names):
            return True
        
        # Test utilities
        if all('test' in f['module'] or 'mock' in f['module'] for f in dup_group):
            return True
        
        return False
    
    def _group_by_module(self) -> Dict[str, List[Dict]]:
        """Group functions by module."""
        modules = defaultdict(list)
        for func_id, func_data in self.function_index.items():
            modules[func_data['module']].append(func_data)
        return modules
    
    def _count_internal_duplicates(self, funcs: List[Dict], ctx: AnalysisContext) -> int:
        """Count duplicate functions within a module."""
        dup_count = 0
        seen = set()
        
        for i, func1 in enumerate(funcs):
            if i in seen:
                continue
            for j, func2 in enumerate(funcs[i+1:], i+1):
                if j in seen:
                    continue
                
                func_id1 = f"{func1['module']}.{func1['name']}"
                func_id2 = f"{func2['module']}.{func2['name']}"
                
                similarity = self._calculate_similarity(func_id1, func_id2, ctx)
                if similarity > 0.85:
                    dup_count += 1
                    seen.add(j)
        
        return dup_count
    
    def _find_similar_modules(self, ctx: AnalysisContext) -> List[Tuple[str, str, float]]:
        """Find pairs of similar modules."""
        similar_pairs = []
        modules = list(self._group_by_module().keys())
        
        for i, mod1 in enumerate(modules):
            for mod2 in modules[i+1:]:
                similarity = self._calculate_module_similarity(mod1, mod2, ctx)
                if similarity > 0.5:  # Moderate threshold
                    similar_pairs.append((mod1, mod2, similarity))
        
        return similar_pairs
    
    def _calculate_module_similarity(self, mod1: str, mod2: str, ctx: AnalysisContext) -> float:
        """Calculate similarity between two modules."""
        funcs1 = [f for f in self.function_index.values() if f['module'] == mod1]
        funcs2 = [f for f in self.function_index.values() if f['module'] == mod2]
        
        if not funcs1 or not funcs2:
            return 0.0
        
        # Compare function names
        names1 = set(f['name'] for f in funcs1)
        names2 = set(f['name'] for f in funcs2)
        
        name_similarity = len(names1 & names2) / max(len(names1), len(names2))
        
        # Compare imports
        imports1 = self.module_boundaries.get(mod1, set())
        imports2 = self.module_boundaries.get(mod2, set())
        
        import_similarity = len(imports1 & imports2) / max(len(imports1 | imports2), 1)
        
        return 0.7 * name_similarity + 0.3 * import_similarity
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
