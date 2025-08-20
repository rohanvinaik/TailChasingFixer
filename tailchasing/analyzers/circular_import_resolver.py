"""
Circular import resolver with SCC analysis and automated fix generation.

This module identifies strongly connected components in the import graph and
generates targeted fixes to break circular dependencies, including shared
module extraction and function-scope import codemods.
"""

import ast
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import networkx as nx

from ..core.issues import Issue


@dataclass
class ImportDependency:
    """Represents an import relationship between modules."""
    source_module: str
    target_module: str
    import_type: str  # 'module', 'from', 'star'
    imported_names: List[str]
    line_number: int
    is_module_level: bool
    ast_node: ast.AST


@dataclass
class SharedSymbol:
    """Represents a symbol that could be extracted to a shared module."""
    name: str
    symbol_type: str  # 'class', 'function', 'constant', 'type_alias'
    used_in_modules: Set[str]
    definition_location: Optional[Tuple[str, int]] = None
    ast_node: Optional[ast.AST] = None
    usage_count: int = 0


@dataclass
class SCCAnalysis:
    """Analysis of a strongly connected component."""
    scc_id: str
    modules: List[str]
    import_edges: List[ImportDependency]
    shared_symbols: List[SharedSymbol]
    suggested_shared_module: str
    break_points: List[Tuple[str, str]]  # (source, target) edges to break
    complexity_score: float


@dataclass
class CircularImportFix:
    """Represents a fix for circular import issues."""
    fix_id: str
    fix_type: str  # 'extract_shared', 'function_scope', 'lazy_import'
    target_file: str
    description: str
    codemod: str
    risk_level: str
    dependencies: List[str]  # Other fixes this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'fix_id': self.fix_id,
            'fix_type': self.fix_type,
            'target_file': self.target_file,
            'description': self.description,
            'codemod': self.codemod,
            'risk_level': self.risk_level,
            'dependencies': self.dependencies
        }


class ImportGraphBuilder:
    """Builds import dependency graph from AST analysis."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.import_details: Dict[Tuple[str, str], List[ImportDependency]] = defaultdict(list)
        self.module_symbols: Dict[str, Dict[str, ast.AST]] = defaultdict(dict)
        
    def build_graph(self, ast_index: Dict[str, ast.AST]) -> nx.DiGraph:
        """Build import graph from AST index."""
        for file_path, tree in ast_index.items():
            module_name = self._file_to_module(file_path)
            self.graph.add_node(module_name, file_path=file_path)
            
            # Extract imports and symbols
            imports = self._extract_imports(tree, module_name, file_path)
            symbols = self._extract_module_symbols(tree, module_name)
            
            self.module_symbols[module_name].update(symbols)
            
            # Add edges for imports
            for import_dep in imports:
                if import_dep.target_module != module_name:  # Avoid self-imports
                    edge_key = (import_dep.source_module, import_dep.target_module)
                    self.import_details[edge_key].append(import_dep)
                    
                    # Add edge with weight based on import complexity
                    weight = self._calculate_import_weight(import_dep)
                    if self.graph.has_edge(import_dep.source_module, import_dep.target_module):
                        # Accumulate weights for multiple imports
                        self.graph[import_dep.source_module][import_dep.target_module]['weight'] += weight
                    else:
                        self.graph.add_edge(
                            import_dep.source_module,
                            import_dep.target_module,
                            weight=weight
                        )
        
        return self.graph
    
    def _file_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        # Remove .py extension and convert path separators
        module = str(Path(file_path)).replace('/', '.').replace('\\', '.')
        if module.endswith('.py'):
            module = module[:-3]
        if module.endswith('.__init__'):
            module = module[:-9]
        return module
    
    def _extract_imports(self, tree: ast.AST, module_name: str, file_path: str) -> List[ImportDependency]:
        """Extract import dependencies from AST."""
        imports = []
        
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self):
                self.imports = []
                self.in_function = False
                self.function_depth = 0
            
            def visit_FunctionDef(self, node):
                self.function_depth += 1
                old_in_function = self.in_function
                self.in_function = True
                self.generic_visit(node)
                self.in_function = old_in_function
                self.function_depth -= 1
            
            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)
            
            def visit_ClassDef(self, node):
                # Treat class body as module-level for import purposes
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    target_module = alias.name.split('.')[0]  # Get root module
                    self.imports.append(ImportDependency(
                        source_module=module_name,
                        target_module=target_module,
                        import_type='module',
                        imported_names=[alias.name],
                        line_number=node.lineno,
                        is_module_level=not self.in_function,
                        ast_node=node
                    ))
            
            def visit_ImportFrom(self, node):
                if node.module:
                    target_module = node.module.split('.')[0]  # Get root module
                    imported_names = [alias.name for alias in node.names]
                    import_type = 'star' if any(name == '*' for name in imported_names) else 'from'
                    
                    self.imports.append(ImportDependency(
                        source_module=module_name,
                        target_module=target_module,
                        import_type=import_type,
                        imported_names=imported_names,
                        line_number=node.lineno,
                        is_module_level=not self.in_function,
                        ast_node=node
                    ))
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        return visitor.imports
    
    def _extract_module_symbols(self, tree: ast.AST, module_name: str) -> Dict[str, ast.AST]:
        """Extract module-level symbols."""
        symbols = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbols[node.name] = node
            elif isinstance(node, ast.ClassDef):
                symbols[node.name] = node
            elif isinstance(node, ast.Assign):
                # Look for module-level constants
                if self._is_module_level_assignment(node, tree):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            symbols[target.id] = node
        
        return symbols
    
    def _is_module_level_assignment(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if assignment is at module level."""
        # Simple heuristic: if it's a direct child of Module
        for child in ast.walk(tree):
            if isinstance(child, ast.Module) and node in child.body:
                return True
        return False
    
    def _calculate_import_weight(self, import_dep: ImportDependency) -> float:
        """Calculate weight for import edge based on complexity."""
        base_weight = 1.0
        
        # Module-level imports are heavier (harder to break)
        if import_dep.is_module_level:
            base_weight *= 2.0
        
        # Star imports are heaviest
        if import_dep.import_type == 'star':
            base_weight *= 3.0
        elif import_dep.import_type == 'from':
            # Weight based on number of imported names
            base_weight *= (1.0 + len(import_dep.imported_names) * 0.2)
        
        return base_weight


class SCCAnalyzer:
    """Analyzes strongly connected components in import graph."""
    
    def __init__(self, graph_builder: ImportGraphBuilder):
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        
    def find_and_analyze_sccs(self) -> List[SCCAnalysis]:
        """Find and analyze all SCCs in the import graph."""
        sccs = list(nx.strongly_connected_components(self.graph))
        analyses = []
        
        for i, scc in enumerate(sccs):
            if len(scc) > 1:  # Only analyze true cycles
                analysis = self._analyze_scc(scc, f"SCC_{i:03d}")
                analyses.append(analysis)
        
        # Sort by complexity (most complex first)
        analyses.sort(key=lambda a: a.complexity_score, reverse=True)
        return analyses
    
    def _analyze_scc(self, scc: Set[str], scc_id: str) -> SCCAnalysis:
        """Analyze a single SCC."""
        modules = list(scc)
        
        # Extract import edges within the SCC
        import_edges = []
        for source, target in self.graph.edges():
            if source in scc and target in scc:
                edges = self.graph_builder.import_details.get((source, target), [])
                import_edges.extend(edges)
        
        # Find shared symbols across the cycle
        shared_symbols = self._find_shared_symbols(modules)
        
        # Suggest shared module name
        suggested_shared = self._suggest_shared_module_name(modules)
        
        # Identify break points
        break_points = self._identify_break_points(scc, import_edges)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(scc, import_edges, shared_symbols)
        
        return SCCAnalysis(
            scc_id=scc_id,
            modules=modules,
            import_edges=import_edges,
            shared_symbols=shared_symbols,
            suggested_shared_module=suggested_shared,
            break_points=break_points,
            complexity_score=complexity_score
        )
    
    def _find_shared_symbols(self, modules: List[str]) -> List[SharedSymbol]:
        """Find symbols that are used across multiple modules in the SCC."""
        # Collect all symbols and their usage
        symbol_usage: Dict[str, Set[str]] = defaultdict(set)
        symbol_definitions: Dict[str, Tuple[str, ast.AST]] = {}
        
        # Analyze imports to see what symbols are used where
        for module in modules:
            for (source, target), import_deps in self.graph_builder.import_details.items():
                if source in modules and target in modules:
                    for import_dep in import_deps:
                        for name in import_dep.imported_names:
                            if name != '*':  # Skip star imports for now
                                symbol_usage[name].add(source)
                                if target in self.graph_builder.module_symbols:
                                    if name in self.graph_builder.module_symbols[target]:
                                        symbol_definitions[name] = (
                                            target,
                                            self.graph_builder.module_symbols[target][name]
                                        )
        
        # Create SharedSymbol objects for multi-module usage
        shared_symbols = []
        for symbol_name, using_modules in symbol_usage.items():
            if len(using_modules) > 1:  # Used in multiple modules
                definition_info = symbol_definitions.get(symbol_name)
                definition_location = None
                ast_node = None
                
                if definition_info:
                    def_module, def_node = definition_info
                    # Try to get line number from AST node
                    line_num = getattr(def_node, 'lineno', 0)
                    definition_location = (def_module, line_num)
                    ast_node = def_node
                
                # Determine symbol type
                symbol_type = self._infer_symbol_type(symbol_name, ast_node)
                
                shared_symbol = SharedSymbol(
                    name=symbol_name,
                    symbol_type=symbol_type,
                    used_in_modules=using_modules,
                    definition_location=definition_location,
                    ast_node=ast_node,
                    usage_count=len(using_modules)
                )
                shared_symbols.append(shared_symbol)
        
        # Sort by usage count (most used first)
        shared_symbols.sort(key=lambda s: s.usage_count, reverse=True)
        return shared_symbols
    
    def _infer_symbol_type(self, symbol_name: str, ast_node: Optional[ast.AST]) -> str:
        """Infer the type of a symbol from its name and AST node."""
        if ast_node:
            if isinstance(ast_node, ast.ClassDef):
                return 'class'
            elif isinstance(ast_node, ast.FunctionDef):
                return 'function'
            elif isinstance(ast_node, ast.Assign):
                return 'constant' if symbol_name.isupper() else 'variable'
        
        # Fallback to name-based heuristics
        if symbol_name.isupper():
            return 'constant'
        elif symbol_name[0].isupper():
            return 'class'
        else:
            return 'function'
    
    def _suggest_shared_module_name(self, modules: List[str]) -> str:
        """Suggest a name for the shared module."""
        # Find common prefix
        if len(modules) == 1:
            return f"{modules[0]}_shared"
        
        # Find longest common prefix
        common_parts = []
        module_parts = [mod.split('.') for mod in modules]
        
        if module_parts:
            min_len = min(len(parts) for parts in module_parts)
            for i in range(min_len):
                parts_at_i = [parts[i] for parts in module_parts]
                if all(part == parts_at_i[0] for part in parts_at_i):
                    common_parts.append(parts_at_i[0])
                else:
                    break
        
        if common_parts:
            return '.'.join(common_parts + ['shared'])
        else:
            # Fallback: use first module name + shared
            return f"{modules[0].split('.')[0]}_shared"
    
    def _identify_break_points(self, scc: Set[str], import_edges: List[ImportDependency]) -> List[Tuple[str, str]]:
        """Identify the best edges to break to resolve the cycle."""
        # Create subgraph for this SCC
        subgraph = self.graph.subgraph(scc)
        
        # Find edges that, if removed, would break the most cycles
        break_candidates = []
        
        for edge in subgraph.edges():
            # Create temporary graph without this edge
            temp_graph = subgraph.copy()
            temp_graph.remove_edge(*edge)
            
            # Check if removing this edge breaks the SCC
            temp_sccs = list(nx.strongly_connected_components(temp_graph))
            if len(temp_sccs) > 1 or all(len(scc) == 1 for scc in temp_sccs):
                # This edge removal helps break the cycle
                weight = subgraph[edge[0]][edge[1]].get('weight', 1.0)
                break_candidates.append((edge, weight))
        
        # Sort by weight (prefer breaking lighter edges)
        break_candidates.sort(key=lambda x: x[1])
        
        # Return the best break points
        return [edge for edge, _ in break_candidates[:3]]  # Top 3 candidates
    
    def _calculate_complexity_score(self, scc: Set[str], import_edges: List[ImportDependency], 
                                  shared_symbols: List[SharedSymbol]) -> float:
        """Calculate complexity score for prioritizing SCC fixes."""
        base_score = len(scc) * 10  # Base score from SCC size
        
        # Add complexity from import edges
        import_complexity = sum(
            3 if imp.import_type == 'star' else 
            2 if imp.is_module_level else 1
            for imp in import_edges
        )
        
        # Add complexity from shared symbols
        symbol_complexity = sum(s.usage_count for s in shared_symbols)
        
        return base_score + import_complexity + symbol_complexity


class CircularImportFixGenerator:
    """Generates fixes for circular import issues."""
    
    def __init__(self):
        self.fix_id_counter = 0
        
    def generate_fixes(self, scc_analyses: List[SCCAnalysis]) -> List[CircularImportFix]:
        """Generate fixes for all SCC analyses."""
        all_fixes = []
        
        for analysis in scc_analyses:
            fixes = self._generate_fixes_for_scc(analysis)
            all_fixes.extend(fixes)
        
        # Sort fixes topologically
        all_fixes = self._topological_sort_fixes(all_fixes)
        
        return all_fixes
    
    def _generate_fixes_for_scc(self, analysis: SCCAnalysis) -> List[CircularImportFix]:
        """Generate fixes for a single SCC."""
        fixes = []
        
        # 1. Extract shared module (if beneficial)
        if analysis.shared_symbols and len(analysis.shared_symbols) >= 2:
            shared_fix = self._generate_shared_module_fix(analysis)
            fixes.append(shared_fix)
        
        # 2. Function-scope import fixes
        for import_edge in analysis.import_edges:
            if import_edge.is_module_level and self._can_move_to_function_scope(import_edge):
                func_scope_fix = self._generate_function_scope_fix(import_edge)
                fixes.append(func_scope_fix)
        
        # 3. Lazy import fixes for remaining cases
        for source, target in analysis.break_points:
            lazy_fix = self._generate_lazy_import_fix(source, target, analysis)
            fixes.append(lazy_fix)
        
        return fixes
    
    def _generate_shared_module_fix(self, analysis: SCCAnalysis) -> CircularImportFix:
        """Generate fix to extract shared symbols to a separate module."""
        fix_id = f"extract_shared_{analysis.scc_id}"
        shared_module = analysis.suggested_shared_module
        
        # Generate the shared module content
        shared_content_parts = [
            f'"""Shared symbols extracted from circular import cycle: {", ".join(analysis.modules)}."""',
            "",
            "# This module was automatically generated to break circular imports.",
            "# Original symbols were extracted from the following modules:",
        ]
        
        for module in analysis.modules:
            shared_content_parts.append(f"#   - {module}")
        
        shared_content_parts.extend(["", ""])
        
        # Add shared symbols
        for symbol in analysis.shared_symbols:
            if symbol.definition_location:
                shared_content_parts.append(f"# Originally defined in {symbol.definition_location[0]}")
                if symbol.ast_node:
                    # Try to generate the symbol definition
                    symbol_code = self._ast_to_code(symbol.ast_node)
                    shared_content_parts.append(symbol_code)
                else:
                    shared_content_parts.append(f"# TODO: Move {symbol.name} definition here")
                shared_content_parts.append("")
        
        shared_content = "\n".join(shared_content_parts)
        
        # Generate import replacement codemods
        import_replacements = []
        for module in analysis.modules:
            replacements = [
                f"# Update imports in {module}:",
                f"# Replace cross-module imports with:",
                f"from {shared_module} import " + ", ".join(s.name for s in analysis.shared_symbols),
                ""
            ]
            import_replacements.extend(replacements)
        
        full_codemod = f"""# Step 1: Create shared module
# File: {shared_module.replace('.', '/')}.py
{shared_content}

# Step 2: Update imports in affected modules
{chr(10).join(import_replacements)}

# Step 3: Remove original definitions from their modules
# (Keep only one canonical definition in the shared module)
"""
        
        return CircularImportFix(
            fix_id=fix_id,
            fix_type='extract_shared',
            target_file=f"{shared_module.replace('.', '/')}.py",
            description=f"Extract {len(analysis.shared_symbols)} shared symbols to {shared_module}",
            codemod=full_codemod,
            risk_level='HIGH',
            dependencies=[]
        )
    
    def _generate_function_scope_fix(self, import_edge: ImportDependency) -> CircularImportFix:
        """Generate fix to move import to function scope."""
        self.fix_id_counter += 1
        fix_id = f"function_scope_{self.fix_id_counter:03d}"
        
        # Identify the module file that needs modification
        source_module = import_edge.source_module
        target_file = f"{source_module.replace('.', '/')}.py"
        
        # Generate codemod to move import
        if import_edge.import_type == 'from':
            import_stmt = f"from {import_edge.target_module} import {', '.join(import_edge.imported_names)}"
        else:
            import_stmt = f"import {import_edge.target_module}"
        
        codemod = f"""# Move module-level import to function scope
# File: {target_file}
# Line: {import_edge.line_number}

# REMOVE module-level import:
# {import_stmt}

# ADD to function(s) that use the import:
def your_function():
    {import_stmt}  # Move import here
    # ... rest of function code
"""
        
        return CircularImportFix(
            fix_id=fix_id,
            fix_type='function_scope',
            target_file=target_file,
            description=f"Move import of {import_edge.target_module} to function scope",
            codemod=codemod,
            risk_level='MEDIUM',
            dependencies=[]
        )
    
    def _generate_lazy_import_fix(self, source: str, target: str, analysis: SCCAnalysis) -> CircularImportFix:
        """Generate fix using lazy imports."""
        self.fix_id_counter += 1
        fix_id = f"lazy_import_{self.fix_id_counter:03d}"
        
        target_file = f"{source.replace('.', '/')}.py"
        
        codemod = f"""# Use lazy import to break circular dependency
# File: {target_file}

# REPLACE module-level import:
# import {target}
# OR
# from {target} import some_symbol

# WITH lazy import pattern:
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from {target} import some_symbol

def get_{target.replace('.', '_')}():
    \"\"\"Lazy import of {target} module.\"\"\"
    import {target}
    return {target}

# Usage example:
def some_function():
    target_module = get_{target.replace('.', '_')}()
    # Use target_module.some_symbol instead of direct import
"""
        
        return CircularImportFix(
            fix_id=fix_id,
            fix_type='lazy_import',
            target_file=target_file,
            description=f"Use lazy import for {source} -> {target} dependency",
            codemod=codemod,
            risk_level='LOW',
            dependencies=[]
        )
    
    def _can_move_to_function_scope(self, import_edge: ImportDependency) -> bool:
        """Check if import can safely be moved to function scope."""
        # Simple heuristics for now
        # Don't move star imports or decorator-related imports
        if import_edge.import_type == 'star':
            return False
        
        # Don't move imports of common types/constants
        common_type_imports = {'typing', 'abc', 'enum', 'dataclasses'}
        if import_edge.target_module in common_type_imports:
            return False
        
        return True
    
    def _ast_to_code(self, node: ast.AST) -> str:
        """Convert AST node back to code (simplified)."""
        try:
            import astor
            return astor.to_source(node).strip()
        except ImportError:
            # Fallback to basic representation
            if isinstance(node, ast.FunctionDef):
                return f"def {node.name}(): pass  # TODO: Implement"
            elif isinstance(node, ast.ClassDef):
                return f"class {node.name}: pass  # TODO: Implement"
            elif isinstance(node, ast.Assign):
                if node.targets and isinstance(node.targets[0], ast.Name):
                    return f"{node.targets[0].id} = None  # TODO: Set proper value"
            return "# TODO: Move definition here"
    
    def _topological_sort_fixes(self, fixes: List[CircularImportFix]) -> List[CircularImportFix]:
        """Sort fixes topologically based on dependencies."""
        # Build dependency graph
        fix_graph = nx.DiGraph()
        fix_by_id = {fix.fix_id: fix for fix in fixes}
        
        for fix in fixes:
            fix_graph.add_node(fix.fix_id)
            for dep_id in fix.dependencies:
                if dep_id in fix_by_id:
                    fix_graph.add_edge(dep_id, fix.fix_id)
        
        # Topological sort
        try:
            sorted_fix_ids = list(nx.topological_sort(fix_graph))
            return [fix_by_id[fix_id] for fix_id in sorted_fix_ids if fix_id in fix_by_id]
        except nx.NetworkXError:
            # If there are dependency cycles, return original order
            return fixes


class CircularImportResolver:
    """Main circular import resolver combining all components."""
    
    name = "circular_import_resolver"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('circular_import_resolver', {})
        self.enabled = self.config.get('enabled', True)
        self.min_scc_size = self.config.get('min_scc_size', 2)
        self.generate_fixes = self.config.get('generate_fixes', True)
        
    def run(self, ctx) -> List[Issue]:
        """Run circular import resolution analysis."""
        if not self.enabled:
            return []
        
        issues = []
        
        # Build import graph
        graph_builder = ImportGraphBuilder()
        import_graph = graph_builder.build_graph(ctx.ast_index)
        
        # Find and analyze SCCs
        scc_analyzer = SCCAnalyzer(graph_builder)
        scc_analyses = scc_analyzer.find_and_analyze_sccs()
        
        # Filter by minimum size
        scc_analyses = [scc for scc in scc_analyses if len(scc.modules) >= self.min_scc_size]
        
        if not scc_analyses:
            return issues
        
        # Generate fixes
        fix_generator = CircularImportFixGenerator()
        fixes = fix_generator.generate_fixes(scc_analyses) if self.generate_fixes else []
        
        # Create issues for each SCC
        for scc in scc_analyses:
            issue = self._create_scc_issue(scc, fixes)
            issues.append(issue)
        
        # Create fix orchestration issue
        if fixes:
            fix_issue = self._create_fix_orchestration_issue(fixes, scc_analyses)
            issues.append(fix_issue)
        
        return issues
    
    def _create_scc_issue(self, scc: SCCAnalysis, fixes: List[CircularImportFix]) -> Issue:
        """Create issue for a single SCC."""
        # Find fixes related to this SCC
        scc_fixes = [fix for fix in fixes if scc.scc_id in fix.fix_id or 
                    any(module in fix.target_file for module in scc.modules)]
        
        return Issue(
            kind="circular_import_scc",
            message=f"Circular import SCC detected with {len(scc.modules)} modules: {', '.join(scc.modules[:3])}{'...' if len(scc.modules) > 3 else ''}",
            severity=4 if len(scc.modules) > 2 else 3,
            file=scc.modules[0].replace('.', '/') + '.py',
            line=0,
            evidence={
                "scc_id": scc.scc_id,
                "modules": scc.modules,
                "complexity_score": scc.complexity_score,
                "shared_symbols": [s.name for s in scc.shared_symbols],
                "suggested_shared_module": scc.suggested_shared_module,
                "break_points": scc.break_points,
                "available_fixes": len(scc_fixes)
            },
            suggestions=[
                f"Extract {len(scc.shared_symbols)} shared symbols to {scc.suggested_shared_module}",
                f"Break cycle at {len(scc.break_points)} strategic points",
                "Move module-level imports to function scope where possible",
                "Use lazy imports for remaining dependencies",
                f"Apply {len(scc_fixes)} generated fixes in topological order"
            ]
        )
    
    def _create_fix_orchestration_issue(self, fixes: List[CircularImportFix], 
                                      scc_analyses: List[SCCAnalysis]) -> Issue:
        """Create orchestration issue with all fixes."""
        fix_data = [fix.to_dict() for fix in fixes]
        
        return Issue(
            kind="circular_import_fix_orchestration",
            message=f"Generated {len(fixes)} fixes for {len(scc_analyses)} circular import SCCs",
            severity=1,  # Informational
            file="<orchestration>",
            line=0,
            evidence={
                "total_fixes": len(fixes),
                "total_sccs": len(scc_analyses),
                "fixes": fix_data,
                "execution_order": [fix.fix_id for fix in fixes]
            },
            suggestions=[
                "Apply fixes in the provided topological order",
                "Test each fix independently before proceeding",
                "Create one PR per SCC for easier review",
                "Run tests after each major fix to ensure functionality"
            ]
        )
    
    def generate_fix_script(self, issues: List[Issue], output_path: str) -> str:
        """Generate a comprehensive fix script for all circular import issues."""
        # Extract fixes from orchestration issue
        orchestration_issues = [i for i in issues if i.kind == "circular_import_fix_orchestration"]
        if not orchestration_issues:
            return "# No circular import fixes available"
        
        fixes_data = orchestration_issues[0].evidence.get("fixes", [])
        
        script_parts = [
            "#!/usr/bin/env python3",
            '"""Automated circular import resolution script."""',
            "",
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "def apply_circular_import_fixes():",
            '    """Apply all circular import fixes in topological order."""',
            '    print("Applying circular import fixes...")',
            ""
        ]
        
        for i, fix_data in enumerate(fixes_data, 1):
            script_parts.extend([
                f'    # Fix {i}: {fix_data["description"]}',
                f'    print("Applying fix {i}: {fix_data["fix_type"]}")',
                f'    # Risk level: {fix_data["risk_level"]}',
                f'    # Target: {fix_data["target_file"]}',
                "",
                "    # Codemod:",
                *[f'    # {line}' for line in fix_data["codemod"].split('\n')],
                "",
                f'    print("Completed fix {i}")',
                ""
            ])
        
        script_parts.extend([
            'if __name__ == "__main__":',
            "    apply_circular_import_fixes()",
            '    print("All circular import fixes applied!")'
        ])
        
        script = "\n".join(script_parts)
        
        if output_path:
            Path(output_path).write_text(script)
        
        return script