"""
Canonical module policy system for managing canonical vs shadow implementations.

This module implements a policy layer that identifies canonical implementations
and de-prioritizes or suppresses shadow/experimental duplicates, providing
automatic codemod generation for proper import forwarding.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Union, Any
from pathlib import Path
from collections import defaultdict
import textwrap

from ..core.issues import Issue


@dataclass
class CanonicalSymbol:
    """Represents a canonical symbol definition."""
    name: str
    module_path: str
    file_path: str
    line_number: int
    symbol_type: str  # 'function', 'class', 'variable', 'constant'
    ast_node: ast.AST
    priority: int = 100  # Higher = more canonical
    
    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.module_path}.{self.name}"


@dataclass
class ShadowSymbol:
    """Represents a shadow/duplicate symbol."""
    canonical: CanonicalSymbol
    name: str
    module_path: str
    file_path: str
    line_number: int
    symbol_type: str
    ast_node: ast.AST
    suppressed: bool = False
    
    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.module_path}.{self.name}"


@dataclass
class CanonicalPolicy:
    """Configuration for canonical module policy."""
    canonical_roots: List[str] = field(default_factory=list)
    shadow_roots: List[str] = field(default_factory=list)
    priority_patterns: Dict[str, int] = field(default_factory=dict)
    auto_suppress_shadows: bool = True
    generate_forwarders: bool = True
    
    def is_canonical_path(self, path: str) -> bool:
        """Check if path is in canonical roots."""
        return any(path.startswith(root) for root in self.canonical_roots)
    
    def is_shadow_path(self, path: str) -> bool:
        """Check if path is in shadow roots."""
        return any(path.startswith(root) for root in self.shadow_roots)
    
    def get_path_priority(self, path: str) -> int:
        """Get priority score for a path."""
        if self.is_canonical_path(path):
            base_priority = 100
        elif self.is_shadow_path(path):
            base_priority = 10
        else:
            base_priority = 50
        
        # Apply pattern-based adjustments
        for pattern, adjustment in self.priority_patterns.items():
            if re.search(pattern, path):
                base_priority += adjustment
                
        return base_priority


@dataclass
class CodemodSuggestion:
    """Represents a codemod suggestion for fixing shadow imports."""
    shadow_file: str
    canonical_import: str
    replacement_code: str
    description: str
    risk_level: str = "LOW"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'shadow_file': self.shadow_file,
            'canonical_import': self.canonical_import,
            'replacement_code': self.replacement_code,
            'description': self.description,
            'risk_level': self.risk_level
        }


class CanonicalModuleAnalyzer:
    """Analyzes modules to identify canonical vs shadow implementations."""
    
    def __init__(self, policy: CanonicalPolicy):
        self.policy = policy
        self.canonical_symbols: Dict[str, CanonicalSymbol] = {}
        self.shadow_symbols: List[ShadowSymbol] = []
        self.symbol_registry: Dict[str, List[CanonicalSymbol]] = defaultdict(list)
        
    def analyze_symbols(self, ast_index: Dict[str, ast.AST]) -> Tuple[Dict[str, CanonicalSymbol], List[ShadowSymbol]]:
        """Analyze all modules to identify canonical and shadow symbols."""
        # First pass: collect all symbols
        all_symbols = []
        for file_path, tree in ast_index.items():
            module_path = self._file_to_module(file_path)
            symbols = self._extract_symbols(tree, file_path, module_path)
            all_symbols.extend(symbols)
        
        # Second pass: classify as canonical or shadow
        symbol_groups = defaultdict(list)
        for symbol in all_symbols:
            symbol_groups[symbol.name].append(symbol)
        
        # Determine canonical vs shadow for each symbol name
        for symbol_name, symbols in symbol_groups.items():
            if len(symbols) == 1:
                # Single implementation - it's canonical
                canonical = symbols[0]
                self.canonical_symbols[canonical.full_name] = canonical
                self.symbol_registry[symbol_name].append(canonical)
            else:
                # Multiple implementations - determine canonical
                canonical, shadows = self._classify_symbols(symbols)
                if canonical:
                    self.canonical_symbols[canonical.full_name] = canonical
                    self.symbol_registry[symbol_name].append(canonical)
                    
                    # Create shadow symbols
                    for shadow_ast in shadows:
                        shadow = ShadowSymbol(
                            canonical=canonical,
                            name=shadow_ast.name,
                            module_path=shadow_ast.module_path,
                            file_path=shadow_ast.file_path,
                            line_number=shadow_ast.line_number,
                            symbol_type=shadow_ast.symbol_type,
                            ast_node=shadow_ast.ast_node,
                            suppressed=self.policy.auto_suppress_shadows
                        )
                        self.shadow_symbols.append(shadow)
        
        return self.canonical_symbols, self.shadow_symbols
    
    def _file_to_module(self, file_path: str) -> str:
        """Convert file path to module path."""
        # Remove .py extension and convert path separators
        module = str(Path(file_path)).replace('/', '.').replace('\\', '.')
        if module.endswith('.py'):
            module = module[:-3]
        if module.endswith('.__init__'):
            module = module[:-9]
        return module
    
    def _extract_symbols(self, tree: ast.AST, file_path: str, module_path: str) -> List[CanonicalSymbol]:
        """Extract symbols from AST."""
        symbols = []
        
        for node in ast.walk(tree):
            symbol = None
            
            if isinstance(node, ast.FunctionDef):
                symbol = CanonicalSymbol(
                    name=node.name,
                    module_path=module_path,
                    file_path=file_path,
                    line_number=node.lineno,
                    symbol_type='function',
                    ast_node=node,
                    priority=self.policy.get_path_priority(file_path)
                )
            elif isinstance(node, ast.ClassDef):
                symbol = CanonicalSymbol(
                    name=node.name,
                    module_path=module_path,
                    file_path=file_path,
                    line_number=node.lineno,
                    symbol_type='class',
                    ast_node=node,
                    priority=self.policy.get_path_priority(file_path)
                )
            elif isinstance(node, ast.Assign):
                # Look for module-level assignments
                if hasattr(node, 'lineno') and self._is_module_level(node, tree):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            symbol = CanonicalSymbol(
                                name=target.id,
                                module_path=module_path,
                                file_path=file_path,
                                line_number=node.lineno,
                                symbol_type='variable',
                                ast_node=node,
                                priority=self.policy.get_path_priority(file_path)
                            )
                            symbols.append(symbol)
            
            if symbol:
                symbols.append(symbol)
                
        return symbols
    
    def _is_module_level(self, node: ast.AST, tree: ast.AST) -> bool:
        """Check if node is at module level."""
        # Simple check - if parent is Module, it's module level
        for parent in ast.walk(tree):
            if isinstance(parent, ast.Module):
                if node in parent.body:
                    return True
        return False
    
    def _classify_symbols(self, symbols: List[CanonicalSymbol]) -> Tuple[Optional[CanonicalSymbol], List[CanonicalSymbol]]:
        """Classify symbols as canonical vs shadows based on priority."""
        # Sort by priority (higher = more canonical)
        sorted_symbols = sorted(symbols, key=lambda s: s.priority, reverse=True)
        
        if not sorted_symbols:
            return None, []
            
        # Highest priority is canonical
        canonical = sorted_symbols[0]
        shadows = sorted_symbols[1:]
        
        # Additional heuristics for tie-breaking
        if len(shadows) > 0 and canonical.priority == shadows[0].priority:
            canonical, shadows = self._apply_tiebreaker_rules(sorted_symbols)
            
        return canonical, shadows
    
    def _apply_tiebreaker_rules(self, symbols: List[CanonicalSymbol]) -> Tuple[CanonicalSymbol, List[CanonicalSymbol]]:
        """Apply tiebreaker rules when priorities are equal."""
        # Rule 1: Prefer non-test files
        non_test = [s for s in symbols if 'test' not in s.file_path.lower()]
        if non_test:
            symbols = non_test
            
        # Rule 2: Prefer files with fewer underscores (less "internal")
        min_underscores = min(s.file_path.count('_') for s in symbols)
        symbols = [s for s in symbols if s.file_path.count('_') == min_underscores]
        
        # Rule 3: Prefer shorter paths (closer to root)
        min_depth = min(s.file_path.count('/') for s in symbols)
        symbols = [s for s in symbols if s.file_path.count('/') == min_depth]
        
        # Rule 4: Alphabetical order as final tiebreaker
        symbols.sort(key=lambda s: s.file_path)
        
        return symbols[0], symbols[1:]


class CodemodGenerator:
    """Generates codemods for replacing shadow implementations with imports."""
    
    def __init__(self, policy: CanonicalPolicy):
        self.policy = policy
        
    def generate_forwarder_codemod(self, shadow: ShadowSymbol) -> CodemodSuggestion:
        """Generate a codemod to replace shadow with import forwarder."""
        if shadow.symbol_type == 'function':
            return self._generate_function_forwarder(shadow)
        elif shadow.symbol_type == 'class':
            return self._generate_class_forwarder(shadow)
        elif shadow.symbol_type == 'variable':
            return self._generate_variable_forwarder(shadow)
        else:
            return self._generate_generic_forwarder(shadow)
    
    def _generate_function_forwarder(self, shadow: ShadowSymbol) -> CodemodSuggestion:
        """Generate function forwarder."""
        canonical = shadow.canonical
        import_path = canonical.module_path
        
        # Extract function signature
        func_node = shadow.ast_node
        if isinstance(func_node, ast.FunctionDef):
            # Generate signature
            args = []
            for arg in func_node.args.args:
                args.append(arg.arg)
            
            # Handle *args, **kwargs
            if func_node.args.vararg:
                args.append(f"*{func_node.args.vararg.arg}")
            if func_node.args.kwarg:
                args.append(f"**{func_node.args.kwarg.arg}")
                
            signature = ", ".join(args)
            
            replacement = textwrap.dedent(f"""
                # DEPRECATED: This is a shadow implementation. Use {canonical.full_name} instead.
                from {import_path} import {canonical.name}
                
                def {shadow.name}({signature}):
                    \"\"\"Deprecated: Use {canonical.full_name} instead.\"\"\"
                    import warnings
                    warnings.warn(
                        f"{{__name__}}.{shadow.name} is deprecated. Use {canonical.full_name} instead.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                    return {canonical.name}({signature})
            """).strip()
        else:
            # Fallback for complex signatures
            replacement = textwrap.dedent(f"""
                # DEPRECATED: This is a shadow implementation. Use {canonical.full_name} instead.
                from {import_path} import {canonical.name} as {shadow.name}
            """).strip()
        
        return CodemodSuggestion(
            shadow_file=shadow.file_path,
            canonical_import=f"from {import_path} import {canonical.name}",
            replacement_code=replacement,
            description=f"Replace shadow function {shadow.name} with import from canonical {canonical.full_name}",
            risk_level="LOW"
        )
    
    def _generate_class_forwarder(self, shadow: ShadowSymbol) -> CodemodSuggestion:
        """Generate class forwarder."""
        canonical = shadow.canonical
        import_path = canonical.module_path
        
        replacement = textwrap.dedent(f"""
            # DEPRECATED: This is a shadow implementation. Use {canonical.full_name} instead.
            from {import_path} import {canonical.name}
            
            # Backward compatibility alias
            class {shadow.name}({canonical.name}):
                \"\"\"Deprecated: Use {canonical.full_name} instead.\"\"\"
                def __init__(self, *args, **kwargs):
                    import warnings
                    warnings.warn(
                        f"{{self.__class__.__module__}}.{shadow.name} is deprecated. Use {canonical.full_name} instead.",
                        DeprecationWarning,
                        stacklevel=2
                    )
                    super().__init__(*args, **kwargs)
        """).strip()
        
        return CodemodSuggestion(
            shadow_file=shadow.file_path,
            canonical_import=f"from {import_path} import {canonical.name}",
            replacement_code=replacement,
            description=f"Replace shadow class {shadow.name} with inheritance from canonical {canonical.full_name}",
            risk_level="MEDIUM"
        )
    
    def _generate_variable_forwarder(self, shadow: ShadowSymbol) -> CodemodSuggestion:
        """Generate variable forwarder."""
        canonical = shadow.canonical
        import_path = canonical.module_path
        
        replacement = textwrap.dedent(f"""
            # DEPRECATED: This is a shadow implementation. Use {canonical.full_name} instead.
            from {import_path} import {canonical.name} as {shadow.name}
        """).strip()
        
        return CodemodSuggestion(
            shadow_file=shadow.file_path,
            canonical_import=f"from {import_path} import {canonical.name}",
            replacement_code=replacement,
            description=f"Replace shadow variable {shadow.name} with import from canonical {canonical.full_name}",
            risk_level="LOW"
        )
    
    def _generate_generic_forwarder(self, shadow: ShadowSymbol) -> CodemodSuggestion:
        """Generate generic forwarder."""
        canonical = shadow.canonical
        import_path = canonical.module_path
        
        replacement = textwrap.dedent(f"""
            # DEPRECATED: This is a shadow implementation. Use {canonical.full_name} instead.
            from {import_path} import {canonical.name} as {shadow.name}
        """).strip()
        
        return CodemodSuggestion(
            shadow_file=shadow.file_path,
            canonical_import=f"from {import_path} import {canonical.name}",
            replacement_code=replacement,
            description=f"Replace shadow symbol {shadow.name} with import from canonical {canonical.full_name}",
            risk_level="MEDIUM"
        )
    
    def generate_bulk_codemod(self, shadows: List[ShadowSymbol]) -> str:
        """Generate a bulk codemod script for multiple shadows."""
        codemods = []
        for shadow in shadows:
            codemod = self.generate_forwarder_codemod(shadow)
            codemods.append(codemod)
        
        # Group by file
        by_file = defaultdict(list)
        for codemod in codemods:
            by_file[codemod.shadow_file].append(codemod)
        
        script_parts = [
            "#!/usr/bin/env python3",
            '"""Bulk codemod to replace shadow implementations with canonical imports."""',
            "",
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "def apply_codemod():",
            '    """Apply all codemods."""'
        ]
        
        for file_path, file_codemods in by_file.items():
            script_parts.extend([
                f"",
                f'    # File: {file_path}',
                f'    print(f"Processing {file_path}...")',
                f'    with open("{file_path}", "w") as f:'
            ])
            
            for codemod in file_codemods:
                script_parts.extend([
                    f'        f.write("""',
                    f'{codemod.replacement_code}',
                    f'""")'
                ])
        
        script_parts.extend([
            "",
            'if __name__ == "__main__":',
            "    apply_codemod()",
            '    print("Codemod complete!")'
        ])
        
        return "\n".join(script_parts)


class CanonicalPolicyAnalyzer:
    """Main analyzer that applies canonical module policy."""
    
    name = "canonical_policy"
    
    def __init__(self, config: Dict[str, Any]):
        policy_config = config.get('canonical_policy', {})
        self.policy = CanonicalPolicy(
            canonical_roots=policy_config.get('canonical_roots', []),
            shadow_roots=policy_config.get('shadow_roots', []),
            priority_patterns=policy_config.get('priority_patterns', {}),
            auto_suppress_shadows=policy_config.get('auto_suppress_shadows', True),
            generate_forwarders=policy_config.get('generate_forwarders', True)
        )
        self.analyzer = CanonicalModuleAnalyzer(self.policy)
        self.codemod_generator = CodemodGenerator(self.policy)
        
    def run(self, ctx) -> List[Issue]:
        """Run canonical policy analysis."""
        issues = []
        
        if not self.policy.canonical_roots and not self.policy.shadow_roots:
            # No policy configured
            return issues
            
        # Analyze symbols
        canonical_symbols, shadow_symbols = self.analyzer.analyze_symbols(ctx.ast_index)
        
        # Generate issues for shadow symbols
        for shadow in shadow_symbols:
            if shadow.suppressed:
                # Create suppressed issue
                issue = Issue(
                    kind="shadow_implementation_suppressed",
                    message=f"Shadow implementation of {shadow.name} suppressed in favor of canonical {shadow.canonical.full_name}",
                    severity=1,  # Low severity since it's auto-handled
                    file=shadow.file_path,
                    line=shadow.line_number,
                    evidence={
                        "shadow_symbol": shadow.full_name,
                        "canonical_symbol": shadow.canonical.full_name,
                        "canonical_file": shadow.canonical.file_path
                    },
                    suggestions=[
                        f"Use canonical implementation: {shadow.canonical.full_name}",
                        f"Import: from {shadow.canonical.module_path} import {shadow.canonical.name}",
                        "Consider removing this shadow implementation"
                    ]
                )
            else:
                # Create warning issue
                issue = Issue(
                    kind="shadow_implementation_detected",
                    message=f"Shadow implementation of {shadow.name} detected. Canonical version exists at {shadow.canonical.full_name}",
                    severity=2,
                    file=shadow.file_path,
                    line=shadow.line_number,
                    evidence={
                        "shadow_symbol": shadow.full_name,
                        "canonical_symbol": shadow.canonical.full_name,
                        "canonical_file": shadow.canonical.file_path,
                        "priority_diff": shadow.canonical.priority - 10  # Assume shadow priority is lower
                    },
                    suggestions=[
                        f"Use canonical implementation: {shadow.canonical.full_name}",
                        f"Import: from {shadow.canonical.module_path} import {shadow.canonical.name}",
                        "Replace with import forwarder",
                        "Remove shadow implementation"
                    ]
                )
            
            issues.append(issue)
            
        # Generate codemod suggestions if requested
        if self.policy.generate_forwarders and shadow_symbols:
            codemod_issue = self._create_codemod_issue(shadow_symbols)
            issues.append(codemod_issue)
            
        return issues
    
    def _create_codemod_issue(self, shadows: List[ShadowSymbol]) -> Issue:
        """Create an issue with codemod suggestions."""
        codemods = []
        for shadow in shadows:
            codemod = self.codemod_generator.generate_forwarder_codemod(shadow)
            codemods.append(codemod.to_dict())
        
        return Issue(
            kind="canonical_policy_codemod",
            message=f"Generated {len(codemods)} codemod suggestions for shadow implementations",
            severity=1,
            file="<policy>",
            line=0,
            evidence={
                "codemod_count": len(codemods),
                "codemods": codemods
            },
            suggestions=[
                f"Apply {len(codemods)} generated codemods to replace shadows with imports",
                "Review each codemod before applying",
                "Test thoroughly after applying codemods",
                "Consider updating documentation to reference canonical implementations"
            ]
        )
    
    def generate_codemod_script(self, ast_index: Dict[str, ast.AST], output_path: str) -> str:
        """Generate a complete codemod script."""
        canonical_symbols, shadow_symbols = self.analyzer.analyze_symbols(ast_index)
        
        if not shadow_symbols:
            return "# No shadow implementations detected"
            
        script = self.codemod_generator.generate_bulk_codemod(shadow_symbols)
        
        if output_path:
            Path(output_path).write_text(script)
            
        return script