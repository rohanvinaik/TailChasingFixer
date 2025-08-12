"""Advanced phantom-stub triage system with security classification."""

import ast
import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from .base import Analyzer, AnalysisContext
from ..core.issues import Issue


class PhantomPriority(Enum):
    """Priority levels for phantom stubs."""
    P0_SECURITY = "P0_SECURITY"      # Security/Correctness - Block CI
    P1_FUNCTIONAL = "P1_FUNCTIONAL"  # Functional - Needs implementation
    P3_EXPERIMENTAL = "P3_EXPERIMENTAL"  # Experimental/Tooling - Allowlist


@dataclass
class PhantomStub:
    """Represents a detected phantom stub with classification."""
    function_name: str
    class_name: Optional[str]
    file_path: str
    line_number: int
    signature: str
    body_content: str
    priority: PhantomPriority
    risk_factors: List[str]
    suggested_action: str
    is_allowed: bool = False
    is_blocked: bool = False


class PhantomStubClassifier:
    """Classifies phantom stubs by security and functional importance."""
    
    # P0 Security/Correctness patterns
    SECURITY_PATTERNS = {
        'crypto_verify': [
            r'.*verify.*',
            r'.*authenticate.*', 
            r'.*validate.*signature.*',
            r'.*check.*hash.*',
            r'.*digest.*',
        ],
        'hsm_operations': [
            r'.*hsm.*',
            r'.*hardware.*security.*',
            r'.*key.*management.*',
            r'.*secure.*element.*',
        ],
        'crypto_primitives': [
            r'.*encrypt.*',
            r'.*decrypt.*', 
            r'.*sign.*',
            r'.*hash.*',
            r'.*random.*',
            r'.*entropy.*',
        ],
        'math_operations': [
            r'.*stark.*',
            r'.*proof.*',
            r'.*commitment.*',
            r'.*merkle.*',
            r'.*polynomial.*',
            r'.*field.*arithmetic.*',
        ],
        'post_quantum': [
            r'.*dilithium.*',
            r'.*kyber.*',
            r'.*falcon.*',
            r'.*sphincs.*',
            r'.*lattice.*',
        ]
    }
    
    # P1 Functional patterns
    FUNCTIONAL_PATTERNS = {
        'config_getters': [
            r'.*get.*config.*',
            r'.*load.*settings.*',
            r'.*read.*properties.*',
        ],
        'id_generators': [
            r'.*generate.*id.*',
            r'.*create.*uuid.*',
            r'.*get.*identifier.*',
        ],
        'data_access': [
            r'.*get.*data.*',
            r'.*fetch.*',
            r'.*retrieve.*',
            r'.*load.*',
        ]
    }
    
    # P3 Experimental patterns
    EXPERIMENTAL_PATTERNS = {
        'code_generation': [
            r'.*generate.*verilog.*',
            r'.*generate.*fpga.*',
            r'.*compile.*to.*',
            r'.*transpile.*',
        ],
        'tooling': [
            r'.*benchmark.*',
            r'.*profile.*',
            r'.*debug.*',
            r'.*visualize.*',
        ],
        'research': [
            r'.*experimental.*',
            r'.*prototype.*',
            r'.*research.*',
            r'.*poc.*',
        ]
    }
    
    def classify_stub(self, stub: PhantomStub, context: AnalysisContext) -> PhantomPriority:
        """Classify a phantom stub by priority level."""
        function_key = f"{stub.class_name}.{stub.function_name}" if stub.class_name else stub.function_name
        full_key = f"{stub.file_path}::{function_key}"
        
        # Check if explicitly blocked
        blocked_patterns = context.config.get('placeholders', {}).get('block', [])
        for pattern in blocked_patterns:
            if self._matches_pattern(full_key, pattern):
                stub.is_blocked = True
                return PhantomPriority.P0_SECURITY
        
        # Check if explicitly allowed
        allowed_patterns = context.config.get('placeholders', {}).get('allow', [])
        for pattern in allowed_patterns:
            if self._matches_pattern(full_key, pattern):
                stub.is_allowed = True
                return PhantomPriority.P3_EXPERIMENTAL
        
        # Classify by content patterns
        combined_text = f"{stub.function_name} {stub.signature} {stub.body_content}".lower()
        
        # Check P0 Security patterns
        for category, patterns in self.SECURITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    stub.risk_factors.append(f"Security risk: {category}")
                    return PhantomPriority.P0_SECURITY
        
        # Check for security-related file paths
        if any(keyword in stub.file_path.lower() for keyword in 
               ['crypto', 'security', 'auth', 'hsm', 'quantum', 'proof']):
            stub.risk_factors.append("Security-related file path")
            return PhantomPriority.P0_SECURITY
        
        # Check P1 Functional patterns
        for category, patterns in self.FUNCTIONAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    stub.risk_factors.append(f"Functional requirement: {category}")
                    return PhantomPriority.P1_FUNCTIONAL
        
        # Check P3 Experimental patterns
        for category, patterns in self.EXPERIMENTAL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, combined_text):
                    stub.risk_factors.append(f"Experimental feature: {category}")
                    return PhantomPriority.P3_EXPERIMENTAL
        
        # Default to P1 for unclassified stubs
        return PhantomPriority.P1_FUNCTIONAL
    
    def _matches_pattern(self, target: str, pattern: str) -> bool:
        """Check if target matches a glob-like pattern."""
        # Simple glob matching - convert to regex
        regex_pattern = pattern.replace('*', '.*').replace('?', '.')
        # Use search instead of match to find pattern anywhere in target
        return bool(re.search(regex_pattern, target))


class PhantomStubGenerator:
    """Generates appropriate implementations for phantom stubs."""
    
    def generate_p0_security_implementation(self, stub: PhantomStub) -> str:
        """Generate security-conscious implementation for P0 stubs."""
        if stub.is_blocked:
            return f'''def {stub.function_name}{stub.signature}:
    """SECURITY CRITICAL: This function requires proper implementation.
    
    This stub has been identified as security-critical and must not be used
    in production. Implement the actual cryptographic/security logic here.
    """
    raise NotImplementedError(
        f"SECURITY CRITICAL: {{stub.function_name}} is not implemented. "
        f"Risk factors: {{', '.join(stub.risk_factors)}}"
    )'''
        
        # Generate scaffolding for security functions
        if 'verify' in stub.function_name.lower():
            return f'''def {stub.function_name}{stub.signature}:
    """Security-critical verification function - requires implementation.
    
    TODO: Implement proper cryptographic verification logic.
    This scaffolding is provided to prevent silent failures.
    """
    import warnings
    warnings.warn(
        f"{{stub.function_name}} is using a stub implementation. "
        f"Do not use in production!",
        UserWarning,
        stacklevel=2
    )
    
    # Scaffold implementation - always returns False for safety
    return False  # SAFE DEFAULT: Fail closed for security'''
        
        return f'''def {stub.function_name}{stub.signature}:
    """Security-critical function requiring implementation."""
    raise NotImplementedError(
        f"Security-critical function {{stub.function_name}} must be implemented before use"
    )'''
    
    def generate_p1_functional_implementation(self, stub: PhantomStub) -> str:
        """Generate minimal correct implementation for P1 stubs."""
        if 'get' in stub.function_name.lower() and 'config' in stub.function_name.lower():
            return f'''def {stub.function_name}{stub.signature}:
    """Minimal configuration getter implementation."""
    import os
    import json
    from pathlib import Path
    
    # Try common config locations
    config_paths = [
        Path.cwd() / "config.json",
        Path.home() / ".config" / "app.json",
        Path("/etc/app/config.json")
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    
    # Fallback to environment variables
    return {{
        key.lower(): value for key, value in os.environ.items()
        if key.startswith('APP_')
    }}'''
        
        if 'id' in stub.function_name.lower():
            return f'''def {stub.function_name}{stub.signature}:
    """Minimal ID generation implementation."""
    import uuid
    import hashlib
    import time
    
    # Generate deterministic ID based on context
    context_data = f"{{time.time()}}{{id(self) if hasattr(self, '__class__') else 'global'}}"
    return hashlib.sha256(context_data.encode()).hexdigest()[:16]'''
        
        return f'''def {stub.function_name}{stub.signature}:
    """Functional stub - minimal implementation provided."""
    # TODO: Implement proper functionality
    import warnings
    warnings.warn(f"Using stub implementation for {{stub.function_name}}", UserWarning)
    return None'''
    
    def generate_p3_experimental_implementation(self, stub: PhantomStub) -> str:
        """Generate allowlist implementation for P3 stubs."""
        return f'''def {stub.function_name}{stub.signature}:
    """Experimental/tooling function - implementation pending."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Calling experimental function: {{stub.function_name}}")
    
    # Return safe placeholder for experimental features
    return None  # Placeholder for experimental feature'''


class PhantomTriageAnalyzer(Analyzer):
    """Enhanced phantom stub analyzer with security triage."""
    
    name = "phantom_triage"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.classifier = PhantomStubClassifier()
        self.generator = PhantomStubGenerator()
        self._detected_stubs: List[PhantomStub] = []
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Run phantom stub triage analysis."""
        issues = []
        self._detected_stubs.clear()
        
        for file_path, tree in ctx.ast_index.items():
            stubs = self._analyze_file(file_path, tree, ctx)
            self._detected_stubs.extend(stubs)
            
            # Create issues for each stub
            for stub in stubs:
                issue = self._create_issue_for_stub(stub, ctx)
                issues.append(issue)
        
        return issues
    
    def _analyze_file(self, file_path: str, tree: ast.AST, ctx: AnalysisContext) -> List[PhantomStub]:
        """Analyze a single file for phantom stubs."""
        stubs = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                stub = self._analyze_function(file_path, node, ctx)
                if stub:
                    stubs.append(stub)
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        stub = self._analyze_method(file_path, node.name, item, ctx)
                        if stub:
                            stubs.append(stub)
        
        return stubs
    
    def _analyze_function(self, file_path: str, node: ast.FunctionDef, ctx: AnalysisContext) -> Optional[PhantomStub]:
        """Analyze a function for phantom stub patterns."""
        if not self._is_phantom_stub(node):
            return None
        
        signature = self._extract_signature(node)
        body_content = self._extract_body_content(node)
        
        stub = PhantomStub(
            function_name=node.name,
            class_name=None,
            file_path=file_path,
            line_number=node.lineno,
            signature=signature,
            body_content=body_content,
            priority=PhantomPriority.P1_FUNCTIONAL,  # Will be classified
            risk_factors=[],
            suggested_action=""
        )
        
        # Classify the stub
        stub.priority = self.classifier.classify_stub(stub, ctx)
        stub.suggested_action = self._generate_suggested_action(stub)
        
        return stub
    
    def _analyze_method(self, file_path: str, class_name: str, node: ast.FunctionDef, ctx: AnalysisContext) -> Optional[PhantomStub]:
        """Analyze a class method for phantom stub patterns."""
        if not self._is_phantom_stub(node):
            return None
        
        signature = self._extract_signature(node)
        body_content = self._extract_body_content(node)
        
        stub = PhantomStub(
            function_name=node.name,
            class_name=class_name,
            file_path=file_path,
            line_number=node.lineno,
            signature=signature,
            body_content=body_content,
            priority=PhantomPriority.P1_FUNCTIONAL,
            risk_factors=[],
            suggested_action=""
        )
        
        stub.priority = self.classifier.classify_stub(stub, ctx)
        stub.suggested_action = self._generate_suggested_action(stub)
        
        return stub
    
    def _is_phantom_stub(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a phantom stub."""
        if not node.body:
            return False
        
        # Check for common stub patterns
        for stmt in node.body:
            if isinstance(stmt, ast.Pass):
                return True
            elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                    return True
            elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value is None:
                    return True
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Docstring only
                continue
            else:
                # Has actual implementation
                return False
        
        return True
    
    def _extract_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature as string."""
        args = []
        
        for arg in node.args.args:
            args.append(arg.arg)
        
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        
        for arg in node.args.kwonlyargs:
            args.append(arg.arg)
        
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
        
        return f"({', '.join(args)})"
    
    def _extract_body_content(self, node: ast.FunctionDef) -> str:
        """Extract function body content for analysis."""
        body_parts = []
        
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Docstring
                body_parts.append(str(stmt.value.value))
            elif isinstance(stmt, ast.Pass):
                body_parts.append("pass")
            elif isinstance(stmt, ast.Raise):
                body_parts.append("raise NotImplementedError")
            elif isinstance(stmt, ast.Return):
                body_parts.append("return None")
        
        return " ".join(body_parts)
    
    def _generate_suggested_action(self, stub: PhantomStub) -> str:
        """Generate suggested action based on stub priority."""
        if stub.priority == PhantomPriority.P0_SECURITY:
            if stub.is_blocked:
                return "BLOCK CI: Implement security-critical function or add explicit failure"
            return "URGENT: Scaffold real implementation with secure defaults"
        elif stub.priority == PhantomPriority.P1_FUNCTIONAL:
            return "Provide minimal, correct implementation"
        else:  # P3_EXPERIMENTAL
            return "Allowlist until implementation needed"
    
    def _create_issue_for_stub(self, stub: PhantomStub, ctx: AnalysisContext) -> Issue:
        """Create an issue for a phantom stub."""
        severity_map = {
            PhantomPriority.P0_SECURITY: 5,  # Critical
            PhantomPriority.P1_FUNCTIONAL: 3,  # High
            PhantomPriority.P3_EXPERIMENTAL: 1  # Low
        }
        
        message = f"{stub.priority.value}: {stub.function_name}"
        if stub.class_name:
            message = f"{stub.priority.value}: {stub.class_name}.{stub.function_name}"
        
        if stub.risk_factors:
            message += f" ({', '.join(stub.risk_factors)})"
        
        evidence = {
            'priority': stub.priority.value,
            'function_name': stub.function_name,
            'class_name': stub.class_name,
            'signature': stub.signature,
            'risk_factors': stub.risk_factors,
            'suggested_action': stub.suggested_action,
            'is_allowed': stub.is_allowed,
            'is_blocked': stub.is_blocked,
            'generated_implementation': self._generate_implementation(stub)
        }
        
        return Issue(
            kind="phantom_stub_triage",
            message=message,
            severity=severity_map[stub.priority],
            file=stub.file_path,
            line=stub.line_number,
            evidence=evidence
        )
    
    def _generate_implementation(self, stub: PhantomStub) -> str:
        """Generate implementation code for the stub."""
        if stub.priority == PhantomPriority.P0_SECURITY:
            return self.generator.generate_p0_security_implementation(stub)
        elif stub.priority == PhantomPriority.P1_FUNCTIONAL:
            return self.generator.generate_p1_functional_implementation(stub)
        else:
            return self.generator.generate_p3_experimental_implementation(stub)
    
    def get_detected_stubs(self) -> List[PhantomStub]:
        """Get all detected phantom stubs from the last run."""
        return self._detected_stubs.copy()
    
    def generate_triage_report(self) -> str:
        """Generate a comprehensive triage report."""
        if not self._detected_stubs:
            return "No phantom stubs detected."
        
        # Group by priority
        by_priority = {}
        for stub in self._detected_stubs:
            priority = stub.priority.value
            if priority not in by_priority:
                by_priority[priority] = []
            by_priority[priority].append(stub)
        
        report_lines = [
            "PHANTOM STUB TRIAGE REPORT",
            "=" * 50,
            f"Total stubs detected: {len(self._detected_stubs)}",
            ""
        ]
        
        for priority in ["P0_SECURITY", "P1_FUNCTIONAL", "P3_EXPERIMENTAL"]:
            stubs = by_priority.get(priority, [])
            if not stubs:
                continue
            
            report_lines.extend([
                f"{priority}: {len(stubs)} stubs",
                "-" * 30
            ])
            
            for stub in stubs:
                function_name = f"{stub.class_name}.{stub.function_name}" if stub.class_name else stub.function_name
                report_lines.extend([
                    f"  {function_name} ({stub.file_path}:{stub.line_number})",
                    f"    Action: {stub.suggested_action}"
                ])
                
                if stub.risk_factors:
                    report_lines.append(f"    Risks: {', '.join(stub.risk_factors)}")
                
                if stub.is_blocked:
                    report_lines.append("    ⛔ BLOCKED - Will fail CI")
                elif stub.is_allowed:
                    report_lines.append("    ✅ ALLOWED - Explicitly allowlisted")
                
                report_lines.append("")
            
            report_lines.append("")
        
        return "\n".join(report_lines)