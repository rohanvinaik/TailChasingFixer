"""
Validation logic for patches and fixes.

Provides syntax validation, risk assessment, and safety checks.
"""

import ast
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Protocol
import tempfile
import logging

from .state import PatchInfo, RiskLevel
from .errors import ValidationError


class ValidatorProtocol(Protocol):
    """Protocol for validators."""
    
    def validate(self, patch: PatchInfo) -> bool:
        """Validate a patch."""
        ...
    
    def get_risk_level(self, patch: PatchInfo) -> RiskLevel:
        """Assess risk level of a patch."""
        ...


class SyntaxValidator:
    """
    Validates Python syntax in patches.
    
    Ensures patches don't introduce syntax errors.
    """
    
    def __init__(self):
        """Initialize syntax validator."""
        self.logger = logging.getLogger(__name__)
    
    def validate_syntax(self, code: str, file_path: str = "<string>") -> bool:
        """
        Validate Python syntax.
        
        Args:
            code: Python code to validate
            file_path: Path for error reporting
            
        Returns:
            True if syntax is valid
            
        Raises:
            ValidationError: If syntax is invalid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            raise ValidationError(
                f"Syntax error in {file_path}: {e.msg}",
                validation_type='syntax',
                file_path=file_path,
                line_number=e.lineno,
                details={'error': str(e)}
            )
    
    def validate_patch(self, patch: PatchInfo) -> bool:
        """
        Validate syntax in a patch.
        
        Args:
            patch: Patch to validate
            
        Returns:
            True if patch syntax is valid
        """
        return self.validate_syntax(patch.patched_content, patch.file_path)


class RiskAssessor:
    """
    Assesses risk level of patches.
    
    Evaluates potential impact and safety of changes.
    """
    
    def __init__(self, risk_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize risk assessor.
        
        Args:
            risk_thresholds: Custom risk thresholds
        """
        self.risk_thresholds = risk_thresholds or {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 1.0
        }
        self.logger = logging.getLogger(__name__)
    
    def assess_risk(self, patch: PatchInfo) -> RiskLevel:
        """
        Assess risk level of a patch.
        
        Args:
            patch: Patch to assess
            
        Returns:
            Risk level
        """
        risk_score = self._calculate_risk_score(patch)
        
        if risk_score < self.risk_thresholds['low']:
            return RiskLevel.LOW
        elif risk_score < self.risk_thresholds['medium']:
            return RiskLevel.MEDIUM
        elif risk_score < self.risk_thresholds['high']:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _calculate_risk_score(self, patch: PatchInfo) -> float:
        """
        Calculate numeric risk score.
        
        Args:
            patch: Patch to score
            
        Returns:
            Risk score between 0 and 1
        """
        score = 0.0
        
        # Check size of change
        lines_changed = len(patch.patched_content.splitlines()) - \
                       len(patch.original_content.splitlines())
        
        if abs(lines_changed) > 50:
            score += 0.3
        elif abs(lines_changed) > 20:
            score += 0.2
        elif abs(lines_changed) > 10:
            score += 0.1
        
        # Check for deletions (risky)
        if len(patch.patched_content) < len(patch.original_content) * 0.5:
            score += 0.3
        
        # Check for import changes (moderate risk)
        if 'import' in patch.patched_content and 'import' not in patch.original_content:
            score += 0.2
        
        # Check for class/function deletions (high risk)
        orig_ast = self._safe_parse(patch.original_content)
        new_ast = self._safe_parse(patch.patched_content)
        
        if orig_ast and new_ast:
            orig_defs = self._count_definitions(orig_ast)
            new_defs = self._count_definitions(new_ast)
            
            if new_defs < orig_defs:
                score += 0.3
        
        # Factor in confidence (inverse relationship)
        score *= (2.0 - patch.confidence)
        
        return min(score, 1.0)
    
    def _safe_parse(self, code: str) -> Optional[ast.AST]:
        """Safely parse code to AST."""
        try:
            return ast.parse(code)
        except:
            return None
    
    def _count_definitions(self, tree: ast.AST) -> int:
        """Count function and class definitions."""
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                count += 1
        return count


class PatchValidator:
    """
    Comprehensive patch validation.
    
    Combines syntax validation, risk assessment, and safety checks.
    """
    
    def __init__(self, max_risk: RiskLevel = RiskLevel.HIGH,
                 test_command: Optional[str] = None):
        """
        Initialize patch validator.
        
        Args:
            max_risk: Maximum acceptable risk level
            test_command: Command to run tests
        """
        self.max_risk = max_risk
        self.test_command = test_command
        self.syntax_validator = SyntaxValidator()
        self.risk_assessor = RiskAssessor()
        self.logger = logging.getLogger(__name__)
    
    def validate(self, patch: PatchInfo) -> bool:
        """
        Validate a patch comprehensively.
        
        Args:
            patch: Patch to validate
            
        Returns:
            True if patch is valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check syntax
        self.syntax_validator.validate_patch(patch)
        
        # Assess risk
        risk_level = self.risk_assessor.assess_risk(patch)
        patch.risk_level = risk_level
        
        # Check risk threshold
        if risk_level.to_numeric() > self.max_risk.to_numeric():
            raise ValidationError(
                f"Patch risk level {risk_level.value} exceeds maximum {self.max_risk.value}",
                validation_type='risk',
                file_path=patch.file_path,
                risk_level=risk_level.to_numeric()
            )
        
        # Run tests if configured
        if self.test_command:
            self._run_tests(patch)
        
        return True
    
    def _run_tests(self, patch: PatchInfo) -> bool:
        """Run tests on patched code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(patch.patched_content)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                self.test_command, shell=True, capture_output=True,
                text=True, timeout=30
            )
            
            if result.returncode != 0:
                raise ValidationError(
                    f"Tests failed for patch: {result.stderr}",
                    validation_type='test',
                    file_path=patch.file_path,
                    details={'stdout': result.stdout, 'stderr': result.stderr,
                             'returncode': result.returncode}
                )
            return True
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def validate_batch(self, patches: List[PatchInfo]) -> List[bool]:
        """
        Validate multiple patches.
        
        Args:
            patches: Patches to validate
            
        Returns:
            List of validation results
        """
        results = []
        for patch in patches:
            try:
                results.append(self.validate(patch))
            except ValidationError:
                results.append(False)
        return results