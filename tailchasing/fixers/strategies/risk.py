"""
Risk assessment and scoring strategies.

Handles risk evaluation, impact analysis, and confidence scoring for fixes.
Extracted from fix_strategies.py to reduce context window thrashing.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .base import BaseFixStrategy, Action, Patch, RiskLevel
from ...core.issues import Issue


@dataclass
class RiskFactor:
    """Represents a single risk factor in fix assessment."""
    name: str
    weight: float  # 0.0 - 1.0
    score: float   # 0.0 - 1.0  
    description: str
    mitigation: str = ""


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment for a fix."""
    overall_risk: RiskLevel
    confidence_score: float  # 0.0 - 1.0
    risk_factors: List[RiskFactor]
    impact_analysis: Dict[str, Any]
    recommendations: List[str]
    estimated_time: float
    rollback_difficulty: RiskLevel


class RiskAnalysisStrategy(BaseFixStrategy):
    """
    Strategy for comprehensive risk analysis of proposed fixes.
    
    Evaluates:
    - Code complexity impact
    - Dependency chain effects  
    - Performance implications
    - Security considerations
    - Maintainability impact
    """
    
    def __init__(self):
        super().__init__("RiskAnalysis")
        
        # Risk weights for different factors
        self.risk_weights = {
            'code_complexity': 0.2,
            'dependency_impact': 0.25,
            'performance_impact': 0.15,
            'security_impact': 0.3,
            'maintainability': 0.1
        }
    
    def can_handle(self, issue: Issue) -> bool:
        """Can analyze risk for any issue type."""
        return True
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose risk analysis documentation for a fix."""
        actions = self._generate_risk_analysis_actions(issue, context)
        
        if not actions:
            return None
        
        return Patch(
            actions=actions,
            description=f"Generate risk analysis for: {issue.kind}",
            confidence=0.95,  # Risk analysis is reliable
            risk_level=RiskLevel.LOW,  # Analysis doesn't change code
            estimated_time=self._estimate_time(actions),
            dependencies=[],
            rollback_plan=[self.create_backup_action(action) for action in actions],
            validation_tests=[],
            side_effects=["Creates risk analysis documentation"]
        )
    
    def analyze_fix_risk(self, issue: Issue, proposed_patch: Patch, context: Optional[Dict[str, Any]] = None) -> RiskAssessment:
        """
        Perform comprehensive risk analysis of a proposed fix.
        
        Args:
            issue: The original issue
            proposed_patch: The proposed fix
            context: Additional context information
            
        Returns:
            Comprehensive risk assessment
        """
        risk_factors = []
        
        # Analyze different risk dimensions
        risk_factors.append(self._assess_code_complexity_risk(issue, proposed_patch, context))
        risk_factors.append(self._assess_dependency_risk(issue, proposed_patch, context))
        risk_factors.append(self._assess_performance_risk(issue, proposed_patch, context))
        risk_factors.append(self._assess_security_risk(issue, proposed_patch, context))
        risk_factors.append(self._assess_maintainability_risk(issue, proposed_patch, context))
        
        # Calculate overall risk
        overall_risk, confidence_score = self._calculate_overall_risk(risk_factors)
        
        # Generate impact analysis
        impact_analysis = self._analyze_impact(issue, proposed_patch, risk_factors)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors, overall_risk)
        
        # Assess rollback difficulty
        rollback_difficulty = self._assess_rollback_difficulty(proposed_patch)
        
        return RiskAssessment(
            overall_risk=overall_risk,
            confidence_score=confidence_score,
            risk_factors=risk_factors,
            impact_analysis=impact_analysis,
            recommendations=recommendations,
            estimated_time=proposed_patch.estimated_time,
            rollback_difficulty=rollback_difficulty
        )
    
    def _assess_code_complexity_risk(self, issue: Issue, patch: Patch, context: Optional[Dict[str, Any]]) -> RiskFactor:
        """Assess risk related to code complexity changes."""
        score = 0.0
        
        # Check number of files affected
        files_affected = len(set(action.target for action in patch.actions))
        if files_affected > 5:
            score += 0.4
        elif files_affected > 2:
            score += 0.2
        
        # Check lines of code changes
        total_changes = sum(len(action.content.split('\n')) if action.content else 0 
                          for action in patch.actions if action.type == "modify_file")
        if total_changes > 100:
            score += 0.3
        elif total_changes > 50:
            score += 0.2
        elif total_changes > 20:
            score += 0.1
        
        # Check for new dependencies
        for action in patch.actions:
            if action.content and ('import ' in action.content or 'from ' in action.content):
                score += 0.1
        
        return RiskFactor(
            name="code_complexity",
            weight=self.risk_weights['code_complexity'],
            score=min(score, 1.0),
            description=f"Code complexity impact: {files_affected} files, ~{total_changes} lines changed",
            mitigation="Review changes carefully, add comprehensive tests"
        )
    
    def _assess_dependency_risk(self, issue: Issue, patch: Patch, context: Optional[Dict[str, Any]]) -> RiskFactor:
        """Assess risk related to dependency changes."""
        score = 0.0
        
        # Check for circular dependency fixes (high risk but necessary)
        if issue.kind in ["circular_import", "circular_dependency"]:
            score += 0.6
        
        # Check for import changes
        import_changes = 0
        for action in patch.actions:
            if action.content:
                import_changes += action.content.count('import ')
                import_changes += action.content.count('from ')
        
        if import_changes > 10:
            score += 0.4
        elif import_changes > 5:
            score += 0.2
        elif import_changes > 0:
            score += 0.1
        
        # Check for new file creation (can affect imports)
        new_files = sum(1 for action in patch.actions if action.type == "create_file")
        if new_files > 0:
            score += 0.1 * new_files
        
        return RiskFactor(
            name="dependency_impact",
            weight=self.risk_weights['dependency_impact'],
            score=min(score, 1.0),
            description=f"Dependency impact: {import_changes} import changes, {new_files} new files",
            mitigation="Test all dependent modules, update documentation"
        )
    
    def _assess_performance_risk(self, issue: Issue, patch: Patch, context: Optional[Dict[str, Any]]) -> RiskFactor:
        """Assess potential performance impact."""
        score = 0.0
        
        # Check for generated implementations (may be inefficient)
        if issue.kind in ["phantom_function", "placeholder"]:
            score += 0.3
        
        # Check for complex operations in generated code
        for action in patch.actions:
            if action.content:
                content = action.content.lower()
                
                # Check for potentially expensive operations
                if 'while' in content or 'for' in content:
                    score += 0.1
                if 'recursive' in content or 'recursion' in content:
                    score += 0.2
                if 'import' in content and ('requests' in content or 'urllib' in content):
                    score += 0.1  # Network operations
                if 'open(' in content or 'file' in content:
                    score += 0.1  # File I/O
        
        # Check patch complexity
        if patch.estimated_time > 30:  # seconds
            score += 0.2
        
        return RiskFactor(
            name="performance_impact",
            weight=self.risk_weights['performance_impact'],
            score=min(score, 1.0),
            description="Performance impact analysis based on operations and complexity",
            mitigation="Add performance tests, profile critical paths"
        )
    
    def _assess_security_risk(self, issue: Issue, patch: Patch, context: Optional[Dict[str, Any]]) -> RiskFactor:
        """Assess security implications of the fix."""
        score = 0.0
        
        # Generated code has higher security risk
        if issue.kind in ["phantom_function", "placeholder", "todo_implementation"]:
            score += 0.5
        
        # Check for security-sensitive patterns
        for action in patch.actions:
            if action.content:
                content = action.content.lower()
                
                # Security risk indicators
                security_patterns = [
                    'password', 'secret', 'token', 'key', 'auth',
                    'sql', 'query', 'database', 'db',
                    'eval(', 'exec(', 'subprocess', 'os.system',
                    'pickle', 'marshal', 'shelve',
                    'input(', 'raw_input(',
                    'open(', 'file(', 'write',
                    'requests', 'urllib', 'http',
                ]
                
                for pattern in security_patterns:
                    if pattern in content:
                        score += 0.1
        
        # Import changes can introduce vulnerabilities
        if issue.kind in ["missing_import", "import_anxiety"]:
            score += 0.2
        
        return RiskFactor(
            name="security_impact",
            weight=self.risk_weights['security_impact'],
            score=min(score, 1.0),
            description="Security risk assessment based on code patterns and operations",
            mitigation="Security review required, add input validation, audit dependencies"
        )
    
    def _assess_maintainability_risk(self, issue: Issue, patch: Patch, context: Optional[Dict[str, Any]]) -> RiskFactor:
        """Assess impact on code maintainability."""
        score = 0.0
        
        # Generated implementations may be harder to maintain
        if issue.kind in ["phantom_function", "placeholder"]:
            score += 0.3
        
        # Check for TODO comments in generated code
        todo_count = 0
        for action in patch.actions:
            if action.content:
                todo_count += action.content.count('TODO')
                todo_count += action.content.count('FIXME')
        
        if todo_count > 5:
            score += 0.3
        elif todo_count > 2:
            score += 0.2
        elif todo_count > 0:
            score += 0.1
        
        # Complex actions are harder to maintain
        if len(patch.actions) > 10:
            score += 0.2
        elif len(patch.actions) > 5:
            score += 0.1
        
        return RiskFactor(
            name="maintainability",
            weight=self.risk_weights['maintainability'],
            score=min(score, 1.0),
            description=f"Maintainability impact: {todo_count} TODOs, {len(patch.actions)} actions",
            mitigation="Add comprehensive documentation, review generated code"
        )
    
    def _calculate_overall_risk(self, risk_factors: List[RiskFactor]) -> Tuple[RiskLevel, float]:
        """Calculate overall risk level and confidence score."""
        # Weighted average of risk scores
        total_weighted_score = sum(factor.score * factor.weight for factor in risk_factors)
        total_weight = sum(factor.weight for factor in risk_factors)
        
        if total_weight == 0:
            weighted_average = 0.0
        else:
            weighted_average = total_weighted_score / total_weight
        
        # Convert to risk level
        if weighted_average < 0.3:
            risk_level = RiskLevel.LOW
        elif weighted_average < 0.6:
            risk_level = RiskLevel.MEDIUM
        elif weighted_average < 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        # Confidence is inverse of variance in risk scores
        scores = [factor.score for factor in risk_factors]
        if len(scores) > 1:
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            confidence = max(0.1, 1.0 - variance)
        else:
            confidence = 0.8  # Default confidence
        
        return risk_level, confidence
    
    def _analyze_impact(self, issue: Issue, patch: Patch, risk_factors: List[RiskFactor]) -> Dict[str, Any]:
        """Analyze the broader impact of the fix."""
        return {
            "files_affected": len(set(action.target for action in patch.actions)),
            "lines_changed": sum(len(action.content.split('\n')) if action.content else 0 
                               for action in patch.actions),
            "new_files_created": sum(1 for action in patch.actions if action.type == "create_file"),
            "imports_modified": any('import' in (action.content or '') for action in patch.actions),
            "risk_distribution": {factor.name: factor.score for factor in risk_factors},
            "estimated_review_time": self._estimate_review_time(patch, risk_factors),
            "testing_requirements": self._determine_testing_requirements(issue, risk_factors)
        }
    
    def _generate_recommendations(self, risk_factors: List[RiskFactor], overall_risk: RiskLevel) -> List[str]:
        """Generate recommendations based on risk assessment."""
        recommendations = []
        
        # General recommendations based on overall risk
        if overall_risk == RiskLevel.CRITICAL:
            recommendations.extend([
                "🚨 CRITICAL: This fix requires extensive review and testing",
                "Consider breaking into smaller, incremental changes",
                "Implement comprehensive rollback plan",
                "Schedule deployment during maintenance window"
            ])
        elif overall_risk == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️  HIGH RISK: Thorough testing and review required",
                "Deploy to staging environment first",
                "Have rollback plan ready"
            ])
        elif overall_risk == RiskLevel.MEDIUM:
            recommendations.extend([
                "⚡ MEDIUM RISK: Standard review and testing procedures",
                "Monitor deployment closely"
            ])
        else:
            recommendations.append("✅ LOW RISK: Standard deployment procedures")
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            if factor.score > 0.7:
                recommendations.append(f"📋 {factor.name.title()}: {factor.mitigation}")
        
        # Add specific recommendations based on issue type
        if any(factor.name == "security_impact" and factor.score > 0.5 for factor in risk_factors):
            recommendations.extend([
                "🔒 Security review required before deployment",
                "Run security scanning tools",
                "Validate input sanitization"
            ])
        
        return recommendations
    
    def _assess_rollback_difficulty(self, patch: Patch) -> RiskLevel:
        """Assess how difficult it would be to rollback this fix."""
        difficulty_score = 0.0
        
        # File modifications are easier to rollback than creations
        creates = sum(1 for action in patch.actions if action.type == "create_file")
        modifies = sum(1 for action in patch.actions if action.type == "modify_file")
        
        if creates > 3:
            difficulty_score += 0.3
        elif creates > 0:
            difficulty_score += 0.1
        
        if modifies > 10:
            difficulty_score += 0.4
        elif modifies > 5:
            difficulty_score += 0.2
        
        # Database or schema changes are harder to rollback
        for action in patch.actions:
            if action.content and ('CREATE TABLE' in action.content.upper() or 
                                 'ALTER TABLE' in action.content.upper()):
                difficulty_score += 0.5
        
        # Convert to risk level
        if difficulty_score < 0.3:
            return RiskLevel.LOW
        elif difficulty_score < 0.6:
            return RiskLevel.MEDIUM
        elif difficulty_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _estimate_review_time(self, patch: Patch, risk_factors: List[RiskFactor]) -> float:
        """Estimate time required for code review in hours."""
        base_time = 0.5  # 30 minutes base
        
        # Add time based on complexity
        lines_changed = sum(len(action.content.split('\n')) if action.content else 0 
                          for action in patch.actions)
        base_time += lines_changed / 100  # 1 hour per 100 lines
        
        # Add time based on risk factors
        high_risk_factors = sum(1 for factor in risk_factors if factor.score > 0.7)
        base_time += high_risk_factors * 0.5  # 30 minutes per high-risk factor
        
        return round(base_time, 1)
    
    def _determine_testing_requirements(self, issue: Issue, risk_factors: List[RiskFactor]) -> List[str]:
        """Determine what types of testing are required."""
        requirements = ["Unit tests"]
        
        # Add requirements based on risk factors
        for factor in risk_factors:
            if factor.name == "performance_impact" and factor.score > 0.5:
                requirements.append("Performance tests")
            elif factor.name == "security_impact" and factor.score > 0.5:
                requirements.append("Security tests")
            elif factor.name == "dependency_impact" and factor.score > 0.5:
                requirements.append("Integration tests")
        
        # Add requirements based on issue type
        if issue.kind in ["circular_import", "circular_dependency"]:
            requirements.append("Import resolution tests")
        
        if issue.kind in ["phantom_function", "placeholder"]:
            requirements.append("Generated code validation")
        
        return list(set(requirements))  # Remove duplicates
    
    def _generate_risk_analysis_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to create risk analysis documentation."""
        actions = []
        
        # Create a placeholder patch to analyze
        placeholder_patch = Patch(
            actions=[],
            description="Risk analysis placeholder",
            confidence=0.5,
            risk_level=RiskLevel.MEDIUM,
            estimated_time=10.0
        )
        
        # Perform risk analysis
        assessment = self.analyze_fix_risk(issue, placeholder_patch, context)
        
        # Generate risk analysis report
        report_content = self._generate_risk_report(issue, assessment)
        
        actions.append(Action(
            type="create_file",
            target=f"risk_analysis_{issue.kind}_{issue.file.replace('/', '_').replace('.py', '')}.md",
            content=report_content,
            metadata={
                "type": "risk_analysis",
                "overall_risk": assessment.overall_risk.name,
                "confidence": assessment.confidence_score
            }
        ))
        
        return actions
    
    def _generate_risk_report(self, issue: Issue, assessment: RiskAssessment) -> str:
        """Generate a comprehensive risk analysis report."""
        recommendations_text = '\n'.join(f"- {rec}" for rec in assessment.recommendations)
        
        risk_factors_text = ""
        for factor in assessment.risk_factors:
            risk_factors_text += f"""
### {factor.name.replace('_', ' ').title()}
- **Score**: {factor.score:.2f} (Weight: {factor.weight:.2f})
- **Description**: {factor.description}
- **Mitigation**: {factor.mitigation}
"""
        
        return f"""# Risk Analysis Report

## Issue Information
- **Type**: {issue.kind}
- **File**: {issue.file}
- **Symbol**: {issue.symbol or 'N/A'}
- **Message**: {issue.message or 'N/A'}

## Overall Assessment
- **Risk Level**: {assessment.overall_risk.name}
- **Confidence Score**: {assessment.confidence_score:.2f}
- **Estimated Review Time**: {assessment.impact_analysis.get('estimated_review_time', 'Unknown')} hours
- **Rollback Difficulty**: {assessment.rollback_difficulty.name}

## Risk Factors
{risk_factors_text}

## Impact Analysis
- **Files Affected**: {assessment.impact_analysis.get('files_affected', 0)}
- **Lines Changed**: {assessment.impact_analysis.get('lines_changed', 0)}
- **New Files Created**: {assessment.impact_analysis.get('new_files_created', 0)}
- **Imports Modified**: {assessment.impact_analysis.get('imports_modified', False)}

## Testing Requirements
{chr(10).join(f"- {req}" for req in assessment.impact_analysis.get('testing_requirements', []))}

## Recommendations
{recommendations_text}

---
*Generated by TailChasingFixer RiskAnalysisStrategy*
"""


class ConfidenceScorer(BaseFixStrategy):
    """
    Strategy for scoring confidence in fix quality.
    
    Evaluates factors that contribute to fix reliability.
    """
    
    def __init__(self):
        super().__init__("ConfidenceScorer")
    
    def can_handle(self, issue: Issue) -> bool:
        """Can score confidence for any issue type."""
        return True
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """ConfidenceScorer doesn't propose fixes, only scores them."""
        return None
    
    def calculate_confidence_score(self, issue: Issue, patch: Patch, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate confidence score for a proposed fix.
        
        Returns a score between 0.0 and 1.0 indicating confidence in the fix.
        """
        factors = {}
        
        # Issue type confidence
        factors['issue_type'] = self._score_issue_type_confidence(issue)
        
        # Patch quality confidence  
        factors['patch_quality'] = self._score_patch_quality(patch)
        
        # Context availability confidence
        factors['context_quality'] = self._score_context_quality(issue, context)
        
        # Evidence quality confidence
        factors['evidence_quality'] = self._score_evidence_quality(issue)
        
        # Calculate weighted confidence
        return self.calculate_confidence(factors, {
            'issue_type': 0.3,
            'patch_quality': 0.4,
            'context_quality': 0.2,
            'evidence_quality': 0.1
        })
    
    def _score_issue_type_confidence(self, issue: Issue) -> float:
        """Score confidence based on issue type."""
        # Some issue types are more reliably fixable
        confidence_map = {
            'missing_import': 0.9,
            'missing_symbol': 0.85,
            'unused_import': 0.95,
            'import_anxiety': 0.8,
            'duplicate_function': 0.7,
            'semantic_duplicate_function': 0.6,
            'circular_import': 0.5,
            'phantom_function': 0.4,
            'placeholder': 0.3,
            'todo_implementation': 0.3,
        }
        
        return confidence_map.get(issue.kind, 0.5)
    
    def _score_patch_quality(self, patch: Patch) -> float:
        """Score confidence based on patch characteristics."""
        score = 0.5  # Base score
        
        # Well-structured patches get higher scores
        if patch.description and len(patch.description) > 10:
            score += 0.1
        
        if patch.validation_tests:
            score += 0.2
        
        if patch.rollback_plan:
            score += 0.1
        
        # Complex patches are less reliable
        if len(patch.actions) > 10:
            score -= 0.2
        elif len(patch.actions) > 5:
            score -= 0.1
        
        # Risk level affects confidence
        if patch.risk_level == RiskLevel.LOW:
            score += 0.1
        elif patch.risk_level == RiskLevel.HIGH:
            score -= 0.1
        elif patch.risk_level == RiskLevel.CRITICAL:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_context_quality(self, issue: Issue, context: Optional[Dict[str, Any]]) -> float:
        """Score confidence based on available context."""
        if not context:
            return 0.3
        
        score = 0.5
        
        # More context information increases confidence
        if 'source_lines' in context:
            score += 0.2
        
        if 'ast' in context:
            score += 0.2
        
        if 'symbol_table' in context:
            score += 0.1
        
        return min(1.0, score)
    
    def _score_evidence_quality(self, issue: Issue) -> float:
        """Score confidence based on issue evidence quality."""
        if not issue.evidence:
            return 0.2
        
        score = 0.5
        
        # More detailed evidence increases confidence
        if isinstance(issue.evidence, dict):
            if len(issue.evidence) > 3:
                score += 0.3
            elif len(issue.evidence) > 1:
                score += 0.2
            else:
                score += 0.1
        
        return min(1.0, score)