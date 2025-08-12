"""
Pipeline analyzer for CI/CD integration.

Analyzes pull requests to detect new tail-chasing issues, track fixes,
calculate risk trajectories, and identify AI-assistance patterns.
"""

import json
import logging
import subprocess
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import difflib
import re

from ..core.detector import TailChasingDetector
from ..core.issues import Issue
from ..analyzers.explainer import TailChasingExplainer
from ..llm_integration.feedback_generator import FeedbackGenerator

logger = logging.getLogger(__name__)


@dataclass
class PRAnalysis:
    """Analysis results for a pull request."""
    
    pr_number: int
    branch: str
    base_branch: str
    
    # Issue tracking
    new_issues: List[Issue] = field(default_factory=list)
    fixed_issues: List[Issue] = field(default_factory=list)
    existing_issues: List[Issue] = field(default_factory=list)
    
    # Risk metrics
    risk_score_before: float = 0.0
    risk_score_after: float = 0.0
    risk_trajectory: float = 0.0  # Positive = increasing risk
    
    # AI detection
    ai_patterns_detected: List[Dict[str, Any]] = field(default_factory=list)
    ai_confidence: float = 0.0
    ai_assistance_likely: bool = False
    
    # File changes
    files_changed: List[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    
    # Recommendations
    blocking_issues: List[Issue] = field(default_factory=list)
    suggested_fixes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    analyzed_at: datetime = field(default_factory=datetime.now)
    analysis_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pr_number': self.pr_number,
            'branch': self.branch,
            'base_branch': self.base_branch,
            'new_issues': [issue.to_dict() for issue in self.new_issues],
            'fixed_issues': [issue.to_dict() for issue in self.fixed_issues],
            'existing_issues': [issue.to_dict() for issue in self.existing_issues],
            'risk_score_before': self.risk_score_before,
            'risk_score_after': self.risk_score_after,
            'risk_trajectory': self.risk_trajectory,
            'ai_patterns_detected': self.ai_patterns_detected,
            'ai_confidence': self.ai_confidence,
            'ai_assistance_likely': self.ai_assistance_likely,
            'files_changed': self.files_changed,
            'lines_added': self.lines_added,
            'lines_removed': self.lines_removed,
            'blocking_issues': [issue.to_dict() for issue in self.blocking_issues],
            'suggested_fixes': self.suggested_fixes,
            'analyzed_at': self.analyzed_at.isoformat(),
            'analysis_time': self.analysis_time
        }
    
    def should_block_merge(self, threshold: float = 10.0) -> bool:
        """Determine if PR should be blocked from merging."""
        # Block if risk increased significantly
        if self.risk_trajectory > threshold:
            return True
        
        # Block if there are high-severity new issues
        high_severity_new = [i for i in self.new_issues if i.severity >= 4]
        if high_severity_new:
            return True
        
        # Block if there are explicit blocking issues
        if self.blocking_issues:
            return True
        
        return False


class PipelineAnalyzer:
    """
    Analyze pull requests for tail-chasing patterns.
    
    Provides comprehensive PR analysis including issue detection,
    risk assessment, and AI-assistance pattern identification.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline analyzer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        self.detector = TailChasingDetector(config)
        self.explainer = TailChasingExplainer(config)
        self.feedback_generator = FeedbackGenerator(config)
        
        # Configuration
        self.risk_threshold = self.config.get('risk_threshold', 10.0)
        self.ai_detection_enabled = self.config.get('ai_detection_enabled', True)
        self.generate_fixes = self.config.get('generate_fixes', True)
        self.cache_results = self.config.get('cache_results', True)
        
        # Cache directory
        if self.cache_results:
            self.cache_dir = Path(self.config.get('cache_dir', '.tailchasing_cache'))
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # AI pattern detection patterns
        self.ai_patterns = self._initialize_ai_patterns()
    
    def analyze_pr(
        self,
        pr_number: int,
        branch: str,
        base_branch: str = 'main',
        repo_path: Optional[Path] = None
    ) -> PRAnalysis:
        """
        Analyze a pull request for tail-chasing patterns.
        
        Args:
            pr_number: PR number
            branch: PR branch name
            base_branch: Base branch name
            repo_path: Repository path (defaults to current directory)
            
        Returns:
            PRAnalysis object with comprehensive results
        """
        start_time = datetime.now()
        repo_path = repo_path or Path.cwd()
        
        analysis = PRAnalysis(
            pr_number=pr_number,
            branch=branch,
            base_branch=base_branch
        )
        
        try:
            # Get changed files
            analysis.files_changed = self._get_changed_files(
                branch, base_branch, repo_path
            )
            
            # Get diff statistics
            stats = self._get_diff_stats(branch, base_branch, repo_path)
            analysis.lines_added = stats['added']
            analysis.lines_removed = stats['removed']
            
            # Analyze base branch
            logger.info(f"Analyzing base branch: {base_branch}")
            base_issues = self._analyze_branch(base_branch, repo_path)
            analysis.risk_score_before = self._calculate_risk_score(base_issues)
            
            # Analyze PR branch
            logger.info(f"Analyzing PR branch: {branch}")
            pr_issues = self._analyze_branch(branch, repo_path)
            analysis.risk_score_after = self._calculate_risk_score(pr_issues)
            
            # Calculate risk trajectory
            analysis.risk_trajectory = (
                analysis.risk_score_after - analysis.risk_score_before
            )
            
            # Categorize issues
            self._categorize_issues(analysis, base_issues, pr_issues)
            
            # Detect AI patterns if enabled
            if self.ai_detection_enabled:
                self._detect_ai_patterns(analysis, repo_path)
            
            # Identify blocking issues
            analysis.blocking_issues = self._identify_blocking_issues(analysis)
            
            # Generate fix suggestions if enabled
            if self.generate_fixes and analysis.new_issues:
                analysis.suggested_fixes = self._generate_fix_suggestions(
                    analysis.new_issues
                )
            
            # Cache results if enabled
            if self.cache_results:
                self._cache_analysis(analysis)
            
        except Exception as e:
            logger.error(f"Error analyzing PR {pr_number}: {e}")
            raise
        
        finally:
            analysis.analysis_time = (datetime.now() - start_time).total_seconds()
        
        return analysis
    
    def analyze_commit(
        self,
        commit_sha: str,
        repo_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single commit for tail-chasing patterns.
        
        Args:
            commit_sha: Commit SHA
            repo_path: Repository path
            
        Returns:
            Analysis results
        """
        repo_path = repo_path or Path.cwd()
        
        # Get commit diff
        diff = self._get_commit_diff(commit_sha, repo_path)
        
        # Analyze patterns in diff
        patterns = self._analyze_diff_patterns(diff)
        
        # Check for AI indicators
        ai_indicators = self._check_commit_message_ai_indicators(
            commit_sha, repo_path
        )
        
        return {
            'commit_sha': commit_sha,
            'patterns_detected': patterns,
            'ai_indicators': ai_indicators,
            'risk_level': self._assess_commit_risk(patterns)
        }
    
    def calculate_trend(
        self,
        analyses: List[PRAnalysis],
        window: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate trends across multiple PRs.
        
        Args:
            analyses: List of PR analyses
            window: Number of PRs to consider
            
        Returns:
            Trend analysis
        """
        if not analyses:
            return {'trend': 'unknown', 'metrics': {}}
        
        # Sort by date
        analyses = sorted(analyses, key=lambda a: a.analyzed_at)
        
        # Use sliding window
        recent = analyses[-window:] if len(analyses) > window else analyses
        
        # Calculate metrics
        risk_scores = [a.risk_score_after for a in recent]
        ai_confidence = [a.ai_confidence for a in recent]
        new_issue_counts = [len(a.new_issues) for a in recent]
        
        # Determine trend
        if len(risk_scores) >= 3:
            # Simple linear regression for trend
            x = list(range(len(risk_scores)))
            x_mean = sum(x) / len(x)
            y_mean = sum(risk_scores) / len(risk_scores)
            
            numerator = sum((x[i] - x_mean) * (risk_scores[i] - y_mean) 
                          for i in range(len(x)))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
            
            slope = numerator / denominator if denominator != 0 else 0
            
            if slope > 0.5:
                trend = 'worsening'
            elif slope < -0.5:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'metrics': {
                'avg_risk_score': sum(risk_scores) / len(risk_scores) if risk_scores else 0,
                'avg_ai_confidence': sum(ai_confidence) / len(ai_confidence) if ai_confidence else 0,
                'avg_new_issues': sum(new_issue_counts) / len(new_issue_counts) if new_issue_counts else 0,
                'total_prs_analyzed': len(recent)
            }
        }
    
    def _initialize_ai_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize patterns for detecting AI assistance."""
        return {
            'comment_patterns': [
                re.compile(r'#\s*TODO:?\s*implement', re.IGNORECASE),
                re.compile(r'#\s*FIXME:?\s*later', re.IGNORECASE),
                re.compile(r'#\s*Note:?\s*placeholder', re.IGNORECASE),
                re.compile(r'#\s*This (function|method|class) is not implemented', re.IGNORECASE),
                re.compile(r'#\s*Helper (function|method) for', re.IGNORECASE),
                re.compile(r'#\s*Utility (function|method) to', re.IGNORECASE),
            ],
            'code_patterns': [
                re.compile(r'def \w+\([^)]*\):\s*pass'),  # Empty functions
                re.compile(r'def \w+\([^)]*\):\s*raise NotImplementedError'),  # Stub functions
                re.compile(r'def \w+_helper\('),  # Helper function pattern
                re.compile(r'def \w+_util\('),  # Utility function pattern
                re.compile(r'import \*'),  # Wildcard imports
                re.compile(r'from .+ import \*'),  # Wildcard from imports
            ],
            'naming_patterns': [
                re.compile(r'(get|fetch|retrieve|load)_\w+_data'),  # Repetitive data getters
                re.compile(r'process_\w+_(v\d+|alt|new|old)'),  # Versioned functions
                re.compile(r'\w+_(handler|processor|manager)'),  # Generic handlers
            ],
            'structure_patterns': [
                re.compile(r'class \w+Base[\w]*:'),  # Base classes
                re.compile(r'class \w+Abstract[\w]*:'),  # Abstract classes
                re.compile(r'class \w+Interface[\w]*:'),  # Interface classes
            ]
        }
    
    def _get_changed_files(
        self,
        branch: str,
        base_branch: str,
        repo_path: Path
    ) -> List[str]:
        """Get list of changed files between branches."""
        try:
            cmd = [
                'git', 'diff', '--name-only',
                f'{base_branch}...{branch}'
            ]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = result.stdout.strip().split('\n')
            # Filter Python files
            return [f for f in files if f.endswith('.py') and f]
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get changed files: {e}")
            return []
    
    def _get_diff_stats(
        self,
        branch: str,
        base_branch: str,
        repo_path: Path
    ) -> Dict[str, int]:
        """Get diff statistics between branches."""
        try:
            cmd = [
                'git', 'diff', '--numstat',
                f'{base_branch}...{branch}'
            ]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            added = 0
            removed = 0
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            added += int(parts[0])
                            removed += int(parts[1])
                        except ValueError:
                            continue
            
            return {'added': added, 'removed': removed}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get diff stats: {e}")
            return {'added': 0, 'removed': 0}
    
    def _analyze_branch(
        self,
        branch: str,
        repo_path: Path
    ) -> List[Issue]:
        """Analyze a specific branch for issues."""
        try:
            # Checkout branch
            subprocess.run(
                ['git', 'checkout', branch],
                cwd=repo_path,
                capture_output=True,
                check=True
            )
            
            # Run detector
            issues = self.detector.detect(repo_path)
            
            return issues
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to analyze branch {branch}: {e}")
            return []
    
    def _calculate_risk_score(self, issues: List[Issue]) -> float:
        """Calculate overall risk score from issues."""
        if not issues:
            return 0.0
        
        # Weight by severity
        weights = {1: 1, 2: 2, 3: 5, 4: 10, 5: 20}
        
        total_score = sum(
            weights.get(issue.severity, 1) for issue in issues
        )
        
        # Normalize (somewhat arbitrary, adjust as needed)
        return min(total_score, 100.0)
    
    def _categorize_issues(
        self,
        analysis: PRAnalysis,
        base_issues: List[Issue],
        pr_issues: List[Issue]
    ):
        """Categorize issues as new, fixed, or existing."""
        # Create issue signatures for comparison
        def issue_signature(issue: Issue) -> str:
            return f"{issue.kind}:{issue.file}:{issue.line}:{issue.symbol}"
        
        base_sigs = {issue_signature(i): i for i in base_issues}
        pr_sigs = {issue_signature(i): i for i in pr_issues}
        
        # Find new issues
        for sig, issue in pr_sigs.items():
            if sig not in base_sigs:
                analysis.new_issues.append(issue)
            else:
                analysis.existing_issues.append(issue)
        
        # Find fixed issues
        for sig, issue in base_sigs.items():
            if sig not in pr_sigs:
                analysis.fixed_issues.append(issue)
    
    def _detect_ai_patterns(
        self,
        analysis: PRAnalysis,
        repo_path: Path
    ):
        """Detect patterns suggesting AI assistance."""
        patterns_found = []
        confidence_scores = []
        
        for file_path in analysis.files_changed:
            full_path = repo_path / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                
                # Check comment patterns
                for pattern in self.ai_patterns['comment_patterns']:
                    matches = pattern.findall(content)
                    if matches:
                        patterns_found.append({
                            'type': 'comment',
                            'pattern': pattern.pattern,
                            'file': file_path,
                            'matches': len(matches)
                        })
                        confidence_scores.append(0.1 * len(matches))
                
                # Check code patterns
                for pattern in self.ai_patterns['code_patterns']:
                    matches = pattern.findall(content)
                    if matches:
                        patterns_found.append({
                            'type': 'code',
                            'pattern': pattern.pattern,
                            'file': file_path,
                            'matches': len(matches)
                        })
                        confidence_scores.append(0.15 * len(matches))
                
                # Check naming patterns
                for pattern in self.ai_patterns['naming_patterns']:
                    matches = pattern.findall(content)
                    if matches:
                        patterns_found.append({
                            'type': 'naming',
                            'pattern': pattern.pattern,
                            'file': file_path,
                            'matches': len(matches)
                        })
                        confidence_scores.append(0.2 * len(matches))
                
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        # Additional indicators
        if len(analysis.new_issues) > 5:
            confidence_scores.append(0.3)
            patterns_found.append({
                'type': 'issue_count',
                'description': f'High number of new issues: {len(analysis.new_issues)}'
            })
        
        # Check for repetitive function creation
        duplicate_issues = [i for i in analysis.new_issues 
                          if 'duplicate' in i.kind.lower()]
        if duplicate_issues:
            confidence_scores.append(0.4 * len(duplicate_issues))
            patterns_found.append({
                'type': 'duplicates',
                'description': f'Multiple duplicate functions: {len(duplicate_issues)}'
            })
        
        # Calculate overall confidence
        analysis.ai_patterns_detected = patterns_found
        analysis.ai_confidence = min(sum(confidence_scores), 1.0)
        analysis.ai_assistance_likely = analysis.ai_confidence > 0.5
    
    def _identify_blocking_issues(
        self,
        analysis: PRAnalysis
    ) -> List[Issue]:
        """Identify issues that should block merge."""
        blocking = []
        
        for issue in analysis.new_issues:
            # High severity issues
            if issue.severity >= 4:
                blocking.append(issue)
            
            # Circular imports
            elif issue.kind == 'circular_import':
                blocking.append(issue)
            
            # Large hallucination cascades
            elif issue.kind == 'hallucination_cascade':
                if hasattr(issue, 'evidence'):
                    components = issue.evidence.get('components', [])
                    if len(components) > 5:
                        blocking.append(issue)
        
        return blocking
    
    def _generate_fix_suggestions(
        self,
        issues: List[Issue]
    ) -> List[Dict[str, Any]]:
        """Generate fix suggestions for issues."""
        suggestions = []
        
        for issue in issues[:10]:  # Limit to top 10
            explanation = self.explainer.explain_issue_enhanced(issue)
            
            suggestion = {
                'issue_type': issue.kind,
                'file': issue.file,
                'line': issue.line,
                'summary': explanation.summary,
                'remediation_steps': explanation.remediation_steps,
                'priority': 'high' if issue.severity >= 4 else 'medium'
            }
            
            # Add specific fix code if available
            if issue.suggestions:
                suggestion['fix_code'] = issue.suggestions[0]
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _get_commit_diff(
        self,
        commit_sha: str,
        repo_path: Path
    ) -> str:
        """Get diff for a specific commit."""
        try:
            cmd = ['git', 'diff', f'{commit_sha}^', commit_sha]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError:
            return ""
    
    def _analyze_diff_patterns(self, diff: str) -> List[Dict[str, Any]]:
        """Analyze patterns in a diff."""
        patterns = []
        
        # Check for duplicate function additions
        added_functions = re.findall(r'\+def (\w+)\(', diff)
        if len(added_functions) != len(set(added_functions)):
            patterns.append({
                'type': 'duplicate_functions',
                'description': 'Multiple functions with same name added'
            })
        
        # Check for stub functions
        stub_pattern = re.compile(r'\+def \w+\([^)]*\):\s*\+\s*pass')
        if stub_pattern.search(diff):
            patterns.append({
                'type': 'stub_functions',
                'description': 'Empty stub functions added'
            })
        
        return patterns
    
    def _check_commit_message_ai_indicators(
        self,
        commit_sha: str,
        repo_path: Path
    ) -> List[str]:
        """Check commit message for AI indicators."""
        indicators = []
        
        try:
            cmd = ['git', 'log', '--format=%B', '-n', '1', commit_sha]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            message = result.stdout.lower()
            
            # Check for AI-related keywords
            ai_keywords = [
                'ai-generated', 'auto-generated', 'copilot',
                'assistant', 'suggested', 'automated fix'
            ]
            
            for keyword in ai_keywords:
                if keyword in message:
                    indicators.append(f"Commit message contains '{keyword}'")
            
        except subprocess.CalledProcessError:
            pass
        
        return indicators
    
    def _assess_commit_risk(
        self,
        patterns: List[Dict[str, Any]]
    ) -> str:
        """Assess risk level of a commit."""
        if not patterns:
            return 'low'
        elif len(patterns) <= 2:
            return 'medium'
        else:
            return 'high'
    
    def _cache_analysis(self, analysis: PRAnalysis):
        """Cache analysis results."""
        if not self.cache_results:
            return
        
        cache_file = self.cache_dir / f"pr_{analysis.pr_number}_{analysis.analyzed_at.timestamp()}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(analysis.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")
    
    def load_cached_analyses(
        self,
        pr_number: Optional[int] = None
    ) -> List[PRAnalysis]:
        """Load cached analyses."""
        if not self.cache_results or not self.cache_dir.exists():
            return []
        
        analyses = []
        
        pattern = f"pr_{pr_number}_*.json" if pr_number else "pr_*.json"
        
        for cache_file in self.cache_dir.glob(pattern):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Note: This is simplified, would need proper deserialization
                    analyses.append(data)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")