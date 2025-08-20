"""
GitHub Actions integration for tail-chasing detection.

Provides GitHub Actions workflow integration with PR comments,
merge blocking, fix suggestions, and automated fix commits.
"""

import json
import logging
import os
import hashlib
import hmac
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import subprocess

from .pipeline_analyzer import PipelineAnalyzer, PRAnalysis
from ..fixers.suggestion_generator import SuggestionGenerator
from ..fixers.fix_applier import FixApplier
from ..llm_integration.feedback_generator import FeedbackGenerator

logger = logging.getLogger(__name__)


@dataclass
class GitHubContext:
    """GitHub Actions context information."""
    
    repository: str
    pr_number: int
    head_ref: str  # PR branch
    base_ref: str  # Base branch
    sha: str
    actor: str
    event_name: str
    workflow: str
    run_id: str
    run_number: str
    
    @classmethod
    def from_env(cls) -> 'GitHubContext':
        """Create from GitHub Actions environment variables."""
        return cls(
            repository=os.getenv('GITHUB_REPOSITORY', ''),
            pr_number=int(os.getenv('GITHUB_PR_NUMBER', '0')),
            head_ref=os.getenv('GITHUB_HEAD_REF', ''),
            base_ref=os.getenv('GITHUB_BASE_REF', 'main'),
            sha=os.getenv('GITHUB_SHA', ''),
            actor=os.getenv('GITHUB_ACTOR', ''),
            event_name=os.getenv('GITHUB_EVENT_NAME', ''),
            workflow=os.getenv('GITHUB_WORKFLOW', ''),
            run_id=os.getenv('GITHUB_RUN_ID', ''),
            run_number=os.getenv('GITHUB_RUN_NUMBER', '')
        )


class GitHubIntegration:
    """
    GitHub Actions integration for tail-chasing detection.
    
    Provides PR commenting, merge blocking, and automated fixes.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GitHub integration.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Initialize components
        self.analyzer = PipelineAnalyzer(config)
        self.suggestion_generator = SuggestionGenerator(config)
        self.fix_applier = FixApplier(config)
        self.feedback_generator = FeedbackGenerator(config)
        
        # GitHub configuration
        self.github_token = os.getenv('GITHUB_TOKEN') or self.config.get('github_token')
        self.repository = self.config.get('repository') or os.getenv('GITHUB_REPOSITORY')
        
        # Behavior configuration
        self.auto_comment = self.config.get('auto_comment', True)
        self.block_on_risk = self.config.get('block_on_risk', True)
        self.risk_threshold = self.config.get('risk_threshold', 10.0)
        self.auto_fix = self.config.get('auto_fix', False)
        self.detailed_report = self.config.get('detailed_report', True)
        
        # Comment templates
        self.comment_templates = self._initialize_comment_templates()
    
    def run_pr_check(
        self,
        context: Optional[GitHubContext] = None
    ) -> Dict[str, Any]:
        """
        Run PR check workflow.
        
        Args:
            context: GitHub context (defaults to environment)
            
        Returns:
            Check results
        """
        # Get context
        if not context:
            context = GitHubContext.from_env()
        
        logger.info(f"Running PR check for {context.repository}#{context.pr_number}")
        
        # Analyze PR
        analysis = self.analyzer.analyze_pr(
            pr_number=context.pr_number,
            branch=context.head_ref,
            base_branch=context.base_ref
        )
        
        # Generate report
        report = self._generate_pr_report(analysis)
        
        # Post comment if enabled
        if self.auto_comment:
            self._post_pr_comment(context, report, analysis)
        
        # Set check status
        check_status = self._determine_check_status(analysis)
        self._set_check_status(context, check_status, analysis)
        
        # Generate fixes if requested
        fix_results = None
        if self.auto_fix and analysis.new_issues:
            fix_results = self._generate_and_apply_fixes(context, analysis)
        
        # Output for GitHub Actions
        self._set_action_outputs(analysis, check_status)
        
        return {
            'analysis': analysis.to_dict(),
            'check_status': check_status,
            'report': report,
            'fix_results': fix_results
        }
    
    def handle_webhook(
        self,
        event_type: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle GitHub webhook events.
        
        Args:
            event_type: GitHub event type
            payload: Webhook payload
            signature: GitHub signature for validation
            
        Returns:
            Processing results
        """
        # Validate signature if provided
        if signature and not self._validate_webhook_signature(payload, signature):
            return {'error': 'Invalid signature'}
        
        logger.info(f"Handling webhook event: {event_type}")
        
        # Route based on event type
        if event_type == 'pull_request':
            return self._handle_pr_event(payload)
        elif event_type == 'push':
            return self._handle_push_event(payload)
        elif event_type == 'issue_comment':
            return self._handle_comment_event(payload)
        else:
            logger.warning(f"Unhandled event type: {event_type}")
            return {'status': 'ignored', 'reason': 'Unhandled event type'}
    
    def generate_workflow_yaml(self) -> str:
        """
        Generate GitHub Actions workflow YAML.
        
        Returns:
            Workflow YAML content
        """
        workflow = """
name: Tail-Chasing Detection

on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main, develop]

jobs:
  tail-chasing-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Full history for better analysis
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install TailChasingFixer
      run: |
        pip install tailchasing
        # Or install from source:
        # pip install -e .
    
    - name: Run Tail-Chasing Analysis
      id: analysis
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        GITHUB_PR_NUMBER: ${{ github.event.pull_request.number }}
      run: |
        python -m tailchasing.ci.run_check
    
    - name: Comment on PR
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        github-token: ${{ secrets.GITHUB_TOKEN }}
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('tailchasing_report.md', 'utf8');
          
          github.rest.issues.createComment({
            owner: context.repo.owner,
            repo: context.repo.repo,
            issue_number: context.issue.number,
            body: report
          });
    
    - name: Set Check Status
      if: always()
      uses: actions/github-script@v6
      with:
        github-token: ${{ secrets.GITHUB_TOKEN }}
        script: |
          const conclusion = '${{ steps.analysis.outputs.conclusion }}';
          const summary = '${{ steps.analysis.outputs.summary }}';
          
          github.rest.checks.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            name: 'Tail-Chasing Detection',
            head_sha: context.sha,
            status: 'completed',
            conclusion: conclusion,
            output: {
              title: 'Tail-Chasing Analysis',
              summary: summary
            }
          });
"""
        return workflow.strip()
    
    def _initialize_comment_templates(self) -> Dict[str, str]:
        """Initialize comment templates."""
        return {
            'header': "## 🔍 Tail-Chasing Detection Report\n\n",
            
            'summary_clean': (
                "✅ **No tail-chasing patterns detected!**\n\n"
                "Your code changes look clean and well-structured."
            ),
            
            'summary_issues': (
                "⚠️ **Tail-chasing patterns detected**\n\n"
                "Found {new_count} new issue(s) and fixed {fixed_count} issue(s).\n"
                "Risk trajectory: {trajectory:+.1f} ({trend})\n"
            ),
            
            'ai_detection': (
                "\n### 🤖 AI Assistance Detection\n"
                "**Confidence:** {confidence:.0%}\n"
                "**Likely AI-assisted:** {likely}\n"
                "{patterns}\n"
            ),
            
            'blocking': (
                "\n### 🚫 Blocking Issues\n"
                "The following issues must be resolved before merging:\n{issues}\n"
            ),
            
            'suggestions': (
                "\n### 💡 Suggested Fixes\n"
                "{suggestions}\n"
            ),
            
            'details': (
                "\n<details>\n<summary>📊 Detailed Analysis</summary>\n\n"
                "{content}\n"
                "</details>\n"
            ),
            
            'footer': (
                "\n---\n"
                "*Generated by [TailChasingFixer](https://github.com/your-org/tailchasing) "
                "• [Documentation](https://docs.example.com) "
                "• [Report Issue](https://github.com/your-org/tailchasing/issues)*"
            )
        }
    
    def _generate_pr_report(self, analysis: PRAnalysis) -> str:
        """Generate PR report comment."""
        parts = [self.comment_templates['header']]
        
        # Summary
        if not analysis.new_issues and not analysis.existing_issues:
            parts.append(self.comment_templates['summary_clean'])
        else:
            trend = 'increasing' if analysis.risk_trajectory > 0 else 'decreasing'
            summary = self.comment_templates['summary_issues'].format(
                new_count=len(analysis.new_issues),
                fixed_count=len(analysis.fixed_issues),
                trajectory=analysis.risk_trajectory,
                trend=trend
            )
            parts.append(summary)
        
        # Risk metrics
        parts.append(f"\n### 📈 Risk Metrics\n")
        parts.append(f"- **Before:** {analysis.risk_score_before:.1f}/100")
        parts.append(f"- **After:** {analysis.risk_score_after:.1f}/100")
        parts.append(f"- **Change:** {analysis.risk_trajectory:+.1f}")
        
        # AI detection if significant
        if analysis.ai_confidence > 0.3:
            ai_patterns = "\n".join([
                f"- {p['type']}: {p.get('description', p.get('pattern', 'detected'))}"
                for p in analysis.ai_patterns_detected[:5]
            ])
            
            ai_section = self.comment_templates['ai_detection'].format(
                confidence=analysis.ai_confidence,
                likely="Yes" if analysis.ai_assistance_likely else "No",
                patterns=ai_patterns
            )
            parts.append(ai_section)
        
        # New issues
        if analysis.new_issues:
            parts.append(f"\n### 🆕 New Issues ({len(analysis.new_issues)})\n")
            for issue in analysis.new_issues[:10]:  # Limit display
                severity_emoji = self._get_severity_emoji(issue.severity)
                parts.append(
                    f"- {severity_emoji} **{issue.kind}** in `{issue.file}:{issue.line}` "
                    f"- {issue.message[:100]}"
                )
            
            if len(analysis.new_issues) > 10:
                parts.append(f"\n*... and {len(analysis.new_issues) - 10} more*")
        
        # Fixed issues
        if analysis.fixed_issues:
            parts.append(f"\n### ✅ Fixed Issues ({len(analysis.fixed_issues)})\n")
            for issue in analysis.fixed_issues[:5]:
                parts.append(f"- ~~{issue.kind} in `{issue.file}`~~")
        
        # Blocking issues
        if analysis.blocking_issues:
            blocking_list = "\n".join([
                f"1. **{issue.kind}** - {issue.message[:100]}"
                for issue in analysis.blocking_issues
            ])
            parts.append(
                self.comment_templates['blocking'].format(issues=blocking_list)
            )
        
        # Suggestions
        if analysis.suggested_fixes:
            suggestions_text = self._format_suggestions(analysis.suggested_fixes[:5])
            parts.append(
                self.comment_templates['suggestions'].format(
                    suggestions=suggestions_text
                )
            )
        
        # Detailed report if enabled
        if self.detailed_report and (analysis.new_issues or analysis.existing_issues):
            details = self._generate_detailed_report(analysis)
            parts.append(
                self.comment_templates['details'].format(content=details)
            )
        
        # Footer
        parts.append(self.comment_templates['footer'])
        
        return "\n".join(parts)
    
    def _format_suggestions(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format fix suggestions."""
        formatted = []
        
        for i, suggestion in enumerate(suggestions, 1):
            formatted.append(f"\n**{i}. {suggestion['issue_type']}** (`{suggestion['file']}`)")
            formatted.append(f"   {suggestion['summary']}")
            
            if suggestion.get('remediation_steps'):
                for step in suggestion['remediation_steps'][:2]:
                    formatted.append(f"   - {step}")
            
            if suggestion.get('fix_code'):
                formatted.append("   ```python")
                formatted.append(f"   {suggestion['fix_code'][:200]}")
                formatted.append("   ```")
        
        return "\n".join(formatted)
    
    def _generate_detailed_report(self, analysis: PRAnalysis) -> str:
        """Generate detailed analysis report."""
        parts = []
        
        # Issue breakdown by type
        issue_types = {}
        for issue in analysis.new_issues + analysis.existing_issues:
            issue_types.setdefault(issue.kind, []).append(issue)
        
        parts.append("#### Issue Breakdown by Type\n")
        for issue_type, issues in issue_types.items():
            parts.append(f"- **{issue_type}**: {len(issues)} occurrence(s)")
        
        # File risk map
        file_risks = {}
        for issue in analysis.new_issues:
            if issue.file:
                file_risks.setdefault(issue.file, 0)
                file_risks[issue.file] += issue.severity
        
        if file_risks:
            parts.append("\n#### High-Risk Files\n")
            sorted_files = sorted(file_risks.items(), key=lambda x: x[1], reverse=True)
            for file, risk in sorted_files[:5]:
                parts.append(f"- `{file}`: Risk score {risk}")
        
        # Pattern analysis
        if analysis.ai_patterns_detected:
            parts.append("\n#### AI Pattern Analysis\n")
            pattern_types = {}
            for pattern in analysis.ai_patterns_detected:
                pattern_types.setdefault(pattern['type'], 0)
                pattern_types[pattern['type']] += pattern.get('matches', 1)
            
            for ptype, count in pattern_types.items():
                parts.append(f"- {ptype}: {count} match(es)")
        
        return "\n".join(parts)
    
    def _get_severity_emoji(self, severity: int) -> str:
        """Get emoji for severity level."""
        emojis = {
            1: "💚",  # Low
            2: "💛",  # Medium-low
            3: "🟠",  # Medium
            4: "🔴",  # High
            5: "🚨"   # Critical
        }
        return emojis.get(severity, "⚪")
    
    def _post_pr_comment(
        self,
        context: GitHubContext,
        report: str,
        analysis: PRAnalysis
    ):
        """Post comment on PR."""
        if not self.github_token:
            logger.warning("No GitHub token available, skipping comment")
            return
        
        try:
            # Use GitHub CLI if available
            cmd = [
                'gh', 'pr', 'comment', str(context.pr_number),
                '--body', report
            ]
            
            subprocess.run(
                cmd,
                env={**os.environ, 'GH_TOKEN': self.github_token},
                check=True,
                capture_output=True
            )
            
            logger.info(f"Posted comment on PR #{context.pr_number}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to post PR comment: {e}")
        except FileNotFoundError:
            logger.warning("GitHub CLI not found, skipping comment")
    
    def _determine_check_status(
        self,
        analysis: PRAnalysis
    ) -> Dict[str, str]:
        """Determine check status from analysis."""
        if analysis.should_block_merge(self.risk_threshold):
            return {
                'conclusion': 'failure',
                'title': '❌ Tail-Chasing Issues Detected',
                'summary': f'Found {len(analysis.blocking_issues)} blocking issue(s)'
            }
        elif analysis.new_issues:
            return {
                'conclusion': 'neutral',
                'title': '⚠️ Non-blocking Issues Found',
                'summary': f'Found {len(analysis.new_issues)} new issue(s)'
            }
        else:
            return {
                'conclusion': 'success',
                'title': '✅ No Issues Detected',
                'summary': 'Code looks clean!'
            }
    
    def _set_check_status(
        self,
        context: GitHubContext,
        status: Dict[str, str],
        analysis: PRAnalysis
    ):
        """Set GitHub check status."""
        if not self.github_token:
            logger.warning("No GitHub token available, skipping check status")
            return
        
        try:
            # Create check run using GitHub CLI
            cmd = [
                'gh', 'api',
                f'/repos/{context.repository}/check-runs',
                '-X', 'POST',
                '-f', f'name=Tail-Chasing Detection',
                '-f', f'head_sha={context.sha}',
                '-f', 'status=completed',
                '-f', f'conclusion={status["conclusion"]}',
                '-f', f'output[title]={status["title"]}',
                '-f', f'output[summary]={status["summary"]}'
            ]
            
            subprocess.run(
                cmd,
                env={**os.environ, 'GH_TOKEN': self.github_token},
                check=True,
                capture_output=True
            )
            
            logger.info(f"Set check status: {status['conclusion']}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set check status: {e}")
    
    def _set_action_outputs(
        self,
        analysis: PRAnalysis,
        check_status: Dict[str, str]
    ):
        """Set GitHub Actions output variables."""
        # Write to GITHUB_OUTPUT if available
        output_file = os.getenv('GITHUB_OUTPUT')
        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"conclusion={check_status['conclusion']}\n")
                f.write(f"summary={check_status['summary']}\n")
                f.write(f"risk_score={analysis.risk_score_after}\n")
                f.write(f"new_issues={len(analysis.new_issues)}\n")
                f.write(f"should_block={analysis.should_block_merge(self.risk_threshold)}\n")
        
        # Also output to stdout for visibility
        print(f"::set-output name=conclusion::{check_status['conclusion']}")
        print(f"::set-output name=risk_score::{analysis.risk_score_after}")
        
        # Save report to file
        report_file = Path('tailchasing_report.md')
        report = self._generate_pr_report(analysis)
        report_file.write_text(report)
        
        # Save JSON analysis
        analysis_file = Path('tailchasing_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2, default=str)
    
    def _generate_and_apply_fixes(
        self,
        context: GitHubContext,
        analysis: PRAnalysis
    ) -> Dict[str, Any]:
        """Generate and apply automated fixes."""
        if not self.auto_fix:
            return {'status': 'skipped', 'reason': 'Auto-fix disabled'}
        
        fixes_applied = []
        
        for issue in analysis.new_issues[:5]:  # Limit fixes
            if issue.severity < 3:  # Only fix lower severity issues
                suggestions = self.suggestion_generator.generate_suggestions([issue])
                if suggestions:
                    suggestion = suggestions[0]
                    success = self.fix_applier.apply_fix(
                        suggestion,
                        Path(issue.file)
                    )
                    if success:
                        fixes_applied.append({
                            'issue': issue.kind,
                            'file': issue.file,
                            'line': issue.line
                        })
        
        # Create fix commit if any fixes applied
        if fixes_applied:
            self._create_fix_commit(context, fixes_applied)
        
        return {
            'status': 'success',
            'fixes_applied': fixes_applied
        }
    
    def _create_fix_commit(
        self,
        context: GitHubContext,
        fixes: List[Dict[str, Any]]
    ):
        """Create a commit with automated fixes."""
        try:
            # Stage changes
            subprocess.run(['git', 'add', '-A'], check=True)
            
            # Create commit message
            message = "🤖 Auto-fix tail-chasing issues\n\n"
            message += "Fixed issues:\n"
            for fix in fixes:
                message += f"- {fix['issue']} in {fix['file']}\n"
            message += "\nGenerated by TailChasingFixer"
            
            # Commit
            subprocess.run(
                ['git', 'commit', '-m', message],
                check=True
            )
            
            # Push to PR branch
            subprocess.run(
                ['git', 'push', 'origin', context.head_ref],
                check=True
            )
            
            logger.info(f"Created fix commit with {len(fixes)} fixes")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create fix commit: {e}")
    
    def _validate_webhook_signature(
        self,
        payload: Dict[str, Any],
        signature: str
    ) -> bool:
        """Validate GitHub webhook signature."""
        if not self.config.get('webhook_secret'):
            return True  # No secret configured, skip validation
        
        secret = self.config['webhook_secret'].encode()
        payload_bytes = json.dumps(payload).encode()
        
        expected = 'sha256=' + hmac.new(
            secret,
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected, signature)
    
    def _handle_pr_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pull_request webhook event."""
        action = payload.get('action')
        pr = payload.get('pull_request', {})
        
        if action not in ['opened', 'synchronize', 'reopened']:
            return {'status': 'ignored', 'reason': f'Action {action} not handled'}
        
        # Create context from payload
        context = GitHubContext(
            repository=payload.get('repository', {}).get('full_name', ''),
            pr_number=pr.get('number', 0),
            head_ref=pr.get('head', {}).get('ref', ''),
            base_ref=pr.get('base', {}).get('ref', 'main'),
            sha=pr.get('head', {}).get('sha', ''),
            actor=payload.get('sender', {}).get('login', ''),
            event_name='pull_request',
            workflow='webhook',
            run_id='',
            run_number=''
        )
        
        # Run check
        return self.run_pr_check(context)
    
    def _handle_push_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle push webhook event."""
        ref = payload.get('ref', '')
        
        # Only analyze main branches
        if not any(branch in ref for branch in ['main', 'master', 'develop']):
            return {'status': 'ignored', 'reason': 'Not a main branch'}
        
        commits = payload.get('commits', [])
        results = []
        
        for commit in commits[-5:]:  # Analyze last 5 commits
            result = self.analyzer.analyze_commit(
                commit['id'],
                Path.cwd()
            )
            results.append(result)
        
        return {
            'status': 'success',
            'commits_analyzed': len(results),
            'results': results
        }
    
    def _handle_comment_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle issue_comment webhook event."""
        comment = payload.get('comment', {})
        body = comment.get('body', '')
        
        # Check for bot commands
        if '/tailchasing' not in body.lower():
            return {'status': 'ignored', 'reason': 'No bot command found'}
        
        # Parse command
        if '/tailchasing analyze' in body.lower():
            pr = payload.get('issue', {})
            if pr.get('pull_request'):
                # Trigger analysis
                context = GitHubContext(
                    repository=payload.get('repository', {}).get('full_name', ''),
                    pr_number=pr.get('number', 0),
                    head_ref='',  # Would need to fetch
                    base_ref='main',
                    sha='',
                    actor=comment.get('user', {}).get('login', ''),
                    event_name='issue_comment',
                    workflow='webhook',
                    run_id='',
                    run_number=''
                )
                return self.run_pr_check(context)
        
        return {'status': 'success', 'action': 'command_processed'}