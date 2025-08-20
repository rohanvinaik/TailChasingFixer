"""Git history analyzer for detecting tail-chasing patterns over time."""

import re
import subprocess
from datetime import datetime, timedelta
from typing import List, Dict, Iterable, Any
from collections import defaultdict
from pathlib import Path

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue


class GitChainAnalyzer(BaseAnalyzer):
    """Analyzes git history to detect temporal tail-chasing patterns."""
    
    name = "git_chains"
    
    # Patterns that suggest tail-chasing fixes
    SUSPECT_PATTERNS = [
        r"fix\s+import",
        r"add\s+missing",
        r"rename.*to\s+match",
        r"fix.*error",
        r"resolve.*issue",
        r"temporary",
        r"temp\s+fix",
        r"quick\s+fix",
        r"typo",
        r"oops",
        r"forgot",
        r"missing\s+file",
        r"undefined",
        r"not\s+found"
    ]
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Analyze git history for tail-chasing patterns."""
        if not ctx.config.get("git", {}).get("enable", True):
            return
            
        # Check if we're in a git repository
        if not self._is_git_repo(ctx.root_dir):
            return
            
        # Get recent commits
        commits = self._get_recent_commits(ctx.root_dir, days=30)
        
        # Analyze commit patterns
        chains = self._find_fix_chains(commits, ctx)
        
        # Create issues for suspicious chains
        for chain in chains:
            yield self._create_chain_issue(chain, ctx)
            
    def _is_git_repo(self, path: Path) -> bool:
        """Check if the path is inside a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
            
    def _get_recent_commits(self, path: Path, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent commits with their metadata."""
        try:
            # Get commit info with file changes
            result = subprocess.run(
                [
                    "git", "log",
                    f"--since={days}.days.ago",
                    "--name-status",
                    "--format=%H|%ai|%s|%an"
                ],
                cwd=path,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return []
                
            commits = []
            current_commit = None
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                if '|' in line:
                    # New commit
                    if current_commit:
                        commits.append(current_commit)
                        
                    parts = line.split('|', 3)
                    current_commit = {
                        "hash": parts[0],
                        "date": parts[1],
                        "message": parts[2],
                        "author": parts[3],
                        "files": []
                    }
                elif line[0] in "AMDRTUX":
                    # File change
                    if current_commit:
                        status = line[0]
                        file_path = line[1:].strip()
                        current_commit["files"].append({
                            "status": status,
                            "path": file_path
                        })
                        
            if current_commit:
                commits.append(current_commit)
                
            return commits
            
        except (subprocess.SubprocessError, FileNotFoundError):
            return []
            
    def _find_fix_chains(self, commits: List[Dict], ctx: AnalysisContext) -> List[List[Dict]]:
        """Find chains of suspicious fix commits."""
        chains = []
        
        # Group commits by file
        file_commits = defaultdict(list)
        for commit in commits:
            for file_info in commit["files"]:
                file_commits[file_info["path"]].append(commit)
                
        # Look for files with multiple suspicious commits
        for file_path, file_commit_list in file_commits.items():
            if len(file_commit_list) < 3:  # Need at least 3 commits to be suspicious
                continue
                
            # Check for suspicious patterns
            suspicious_commits = []
            for commit in file_commit_list:
                if self._is_suspicious_commit(commit):
                    suspicious_commits.append(commit)
                    
            # If we have a chain of suspicious commits, record it
            if len(suspicious_commits) >= 3:
                # Check if they're close in time (within 24 hours of each other)
                time_chain = self._find_time_clusters(suspicious_commits, hours=24)
                chains.extend(time_chain)
                
        return chains
        
    def _is_suspicious_commit(self, commit: Dict) -> bool:
        """Check if a commit message suggests tail-chasing."""
        message = commit["message"].lower()
        
        for pattern in self.SUSPECT_PATTERNS:
            if re.search(pattern, message):
                return True
                
        # Also check for very short messages (often "fix", "update", etc.)
        if len(message) < 10 and message not in ["merge", "revert"]:
            return True
            
        return False
        
    def _find_time_clusters(self, commits: List[Dict], hours: int) -> List[List[Dict]]:
        """Find commits that are close together in time."""
        # Sort by date
        sorted_commits = sorted(commits, key=lambda c: c["date"])
        
        clusters = []
        current_cluster = []
        
        for i, commit in enumerate(sorted_commits):
            if not current_cluster:
                current_cluster.append(commit)
            else:
                # Parse dates
                prev_date = datetime.fromisoformat(current_cluster[-1]["date"].replace(' ', 'T'))
                curr_date = datetime.fromisoformat(commit["date"].replace(' ', 'T'))
                
                # Check if within time window
                if curr_date - prev_date <= timedelta(hours=hours):
                    current_cluster.append(commit)
                else:
                    # Start new cluster
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [commit]
                    
        # Don't forget the last cluster
        if len(current_cluster) >= 3:
            clusters.append(current_cluster)
            
        return clusters
        
    def _create_chain_issue(self, chain: List[Dict], ctx: AnalysisContext) -> Issue:
        """Create an issue for a tail-chasing chain."""
        # Extract affected files
        affected_files = set()
        for commit in chain:
            for file_info in commit["files"]:
                affected_files.add(file_info["path"])
                
        # Build a description of the chain
        commit_summaries = []
        for commit in chain[:5]:  # Show first 5
            date = commit["date"].split(' ')[0]  # Just the date part
            msg = commit["message"][:50]  # First 50 chars
            commit_summaries.append(f"{date}: {msg}")
            
        if len(chain) > 5:
            commit_summaries.append(f"... and {len(chain) - 5} more")
            
        message = (
            f"Suspicious chain of {len(chain)} fixes in {len(affected_files)} file(s) "
            f"over {self._calculate_duration(chain)} - possible tail-chasing"
        )
        
        issue = Issue(
            kind="tail_chasing_chain",
            message=message,
            severity=min(3 + len(chain) // 5, 4),  # Higher severity for longer chains
            evidence={
                "chain_length": len(chain),
                "commits": commit_summaries,
                "affected_files": list(affected_files)[:10],  # First 10 files
                "duration": self._calculate_duration(chain),
                "patterns_found": self._extract_patterns(chain)
            },
            suggestions=[
                "Review the architecture to identify root causes",
                "Consider refactoring instead of applying quick fixes",
                "Document the actual requirements before making changes",
                "Use version control to revert and start fresh if needed"
            ],
            confidence=self._calculate_chain_confidence(chain)
        )
        
        # Set file to the most frequently modified
        file_counts = defaultdict(int)
        for commit in chain:
            for file_info in commit["files"]:
                file_counts[file_info["path"]] += 1
                
        if file_counts:
            most_modified = max(file_counts.items(), key=lambda x: x[1])
            issue.file = most_modified[0]
            
        return issue
        
    def _calculate_duration(self, chain: List[Dict]) -> str:
        """Calculate the duration of a commit chain."""
        if not chain:
            return "0 days"
            
        dates = [datetime.fromisoformat(c["date"].replace(' ', 'T')) for c in chain]
        duration = max(dates) - min(dates)
        
        if duration.days > 0:
            return f"{duration.days} days"
        else:
            hours = duration.seconds // 3600
            return f"{hours} hours"
            
    def _extract_patterns(self, chain: List[Dict]) -> List[str]:
        """Extract the patterns found in commit messages."""
        patterns_found = set()
        
        for commit in chain:
            message = commit["message"].lower()
            for pattern in self.SUSPECT_PATTERNS:
                if re.search(pattern, message):
                    # Clean up the pattern for display
                    display_pattern = pattern.replace(r'\s+', ' ').replace('.*', '...')
                    patterns_found.add(display_pattern)
                    
        return sorted(patterns_found)
        
    def _calculate_chain_confidence(self, chain: List[Dict]) -> float:
        """Calculate confidence that this is a tail-chasing chain."""
        confidence = 0.6  # Base confidence
        
        # More commits = higher confidence
        if len(chain) >= 5:
            confidence += 0.1
        if len(chain) >= 10:
            confidence += 0.1
            
        # Multiple patterns = higher confidence
        patterns = self._extract_patterns(chain)
        if len(patterns) >= 3:
            confidence += 0.1
            
        # Short time span = higher confidence
        duration = self._calculate_duration(chain)
        if "hours" in duration:
            confidence += 0.1
            
        return min(confidence, 0.95)
