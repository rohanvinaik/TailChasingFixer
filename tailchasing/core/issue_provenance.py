"""Issue provenance tracking and regression detection system."""

import hashlib
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import ast

from .issues import Issue


class IssueFingerprint:
    """Stable fingerprint for an issue based on content and location."""
    
    def __init__(
        self,
        issue_id: str,
        content_hash: str,
        location_hash: str,
        normalized_ast_hash: Optional[str],
        file_path: str,
        line_range: Tuple[int, int],
        issue_kind: str
    ):
        self.issue_id = issue_id
        self.content_hash = content_hash
        self.location_hash = location_hash
        self.normalized_ast_hash = normalized_ast_hash
        self.file_path = file_path
        self.line_range = line_range
        self.issue_kind = issue_kind
    
    @classmethod
    def from_issue(cls, issue: Issue, ast_node: Optional[ast.AST] = None) -> "IssueFingerprint":
        """Create fingerprint from an issue."""
        # Normalize message for content hash
        normalized_message = cls._normalize_message(issue.message)
        
        # Create content hash from normalized message + evidence
        content_components = [
            issue.kind,
            normalized_message,
            json.dumps(issue.evidence, sort_keys=True) if issue.evidence else "{}"
        ]
        content_hash = hashlib.sha256('|'.join(content_components).encode()).hexdigest()[:16]
        
        # Create location hash from file + line
        location_components = [
            str(Path(issue.file or "").resolve()),
            str(issue.line or 0)
        ]
        location_hash = hashlib.sha256('|'.join(location_components).encode()).hexdigest()[:8]
        
        # Create AST hash if available
        ast_hash = None
        if ast_node:
            try:
                ast_str = ast.dump(ast_node)
                # Normalize AST string by removing line numbers
                normalized_ast = re.sub(r'lineno=\d+', 'lineno=N', ast_str)
                normalized_ast = re.sub(r'col_offset=\d+', 'col_offset=N', normalized_ast)
                ast_hash = hashlib.sha256(normalized_ast.encode()).hexdigest()[:16]
            except:
                pass
        
        # Create stable issue ID
        issue_id = f"TC_{issue.kind}_{content_hash}_{location_hash}"
        
        return cls(
            issue_id=issue_id,
            content_hash=content_hash,
            location_hash=location_hash,
            normalized_ast_hash=ast_hash,
            file_path=issue.file or "",
            line_range=(issue.line or 0, issue.line or 0),
            issue_kind=issue.kind
        )
    
    @staticmethod
    def _normalize_message(message: str) -> str:
        """Normalize issue message for stable content hashing."""
        # Replace numbers with placeholder
        normalized = re.sub(r'\d+', 'N', message)
        # Replace quoted strings with placeholder
        normalized = re.sub(r'"[^"]*"', '""', normalized)
        normalized = re.sub(r"'[^']*'", "''", normalized)
        # Replace file paths with placeholder
        normalized = re.sub(r'/[^\s]*\.py', '/PATH.py', normalized)
        return normalized


class IssueEvent:
    """Records an event in an issue's lifecycle."""
    
    def __init__(
        self,
        event_type: str,
        timestamp: datetime,
        git_commit: Optional[str] = None,
        git_branch: Optional[str] = None,
        analysis_run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.event_type = event_type  # 'detected', 'resolved', 'regressed'
        self.timestamp = timestamp
        self.git_commit = git_commit
        self.git_branch = git_branch
        self.analysis_run_id = analysis_run_id
        self.metadata = metadata or {}


class IssueHistory:
    """Complete history of an issue including all events."""
    
    def __init__(
        self,
        fingerprint: IssueFingerprint,
        first_detected: datetime,
        last_seen: datetime,
        resolved_in_commit: Optional[str] = None,
        events: Optional[List[IssueEvent]] = None
    ):
        self.fingerprint = fingerprint
        self.first_detected = first_detected
        self.last_seen = last_seen
        self.resolved_in_commit = resolved_in_commit
        self.events = events or []
        self.regression_count = 0
        
        # Calculate regression count
        for event in self.events:
            if event.event_type == 'regressed':
                self.regression_count += 1
    
    def add_event(self, event: IssueEvent):
        """Add an event to this issue's history."""
        self.events.append(event)
        self.last_seen = event.timestamp
        
        if event.event_type == 'regressed':
            self.regression_count += 1
            self.resolved_in_commit = None  # Clear resolution if regressed
        elif event.event_type == 'resolved':
            self.resolved_in_commit = event.git_commit


class GitIntegration:
    """Integration with Git for commit tracking."""
    
    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()
        self._git_available = self._check_git_available()
    
    def _check_git_available(self) -> bool:
        """Check if git is available and we're in a repo."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def get_current_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        if not self._git_available:
            return None
        
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
        except subprocess.TimeoutExpired:
            pass
        
        return None
    
    def get_current_branch(self) -> Optional[str]:
        """Get current git branch."""
        if not self._git_available:
            return None
        
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
        except subprocess.TimeoutExpired:
            pass
        
        return None
    
    def blame_line(self, file_path: str, line_number: int) -> Optional[Dict[str, Any]]:
        """Get git blame info for a specific line."""
        if not self._git_available:
            return None
        
        try:
            result = subprocess.run(
                ['git', 'blame', '--porcelain', '-L', f'{line_number},{line_number}', file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    commit_hash = lines[0].split()[0]
                    return {
                        'commit': commit_hash,
                        'file': file_path,
                        'line': line_number
                    }
        except subprocess.TimeoutExpired:
            pass
        
        return None
    
    def find_commit_that_introduced_file(self, file_path: str) -> Optional[str]:
        """Find the commit that first introduced a file."""
        if not self._git_available:
            return None
        
        try:
            result = subprocess.run(
                ['git', 'log', '--reverse', '--format=%H', '--', file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except subprocess.TimeoutExpired:
            pass
        
        return None


class IssueDatabase:
    """SQLite database for storing issue history and provenance."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS issue_fingerprints (
                    issue_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    location_hash TEXT NOT NULL,
                    normalized_ast_hash TEXT,
                    file_path TEXT NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    issue_kind TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS issue_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    issue_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    git_commit TEXT,
                    git_branch TEXT,
                    analysis_run_id TEXT,
                    metadata TEXT,  -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (issue_id) REFERENCES issue_fingerprints (issue_id)
                );
                
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    git_commit TEXT,
                    git_branch TEXT,
                    total_issues INTEGER,
                    config_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_issue_events_issue_id ON issue_events (issue_id);
                CREATE INDEX IF NOT EXISTS idx_issue_events_timestamp ON issue_events (timestamp);
                CREATE INDEX IF NOT EXISTS idx_issue_fingerprints_file_path ON issue_fingerprints (file_path);
                CREATE INDEX IF NOT EXISTS idx_analysis_runs_timestamp ON analysis_runs (timestamp);
            ''')
    
    def store_fingerprint(self, fingerprint: IssueFingerprint):
        """Store or update issue fingerprint."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO issue_fingerprints 
                (issue_id, content_hash, location_hash, normalized_ast_hash, 
                 file_path, line_start, line_end, issue_kind, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                fingerprint.issue_id,
                fingerprint.content_hash,
                fingerprint.location_hash,
                fingerprint.normalized_ast_hash,
                fingerprint.file_path,
                fingerprint.line_range[0],
                fingerprint.line_range[1],
                fingerprint.issue_kind
            ))
    
    def store_event(self, issue_id: str, event: IssueEvent):
        """Store an issue event."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                INSERT INTO issue_events 
                (issue_id, event_type, timestamp, git_commit, git_branch, analysis_run_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                issue_id,
                event.event_type,
                event.timestamp.isoformat(),
                event.git_commit,
                event.git_branch,
                event.analysis_run_id,
                json.dumps(event.metadata)
            ))
    
    def get_issue_history(self, issue_id: str) -> Optional[IssueHistory]:
        """Get complete history for an issue."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get fingerprint
            fp_row = conn.execute('''
                SELECT * FROM issue_fingerprints WHERE issue_id = ?
            ''', (issue_id,)).fetchone()
            
            if not fp_row:
                return None
            
            fingerprint = IssueFingerprint(
                issue_id=fp_row['issue_id'],
                content_hash=fp_row['content_hash'],
                location_hash=fp_row['location_hash'],
                normalized_ast_hash=fp_row['normalized_ast_hash'],
                file_path=fp_row['file_path'],
                line_range=(fp_row['line_start'], fp_row['line_end']),
                issue_kind=fp_row['issue_kind']
            )
            
            # Get events
            event_rows = conn.execute('''
                SELECT * FROM issue_events 
                WHERE issue_id = ? 
                ORDER BY timestamp ASC
            ''', (issue_id,)).fetchall()
            
            events = []
            for row in event_rows:
                event = IssueEvent(
                    event_type=row['event_type'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    git_commit=row['git_commit'],
                    git_branch=row['git_branch'],
                    analysis_run_id=row['analysis_run_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                events.append(event)
            
            # Determine first/last seen and resolution
            first_detected = events[0].timestamp if events else datetime.fromisoformat(fp_row['created_at'])
            last_seen = events[-1].timestamp if events else first_detected
            resolved_in_commit = None
            
            for event in reversed(events):
                if event.event_type == 'resolved':
                    resolved_in_commit = event.git_commit
                    break
                elif event.event_type == 'regressed':
                    break  # Most recent regression clears resolution
            
            return IssueHistory(
                fingerprint=fingerprint,
                first_detected=first_detected,
                last_seen=last_seen,
                resolved_in_commit=resolved_in_commit,
                events=events
            )
    
    def find_similar_issues(self, fingerprint: IssueFingerprint) -> List[str]:
        """Find issues with similar content hash."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute('''
                SELECT issue_id FROM issue_fingerprints 
                WHERE content_hash = ? AND issue_id != ?
            ''', (fingerprint.content_hash, fingerprint.issue_id)).fetchall()
            
            return [row[0] for row in rows]
    
    def get_recent_regressions(self, days: int) -> List[Tuple[str, IssueEvent]]:
        """Get issues that regressed in the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            rows = conn.execute('''
                SELECT issue_id, event_type, timestamp, git_commit, git_branch, analysis_run_id, metadata
                FROM issue_events 
                WHERE event_type = 'regressed' AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff.isoformat(),)).fetchall()
            
            results = []
            for row in rows:
                event = IssueEvent(
                    event_type=row['event_type'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    git_commit=row['git_commit'],
                    git_branch=row['git_branch'],
                    analysis_run_id=row['analysis_run_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                )
                results.append((row['issue_id'], event))
            
            return results
    
    def store_analysis_run(self, run_id: str, timestamp: datetime, 
                          git_commit: Optional[str], git_branch: Optional[str], 
                          total_issues: int, config_hash: str):
        """Store analysis run metadata."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                INSERT INTO analysis_runs 
                (run_id, timestamp, git_commit, git_branch, total_issues, config_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (run_id, timestamp.isoformat(), git_commit, git_branch, total_issues, config_hash))


class IssueProvenanceTracker:
    """Main provenance tracking system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('issue_provenance', {})
        self.enabled = self.config.get('enabled', False)
        self.track_regressions = self.config.get('track_regressions', True)
        self.git_integration_enabled = self.config.get('git_integration', True)
        
        if not self.enabled:
            self.db = None
            self.git = None
            return
        
        # Initialize database
        db_path_str = self.config.get('db_path', '.tailchasing_history.db')
        self.db_path = Path(db_path_str)
        self.db = IssueDatabase(self.db_path)
        
        # Initialize git integration
        self.git = GitIntegration() if self.git_integration_enabled else None
        
        # Generate run ID for this analysis
        self.run_id = self._generate_run_id()
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID for this analysis."""
        timestamp = datetime.now().isoformat()
        components = [timestamp, str(self.db_path)]
        return hashlib.sha256('|'.join(components).encode()).hexdigest()[:16]
    
    def process_issues(self, issues: List[Issue], ast_index: Dict[str, ast.AST]) -> List[Issue]:
        """Process issues and add provenance information."""
        if not self.enabled or not issues:
            return issues
        
        enhanced_issues = []
        current_commit = self.git.get_current_commit() if self.git else None
        current_branch = self.git.get_current_branch() if self.git else None
        
        for issue in issues:
            # Create fingerprint
            ast_node = None
            if issue.file in ast_index:
                # Try to find relevant AST node (simplified)
                ast_node = ast_index[issue.file]
            
            fingerprint = IssueFingerprint.from_issue(issue, ast_node)
            
            # Store fingerprint
            self.db.store_fingerprint(fingerprint)
            
            # Check if this is a known issue
            existing_history = self.db.get_issue_history(fingerprint.issue_id)
            is_regression = False
            
            if existing_history:
                # This is a known issue - check for regression
                if self.track_regressions:
                    # If it was previously resolved but is appearing again
                    if existing_history.resolved_in_commit:
                        is_regression = True
                        
                        # Store regression event
                        regression_event = IssueEvent(
                            event_type='regressed',
                            timestamp=datetime.now(),
                            git_commit=current_commit,
                            git_branch=current_branch,
                            analysis_run_id=self.run_id,
                            metadata={'severity': issue.severity}
                        )
                        self.db.store_event(fingerprint.issue_id, regression_event)
            else:
                # New issue - store detection event
                detection_event = IssueEvent(
                    event_type='detected',
                    timestamp=datetime.now(),
                    git_commit=current_commit,
                    git_branch=current_branch,
                    analysis_run_id=self.run_id,
                    metadata={'severity': issue.severity}
                )
                self.db.store_event(fingerprint.issue_id, detection_event)
            
            # Enhance issue with provenance information
            enhanced_issue = Issue(
                kind=issue.kind,
                message=issue.message,
                severity=issue.severity,
                file=issue.file,
                line=issue.line,
                evidence=issue.evidence.copy() if issue.evidence else {}
            )
            
            # Add provenance metadata
            enhanced_issue.evidence.update({
                'stable_id': fingerprint.issue_id,
                'content_hash': fingerprint.content_hash,
                'provenance': {
                    'tracking_enabled': True,
                    'git_commit': current_commit,
                    'git_branch': current_branch,
                    'run_id': self.run_id,
                    'is_regression': is_regression
                }
            })
            
            # Update message for regressions
            if is_regression:
                enhanced_issue.message = f"[REGRESSED] {enhanced_issue.message}"
            
            # Find similar issues
            similar = self.db.find_similar_issues(fingerprint)
            if similar:
                enhanced_issue.evidence['similar_issues'] = similar
            
            enhanced_issues.append(enhanced_issue)
        
        # Store analysis run metadata
        config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:8]
        self.db.store_analysis_run(
            run_id=self.run_id,
            timestamp=datetime.now(),
            git_commit=current_commit,
            git_branch=current_branch,
            total_issues=len(issues),
            config_hash=config_hash
        )
        
        return enhanced_issues
    
    def get_regression_report(self, days: int = 7) -> str:
        """Generate a regression report."""
        if not self.enabled or not self.db:
            return "Issue provenance tracking is disabled."
        
        regressions = self.db.get_recent_regressions(days)
        
        report_lines = [
            f"REGRESSION REPORT ({days} days)",
            "=" * 50,
            f"Total regressions: {len(regressions)}",
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if not regressions:
            report_lines.append("No regressions detected in the specified period.")
        else:
            for issue_id, event in regressions:
                history = self.db.get_issue_history(issue_id)
                if history:
                    report_lines.extend([
                        f"Issue: {issue_id}",
                        f"  Kind: {history.fingerprint.issue_kind}",
                        f"  File: {history.fingerprint.file_path}:{history.fingerprint.line_range[0]}",
                        f"  First detected: {history.first_detected.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"  Regressed: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        f"  Commit: {event.git_commit or 'unknown'}",
                        f"  Branch: {event.git_branch or 'unknown'}",
                        f"  Regression count: {history.regression_count}",
                        ""
                    ])
        
        return "\n".join(report_lines)