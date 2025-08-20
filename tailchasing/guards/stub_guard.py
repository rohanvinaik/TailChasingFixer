"""
Stub Guard - Prevents incomplete implementations from entering critical paths.

This guard acts as a gatekeeper to ensure no stub functions, TODOs, or
incomplete implementations exist in critical code paths. Perfect for CI/CD
pipelines and pre-commit hooks.
"""

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class StubPattern:
    """Definition of a stub pattern to detect."""
    
    name: str
    pattern: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'not_implemented', 'todo', 'placeholder', 'debug'
    description: str
    auto_fixable: bool = False
    fix_suggestion: Optional[str] = None


@dataclass
class StubViolation:
    """A detected stub violation."""
    
    file_path: Path
    line_number: int
    pattern: StubPattern
    line_content: str
    context_lines: List[str] = field(default_factory=list)
    in_critical_path: bool = False
    in_test: bool = False
    added_recently: bool = False  # From git history
    author: Optional[str] = None  # From git blame


@dataclass
class GuardConfig:
    """Configuration for stub guard."""
    
    critical_paths: List[str] = field(default_factory=lambda: ["src", "lib", "core"])
    test_paths: List[str] = field(default_factory=lambda: ["tests", "test"])
    exclude_paths: Set[str] = field(default_factory=lambda: {
        ".git", ".venv", "venv", "__pycache__", "build", "dist", 
        "node_modules", ".tox", "*.egg-info", "migrations"
    })
    
    # Control strictness
    fail_on_critical: bool = True
    fail_on_todos: bool = False
    allow_in_tests: bool = True
    allow_in_examples: bool = True
    
    # Severity thresholds
    max_critical_violations: int = 0
    max_high_violations: int = 0
    max_medium_violations: int = 10
    max_low_violations: int = 50
    
    # Time-based allowances
    grace_period_days: int = 7  # Allow stubs for this many days
    warn_after_days: int = 30   # Warn about old stubs
    
    # Reporting
    verbose: bool = False
    output_format: str = "text"  # 'text', 'json', 'junit'


class StubGuard:
    """
    Guards against incomplete implementations in critical code paths.
    
    Features:
    1. Detects various stub patterns (NotImplementedError, TODO, pass, etc.)
    2. Configurable critical paths and severity levels
    3. Git integration for tracking stub age and authors
    4. Progressive enforcement (grace periods for new stubs)
    5. CI/CD integration with multiple output formats
    """
    
    # Default stub patterns
    DEFAULT_PATTERNS = [
        StubPattern(
            name="not_implemented",
            pattern=r"\braise\s+NotImplementedError\b",
            severity="critical",
            category="not_implemented",
            description="Raises NotImplementedError",
            auto_fixable=False,
            fix_suggestion="Implement the function or mark as abstract"
        ),
        StubPattern(
            name="pass_only",
            pattern=r"^\s*pass\s*$",
            severity="high",
            category="placeholder",
            description="Function body contains only 'pass'",
            auto_fixable=False,
            fix_suggestion="Add implementation or remove if unused"
        ),
        StubPattern(
            name="ellipsis",
            pattern=r"^\s*\.\.\.\s*$",
            severity="high",
            category="placeholder",
            description="Function body contains only ellipsis",
            auto_fixable=False,
            fix_suggestion="Replace with actual implementation"
        ),
        StubPattern(
            name="todo_comment",
            pattern=r"\b(TODO|FIXME|XXX|HACK|WIP|REFACTOR)\b",
            severity="medium",
            category="todo",
            description="Contains TODO/FIXME comment",
            auto_fixable=False,
            fix_suggestion="Complete the TODO item"
        ),
        StubPattern(
            name="debug_print",
            pattern=r"\b(print|console\.log|debug|pdb\.set_trace|breakpoint)\s*\(",
            severity="low",
            category="debug",
            description="Contains debug statement",
            auto_fixable=True,
            fix_suggestion="Remove debug statements"
        ),
        StubPattern(
            name="hardcoded_value",
            pattern=r"(return\s+(True|False|None|0|1|-1|\[\]|\{\}|\"\"|\'\'))\s*$",
            severity="low",
            category="placeholder",
            description="Returns hardcoded placeholder value",
            auto_fixable=False,
            fix_suggestion="Implement actual logic"
        ),
    ]
    
    def __init__(self, config: Optional[GuardConfig] = None):
        self.config = config or GuardConfig()
        self.patterns = self.DEFAULT_PATTERNS.copy()
        self.violations: List[StubViolation] = []
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        for pattern in self.patterns:
            self._compiled_patterns[pattern.name] = re.compile(
                pattern.pattern, 
                re.MULTILINE | re.IGNORECASE
            )
    
    def add_pattern(self, pattern: StubPattern):
        """Add a custom stub pattern."""
        self.patterns.append(pattern)
        self._compiled_patterns[pattern.name] = re.compile(
            pattern.pattern,
            re.MULTILINE | re.IGNORECASE
        )
    
    def scan_file(self, file_path: Path) -> List[StubViolation]:
        """Scan a single file for stub violations."""
        violations = []
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            
            for pattern in self.patterns:
                regex = self._compiled_patterns[pattern.name]
                
                for match in regex.finditer(content):
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    # Get context lines
                    context_start = max(0, line_num - 3)
                    context_end = min(len(lines), line_num + 2)
                    context = lines[context_start:context_end]
                    
                    violation = StubViolation(
                        file_path=file_path,
                        line_number=line_num,
                        pattern=pattern,
                        line_content=line_content.strip(),
                        context_lines=context,
                        in_critical_path=self._is_critical_path(file_path),
                        in_test=self._is_test_path(file_path)
                    )
                    
                    # Add git information if available
                    self._add_git_info(violation)
                    
                    violations.append(violation)
        
        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")
        
        return violations
    
    def scan_directory(self, root_path: Path) -> List[StubViolation]:
        """Scan directory tree for stub violations."""
        self.violations.clear()
        
        for path in self._get_python_files(root_path):
            if not self._should_check_file(path):
                continue
            
            file_violations = self.scan_file(path)
            self.violations.extend(file_violations)
        
        return self.violations
    
    def _get_python_files(self, root: Path) -> List[Path]:
        """Get all Python files to check."""
        python_files = []
        
        for pattern in ["**/*.py", "**/*.pyw"]:
            for path in root.glob(pattern):
                if not any(excluded in str(path) for excluded in self.config.exclude_paths):
                    python_files.append(path)
        
        return python_files
    
    def _should_check_file(self, path: Path) -> bool:
        """Determine if file should be checked."""
        path_str = str(path)
        
        # Skip excluded paths
        for excluded in self.config.exclude_paths:
            if excluded in path_str:
                return False
        
        # Skip __pycache__ and compiled files
        if "__pycache__" in path_str or path.suffix == ".pyc":
            return False
        
        # Skip migrations and auto-generated files
        if "migrations" in path_str or "pb2" in path.name:
            return False
        
        return True
    
    def _is_critical_path(self, path: Path) -> bool:
        """Check if path is in critical code paths."""
        path_str = str(path)
        return any(critical in path_str for critical in self.config.critical_paths)
    
    def _is_test_path(self, path: Path) -> bool:
        """Check if path is in test directories."""
        path_str = str(path)
        return any(test in path_str for test in self.config.test_paths)
    
    def _add_git_info(self, violation: StubViolation):
        """Add git blame and history information."""
        try:
            import subprocess
            
            # Get last modified date
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ai", "--", str(violation.file_path)],
                capture_output=True,
                text=True,
                cwd=violation.file_path.parent
            )
            
            if result.returncode == 0:
                date_str = result.stdout.strip()
                if date_str:
                    commit_date = datetime.fromisoformat(date_str.split()[0])
                    days_old = (datetime.now() - commit_date).days
                    violation.added_recently = days_old <= self.config.grace_period_days
            
            # Get author of line
            result = subprocess.run(
                ["git", "blame", "-L", f"{violation.line_number},{violation.line_number}", 
                 "--porcelain", str(violation.file_path)],
                capture_output=True,
                text=True,
                cwd=violation.file_path.parent
            )
            
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith("author "):
                        violation.author = line[7:]
                        break
        
        except Exception:
            pass  # Git info is optional
    
    def check_violations(self) -> Tuple[bool, str]:
        """
        Check if violations exceed configured thresholds.
        
        Returns:
            Tuple of (pass/fail, message)
        """
        if not self.violations:
            return True, "✅ No stub violations found"
        
        # Group by severity
        by_severity = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": []
        }
        
        for violation in self.violations:
            # Skip allowed violations
            if violation.in_test and self.config.allow_in_tests:
                continue
            if violation.added_recently and violation.pattern.severity != "critical":
                continue  # Grace period for non-critical
            
            by_severity[violation.pattern.severity].append(violation)
        
        # Check thresholds
        failed = False
        messages = []
        
        if len(by_severity["critical"]) > self.config.max_critical_violations:
            failed = True
            messages.append(f"❌ {len(by_severity['critical'])} critical violations (max: {self.config.max_critical_violations})")
        
        if len(by_severity["high"]) > self.config.max_high_violations:
            failed = True
            messages.append(f"❌ {len(by_severity['high'])} high violations (max: {self.config.max_high_violations})")
        
        if len(by_severity["medium"]) > self.config.max_medium_violations:
            failed = True
            messages.append(f"⚠️ {len(by_severity['medium'])} medium violations (max: {self.config.max_medium_violations})")
        
        if len(by_severity["low"]) > self.config.max_low_violations:
            messages.append(f"ℹ️ {len(by_severity['low'])} low violations (max: {self.config.max_low_violations})")
        
        # Fail on critical paths
        if self.config.fail_on_critical:
            critical_path_violations = [v for v in self.violations if v.in_critical_path]
            if critical_path_violations:
                failed = True
                messages.append(f"❌ {len(critical_path_violations)} violations in critical paths")
        
        return not failed, "\n".join(messages) if messages else "✅ All checks passed"
    
    def generate_report(self, format: str = "text") -> str:
        """Generate violation report."""
        if format == "text":
            return self._generate_text_report()
        elif format == "json":
            return self._generate_json_report()
        elif format == "junit":
            return self._generate_junit_report()
        else:
            return self._generate_text_report()
    
    def _generate_text_report(self) -> str:
        """Generate human-readable text report."""
        if not self.violations:
            return "No stub violations found."
        
        report = ["Stub Guard Report", "=" * 50, ""]
        
        # Group by file
        by_file = {}
        for violation in self.violations:
            if violation.file_path not in by_file:
                by_file[violation.file_path] = []
            by_file[violation.file_path].append(violation)
        
        for file_path, violations in by_file.items():
            report.append(f"\n{file_path}:")
            
            for v in violations:
                marker = "🔴" if v.pattern.severity == "critical" else \
                        "🟠" if v.pattern.severity == "high" else \
                        "🟡" if v.pattern.severity == "medium" else "⚪"
                
                report.append(f"  {marker} Line {v.line_number}: {v.pattern.description}")
                report.append(f"     {v.line_content}")
                
                if v.pattern.fix_suggestion:
                    report.append(f"     💡 {v.pattern.fix_suggestion}")
                
                if v.author:
                    report.append(f"     👤 {v.author}")
        
        # Summary
        report.append("")
        report.append("Summary:")
        report.append(f"  Total violations: {len(self.violations)}")
        
        severity_counts = {}
        for v in self.violations:
            severity_counts[v.pattern.severity] = severity_counts.get(v.pattern.severity, 0) + 1
        
        for severity, count in severity_counts.items():
            report.append(f"  {severity.capitalize()}: {count}")
        
        return "\n".join(report)
    
    def _generate_json_report(self) -> str:
        """Generate JSON report for CI/CD integration."""
        import json
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_violations": len(self.violations),
            "violations": []
        }
        
        for v in self.violations:
            report["violations"].append({
                "file": str(v.file_path),
                "line": v.line_number,
                "severity": v.pattern.severity,
                "category": v.pattern.category,
                "pattern": v.pattern.name,
                "description": v.pattern.description,
                "content": v.line_content,
                "fix_suggestion": v.pattern.fix_suggestion,
                "in_critical_path": v.in_critical_path,
                "author": v.author,
                "added_recently": v.added_recently
            })
        
        return json.dumps(report, indent=2)
    
    def _generate_junit_report(self) -> str:
        """Generate JUnit XML report for CI/CD."""
        from xml.etree import ElementTree as ET
        
        testsuites = ET.Element("testsuites")
        testsuite = ET.SubElement(testsuites, "testsuite", 
                                  name="StubGuard",
                                  tests=str(len(self.violations)),
                                  failures=str(len([v for v in self.violations 
                                                   if v.pattern.severity in ["critical", "high"]])))
        
        for v in self.violations:
            testcase = ET.SubElement(testsuite, "testcase",
                                    classname=str(v.file_path),
                                    name=f"Line_{v.line_number}_{v.pattern.name}")
            
            if v.pattern.severity in ["critical", "high"]:
                failure = ET.SubElement(testcase, "failure",
                                      type=v.pattern.category,
                                      message=v.pattern.description)
                failure.text = f"{v.line_content}\nSuggestion: {v.pattern.fix_suggestion}"
        
        return ET.tostring(testsuites, encoding="unicode")


def main(args: Optional[List[str]] = None) -> int:
    """CLI entry point for stub guard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Guard against stub implementations")
    parser.add_argument("path", nargs="?", default=".", help="Path to check")
    parser.add_argument("--critical-paths", nargs="+", help="Critical code paths")
    parser.add_argument("--fail-on-todos", action="store_true", help="Fail on TODO comments")
    parser.add_argument("--format", choices=["text", "json", "junit"], default="text")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--max-critical", type=int, default=0, help="Max critical violations")
    parser.add_argument("--max-high", type=int, default=0, help="Max high violations")
    
    parsed_args = parser.parse_args(args)
    
    # Create config
    config = GuardConfig(
        critical_paths=parsed_args.critical_paths or ["src", "lib"],
        fail_on_todos=parsed_args.fail_on_todos,
        verbose=parsed_args.verbose,
        output_format=parsed_args.format,
        max_critical_violations=parsed_args.max_critical,
        max_high_violations=parsed_args.max_high
    )
    
    # Run guard
    guard = StubGuard(config)
    violations = guard.scan_directory(Path(parsed_args.path))
    
    # Generate report
    print(guard.generate_report(parsed_args.format))
    
    # Check violations
    passed, message = guard.check_violations()
    if not passed:
        print(f"\n{message}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())