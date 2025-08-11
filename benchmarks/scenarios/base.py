"""
Base classes for benchmark scenarios.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import tempfile
import shutil


@dataclass
class ScenarioResult:
    """Result of running a benchmark scenario."""
    
    scenario_name: str
    success: bool
    steps_taken: int
    expected_steps: Tuple[int, int]  # (min, max) expected steps
    time_elapsed: float  # seconds
    tokens_used: int
    cost_estimate: float  # in USD
    error_message: Optional[str] = None
    regressions_detected: List[str] = field(default_factory=list)
    fixes_applied: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def within_expected_steps(self) -> bool:
        """Check if convergence happened within expected step range."""
        min_steps, max_steps = self.expected_steps
        return min_steps <= self.steps_taken <= max_steps
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (0-100)."""
        if not self.success:
            return 0.0
        
        min_steps, max_steps = self.expected_steps
        if self.steps_taken <= min_steps:
            return 100.0
        elif self.steps_taken >= max_steps:
            return 50.0  # Still successful but inefficient
        else:
            # Linear interpolation between min and max
            range_size = max_steps - min_steps
            position = self.steps_taken - min_steps
            return 100.0 - (position / range_size * 50.0)


class BenchmarkScenario(ABC):
    """Base class for all benchmark scenarios."""
    
    def __init__(self, name: str, description: str, expected_steps: Tuple[int, int]):
        self.name = name
        self.description = description
        self.expected_steps = expected_steps
        self.temp_dir: Optional[Path] = None
    
    @abstractmethod
    def setup(self) -> Path:
        """
        Set up the scenario in a temporary directory.
        
        Returns:
            Path to the temporary directory containing the scenario files.
        """
        pass
    
    @abstractmethod
    def get_initial_code(self) -> Dict[str, str]:
        """
        Get the initial problematic code for the scenario.
        
        Returns:
            Dictionary mapping file paths to their content.
        """
        pass
    
    @abstractmethod
    def get_expected_solution(self) -> Dict[str, str]:
        """
        Get the expected solution after fixing.
        
        Returns:
            Dictionary mapping file paths to their expected content.
        """
        pass
    
    @abstractmethod
    def validate_solution(self, current_code: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate if the current code matches the expected solution.
        
        Args:
            current_code: Dictionary mapping file paths to their current content.
            
        Returns:
            Tuple of (success, list of validation errors)
        """
        pass
    
    def create_temp_directory(self) -> Path:
        """Create a temporary directory for the scenario."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark_{self.name}_"))
        return self.temp_dir
    
    def cleanup(self):
        """Clean up the temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def write_files(self, code_dict: Dict[str, str]):
        """Write files to the temporary directory."""
        if not self.temp_dir:
            raise RuntimeError("Temporary directory not created. Call setup() first.")
        
        for file_path, content in code_dict.items():
            full_path = self.temp_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
    
    def read_files(self) -> Dict[str, str]:
        """Read all Python files from the temporary directory."""
        if not self.temp_dir:
            raise RuntimeError("Temporary directory not created. Call setup() first.")
        
        result = {}
        for py_file in self.temp_dir.rglob("*.py"):
            rel_path = py_file.relative_to(self.temp_dir)
            result[str(rel_path)] = py_file.read_text()
        
        return result
    
    def check_regressions(self, before: Dict[str, str], after: Dict[str, str]) -> List[str]:
        """
        Check for regressions introduced during fixing.
        
        Args:
            before: Code state before fix
            after: Code state after fix
            
        Returns:
            List of regression descriptions
        """
        regressions = []
        
        # Check for deleted files
        for file_path in before:
            if file_path not in after:
                regressions.append(f"File deleted: {file_path}")
        
        # Check for broken imports
        for file_path, content in after.items():
            if "import" in content:
                lines = content.split("\n")
                for i, line in enumerate(lines, 1):
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        if "# type: ignore" in line and "# type: ignore" not in before.get(file_path, ""):
                            regressions.append(f"Type ignore added in {file_path}:{i}")
        
        # Check for syntax errors
        import ast
        for file_path, content in after.items():
            try:
                ast.parse(content)
            except SyntaxError as e:
                regressions.append(f"Syntax error in {file_path}:{e.lineno}: {e.msg}")
        
        return regressions