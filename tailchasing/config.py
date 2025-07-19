"""Configuration handling for tail-chasing detector."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class Config:
    """Configuration manager for tail-chasing detector."""
    
    DEFAULT_CONFIG = {
        "paths": {
            "include": ["."],
            "exclude": ["tests", "test", "build", "dist", "venv", ".venv", "__pycache__"]
        },
        "risk_thresholds": {
            "warn": 15,
            "fail": 30
        },
        "placeholders": {
            "allow": []
        },
        "ignore_issue_types": [],
        "scoring_weights": {
            "missing_symbol": 2,
            "phantom_function": 2,
            "duplicate_function": 2,
            "circular_import": 3,
            "hallucinated_import": 3,
            "wrapper_abstraction": 1,
            "tail_chasing_chain": 4
        },
        "git": {
            "enable": True
        },
        "report": {
            "formats": ["text"],
            "output_dir": "."
        },
        "fix": {
            "enable": False,
            "auto_rename_single_suggestion": True
        }
    }
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with optional config dictionary."""
        self.config = self._merge_configs(self.DEFAULT_CONFIG, config_dict or {})
        
    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from a file.
        
        Supports YAML, JSON, and TOML formats.
        """
        if not path.exists():
            return cls()
            
        content = path.read_text()
        
        if path.suffix in [".yml", ".yaml"]:
            config_dict = yaml.safe_load(content) or {}
        elif path.suffix == ".json":
            config_dict = json.loads(content)
        elif path.suffix == ".toml":
            config_dict = tomllib.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
            
        return cls(config_dict)
        
    @classmethod
    def find_and_load(cls, root: Path) -> "Config":
        """Find and load configuration from project root."""
        config_names = [
            ".tailchasing.yml",
            ".tailchasing.yaml",
            ".tailchasing.json",
            "tailchasing.yml",
            "tailchasing.yaml",
            "tailchasing.json",
            ".tailchasing.toml",
            "tailchasing.toml",
        ]
        
        for name in config_names:
            config_path = root / name
            if config_path.exists():
                return cls.from_file(config_path)
                
        # Check for config in pyproject.toml
        pyproject_path = root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                pyproject = tomllib.loads(pyproject_path.read_text())
                if "tool" in pyproject and "tailchasing" in pyproject["tool"]:
                    return cls(pyproject["tool"]["tailchasing"])
            except Exception:
                pass
                
        return cls()
        
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.config.copy()
