"""
Configuration management for polymer physics parameters.

This module handles loading, saving, and validating configuration
for the polymer physics analysis parameters.
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


@dataclass
class PolymerConfig:
    """Configuration for polymer physics parameters."""
    
    # Contact probability parameters
    alpha: float = 1.2  # Contact decay exponent
    epsilon: float = 1e-6  # Regularization parameter
    kappa: float = 1.0  # Contact strength constant
    tad_penalty: float = 0.7  # Penalty for cross-TAD contacts
    
    # Distance weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        "tok": 1.0,  # Token-level weight
        "ast": 2.0,  # AST-level weight
        "mod": 3.0,  # Module-level weight
        "git": 4.0   # Git-level weight
    })
    
    # Analysis parameters
    min_tad_size: int = 10  # Minimum TAD size
    max_tad_size: int = 1000  # Maximum TAD size
    loop_anchor_threshold: float = 0.5  # Threshold for loop anchors
    contact_matrix_resolution: int = 100  # Matrix resolution
    
    # Performance thresholds
    thrash_risk_threshold: float = 0.7  # Risk threshold
    optimization_threshold: float = 0.3  # Optimization threshold
    
    # Visualization settings
    heatmap_colormap: str = "viridis"
    show_tad_boundaries: bool = True
    show_loop_anchors: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolymerConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        valid = True
        errors = []
        
        # Validate alpha
        if not 0.5 <= self.alpha <= 3.0:
            errors.append(f"alpha must be between 0.5 and 3.0, got {self.alpha}")
            valid = False
        
        # Validate epsilon
        if not 1e-10 <= self.epsilon <= 1.0:
            errors.append(f"epsilon must be between 1e-10 and 1.0, got {self.epsilon}")
            valid = False
        
        # Validate kappa
        if not 0.1 <= self.kappa <= 10.0:
            errors.append(f"kappa must be between 0.1 and 10.0, got {self.kappa}")
            valid = False
        
        # Validate weights
        for name, weight in self.weights.items():
            if not 0.1 <= weight <= 10.0:
                errors.append(f"weight '{name}' must be between 0.1 and 10.0, got {weight}")
                valid = False
        
        # Validate TAD sizes
        if self.min_tad_size >= self.max_tad_size:
            errors.append(f"min_tad_size must be less than max_tad_size")
            valid = False
        
        if errors:
            console = Console()
            console.print("[red]Configuration validation errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
        
        return valid


class ConfigManager:
    """Manages polymer physics configuration."""
    
    DEFAULT_CONFIG_FILE = ".tailchasing_polymer.yml"
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.console = Console()
        self.config_path = config_path or Path(self.DEFAULT_CONFIG_FILE)
        self._config: Optional[PolymerConfig] = None
    
    def load(self) -> PolymerConfig:
        """
        Load configuration from file or create default.
        
        Returns:
            Loaded or default configuration
        """
        if self._config is not None:
            return self._config
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self._config = PolymerConfig.from_dict(data)
                    self.console.print(f"[green]Loaded config from {self.config_path}[/green]")
            except Exception as e:
                self.console.print(f"[red]Error loading config: {e}[/red]")
                self._config = PolymerConfig()
        else:
            self._config = PolymerConfig()
            self.console.print("[yellow]Using default configuration[/yellow]")
        
        # Apply environment overrides
        self._apply_env_overrides()
        
        return self._config
    
    def save(self, config: Optional[PolymerConfig] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if None)
            
        Returns:
            True if successful
        """
        config = config or self._config or PolymerConfig()
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            self.console.print(f"[green]Saved config to {self.config_path}[/green]")
            return True
        except Exception as e:
            self.console.print(f"[red]Error saving config: {e}[/red]")
            return False
    
    def update(self, **kwargs) -> PolymerConfig:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            Updated configuration
        """
        if self._config is None:
            self._config = self.load()
        
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            else:
                self.console.print(f"[yellow]Warning: Unknown parameter '{key}'[/yellow]")
        
        return self._config
    
    def reset(self) -> PolymerConfig:
        """
        Reset to default configuration.
        
        Returns:
            Default configuration
        """
        self._config = PolymerConfig()
        self.console.print("[yellow]Reset to default configuration[/yellow]")
        return self._config
    
    def display(self, config: Optional[PolymerConfig] = None):
        """
        Display configuration in a formatted panel.
        
        Args:
            config: Configuration to display (uses current if None)
        """
        config = config or self._config or self.load()
        
        # Convert to YAML for display
        yaml_str = yaml.dump(config.to_dict(), default_flow_style=False)
        
        # Create syntax-highlighted panel
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
        panel = Panel(
            syntax,
            title="[bold cyan]Polymer Physics Configuration[/bold cyan]",
            border_style="cyan"
        )
        
        self.console.print(panel)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        if self._config is None:
            return
        
        # Check for environment overrides
        env_prefix = "TAILCHASING_POLYMER_"
        
        # Alpha override
        if env_alpha := os.getenv(f"{env_prefix}ALPHA"):
            try:
                self._config.alpha = float(env_alpha)
                self.console.print(f"[cyan]Applied env override: alpha={self._config.alpha}[/cyan]")
            except ValueError:
                self.console.print(f"[red]Invalid env value for alpha: {env_alpha}[/red]")
        
        # Epsilon override
        if env_epsilon := os.getenv(f"{env_prefix}EPSILON"):
            try:
                self._config.epsilon = float(env_epsilon)
                self.console.print(f"[cyan]Applied env override: epsilon={self._config.epsilon}[/cyan]")
            except ValueError:
                self.console.print(f"[red]Invalid env value for epsilon: {env_epsilon}[/red]")
        
        # Kappa override
        if env_kappa := os.getenv(f"{env_prefix}KAPPA"):
            try:
                self._config.kappa = float(env_kappa)
                self.console.print(f"[cyan]Applied env override: kappa={self._config.kappa}[/cyan]")
            except ValueError:
                self.console.print(f"[red]Invalid env value for kappa: {env_kappa}[/red]")
        
        # Weight overrides
        for weight_name in ["tok", "ast", "mod", "git"]:
            if env_weight := os.getenv(f"{env_prefix}WEIGHT_{weight_name.upper()}"):
                try:
                    self._config.weights[weight_name] = float(env_weight)
                    self.console.print(f"[cyan]Applied env override: weight_{weight_name}={self._config.weights[weight_name]}[/cyan]")
                except ValueError:
                    self.console.print(f"[red]Invalid env value for weight_{weight_name}: {env_weight}[/red]")


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """
    Get global config manager instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Config manager instance
    """
    global _config_manager
    
    if _config_manager is None or (config_path and config_path != _config_manager.config_path):
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def get_config(config_path: Optional[Path] = None) -> PolymerConfig:
    """
    Get current configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Current configuration
    """
    manager = get_config_manager(config_path)
    return manager.load()


def save_config(config: PolymerConfig, config_path: Optional[Path] = None) -> bool:
    """
    Save configuration.
    
    Args:
        config: Configuration to save
        config_path: Optional path to config file
        
    Returns:
        True if successful
    """
    manager = get_config_manager(config_path)
    return manager.save(config)


def create_default_config_file(path: Optional[Path] = None) -> bool:
    """
    Create a default configuration file.
    
    Args:
        path: Path for config file
        
    Returns:
        True if successful
    """
    path = path or Path(ConfigManager.DEFAULT_CONFIG_FILE)
    config = PolymerConfig()
    
    try:
        with open(path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        console = Console()
        console.print(f"[green]Created default config file at {path}[/green]")
        return True
    except Exception as e:
        console = Console()
        console.print(f"[red]Error creating config file: {e}[/red]")
        return False


if __name__ == "__main__":
    # Demo: Show config management
    console = Console()
    console.print("[bold cyan]Polymer Physics Configuration Demo[/bold cyan]\n")
    
    # Create manager
    manager = ConfigManager()
    
    # Load config
    config = manager.load()
    
    # Display
    manager.display(config)
    
    # Validate
    if config.validate():
        console.print("\n[green]✓ Configuration is valid[/green]")
    else:
        console.print("\n[red]✗ Configuration has errors[/red]")
    
    # Update a parameter
    console.print("\n[yellow]Updating alpha to 1.5...[/yellow]")
    config = manager.update(alpha=1.5)
    
    console.print(f"New alpha value: {config.alpha}")