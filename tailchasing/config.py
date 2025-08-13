"""
Configuration system for tailchasing polymer physics parameters.

This module defines configuration structures for polymer physics calculations,
including contact decay parameters, distance weights, and TAD patterns.
"""

import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import yaml


@dataclass
class DistanceWeights:
    """
    Distance weighting parameters for different code relationship types.

    These weights control how different types of code relationships contribute
    to the polymer distance calculations, with higher weights indicating
    stronger/closer relationships.
    """

    tok: float = 1.0  # Token-level relationships (same function)
    ast: float = 2.0  # AST-level relationships (same class/module)
    mod: float = 3.0  # Module-level relationships (same package)
    git: float = 4.0  # Git-level relationships (same commit/branch)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary representation."""
        return {"tok": self.tok, "ast": self.ast, "mod": self.mod, "git": self.git}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "DistanceWeights":
        """Create from dictionary representation."""
        return cls(
            tok=data.get("tok", 1.0),
            ast=data.get("ast", 2.0),
            mod=data.get("mod", 3.0),
            git=data.get("git", 4.0),
        )


@dataclass
class PolymerConfig:
    """
    Configuration for polymer physics calculations.

    This configuration controls the behavior of polymer distance calculations,
    contact probability modeling, and TAD boundary detection.
    """

    # Contact decay exponent (1.0 = linear, 1.2 = sublinear, 1.5 = strong decay)
    alpha: float = 1.2

    # Distance weights for different relationship types
    weights: DistanceWeights = field(default_factory=DistanceWeights)

    # Package glob patterns for TAD boundary detection
    tad_patterns: List[str] = field(
        default_factory=lambda: [
            "*.api.*",
            "*.core.*",
            "*.db.*",
            "*.ui.*",
            "*.utils.*",
            "*.models.*",
            "*.services.*",
            "*.controllers.*",
        ]
    )

    # Numerical stability parameter
    epsilon: float = 1e-6

    # Scaling parameter for contact strength normalization
    kappa: float = 1.0

    # Maximum distance for contact calculations
    max_distance: int = 1000

    # Minimum contact threshold for analysis
    min_contact_threshold: float = 0.01

    def __post_init__(self):
        """Validate configuration parameters."""
        if not (1.0 <= self.alpha <= 1.5):
            raise ValueError(f"alpha must be between 1.0 and 1.5, got {self.alpha}")

        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")

        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")

        if self.max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {self.max_distance}")

        if not (0 <= self.min_contact_threshold <= 1):
            raise ValueError(
                f"min_contact_threshold must be between 0 and 1, got {self.min_contact_threshold}"
            )

    def to_dict(self) -> Dict[str, Union[float, int, List[str], Dict[str, float]]]:
        """Convert to dictionary representation."""
        return {
            "alpha": self.alpha,
            "weights": self.weights.to_dict(),
            "tad_patterns": self.tad_patterns,
            "epsilon": self.epsilon,
            "kappa": self.kappa,
            "max_distance": self.max_distance,
            "min_contact_threshold": self.min_contact_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PolymerConfig":
        """Create from dictionary representation."""
        weights_data = data.get("weights", {})
        if isinstance(weights_data, dict):
            weights = DistanceWeights.from_dict(weights_data)
        else:
            weights = DistanceWeights()

        return cls(
            alpha=data.get("alpha", 1.2),
            weights=weights,
            tad_patterns=data.get(
                "tad_patterns",
                [
                    "*.api.*",
                    "*.core.*",
                    "*.db.*",
                    "*.ui.*",
                    "*.utils.*",
                    "*.models.*",
                    "*.services.*",
                    "*.controllers.*",
                ],
            ),
            epsilon=data.get("epsilon", 1e-6),
            kappa=data.get("kappa", 1.0),
            max_distance=data.get("max_distance", 1000),
            min_contact_threshold=data.get("min_contact_threshold", 0.01),
        )

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        file_path = Path(file_path)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "PolymerConfig":
        """Load configuration from YAML file."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data or {})

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        # Look for config in current directory, then home directory
        current_dir_config = Path(".tailchasing_polymer.yml")
        if current_dir_config.exists():
            return current_dir_config

        home_config = Path.home() / ".tailchasing_polymer.yml"
        return home_config

    @classmethod
    def load_or_default(
        cls, config_path: Optional[Union[str, Path]] = None
    ) -> "PolymerConfig":
        """
        Load configuration from file or return default if not found.

        Args:
            config_path: Optional path to configuration file

        Returns:
            PolymerConfig instance
        """
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                return cls.load_from_file(config_path)
        else:
            # Try default locations
            default_path = cls.get_default_config_path()
            if default_path.exists():
                return cls.load_from_file(default_path)

        # Return default configuration
        return cls()


@dataclass
class CalibrationResult:
    """Results from parameter calibration."""

    optimal_alpha: float
    optimal_weights: DistanceWeights
    correlation_score: float
    validation_score: float
    iterations: int
    converged: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "optimal_alpha": self.optimal_alpha,
            "optimal_weights": self.optimal_weights.to_dict(),
            "correlation_score": self.correlation_score,
            "validation_score": self.validation_score,
            "iterations": self.iterations,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CalibrationResult":
        """Create from dictionary representation."""
        return cls(
            optimal_alpha=data["optimal_alpha"],
            optimal_weights=DistanceWeights.from_dict(data["optimal_weights"]),
            correlation_score=data["correlation_score"],
            validation_score=data["validation_score"],
            iterations=data["iterations"],
            converged=data["converged"],
        )


class ConfigManager:
    """
    Manager for polymer physics configuration.

    Provides utilities for loading, saving, and managing configuration files
    with environment variable support and validation.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self._config: Optional[PolymerConfig] = None

    @property
    def config(self) -> PolymerConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def load_config(self) -> PolymerConfig:
        """Load configuration from file or environment."""
        # Check for environment variable override
        env_config_path = os.getenv("TAILCHASING_CONFIG")
        if env_config_path:
            config_path = Path(env_config_path)
            if config_path.exists():
                return PolymerConfig.load_from_file(config_path)

        # Use specified path or default
        if self.config_path and self.config_path.exists():
            return PolymerConfig.load_from_file(self.config_path)

        # Try default locations
        return PolymerConfig.load_or_default()

    def save_config(
        self, config: PolymerConfig, path: Optional[Union[str, Path]] = None
    ) -> None:
        """Save configuration to file."""
        save_path = (
            Path(path)
            if path
            else (self.config_path or PolymerConfig.get_default_config_path())
        )
        config.save_to_file(save_path)
        self._config = config

    def update_from_calibration(self, calibration_result: CalibrationResult) -> None:
        """Update configuration from calibration results."""
        current_config = self.config

        # Update with optimal parameters
        updated_config = PolymerConfig(
            alpha=calibration_result.optimal_alpha,
            weights=calibration_result.optimal_weights,
            tad_patterns=current_config.tad_patterns,
            epsilon=current_config.epsilon,
            kappa=current_config.kappa,
            max_distance=current_config.max_distance,
            min_contact_threshold=current_config.min_contact_threshold,
        )

        self.save_config(updated_config)

    def get_environment_overrides(self) -> Dict[str, Union[float, int, str]]:
        """Get configuration overrides from environment variables."""
        overrides = {}

        # Check for environment variable overrides
        env_mappings = {
            "TAILCHASING_ALPHA": ("alpha", float),
            "TAILCHASING_KAPPA": ("kappa", float),
            "TAILCHASING_EPSILON": ("epsilon", float),
            "TAILCHASING_MAX_DISTANCE": ("max_distance", int),
            "TAILCHASING_MIN_THRESHOLD": ("min_contact_threshold", float),
            "TAILCHASING_WEIGHT_TOK": ("weights.tok", float),
            "TAILCHASING_WEIGHT_AST": ("weights.ast", float),
            "TAILCHASING_WEIGHT_MOD": ("weights.mod", float),
            "TAILCHASING_WEIGHT_GIT": ("weights.git", float),
        }

        for env_var, (config_key, config_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    overrides[config_key] = config_type(env_value)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid environment variable {env_var}={env_value}: {e}"
                    )

        return overrides

    def apply_environment_overrides(self, config: PolymerConfig) -> PolymerConfig:
        """Apply environment variable overrides to configuration."""
        overrides = self.get_environment_overrides()

        if not overrides:
            return config

        # Convert to dict, apply overrides, and convert back
        config_dict = config.to_dict()

        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like "weights.tok"
                parts = key.split(".")
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[key] = value

        return PolymerConfig.from_dict(config_dict)

    def validate_config(self, config: PolymerConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        try:
            # This will raise ValueError if invalid
            config.__post_init__()
        except ValueError as e:
            issues.append(str(e))

        # Additional validation
        if config.alpha < 1.0:
            issues.append("alpha should be >= 1.0 for realistic polymer behavior")

        if config.alpha > 1.5:
            issues.append("alpha > 1.5 may cause numerical instability")

        weight_values = [
            config.weights.tok,
            config.weights.ast,
            config.weights.mod,
            config.weights.git,
        ]
        if any(w <= 0 for w in weight_values):
            issues.append("All distance weights should be positive")

        # Check for reasonable weight ordering (optional warning)
        if not (
            config.weights.tok
            <= config.weights.ast
            <= config.weights.mod
            <= config.weights.git
        ):
            issues.append(
                "Warning: weights don't follow expected ordering (tok <= ast <= mod <= git)"
            )

        return issues


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None or (
        config_path and _config_manager.config_path != Path(config_path)
    ):
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config(config_path: Optional[Union[str, Path]] = None) -> PolymerConfig:
    """Get the current polymer physics configuration."""
    manager = get_config_manager(config_path)
    return manager.config


def save_config(
    config: PolymerConfig, config_path: Optional[Union[str, Path]] = None
) -> None:
    """Save polymer physics configuration."""
    manager = get_config_manager()
    manager.save_config(config, config_path)


# Backward compatibility wrapper for CLI
class Config:
    """Configuration manager for tail-chasing detector (backward compatibility wrapper)."""
    
    DEFAULT_CONFIG = {
        "paths": {
            "include": ["."],
            "exclude": ["tests", "test", "build", "dist", "venv", ".venv", "__pycache__"]
        },
        "risk_thresholds": {
            "warn": 15,
            "fail": 30
        },
        "output": {
            "verbosity": "normal",  # Options: quiet, normal, verbose, json
            "show_progress": True,
            "show_memory_stats": False,
            "show_timing": False,
            "suppress_warnings": False
        },
        "lsh_duplicates": {
            "enabled": True,  # Enable by default for O(n·k) performance
            "num_hashes": 100,
            "bands": 20,
            "rows_per_band": 5,
            "exact_match_threshold": 0.95,
            "semantic_match_threshold": 0.85,
            "use_progressive": False  # Enable to use 3-level progressive refinement
        },
        "placeholders": {
            "allow": [],
            "block": [],
            "triage_enabled": True,
            "security_patterns": {
                "crypto": ["*verify*", "*authenticate*", "*sign*", "*encrypt*", "*decrypt*"],
                "hsm": ["*hsm*", "*hardware*security*", "*secure*element*"],
                "post_quantum": ["*dilithium*", "*kyber*", "*falcon*", "*sphincs*"]
            }
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
        },
        "canonical_policy": {
            "canonical_roots": [],
            "shadow_roots": [],
            "priority_patterns": {
                r".*test.*": -10,
                r".*experimental.*": -20,
                r".*_test\.py": -15,
                r".*/__init__\.py": 5
            },
            "auto_suppress_shadows": True,
            "generate_forwarders": True,
            "codemod_output": "./canonical_codemod.py"
        },
        "circular_import_resolver": {
            "enabled": True,
            "min_scc_size": 2,
            "generate_fixes": True,
            "fix_script_output": "./circular_import_fixes.py"
        },
        "issue_provenance": {
            "enabled": True,
            "db_path": ".tailchasing_history.db",
            "track_regressions": True,
            "git_integration": True
        },
        "playbooks": {
            "enabled": True,
            "auto_generate": True,
            "require_review_for_high_risk": True,
            "backup_before_execution": True,
            "rollback_on_failure": True,
            "output_dir": "./playbooks"
        },
        "resource_limits": {
            "max_duplicate_pairs": 200000,
            "analyzer_timeout_seconds": 600,  # Increased to 10 minutes
            "heartbeat_interval_seconds": 5,
            "max_memory_mb": 8192,
            "lsh_bucket_cap": 2000,
            "semantic_analysis_file_limit": 1000,
            "semantic_analysis_duplicate_limit": 500
        },
        "duplicates": {
            "enabled": True,
            "use_fast_detection": True,
            "lsh_bands": 8,
            "lsh_rows": 16,
            "generate_plan": True
        }
    }
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize with optional config dictionary."""
        self.config = self._merge_configs(self.DEFAULT_CONFIG, config_dict or {})
        # Also initialize polymer config
        self._polymer_config = PolymerConfig()
        
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from a file."""
        path = Path(path)
        if not path.exists():
            return cls()
            
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f) or {}
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
                
        return cls(data)
    
    @classmethod
    def find_and_load(cls, start_path: Path) -> "Config":
        """Find and load configuration from standard locations."""
        # Look for .tailchasing.yml in current and parent directories
        current = Path(start_path).resolve()
        
        while current != current.parent:
            for name in [".tailchasing.yml", ".tailchasing.yaml", "tailchasing.yml", "tailchasing.yaml"]:
                config_path = current / name
                if config_path.exists():
                    return cls.from_file(config_path)
            current = current.parent
            
        # Return default config if no file found
        return cls()
    
    def get(self, key: str, default=None):
        """Get configuration value by dot-separated key."""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
                
        return value
    
    def set(self, key: str, value):
        """Set configuration value by dot-separated key."""
        keys = key.split(".")
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set the final key
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return self.config.copy()
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result


# ----------------------------
# Large Codebase Configuration
# ----------------------------

from typing import Literal

# Optional imports (fall back to stubs if these modules aren't present yet)
try:
    from tailchasing.semantic.lsh_index import LSHParams  # rows_per_band * bands == num_hashes
except Exception:  # pragma: no cover
    @dataclass
    class LSHParams:
        num_hashes: int = 100
        bands: int = 20
        rows_per_band: int = 5
        seed: int = 0x5EED_1DEE

try:
    from tailchasing.performance import ParallelConfig
except Exception:  # pragma: no cover
    @dataclass
    class ParallelConfig:
        max_workers: int = os.cpu_count() or 4
        bucket_timeout_sec: float = 5.0
        lsh_bands: int = 20
        expected_pairs: int = 2_000_000
        quick_gate: float = 0.30
        medium_gate: float = 0.50

SamplingStrategy = Literal["intelligent", "random", "stratified"]


@dataclass
class LargeCodebaseConfig:
    """Unified config for 'large codebase mode' tuning and resource limits."""
    # Thresholds
    large_codebase_threshold: int = 1000     # functions
    huge_codebase_threshold: int = 5000

    # Sampling
    max_sample_size: int = 500
    sampling_strategy: SamplingStrategy = "intelligent"  # or "random", "stratified"

    # LSH parameters
    lsh_bands: int = 20
    lsh_rows: int = 5
    minhash_permutations: int = 100  # = lsh_bands * lsh_rows ideally

    # Performance
    parallel_buckets: bool = True
    max_bucket_size: int = 100
    comparison_timeout: float = 60.0  # seconds per bucket - much more generous

    # Timeout configuration - Much more generous timeouts for large codebases
    analyzer_timeout_sec: float = 600.0  # Total analyzer timeout (10 minutes)
    group_timeout_sec: float = 120.0     # Per-group timeout (2 minutes)
    watchdog_timeout_sec: float = 0.0    # Watchdog timeout (0 = disabled)

    # Memory management
    use_disk_cache: bool = True
    cache_dir: str = ".tailchasing_cache"
    max_memory_mb: int = 2000

    # Misc
    seed: int = 0x5EED_1DEE

    # --- Lifecycle helpers ---

    def validate(self) -> None:
        if self.large_codebase_threshold <= 0:
            raise ValueError("large_codebase_threshold must be > 0")
        if self.huge_codebase_threshold < self.large_codebase_threshold:
            raise ValueError("huge_codebase_threshold must be >= large_codebase_threshold")
        if self.max_sample_size <= 0:
            raise ValueError("max_sample_size must be > 0")
        if self.lsh_bands <= 0 or self.lsh_rows <= 0:
            raise ValueError("lsh_bands and lsh_rows must be > 0")
        if self.minhash_permutations != self.lsh_bands * self.lsh_rows:
            # Not fatal, but warn via exception by default to avoid silent mismatch
            raise ValueError(
                f"minhash_permutations ({self.minhash_permutations}) must equal lsh_bands*lsh_rows "
                f"({self.lsh_bands*self.lsh_rows})"
            )
        if self.max_bucket_size < 2:
            raise ValueError("max_bucket_size must be >= 2")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be > 0")

    # --- Converters into other modules' configs ---

    def to_lsh_params(self) -> LSHParams:
        """Project into tailchasing.semantic.index.LSHParams."""
        return LSHParams(
            num_hashes=self.minhash_permutations,
            bands=self.lsh_bands,
            rows_per_band=self.lsh_rows,
            seed=self.seed,
        )

    def to_parallel_config(self) -> ParallelConfig:
        """Project into tailchasing.performance.ParallelConfig."""
        import os
        return ParallelConfig(
            max_workers=os.cpu_count() or 4,
            bucket_timeout_sec=float(self.comparison_timeout),
            lsh_bands=self.lsh_bands,
            # expected_pairs left as default; adjust upstream if needed
        )

    # --- Environment / dict loaders ---

    @classmethod
    def from_env(cls, prefix: str = "TAILCHASING_") -> "LargeCodebaseConfig":
        """
        Load overrides from environment variables (all optional).
        Example vars:
          TAILCHASING_LARGE_THRESHOLD=1500
          TAILCHASING_SAMPLING_STRATEGY=stratified
          TAILCHASING_LSH_BANDS=32
        """
        def get_int(name: str, default: int) -> int:
            v = os.getenv(prefix + name)
            return int(v) if v is not None else default

        def get_float(name: str, default: float) -> float:
            v = os.getenv(prefix + name)
            return float(v) if v is not None else default

        def get_bool(name: str, default: bool) -> bool:
            v = os.getenv(prefix + name)
            if v is None:
                return default
            return v.strip().lower() in ("1", "true", "yes", "on")

        def get_str(name: str, default: str) -> str:
            v = os.getenv(prefix + name)
            return v if v is not None else default

        cfg = cls(
            large_codebase_threshold=get_int("LARGE_THRESHOLD", 1000),
            huge_codebase_threshold=get_int("HUGE_THRESHOLD", 5000),
            max_sample_size=get_int("MAX_SAMPLE", 500),
            sampling_strategy=get_str("SAMPLING_STRATEGY", "intelligent"),  # type: ignore[arg-type]
            lsh_bands=get_int("LSH_BANDS", 20),
            lsh_rows=get_int("LSH_ROWS", 5),
            minhash_permutations=get_int("MINHASH_PERMUTATIONS", 100),
            parallel_buckets=get_bool("PARALLEL_BUCKETS", True),
            max_bucket_size=get_int("MAX_BUCKET_SIZE", 100),
            comparison_timeout=get_float("COMPARISON_TIMEOUT", 60.0),
            analyzer_timeout_sec=get_float("ANALYZER_TIMEOUT_SEC", 600.0),
            group_timeout_sec=get_float("GROUP_TIMEOUT_SEC", 120.0),
            watchdog_timeout_sec=get_float("WATCHDOG_SEC", 0.0),
            use_disk_cache=get_bool("USE_DISK_CACHE", True),
            cache_dir=get_str("CACHE_DIR", ".tailchasing_cache"),
            max_memory_mb=get_int("MAX_MEMORY_MB", 2000),
            seed=get_int("SEED", 0x5EED_1DEE),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LargeCodebaseConfig":
        """Create config from a (possibly partial) dict of overrides."""
        # Start with defaults, replace provided keys
        from dataclasses import fields as dataclass_fields
        field_names = {f.name for f in dataclass_fields(cls)}
        kwargs = {}
        for k, v in data.items():
            if k in field_names:
                kwargs[k] = v
        cfg = cls(**kwargs)  # type: ignore[arg-type]
        cfg.validate()
        return cfg

    # --- Convenience ---

    def as_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def apply_to_analyzer(self, analyzer: Any) -> None:
        """
        Convenience hook to push relevant knobs into an analyzer instance
        (if it exposes similarly named attributes).
        """
        # LSH params
        if hasattr(analyzer, "lsh_params"):
            analyzer.lsh_params = self.to_lsh_params()
        # parallel/timeout
        if hasattr(analyzer, "config"):
            try:
                pc = analyzer.config  # e.g., ParallelConfig
                if hasattr(pc, "lsh_bands"):
                    pc.lsh_bands = self.lsh_bands
                if hasattr(pc, "bucket_timeout_sec"):
                    pc.bucket_timeout_sec = float(self.comparison_timeout)
            except Exception:
                pass
        # timeout configuration
        for attr, val in [
            ("timeout_seconds", self.analyzer_timeout_sec),
            ("group_timeout_seconds", self.group_timeout_sec),
        ]:
            if hasattr(analyzer, attr):
                setattr(analyzer, attr, val)
        # sampling knobs (if present)
        for attr, val in [
            ("max_sample_size", self.max_sample_size),
            ("sampling_strategy", self.sampling_strategy),
            ("max_bucket_size", self.max_bucket_size),
            ("parallel_buckets", self.parallel_buckets),
        ]:
            if hasattr(analyzer, attr):
                setattr(analyzer, attr, val)