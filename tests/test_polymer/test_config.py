"""Tests for polymer physics configuration module."""

import pytest
import tempfile
import yaml
from pathlib import Path
import os

from tailchasing.polymer.config import (
    PolymerConfig,
    ConfigManager,
    get_config_manager,
    get_config,
    save_config,
    create_default_config_file
)


class TestPolymerConfig:
    """Test PolymerConfig dataclass."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = PolymerConfig()
        
        assert config.alpha == 1.2
        assert config.epsilon == 1e-6
        assert config.kappa == 1.0
        assert config.tad_penalty == 0.7
        
        assert config.weights["tok"] == 1.0
        assert config.weights["ast"] == 2.0
        assert config.weights["mod"] == 3.0
        assert config.weights["git"] == 4.0
        
        assert config.min_tad_size == 10
        assert config.max_tad_size == 1000
        assert config.thrash_risk_threshold == 0.7
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = PolymerConfig(alpha=1.5)
        data = config.to_dict()
        
        assert data["alpha"] == 1.5
        assert "weights" in data
        assert "epsilon" in data
        assert "kappa" in data
    
    def test_config_from_dict(self):
        """Test configuration deserialization."""
        data = {
            "alpha": 2.0,
            "epsilon": 1e-7,
            "kappa": 2.0,
            "weights": {"tok": 2.0, "ast": 3.0, "mod": 4.0, "git": 5.0},
            "min_tad_size": 20,
            "max_tad_size": 500
        }
        
        config = PolymerConfig.from_dict(data)
        
        assert config.alpha == 2.0
        assert config.epsilon == 1e-7
        assert config.kappa == 2.0
        assert config.weights["tok"] == 2.0
        assert config.min_tad_size == 20
        assert config.max_tad_size == 500
    
    def test_config_validation_valid(self):
        """Test validation with valid configuration."""
        config = PolymerConfig()
        assert config.validate() is True
    
    def test_config_validation_invalid_alpha(self, capsys):
        """Test validation with invalid alpha."""
        config = PolymerConfig(alpha=5.0)  # Out of range
        assert config.validate() is False
        
        captured = capsys.readouterr()
        assert "alpha must be between" in captured.out
    
    def test_config_validation_invalid_weights(self, capsys):
        """Test validation with invalid weights."""
        config = PolymerConfig()
        config.weights["tok"] = 20.0  # Out of range
        assert config.validate() is False
        
        captured = capsys.readouterr()
        assert "weight 'tok' must be between" in captured.out
    
    def test_config_validation_invalid_tad_sizes(self, capsys):
        """Test validation with invalid TAD sizes."""
        config = PolymerConfig(min_tad_size=100, max_tad_size=50)
        assert config.validate() is False
        
        captured = capsys.readouterr()
        assert "min_tad_size must be less than max_tad_size" in captured.out


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config_data = {
                "alpha": 1.5,
                "epsilon": 1e-7,
                "kappa": 1.5,
                "weights": {
                    "tok": 1.5,
                    "ast": 2.5,
                    "mod": 3.5,
                    "git": 4.5
                }
            }
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        yield temp_path
        
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    def manager_with_temp_file(self, temp_config_file):
        """Create ConfigManager with temporary file."""
        return ConfigManager(temp_config_file)
    
    def test_load_existing_config(self, manager_with_temp_file):
        """Test loading existing configuration."""
        config = manager_with_temp_file.load()
        
        assert config.alpha == 1.5
        assert config.epsilon == 1e-7
        assert config.kappa == 1.5
        assert config.weights["tok"] == 1.5
    
    def test_load_nonexistent_config(self):
        """Test loading when config file doesn't exist."""
        manager = ConfigManager(Path("nonexistent.yml"))
        config = manager.load()
        
        # Should return default config
        assert config.alpha == 1.2
        assert config.epsilon == 1e-6
    
    def test_save_config(self, temp_config_file):
        """Test saving configuration."""
        manager = ConfigManager(temp_config_file)
        config = PolymerConfig(alpha=2.5)
        
        assert manager.save(config) is True
        
        # Verify saved content
        with open(temp_config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        assert data["alpha"] == 2.5
    
    def test_update_config(self, manager_with_temp_file):
        """Test updating configuration."""
        config = manager_with_temp_file.update(alpha=3.0, kappa=2.5)
        
        assert config.alpha == 3.0
        assert config.kappa == 2.5
    
    def test_update_unknown_parameter(self, manager_with_temp_file, capsys):
        """Test updating with unknown parameter."""
        manager_with_temp_file.update(unknown_param=123)
        
        captured = capsys.readouterr()
        assert "Unknown parameter 'unknown_param'" in captured.out
    
    def test_reset_config(self, manager_with_temp_file):
        """Test resetting to default configuration."""
        # First modify
        manager_with_temp_file.update(alpha=2.5)
        
        # Then reset
        config = manager_with_temp_file.reset()
        
        assert config.alpha == 1.2  # Default value
        assert config.epsilon == 1e-6  # Default value
    
    def test_env_override_alpha(self, manager_with_temp_file, monkeypatch):
        """Test environment variable override for alpha."""
        monkeypatch.setenv("TAILCHASING_POLYMER_ALPHA", "2.8")
        
        config = manager_with_temp_file.load()
        
        assert config.alpha == 2.8
    
    def test_env_override_weights(self, manager_with_temp_file, monkeypatch):
        """Test environment variable override for weights."""
        monkeypatch.setenv("TAILCHASING_POLYMER_WEIGHT_TOK", "5.0")
        monkeypatch.setenv("TAILCHASING_POLYMER_WEIGHT_AST", "6.0")
        
        config = manager_with_temp_file.load()
        
        assert config.weights["tok"] == 5.0
        assert config.weights["ast"] == 6.0
    
    def test_env_override_invalid_value(self, manager_with_temp_file, monkeypatch, capsys):
        """Test environment variable with invalid value."""
        monkeypatch.setenv("TAILCHASING_POLYMER_ALPHA", "not_a_number")
        
        config = manager_with_temp_file.load()
        
        captured = capsys.readouterr()
        assert "Invalid env value for alpha" in captured.out
        # Should keep original value
        assert config.alpha == 1.5  # From temp file


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def test_get_config_manager(self):
        """Test getting global config manager."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_get_config_manager_different_path(self):
        """Test getting config manager with different path."""
        manager1 = get_config_manager(Path("config1.yml"))
        manager2 = get_config_manager(Path("config2.yml"))
        
        # Should create new instance for different path
        assert manager1 is not manager2
    
    def test_get_config(self):
        """Test getting configuration."""
        config = get_config()
        
        assert isinstance(config, PolymerConfig)
    
    def test_save_config_function(self):
        """Test save_config function."""
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config = PolymerConfig(alpha=2.2)
            result = save_config(config, temp_path)
            
            assert result is True
            
            # Verify saved
            with open(temp_path, 'r') as f:
                data = yaml.safe_load(f)
            assert data["alpha"] == 2.2
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_create_default_config_file(self):
        """Test creating default config file."""
        with tempfile.NamedTemporaryFile(suffix='.yml', delete=False) as f:
            temp_path = Path(f.name)
        
        # Remove the file so we can create it
        temp_path.unlink()
        
        try:
            result = create_default_config_file(temp_path)
            
            assert result is True
            assert temp_path.exists()
            
            # Verify content
            with open(temp_path, 'r') as f:
                data = yaml.safe_load(f)
            
            assert data["alpha"] == 1.2  # Default value
            assert data["epsilon"] == 1e-6  # Default value
        finally:
            if temp_path.exists():
                temp_path.unlink()