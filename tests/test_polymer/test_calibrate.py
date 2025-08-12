"""Tests for polymer physics calibration module."""

import pytest
import numpy as np
from typing import List

from tailchasing.polymer.calibrate import (
    ThrashEvent,
    CodebaseMetrics,
    CalibrationResult,
    CalibrationTool
)


class TestThrashEvent:
    """Test ThrashEvent dataclass."""
    
    def test_thrash_event_creation(self):
        """Test ThrashEvent creation."""
        event = ThrashEvent(
            file1="file1.py",
            file2="file2.py",
            distance_type="tok",
            observed_frequency=0.7,
            latency_ms=150.0,
            timestamp=1000.0
        )
        
        assert event.file1 == "file1.py"
        assert event.file2 == "file2.py"
        assert event.distance_type == "tok"
        assert event.observed_frequency == 0.7
        assert event.latency_ms == 150.0
        assert event.timestamp == 1000.0
    
    def test_thrash_event_to_dict(self):
        """Test ThrashEvent serialization."""
        event = ThrashEvent(
            file1="test1.py",
            file2="test2.py",
            distance_type="ast",
            observed_frequency=0.5,
            latency_ms=100.0,
            timestamp=500.0
        )
        
        data = event.to_dict()
        
        assert data["file1"] == "test1.py"
        assert data["file2"] == "test2.py"
        assert data["distance_type"] == "ast"
        assert data["observed_frequency"] == 0.5
        assert data["latency_ms"] == 100.0
        assert data["timestamp"] == 500.0


class TestCodebaseMetrics:
    """Test CodebaseMetrics dataclass."""
    
    def test_codebase_metrics_creation(self):
        """Test CodebaseMetrics creation."""
        metrics = CodebaseMetrics(
            total_files=100,
            total_lines=10000,
            module_count=10,
            avg_file_size=100.0,
            complexity_score=0.6,
            interaction_density=0.3
        )
        
        assert metrics.total_files == 100
        assert metrics.total_lines == 10000
        assert metrics.module_count == 10
        assert metrics.avg_file_size == 100.0
        assert metrics.complexity_score == 0.6
        assert metrics.interaction_density == 0.3
    
    def test_codebase_metrics_to_dict(self):
        """Test CodebaseMetrics serialization."""
        metrics = CodebaseMetrics(
            total_files=50,
            total_lines=5000,
            module_count=5,
            avg_file_size=100.0,
            complexity_score=0.5,
            interaction_density=0.2
        )
        
        data = metrics.to_dict()
        
        assert data["total_files"] == 50
        assert data["total_lines"] == 5000
        assert data["module_count"] == 5
        assert data["avg_file_size"] == 100.0
        assert data["complexity_score"] == 0.5
        assert data["interaction_density"] == 0.2


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""
    
    def test_calibration_result_creation(self):
        """Test CalibrationResult creation."""
        result = CalibrationResult(
            alpha=1.5,
            weights={"tok": 1.0, "ast": 2.0, "mod": 3.0, "git": 4.0},
            epsilon=1e-6,
            kappa=1.0,
            correlation_score=0.85,
            mse=0.1,
            iterations=50,
            converged=True
        )
        
        assert result.alpha == 1.5
        assert result.weights["tok"] == 1.0
        assert result.weights["ast"] == 2.0
        assert result.epsilon == 1e-6
        assert result.kappa == 1.0
        assert result.correlation_score == 0.85
        assert result.mse == 0.1
        assert result.iterations == 50
        assert result.converged is True
    
    def test_calibration_result_to_dict(self):
        """Test CalibrationResult serialization."""
        result = CalibrationResult(
            alpha=1.2,
            weights={"tok": 1.5, "ast": 2.5, "mod": 3.5, "git": 4.5},
            epsilon=1e-7,
            kappa=2.0,
            correlation_score=0.9,
            mse=0.05,
            iterations=100,
            converged=False
        )
        
        data = result.to_dict()
        
        assert data["alpha"] == 1.2
        assert data["weights"]["tok"] == 1.5
        assert data["epsilon"] == 1e-7
        assert data["kappa"] == 2.0
        assert data["correlation_score"] == 0.9
        assert data["mse"] == 0.05
        assert data["iterations"] == 100
        assert data["converged"] is False


class TestCalibrationTool:
    """Test CalibrationTool functionality."""
    
    @pytest.fixture
    def tool(self):
        """Create CalibrationTool instance."""
        return CalibrationTool()
    
    @pytest.fixture
    def sample_events(self) -> List[ThrashEvent]:
        """Create sample thrash events."""
        events = []
        np.random.seed(42)
        
        for i in range(20):
            event = ThrashEvent(
                file1=f"file{i}.py",
                file2=f"file{i+1}.py",
                distance_type=np.random.choice(["tok", "ast", "mod", "git"]),
                observed_frequency=np.random.uniform(0, 1),
                latency_ms=np.random.uniform(10, 500),
                timestamp=i * 100
            )
            events.append(event)
        
        return events
    
    @pytest.fixture
    def sample_codebase(self) -> CodebaseMetrics:
        """Create sample codebase metrics."""
        return CodebaseMetrics(
            total_files=100,
            total_lines=10000,
            module_count=10,
            avg_file_size=100.0,
            complexity_score=0.6,
            interaction_density=0.3
        )
    
    def test_generate_demo_data(self, tool):
        """Test demo data generation."""
        events, codebase = tool.generate_demo_data(n_events=50, seed=42)
        
        assert len(events) == 50
        assert isinstance(codebase, CodebaseMetrics)
        
        # Check event properties
        for event in events:
            assert isinstance(event, ThrashEvent)
            assert event.distance_type in ["tok", "ast", "mod", "git"]
            assert 0 <= event.observed_frequency <= 1
            assert event.latency_ms > 0
    
    def test_fit_parameters_with_data(self, tool, sample_events, sample_codebase):
        """Test parameter fitting with data."""
        result = tool.fit_parameters(
            sample_events,
            sample_codebase,
            initial_alpha=1.2,
            max_iterations=10
        )
        
        assert isinstance(result, CalibrationResult)
        assert 0.5 <= result.alpha <= 3.0
        assert isinstance(result.weights, dict)
        assert all(k in result.weights for k in ["tok", "ast", "mod", "git"])
        assert result.iterations <= 10
    
    def test_fit_parameters_no_data(self, tool, sample_codebase):
        """Test parameter fitting with no data."""
        result = tool.fit_parameters([], sample_codebase)
        
        assert isinstance(result, CalibrationResult)
        assert result.converged is False
        assert result.iterations == 0
    
    def test_fit_parameters_insufficient_data(self, tool, sample_codebase):
        """Test parameter fitting with insufficient data."""
        events = [
            ThrashEvent(
                file1="f1.py",
                file2="f2.py",
                distance_type="tok",
                observed_frequency=0.5,
                latency_ms=100,
                timestamp=0
            )
        ]
        
        result = tool.fit_parameters(events, sample_codebase)
        
        assert isinstance(result, CalibrationResult)
        assert result.converged is False
    
    def test_grid_search(self, tool, sample_events, sample_codebase):
        """Test grid search calibration."""
        result = tool.grid_search(
            sample_events,
            sample_codebase,
            alpha_range=(0.8, 1.5),
            n_grid_points=2
        )
        
        assert isinstance(result, CalibrationResult)
        assert 0.8 <= result.alpha <= 1.5
    
    def test_validate_parameters(self, tool, sample_events):
        """Test parameter validation."""
        # Create a known result
        result = CalibrationResult(
            alpha=1.2,
            weights={"tok": 1.0, "ast": 2.0, "mod": 3.0, "git": 4.0},
            epsilon=1e-6,
            kappa=1.0,
            correlation_score=0.5,
            mse=0.1,
            iterations=10,
            converged=True
        )
        
        # Validate on test events
        test_events = sample_events[:5]
        validation = tool.validate_parameters(result, test_events)
        
        assert "correlation" in validation
        assert "mse" in validation
        assert "mae" in validation
        assert "r_squared" in validation
    
    def test_validate_parameters_no_events(self, tool):
        """Test validation with no events."""
        result = CalibrationResult(
            alpha=1.2,
            weights={"tok": 1.0, "ast": 2.0, "mod": 3.0, "git": 4.0},
            epsilon=1e-6,
            kappa=1.0,
            correlation_score=0.5,
            mse=0.1,
            iterations=10,
            converged=True
        )
        
        validation = tool.validate_parameters(result, [])
        
        assert "error" in validation
    
    def test_objective_function(self, tool):
        """Test the objective function."""
        # Create simple test data
        X = np.array([
            [1, 0, 0, 0, 0.1],
            [0, 1, 0, 0, 0.2],
            [0, 0, 1, 0, 0.3],
            [0, 0, 0, 1, 0.4]
        ])
        y = np.array([0.8, 0.6, 0.4, 0.2])
        
        params = np.array([1.2, 1e-6, 1.0, 1.0, 2.0, 3.0, 4.0])
        
        loss = tool._objective_function(params, X, y)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_predict_contacts(self, tool):
        """Test contact prediction."""
        X = np.array([
            [1, 0, 0, 0, 0.1],
            [0, 1, 0, 0, 0.2]
        ])
        
        params = np.array([1.2, 1e-6, 1.0, 1.0, 2.0, 3.0, 4.0])
        
        predictions = tool._predict_contacts(params, X)
        
        assert len(predictions) == 2
        assert all(0 <= p <= 1 for p in predictions)