"""Tests for polymer physics reporting module."""

import pytest
import numpy as np
from typing import List, Tuple

from tailchasing.polymer.reporting import (
    TAD,
    ThrashCluster,
    HiCHeatmapGenerator,
    PolymerMetricsReport,
    integrate_chromatin_analysis,
    generate_comparative_matrices
)


class TestTAD:
    """Test TAD class functionality."""
    
    def test_tad_creation(self):
        """Test TAD creation and properties."""
        tad = TAD(start=10, end=50, name="test_tad", activity_level=0.7)
        
        assert tad.start == 10
        assert tad.end == 50
        assert tad.name == "test_tad"
        assert tad.activity_level == 0.7
        assert tad.size() == 40
    
    def test_tad_contains(self):
        """Test TAD boundary checking."""
        tad = TAD(start=10, end=50, name="test_tad")
        
        assert tad.contains(10) is True  # Start boundary
        assert tad.contains(50) is True  # End boundary
        assert tad.contains(30) is True  # Inside
        assert tad.contains(9) is False  # Before
        assert tad.contains(51) is False  # After


class TestThrashCluster:
    """Test ThrashCluster functionality."""
    
    def test_cluster_creation(self):
        """Test ThrashCluster creation."""
        cluster = ThrashCluster(
            positions=[10, 20, 30],
            risk_score=0.8,
            frequency=5,
            avg_latency=100.0
        )
        
        assert cluster.positions == [10, 20, 30]
        assert cluster.risk_score == 0.8
        assert cluster.frequency == 5
        assert cluster.avg_latency == 100.0
    
    def test_cluster_center(self):
        """Test cluster center calculation."""
        cluster = ThrashCluster(
            positions=[10, 20, 30],
            risk_score=0.5,
            frequency=3,
            avg_latency=50.0
        )
        
        assert cluster.center() == 20.0
        
        # Test empty cluster
        empty_cluster = ThrashCluster(
            positions=[],
            risk_score=0.0,
            frequency=0,
            avg_latency=0.0
        )
        assert empty_cluster.center() == 0.0


class TestHiCHeatmapGenerator:
    """Test Hi-C heatmap generation."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return HiCHeatmapGenerator()
    
    @pytest.fixture
    def sample_matrix(self):
        """Create sample contact matrix."""
        np.random.seed(42)
        return np.random.rand(10, 10)
    
    def test_heatmap_generation(self, generator, sample_matrix):
        """Test basic heatmap generation."""
        heatmap = generator.generate_contact_heatmap(sample_matrix)
        
        assert isinstance(heatmap, str)
        assert "Performance Contact Matrix" in heatmap
        assert "Scale:" in heatmap
    
    def test_empty_matrix(self, generator):
        """Test handling of empty matrix."""
        empty_matrix = np.array([])
        heatmap = generator.generate_contact_heatmap(empty_matrix)
        
        assert heatmap == "Empty contact matrix"
    
    def test_thrash_cluster_highlighting(self, generator, sample_matrix):
        """Test thrash cluster highlighting."""
        risk_scores = {
            (2, 3): 0.9,  # High risk
            (5, 5): 0.6,  # Medium risk
            (7, 8): 0.3,  # Low risk
        }
        
        result = generator.highlight_thrash_clusters(sample_matrix, risk_scores)
        
        assert isinstance(result, str)
        assert "Thrash Risk Clusters" in result
        assert "Risk indicators:" in result
    
    def test_tad_boundary_visualization(self, generator, sample_matrix):
        """Test TAD boundary visualization."""
        tad_map = {
            "module1": TAD(0, 30, "module1", 0.8),
            "module2": TAD(31, 60, "module2", 0.6)
        }
        
        result = generator.show_tad_boundaries(sample_matrix, tad_map)
        
        assert isinstance(result, str)
        assert "TAD Boundary Visualization" in result
        assert "Activity levels:" in result
        assert "module1" in result
        assert "module2" in result


class TestPolymerMetricsReport:
    """Test polymer metrics reporting."""
    
    @pytest.fixture
    def reporter(self):
        """Create reporter instance."""
        return PolymerMetricsReport(alpha=1.2)
    
    @pytest.fixture
    def sample_tads(self) -> List[TAD]:
        """Create sample TADs."""
        return [
            TAD(start=0, end=30, name="tad1", activity_level=0.8),
            TAD(start=31, end=60, name="tad2", activity_level=0.6),
            TAD(start=61, end=100, name="tad3", activity_level=0.9)
        ]
    
    @pytest.fixture
    def sample_interactions(self) -> List[Tuple[int, int, float]]:
        """Create sample interactions."""
        return [
            (10, 20, 0.8),  # Within tad1
            (35, 45, 0.7),  # Within tad2
            (70, 80, 0.9),  # Within tad3
            (15, 40, 0.4),  # Between tad1 and tad2
            (50, 75, 0.3),  # Between tad2 and tad3
        ]
    
    def test_polymer_distance_calculation(self, reporter, sample_tads, sample_interactions):
        """Test polymer distance metrics calculation."""
        metrics = reporter.calculate_polymer_distances(sample_tads, sample_interactions)
        
        assert "intra_tad_distances" in metrics
        assert "inter_tad_distances" in metrics
        assert "global_metrics" in metrics
        
        # Check intra-TAD metrics
        assert "tad1" in metrics["intra_tad_distances"]
        assert "mean" in metrics["intra_tad_distances"]["tad1"]
        assert "std" in metrics["intra_tad_distances"]["tad1"]
        
        # Check global metrics
        assert "mean_distance" in metrics["global_metrics"]
        assert "total_interactions" in metrics["global_metrics"]
        assert metrics["global_metrics"]["total_interactions"] == 5
    
    def test_contact_probability_calculation(self, reporter, sample_interactions):
        """Test contact probability decay calculation."""
        result = reporter.calculate_contact_probabilities(sample_interactions)
        
        assert "distance_bins" in result
        assert "probabilities" in result
        assert "decay_rate" in result
        assert "statistics" in result
        
        stats = result["statistics"]
        assert "mean_contact_distance" in stats
        assert "contact_decay_rate" in stats
        assert "short_range_fraction" in stats
        assert "long_range_fraction" in stats
    
    def test_thrash_reduction_prediction(self, reporter):
        """Test thrash reduction predictions."""
        strategies = [
            {
                "name": "strategy1",
                "impact_score": 0.8,
                "complexity": 0.3,
                "confidence": 0.9
            },
            {
                "name": "strategy2",
                "impact_score": 0.5,
                "complexity": 0.7,
                "confidence": 0.6
            }
        ]
        
        predictions = reporter.predict_thrash_reduction(strategies)
        
        assert "strategy1" in predictions
        assert "strategy2" in predictions
        
        for name, pred in predictions.items():
            assert "estimated_reduction" in pred
            assert "implementation_risk" in pred
            assert "roi_score" in pred
            assert "recommended_priority" in pred
            assert pred["recommended_priority"] in ["High", "Medium", "Low"]
    
    def test_replication_timing_visualization(self, reporter):
        """Test replication timing visualization."""
        timeline_data = [
            {
                "name": "event1",
                "timestamp": 0,
                "duration": 50,
                "impact": 0.8,
                "status": "completed"
            },
            {
                "name": "event2",
                "timestamp": 30,
                "duration": 40,
                "impact": 0.5,
                "status": "completed"
            }
        ]
        
        result = reporter.visualize_replication_timing(timeline_data)
        
        assert isinstance(result, str)
        assert "Replication Timing Schedule" in result
        assert "event1" in result
        assert "event2" in result
        assert "Time scale:" in result
    
    def test_comprehensive_report_generation(self, reporter, sample_tads, sample_interactions):
        """Test comprehensive report generation."""
        fix_strategies = [
            {
                "name": "fix1",
                "impact_score": 0.7,
                "complexity": 0.4,
                "confidence": 0.8
            }
        ]
        
        timeline_data = [
            {
                "name": "process1",
                "timestamp": 0,
                "duration": 100,
                "impact": 0.6,
                "status": "active"
            }
        ]
        
        report = reporter.generate_comprehensive_report(
            sample_tads,
            sample_interactions,
            fix_strategies,
            timeline_data
        )
        
        assert "polymer_distances" in report
        assert "contact_probabilities" in report
        assert "thrash_predictions" in report
        assert "timeline_analysis" in report
        assert "summary_metrics" in report
        
        summary = report["summary_metrics"]
        assert "overall_health_score" in summary
        assert "optimization_potential" in summary
        assert "stability_index" in summary
        assert 0 <= summary["overall_health_score"] <= 1
        assert 0 <= summary["optimization_potential"] <= 1
        assert 0 <= summary["stability_index"] <= 1


class TestIntegrationFunctions:
    """Test integration functions."""
    
    def test_integrate_chromatin_analysis(self):
        """Test chromatin analysis integration."""
        existing_report = {"existing_data": "value"}
        contact_matrix = np.random.rand(5, 5)
        tads = [TAD(0, 50, "test_tad")]
        interactions = [(10, 20, 0.5)]
        fix_strategies = [{"name": "fix1", "impact_score": 0.7}]
        timeline_data = [{"name": "event1", "timestamp": 0, "duration": 10}]
        
        enhanced = integrate_chromatin_analysis(
            existing_report,
            contact_matrix,
            tads,
            interactions,
            fix_strategies,
            timeline_data
        )
        
        assert "existing_data" in enhanced
        assert "chromatin_analysis" in enhanced
        
        chromatin = enhanced["chromatin_analysis"]
        assert "contact_matrix_summary" in chromatin
        assert "tad_analysis" in chromatin
        assert "polymer_metrics" in chromatin
        assert "risk_analysis" in chromatin
    
    def test_generate_comparative_matrices(self):
        """Test comparative matrix generation."""
        before_matrix = np.ones((10, 10)) * 0.8
        after_matrix = np.ones((10, 10)) * 0.4
        tads = [
            TAD(0, 50, "tad1"),
            TAD(51, 100, "tad2")
        ]
        
        result = generate_comparative_matrices(
            before_matrix,
            after_matrix,
            tads,
            "test_strategy"
        )
        
        assert "strategy_name" in result
        assert result["strategy_name"] == "test_strategy"
        assert "metrics" in result
        assert "visualizations" in result
        assert "tad_specific_improvements" in result
        
        metrics = result["metrics"]
        assert "total_contacts_before" in metrics
        assert "total_contacts_after" in metrics
        assert "reduction_percentage" in metrics
        assert metrics["reduction_percentage"] == 50.0  # 50% reduction from 0.8 to 0.4