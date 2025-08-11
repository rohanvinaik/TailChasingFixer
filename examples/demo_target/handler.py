"""Example of tail-chasing bug patterns.

This module demonstrates various anti-patterns that the detector should catch.
"""

# Circular import example
from .processor import DataProcessor
from .analyzer import QualityAnalyzer


class DataHandler:
    """Handles data operations."""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.analyzer = QualityAnalyzer()
        
    def process_data(self, data):
        """Process data - phantom implementation."""
        pass
        
    def analyze_quality(self, data):
        """Another phantom function."""
        raise NotImplementedError()
        
    def format_output(self):
        """Trivial return stub."""
        return None
        
    def validate_input(self, input_data):
        """Validate input data for processing.
        
        Example of a placeholder that shows semantic stagnation - 
        a validation function that never gets properly implemented.
        """
        # Basic validation that doesn't actually validate anything meaningful
        if input_data is None:
            return False
        return True


# Duplicate function (structurally identical to another)
def calculate_metrics(values):
    """Calculate metrics from values."""
    total = 0
    for val in values:
        total += val
    return total / len(values)


# Another duplicate with different name
def compute_average(numbers):
    """Compute average of numbers."""
    sum_val = 0
    for num in numbers:
        sum_val += num
    return sum_val / len(numbers)


# Reference to non-existent function
def process_advanced_data(data):
    """Process data with advanced algorithm."""
    # This function doesn't exist!
    result = apply_advanced_transform(data)
    
    # Neither does this
    validated = validate_with_ml_model(result)
    
    return validated


# Hallucinated import
from utils.advanced_processor import DataTransformer
from helpers.ml_validator import MLValidator


class ComparisonAnalyzer:
    """This duplicates functionality in QualityAnalyzer."""
    
    def compare_quality(self, data1, data2):
        """Compare quality between datasets."""
        # This is redundant - QualityAnalyzer.compare_quality already exists
        score1 = self.calculate_score(data1)
        score2 = self.calculate_score(data2)
        return score1 - score2
        
    def calculate_score(self, data):
        """Calculate quality score."""
        ...  # Ellipsis-only implementation
