"""Quality analyzer module."""

from .processor import DataProcessor  # Another circular import via handler


class QualityAnalyzer:
    """Analyzes data quality."""
    
    def __init__(self):
        self.processor = DataProcessor()
        
    def compare_quality(self, data1, data2):
        """Compare quality between two datasets."""
        # Actual implementation that ComparisonAnalyzer duplicates
        score1 = self._calculate_quality_score(data1)
        score2 = self._calculate_quality_score(data2)
        return score1 - score2
        
    def _calculate_quality_score(self, data):
        """Calculate quality score for data."""
        if not data:
            return 0
            
        # Some actual logic
        valid_count = sum(1 for item in data if self._is_valid(item))
        return valid_count / len(data)
        
    def _is_valid(self, item):
        """Check if item is valid."""
        # Placeholder
        pass
        
    def analyze(self, data):
        """Analyze data quality."""
        # Reference to non-existent function
        preprocessed = preprocess_data(data)
        return self._calculate_quality_score(preprocessed)
