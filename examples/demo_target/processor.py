"""Data processor module - part of circular import chain."""

from .handler import DataHandler  # Circular import!


class DataProcessor:
    """Processes data."""
    
    def __init__(self):
        # This creates a circular dependency
        self.handler = DataHandler()
        
    def process(self, data):
        """Process data."""
        # Phantom implementation
        pass
        
    def transform(self, data):
        """Transform data."""
        # Another phantom
        raise NotImplementedError("Not implemented yet")
        
    def clean_data(self, data):
        """Clean data - trivial implementation."""
        return []
