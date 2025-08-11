"""
Simple import error scenarios that should be fixed in 1-2 steps.
"""

from typing import Dict, List, Tuple
from .base import BenchmarkScenario


class SimpleImportScenario(BenchmarkScenario):
    """Scenario with simple import errors that should be quickly resolved."""
    
    def __init__(self):
        super().__init__(
            name="simple_import",
            description="Simple import errors with missing or incorrect imports",
            expected_steps=(1, 2)
        )
    
    def setup(self) -> str:
        """Set up the simple import error scenario."""
        temp_dir = self.create_temp_directory()
        self.write_files(self.get_initial_code())
        return temp_dir
    
    def get_initial_code(self) -> Dict[str, str]:
        """Get initial code with import errors."""
        return {
            "main.py": '''"""Main module with missing imports."""

def process_data(data):
    """Process data using pandas DataFrame."""
    # Missing import for pd
    df = pd.DataFrame(data)
    
    # Missing import for np
    mean_value = np.mean(df.values)
    
    return {
        "dataframe": df,
        "mean": mean_value,
        "count": len(df)
    }

def calculate_statistics(numbers):
    """Calculate statistics using math module."""
    # Missing import for math
    std_dev = math.sqrt(sum((x - mean(numbers))**2 for x in numbers) / len(numbers))
    
    # Missing import for statistics
    median_val = statistics.median(numbers)
    
    return {
        "std_dev": std_dev,
        "median": median_val
    }

if __name__ == "__main__":
    data = {"values": [1, 2, 3, 4, 5]}
    result = process_data(data)
    print(f"Results: {result}")
''',
            "utils.py": '''"""Utility functions with import issues."""

def load_json_file(filepath):
    """Load JSON file."""
    # Missing import for json
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_pickle_file(data, filepath):
    """Save data as pickle file."""
    # Missing import for pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def get_current_time():
    """Get current timestamp."""
    # Missing import for datetime
    return datetime.now().isoformat()

def parse_yaml_config(config_str):
    """Parse YAML configuration."""
    # Missing import for yaml
    return yaml.safe_load(config_str)
''',
            "test_main.py": '''"""Test file with missing test imports."""

class TestProcessData:
    """Test cases for data processing."""
    
    def test_process_empty_data(self):
        """Test processing empty data."""
        # Missing import for unittest
        self.assertEqual(process_data({}), {"dataframe": None, "mean": 0, "count": 0})
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        # Missing import for pytest
        with pytest.raises(ValueError):
            calculate_statistics([])
'''
        }
    
    def get_expected_solution(self) -> Dict[str, str]:
        """Get expected solution with all imports fixed."""
        return {
            "main.py": '''"""Main module with missing imports."""

import pandas as pd
import numpy as np
import math
import statistics

def process_data(data):
    """Process data using pandas DataFrame."""
    df = pd.DataFrame(data)
    mean_value = np.mean(df.values)
    
    return {
        "dataframe": df,
        "mean": mean_value,
        "count": len(df)
    }

def calculate_statistics(numbers):
    """Calculate statistics using math module."""
    std_dev = math.sqrt(sum((x - statistics.mean(numbers))**2 for x in numbers) / len(numbers))
    median_val = statistics.median(numbers)
    
    return {
        "std_dev": std_dev,
        "median": median_val
    }

if __name__ == "__main__":
    data = {"values": [1, 2, 3, 4, 5]}
    result = process_data(data)
    print(f"Results: {result}")
''',
            "utils.py": '''"""Utility functions with import issues."""

import json
import pickle
from datetime import datetime
import yaml

def load_json_file(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_pickle_file(data, filepath):
    """Save data as pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def get_current_time():
    """Get current timestamp."""
    return datetime.now().isoformat()

def parse_yaml_config(config_str):
    """Parse YAML configuration."""
    return yaml.safe_load(config_str)
''',
            "test_main.py": '''"""Test file with missing test imports."""

import unittest
import pytest
from main import process_data, calculate_statistics

class TestProcessData(unittest.TestCase):
    """Test cases for data processing."""
    
    def test_process_empty_data(self):
        """Test processing empty data."""
        self.assertEqual(process_data({}), {"dataframe": None, "mean": 0, "count": 0})
    
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        with pytest.raises(ValueError):
            calculate_statistics([])
'''
        }
    
    def validate_solution(self, current_code: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate if imports are correctly added."""
        errors = []
        
        # Check main.py imports
        if "main.py" in current_code:
            main_content = current_code["main.py"]
            required_imports = [
                "import pandas as pd",
                "import numpy as np", 
                "import math",
                "import statistics"
            ]
            for imp in required_imports:
                if imp not in main_content and imp.replace("import ", "from ") not in main_content:
                    errors.append(f"Missing import in main.py: {imp}")
        
        # Check utils.py imports
        if "utils.py" in current_code:
            utils_content = current_code["utils.py"]
            required_imports = [
                "import json",
                "import pickle",
                "datetime",
                "import yaml"
            ]
            for imp in required_imports:
                if imp not in utils_content:
                    errors.append(f"Missing import in utils.py: {imp}")
        
        # Check test_main.py imports
        if "test_main.py" in current_code:
            test_content = current_code["test_main.py"]
            required_imports = [
                "import unittest",
                "import pytest",
                "from main import"
            ]
            for imp in required_imports:
                if imp not in test_content:
                    errors.append(f"Missing import in test_main.py: {imp}")
        
        # Validate syntax
        import ast
        for file_path, content in current_code.items():
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"Syntax error in {file_path}: {e}")
        
        return len(errors) == 0, errors