"""
Semantic duplicate scenarios that should be fixed in 2-3 steps.
"""

from typing import Dict, List, Tuple
from .base import BenchmarkScenario


class SemanticDuplicateScenario(BenchmarkScenario):
    """Scenario with semantically similar functions that should be unified."""
    
    def __init__(self):
        super().__init__(
            name="semantic_duplicate",
            description="Semantically duplicate functions requiring consolidation",
            expected_steps=(2, 3)
        )
    
    def setup(self) -> str:
        """Set up the semantic duplicate scenario."""
        temp_dir = self.create_temp_directory()
        self.write_files(self.get_initial_code())
        return temp_dir
    
    def get_initial_code(self) -> Dict[str, str]:
        """Get initial code with semantic duplicates."""
        return {
            "data_processing.py": '''"""Data processing module with semantic duplicates."""

def process_user_data(user_info):
    """Process user information and return formatted data."""
    if not user_info:
        return None
    
    result = {
        "id": user_info.get("user_id", ""),
        "name": user_info.get("full_name", "").strip(),
        "email": user_info.get("email_address", "").lower(),
        "active": user_info.get("is_active", False),
        "created": user_info.get("created_at", None)
    }
    
    # Validate email
    if "@" not in result["email"]:
        result["email"] = ""
    
    return result

def format_customer_data(customer):
    """Format customer data for display."""
    if customer is None:
        return None
    
    formatted = {
        "id": customer.get("user_id", ""),
        "name": customer.get("full_name", "").strip(),
        "email": customer.get("email_address", "").lower(),
        "active": customer.get("is_active", False),
        "created": customer.get("created_at", None)
    }
    
    # Check email validity
    if "@" not in formatted["email"]:
        formatted["email"] = ""
    
    return formatted

def prepare_member_info(member_data):
    """Prepare member information for storage."""
    if not member_data:
        return None
    
    info = {
        "id": member_data.get("user_id", ""),
        "name": member_data.get("full_name", "").strip(),
        "email": member_data.get("email_address", "").lower(),
        "active": member_data.get("is_active", False),
        "created": member_data.get("created_at", None)
    }
    
    # Email validation
    if "@" not in info["email"]:
        info["email"] = ""
    
    return info
''',
            "file_operations.py": '''"""File operations with semantic duplicates."""

import os
import json

def read_json_from_file(file_path):
    """Read JSON data from a file."""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_json_file(path_to_file):
    """Load JSON file and return parsed data."""
    if not os.path.exists(path_to_file):
        return None
    
    try:
        with open(path_to_file, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return content
    except (IOError, json.JSONDecodeError) as error:
        print(f"Error reading {path_to_file}: {error}")
        return None

def get_json_data(filename):
    """Get JSON data from specified file."""
    if not os.path.exists(filename):
        return None
    
    try:
        with open(filename, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        return json_data
    except (IOError, json.JSONDecodeError) as exc:
        print(f"Error reading {filename}: {exc}")
        return None

def write_json_to_file(data, file_path):
    """Write data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        return True
    except IOError as e:
        print(f"Error writing {file_path}: {e}")
        return False

def save_json_file(content, path_to_file):
    """Save content as JSON file."""
    try:
        with open(path_to_file, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2)
        return True
    except IOError as error:
        print(f"Error writing {path_to_file}: {error}")
        return False
''',
            "string_utils.py": '''"""String utilities with semantic duplicates."""

def clean_string(text):
    """Clean and normalize a string."""
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    cleaned = text.strip()
    
    # Replace multiple spaces with single space
    cleaned = " ".join(cleaned.split())
    
    # Remove special characters
    cleaned = ''.join(c for c in cleaned if c.isalnum() or c.isspace())
    
    return cleaned

def normalize_text(input_text):
    """Normalize text by removing extra spaces and special chars."""
    if not input_text:
        return ""
    
    # Trim whitespace
    normalized = input_text.strip()
    
    # Collapse multiple spaces
    normalized = " ".join(normalized.split())
    
    # Keep only alphanumeric and spaces
    normalized = ''.join(ch for ch in normalized if ch.isalnum() or ch.isspace())
    
    return normalized

def sanitize_string(string_input):
    """Sanitize string by removing unwanted characters."""
    if not string_input:
        return ""
    
    # Strip whitespace
    sanitized = string_input.strip()
    
    # Remove duplicate spaces
    sanitized = " ".join(sanitized.split())
    
    # Filter out non-alphanumeric except spaces
    sanitized = ''.join(char for char in sanitized if char.isalnum() or char.isspace())
    
    return sanitized
'''
        }
    
    def get_expected_solution(self) -> Dict[str, str]:
        """Get expected solution with duplicates consolidated."""
        return {
            "data_processing.py": '''"""Data processing module with consolidated functions."""

def process_user_data(user_info):
    """Process user information and return formatted data.
    
    This function handles user, customer, and member data uniformly.
    """
    if not user_info:
        return None
    
    result = {
        "id": user_info.get("user_id", ""),
        "name": user_info.get("full_name", "").strip(),
        "email": user_info.get("email_address", "").lower(),
        "active": user_info.get("is_active", False),
        "created": user_info.get("created_at", None)
    }
    
    # Validate email
    if "@" not in result["email"]:
        result["email"] = ""
    
    return result

# Create aliases for backward compatibility
format_customer_data = process_user_data
prepare_member_info = process_user_data
''',
            "file_operations.py": '''"""File operations with consolidated functions."""

import os
import json

def read_json_from_file(file_path):
    """Read JSON data from a file.
    
    This is the primary function for JSON file reading.
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Create aliases for backward compatibility
load_json_file = read_json_from_file
get_json_data = read_json_from_file

def write_json_to_file(data, file_path):
    """Write data to JSON file.
    
    This is the primary function for JSON file writing.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
        return True
    except IOError as e:
        print(f"Error writing {file_path}: {e}")
        return False

# Create alias for backward compatibility
save_json_file = write_json_to_file
''',
            "string_utils.py": '''"""String utilities with consolidated functions."""

def clean_string(text):
    """Clean and normalize a string.
    
    This is the primary string cleaning function.
    """
    if not text:
        return ""
    
    # Remove leading/trailing whitespace
    cleaned = text.strip()
    
    # Replace multiple spaces with single space
    cleaned = " ".join(cleaned.split())
    
    # Remove special characters
    cleaned = ''.join(c for c in cleaned if c.isalnum() or c.isspace())
    
    return cleaned

# Create aliases for backward compatibility
normalize_text = clean_string
sanitize_string = clean_string
'''
        }
    
    def validate_solution(self, current_code: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate if semantic duplicates are properly consolidated."""
        errors = []
        
        # Check data_processing.py
        if "data_processing.py" in current_code:
            content = current_code["data_processing.py"]
            
            # Should have one main function and aliases
            main_impl_count = content.count("if not user_info:")
            if main_impl_count > 1:
                errors.append(f"data_processing.py: Found {main_impl_count} duplicate implementations, expected 1 with aliases")
            
            # Check for aliases
            if "format_customer_data = process_user_data" not in content and \
               "def format_customer_data" in content:
                errors.append("data_processing.py: format_customer_data not properly aliased")
            
            if "prepare_member_info = process_user_data" not in content and \
               "def prepare_member_info" in content:
                errors.append("data_processing.py: prepare_member_info not properly aliased")
        
        # Check file_operations.py
        if "file_operations.py" in current_code:
            content = current_code["file_operations.py"]
            
            # Check JSON read functions
            read_impl_count = content.count('json.load(')
            if read_impl_count > 1:
                # Check if they're truly different functions or aliases
                if "load_json_file = read_json_from_file" not in content and \
                   "def load_json_file" in content:
                    errors.append("file_operations.py: JSON read functions not consolidated")
            
            # Check JSON write functions  
            write_impl_count = content.count('json.dump(')
            if write_impl_count > 1:
                if "save_json_file = write_json_to_file" not in content and \
                   "def save_json_file" in content:
                    errors.append("file_operations.py: JSON write functions not consolidated")
        
        # Check string_utils.py
        if "string_utils.py" in current_code:
            content = current_code["string_utils.py"]
            
            # Should have one main implementation
            strip_count = content.count('.strip()')
            # Allow up to 3 (one per original function name in docstrings/comments)
            if strip_count > 3:
                errors.append(f"string_utils.py: Found {strip_count} strip() calls, suggests unconsolidated code")
            
            # Check for aliases
            if "normalize_text = clean_string" not in content and \
               "def normalize_text" in content:
                errors.append("string_utils.py: normalize_text not properly aliased")
            
            if "sanitize_string = clean_string" not in content and \
               "def sanitize_string" in content:
                errors.append("string_utils.py: sanitize_string not properly aliased")
        
        # Validate syntax
        import ast
        for file_path, content in current_code.items():
            if file_path.endswith(".py"):
                try:
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(f"Syntax error in {file_path}: {e}")
        
        return len(errors) == 0, errors