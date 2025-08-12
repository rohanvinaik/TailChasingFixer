"""
Example: Context Window Thrashing Anti-Pattern

This demonstrates how LLMs recreate existing functionality when the
original implementation falls out of their context window.
"""

# This pattern typically appears in large files where similar functions
# are implemented multiple times with slight naming variations

# Beginning of file (line 1-100)
def parse_json_data(json_string):
    """Parse JSON string and return Python object."""
    import json
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def validate_email(email):
    """Validate email address format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def calculate_average(numbers):
    """Calculate average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

# ... 500 lines of other code ...
# (Context window starts to lose earlier functions)

# Middle of file (line 600-700)
# LLM recreates similar functionality with different names

def process_json_string(data_string):
    """Process JSON string into dictionary."""
    import json
    try:
        result = json.loads(data_string)
        return result
    except Exception as err:
        print(f"JSON processing failed: {err}")
        return None

def check_email_validity(email_address):
    """Check if email address is valid."""
    import re
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(email_regex, email_address):
        return True
    return False

def compute_mean(value_list):
    """Compute mean of values."""
    if len(value_list) == 0:
        return 0
    total = sum(value_list)
    return total / len(value_list)

# ... 500 more lines ...
# (Context window completely lost initial functions)

# End of file (line 1200-1300)
# LLM creates third version with yet another naming convention

def deserialize_json(json_text):
    """Deserialize JSON text to Python object."""
    import json
    try:
        obj = json.loads(json_text)
        return obj
    except json.decoder.JSONDecodeError as exc:
        print(f"Deserialization error: {exc}")
        return None

def is_valid_email(email_str):
    """Determine if email string is valid."""
    import re
    regex_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    match = re.match(regex_pattern, email_str)
    return match is not None

def get_average_value(number_array):
    """Get average value from array of numbers."""
    if number_array == []:
        return 0.0
    sum_total = 0
    for num in number_array:
        sum_total += num
    return sum_total / len(number_array)

# Pattern characteristics detected by TailChasingFixer:
#
# 1. Naming drift pattern:
#    - parse_json_data -> process_json_string -> deserialize_json
#    - validate_email -> check_email_validity -> is_valid_email
#    - calculate_average -> compute_mean -> get_average_value
#
# 2. Line distance between duplicates:
#    - First to second: ~500 lines
#    - Second to third: ~500 lines
#
# 3. Semantic similarity scores:
#    - parse_json_data vs process_json_string: 0.94
#    - process_json_string vs deserialize_json: 0.92
#    - All three JSON functions: 0.91 average
#
# How TailChasingFixer detects this:
#
# $ tailchasing analyze --deep
#
# Issue: context_window_thrashing
# Severity: 3 (Medium)
# File: examples/context_window_thrashing.py
#
# Detected context window thrashing pattern:
# - 3 implementations of JSON parsing (lines 6, 32, 68)
# - 3 implementations of email validation (lines 14, 42, 76)
# - 3 implementations of average calculation (lines 20, 52, 84)
#
# Pattern indicators:
# - Physical distance: >500 lines between duplicates
# - Naming evolution detected (gradual drift)
# - High semantic similarity (>0.9)
# - Implementation style variations increase with distance
#
# Recommendation:
# 1. Move common utilities to separate module
# 2. Use consistent naming conventions
# 3. Consider file splitting if >1000 lines