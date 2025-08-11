"""
Example code demonstrating semantic duplicate patterns that the 
hypervector analyzer can detect.
"""

# Example 1: Semantic duplicates with different implementations
def calculate_average(numbers):
    """Calculate the mean of a list of numbers."""
    total = sum(numbers)
    count = len(numbers)
    return total / count if count > 0 else 0

def compute_mean(data_list):
    """Compute arithmetic mean of values."""
    if not data_list:
        return 0
    accumulator = 0
    for value in data_list:
        accumulator += value
    return accumulator / len(data_list)

def get_avg(lst):
    """Get average value."""
    return sum(lst) / len(lst) if lst else 0


# Example 2: Prototype fragmentation - multiple validation functions
def validate_email(email):
    """Check if email is valid."""
    return "@" in email and "." in email

def check_email_format(email_address):
    """Validate email address format."""
    if "@" not in email_address:
        return False
    if "." not in email_address:
        return False
    return True

def is_valid_email(addr):
    """Email validation."""
    return addr.count("@") == 1 and "." in addr.split("@")[1]

def email_validator(email_str):
    """Validates email string."""
    parts = email_str.split("@")
    return len(parts) == 2 and "." in parts[1]


# Example 3: Wrapper proliferation
def process_data(data):
    """Main data processing logic."""
    # Complex processing here
    return [d.strip().lower() for d in data if d]

def process_data_wrapper(data):
    """Process data."""
    return process_data(data)

def handle_data(data):
    """Handle data processing."""
    return process_data(data)

def data_processor(input_data):
    """Processes the input data."""
    return process_data(input_data)


# Example 4: Semantic stagnation - placeholders that never evolve
def analyze_results():
    """Analyze computation results."""
    pass

def generate_report():
    """Generate analysis report."""
    raise NotImplementedError()

def compute_metrics():
    """Compute performance metrics."""
    # Example of semantic stagnation - placeholder that never gets implemented
    # This is detected by the semantic analyzer as a non-evolving stub
    metrics = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0
    }
    return metrics


# Example 5: Rename cascade - same function, different names over time
# (These would typically appear across different commits)
def fetch_user_data(user_id):
    """Fetch user information from database."""
    # Imagine this connects to a database
    return {"id": user_id, "name": f"User{user_id}"}

def get_user_info(uid):
    """Get user info from DB."""
    # Same implementation, just renamed
    return {"id": uid, "name": f"User{uid}"}

def retrieve_user(user_identifier):
    """Retrieve user from database."""
    # Yet another rename
    return {"id": user_identifier, "name": f"User{user_identifier}"}
