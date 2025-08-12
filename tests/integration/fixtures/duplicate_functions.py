"""
Test fixtures for duplicate function detection.

Contains pairs and groups of duplicate functions with varying similarities.
"""


def calculate_user_score_v1(user_data):
    """Calculate user score based on activity."""
    if not user_data:
        return 0
    
    base_score = 100
    activity_bonus = user_data.get('activity_count', 0) * 5
    reputation_bonus = user_data.get('reputation', 0) * 2
    
    total_score = base_score + activity_bonus + reputation_bonus
    
    # Apply penalties
    if user_data.get('violations', 0) > 0:
        total_score -= user_data['violations'] * 10
    
    return max(0, total_score)


def compute_user_rating_v2(user_info):
    """Compute user rating - identical logic to calculate_user_score_v1."""
    if not user_info:
        return 0
    
    initial_score = 100
    activity_points = user_info.get('activity_count', 0) * 5
    rep_points = user_info.get('reputation', 0) * 2
    
    final_score = initial_score + activity_points + rep_points
    
    # Deduct for violations
    if user_info.get('violations', 0) > 0:
        final_score -= user_info['violations'] * 10
    
    return max(0, final_score)


class OrderProcessor:
    """Order processing with duplicate methods."""
    
    def process_standard_order(self, order):
        """Process a standard order."""
        if not order or not order.items:
            return None
        
        total_cost = 0
        processed_items = []
        
        for item in order.items:
            if item.quantity <= 0:
                continue
                
            item_cost = item.price * item.quantity
            
            # Apply discount if applicable
            if hasattr(item, 'discount') and item.discount > 0:
                item_cost = item_cost * (1 - item.discount)
            
            total_cost += item_cost
            processed_items.append({
                'id': item.id,
                'quantity': item.quantity,
                'cost': item_cost
            })
        
        return {
            'total_cost': total_cost,
            'items': processed_items,
            'status': 'processed'
        }
    
    def handle_regular_order(self, order_data):
        """Handle regular order - duplicate of process_standard_order."""
        if not order_data or not order_data.items:
            return None
        
        total_amount = 0
        item_list = []
        
        for product in order_data.items:
            if product.quantity <= 0:
                continue
                
            product_cost = product.price * product.quantity
            
            # Apply discount if applicable
            if hasattr(product, 'discount') and product.discount > 0:
                product_cost = product_cost * (1 - product.discount)
            
            total_amount += product_cost
            item_list.append({
                'id': product.id,
                'quantity': product.quantity,
                'cost': product_cost
            })
        
        return {
            'total_cost': total_amount,
            'items': item_list,
            'status': 'processed'
        }


def validate_email_format(email_address):
    """Validate email format using regex."""
    import re
    
    if not email_address or not isinstance(email_address, str):
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return re.match(pattern, email_address.strip()) is not None


def check_email_validity(email_string):
    """Check email validity - identical to validate_email_format."""
    import re
    
    if not email_string or not isinstance(email_string, str):
        return False
    
    # Email validation pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    return re.match(email_pattern, email_string.strip()) is not None


def is_valid_email_address(email):
    """Alternative email validation - slight variation."""
    import re
    
    if not email or not isinstance(email, str):
        return False
    
    # Slightly different but functionally identical pattern
    regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    match = re.match(regex, email.strip())
    return match is not None


class DataProcessor:
    """Data processor with multiple duplicate methods."""
    
    def clean_text_data(self, text_input):
        """Clean text data by removing extra whitespace and normalizing."""
        if not text_input:
            return ""
        
        # Convert to string and strip
        cleaned = str(text_input).strip()
        
        # Remove extra whitespace
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters except basic punctuation
        cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
        
        return cleaned.lower()
    
    def normalize_text_input(self, input_text):
        """Normalize text input - duplicate of clean_text_data."""
        if not input_text:
            return ""
        
        # String conversion and trimming
        normalized = str(input_text).strip()
        
        # Collapse whitespace
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Filter special characters
        normalized = re.sub(r'[^\w\s.,!?-]', '', normalized)
        
        return normalized.lower()
    
    def sanitize_user_input(self, user_text):
        """Sanitize user input - similar but with additional validation."""
        if not user_text:
            return ""
        
        # Basic validation
        if len(str(user_text)) > 10000:  # Prevent extremely long inputs
            user_text = str(user_text)[:10000]
        
        sanitized = str(user_text).strip()
        
        # Remove extra whitespace
        import re
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove potentially harmful characters
        sanitized = re.sub(r'[^\w\s.,!?-]', '', sanitized)
        
        return sanitized.lower()


# Utility functions that are NOT duplicates (for testing false positive rates)
def generate_unique_id():
    """Generate a unique identifier."""
    import uuid
    return str(uuid.uuid4())


def format_currency(amount, currency_code='USD'):
    """Format amount as currency."""
    if currency_code == 'USD':
        return f"${amount:.2f}"
    elif currency_code == 'EUR':
        return f"€{amount:.2f}"
    else:
        return f"{amount:.2f} {currency_code}"


def calculate_percentage(part, whole):
    """Calculate percentage."""
    if whole == 0:
        return 0
    return (part / whole) * 100