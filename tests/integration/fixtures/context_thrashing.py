"""
Test fixtures for context-window thrashing detection.

Contains functions that are similar but separated by distance,
causing context-window thrashing during LLM-assisted development.
"""


def process_user_registration_standard(user_data):
    """Process standard user registration."""
    # Validate required fields
    if not user_data.get('email'):
        return {'error': 'Email is required'}
    if not user_data.get('username'):
        return {'error': 'Username is required'}
    if not user_data.get('password'):
        return {'error': 'Password is required'}
    
    # Check password strength
    password = user_data['password']
    if len(password) < 8:
        return {'error': 'Password must be at least 8 characters'}
    
    # Create user record
    user_record = {
        'id': generate_user_id(),
        'email': user_data['email'].lower(),
        'username': user_data['username'],
        'password_hash': hash_password(password),
        'created_at': get_current_timestamp(),
        'status': 'active'
    }
    
    # Save to database
    user_id = save_user_to_database(user_record)
    
    # Send welcome email
    send_welcome_email(user_record['email'], user_record['username'])
    
    return {'success': True, 'user_id': user_id}


def calculate_shipping_cost_domestic(order):
    """Calculate shipping cost for domestic orders."""
    base_shipping = 5.99
    weight_rate = 0.50  # per pound
    
    total_weight = 0
    for item in order['items']:
        total_weight += item.get('weight', 0) * item.get('quantity', 1)
    
    shipping_cost = base_shipping + (total_weight * weight_rate)
    
    # Apply free shipping threshold
    if order.get('total', 0) >= 75:
        shipping_cost = 0
    
    return round(shipping_cost, 2)


def validate_credit_card_info(card_data):
    """Validate credit card information."""
    errors = []
    
    # Check card number
    card_number = card_data.get('number', '').replace(' ', '').replace('-', '')
    if not card_number.isdigit() or len(card_number) < 13 or len(card_number) > 19:
        errors.append('Invalid card number format')
    
    # Check expiration date
    exp_month = card_data.get('exp_month')
    exp_year = card_data.get('exp_year')
    if not (1 <= int(exp_month or 0) <= 12):
        errors.append('Invalid expiration month')
    if not (2024 <= int(exp_year or 0) <= 2034):
        errors.append('Invalid expiration year')
    
    # Check CVV
    cvv = card_data.get('cvv', '')
    if not cvv.isdigit() or len(cvv) < 3 or len(cvv) > 4:
        errors.append('Invalid CVV')
    
    return {'valid': len(errors) == 0, 'errors': errors}


# ... Some unrelated functions to create distance ...

class ProductCatalog:
    """Product catalog management."""
    
    def __init__(self):
        self.products = {}
    
    def add_product(self, product_data):
        """Add product to catalog."""
        product_id = generate_product_id()
        self.products[product_id] = {
            'id': product_id,
            'name': product_data['name'],
            'price': product_data['price'],
            'description': product_data.get('description', ''),
            'category': product_data.get('category', 'general'),
            'stock': product_data.get('stock', 0)
        }
        return product_id
    
    def get_product(self, product_id):
        """Get product by ID."""
        return self.products.get(product_id)
    
    def search_products(self, query):
        """Search products by name or description."""
        results = []
        query_lower = query.lower()
        
        for product in self.products.values():
            if (query_lower in product['name'].lower() or 
                query_lower in product['description'].lower()):
                results.append(product)
        
        return results


class OrderManager:
    """Order management system."""
    
    def __init__(self):
        self.orders = {}
    
    def create_order(self, customer_id, items):
        """Create a new order."""
        order_id = generate_order_id()
        
        # Calculate total
        total = 0
        for item in items:
            total += item['price'] * item['quantity']
        
        order = {
            'id': order_id,
            'customer_id': customer_id,
            'items': items,
            'total': total,
            'status': 'pending',
            'created_at': get_current_timestamp()
        }
        
        self.orders[order_id] = order
        return order_id
    
    def get_order(self, order_id):
        """Get order by ID."""
        return self.orders.get(order_id)


def process_premium_user_registration(registration_data):
    """Process premium user registration with additional features."""
    # Validate essential fields
    if not registration_data.get('email'):
        return {'error': 'Email address is required'}
    if not registration_data.get('username'):
        return {'error': 'Username is required'}
    if not registration_data.get('password'):
        return {'error': 'Password is required'}
    
    # Enhanced password validation for premium users
    pwd = registration_data['password']
    if len(pwd) < 10:  # Stricter requirement
        return {'error': 'Premium accounts require passwords of at least 10 characters'}
    
    # Additional premium validation
    if not registration_data.get('phone'):
        return {'error': 'Phone number required for premium accounts'}
    
    # Create premium user record
    premium_user = {
        'id': generate_premium_user_id(),
        'email': registration_data['email'].lower(),
        'username': registration_data['username'],
        'password_hash': hash_password_premium(pwd),
        'phone': registration_data['phone'],
        'tier': 'premium',
        'created_at': get_current_timestamp(),
        'status': 'active',
        'features': ['priority_support', 'advanced_analytics', 'api_access']
    }
    
    # Save premium user
    user_id = save_premium_user_to_database(premium_user)
    
    # Send premium welcome email with additional information
    send_premium_welcome_email(premium_user['email'], premium_user['username'])
    
    # Set up premium features
    setup_premium_features(user_id)
    
    return {'success': True, 'user_id': user_id, 'tier': 'premium'}


def calculate_express_shipping_cost(order_details):
    """Calculate express shipping cost with expedited rates."""
    express_base = 12.99
    weight_multiplier = 0.75  # Higher rate for express
    
    total_weight_lbs = 0
    for product in order_details['items']:
        item_weight = product.get('weight', 0) * product.get('quantity', 1)
        total_weight_lbs += item_weight
    
    express_cost = express_base + (total_weight_lbs * weight_multiplier)
    
    # Express shipping minimum
    if express_cost < 15.99:
        express_cost = 15.99
    
    # Premium customers get 20% express shipping discount
    if order_details.get('customer_tier') == 'premium':
        express_cost *= 0.8
    
    return round(express_cost, 2)


# More unrelated functions...

class NotificationService:
    """Service for sending notifications."""
    
    def send_email(self, recipient, subject, body):
        """Send email notification."""
        # Email sending logic
        pass
    
    def send_sms(self, phone, message):
        """Send SMS notification."""
        # SMS sending logic
        pass


def validate_payment_card_details(payment_info):
    """Validate payment card details with enhanced checks."""
    validation_errors = []
    
    # Validate card number with Luhn algorithm
    card_num = payment_info.get('number', '').replace(' ', '').replace('-', '')
    if not card_num.isdigit() or len(card_num) < 13 or len(card_num) > 19:
        validation_errors.append('Card number format is invalid')
    else:
        # Luhn algorithm check
        if not luhn_check(card_num):
            validation_errors.append('Card number fails checksum validation')
    
    # Enhanced expiration validation
    month = payment_info.get('exp_month')
    year = payment_info.get('exp_year')
    if not (1 <= int(month or 0) <= 12):
        validation_errors.append('Expiration month must be 1-12')
    if not (2024 <= int(year or 0) <= 2034):
        validation_errors.append('Expiration year must be between 2024 and 2034')
    
    # CVV validation with card type detection
    security_code = payment_info.get('cvv', '')
    if not security_code.isdigit():
        validation_errors.append('Security code must be numeric')
    elif len(security_code) < 3 or len(security_code) > 4:
        validation_errors.append('Security code must be 3 or 4 digits')
    
    # Additional validation: cardholder name
    if not payment_info.get('cardholder_name', '').strip():
        validation_errors.append('Cardholder name is required')
    
    return {'is_valid': len(validation_errors) == 0, 'errors': validation_errors}


def handle_international_shipping_calculation(shipment):
    """Calculate international shipping with customs and duties."""
    international_base = 25.99
    weight_charge = 1.25  # per pound for international
    
    package_weight = 0
    for item in shipment['items']:
        item_weight = item.get('weight', 0) * item.get('quantity', 1)
        package_weight += item_weight
    
    shipping_fee = international_base + (package_weight * weight_charge)
    
    # Add customs processing fee
    customs_fee = 15.00
    shipping_fee += customs_fee
    
    # Country-specific adjustments
    destination = shipment.get('country', 'unknown')
    if destination in ['CA', 'Mexico']:  # North America
        shipping_fee *= 0.8  # Reduced rate
    elif destination in ['UK', 'France', 'Germany']:  # Europe
        shipping_fee *= 1.2  # Increased rate
    elif destination in ['Japan', 'Australia']:  # Pacific
        shipping_fee *= 1.5  # Higher rate
    
    return round(shipping_fee, 2)


# Helper functions (these would typically be in separate modules)
def generate_user_id():
    """Generate unique user ID."""
    import uuid
    return str(uuid.uuid4())

def generate_premium_user_id():
    """Generate premium user ID with special prefix."""
    import uuid
    return f"premium_{uuid.uuid4()}"

def generate_product_id():
    """Generate product ID."""
    import uuid
    return f"prod_{uuid.uuid4()}"

def generate_order_id():
    """Generate order ID."""
    import uuid
    return f"order_{uuid.uuid4()}"

def hash_password(password):
    """Hash password with standard algorithm."""
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def hash_password_premium(password):
    """Hash password with enhanced algorithm for premium users."""
    import hashlib
    # Simulate more secure hashing
    salted = password + "premium_salt"
    return hashlib.sha256(salted.encode()).hexdigest()

def get_current_timestamp():
    """Get current timestamp."""
    import time
    return int(time.time())

def save_user_to_database(user_record):
    """Save user to database."""
    # Simulate database save
    return user_record['id']

def save_premium_user_to_database(user_record):
    """Save premium user to database."""
    # Simulate premium database save
    return user_record['id']

def send_welcome_email(email, username):
    """Send welcome email."""
    # Simulate email sending
    pass

def send_premium_welcome_email(email, username):
    """Send premium welcome email."""
    # Simulate premium email sending
    pass

def setup_premium_features(user_id):
    """Set up premium features for user."""
    # Simulate feature setup
    pass

def luhn_check(card_number):
    """Perform Luhn algorithm check."""
    # Simple Luhn check implementation
    digits = [int(d) for d in card_number[::-1]]
    for i in range(1, len(digits), 2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
    return sum(digits) % 10 == 0