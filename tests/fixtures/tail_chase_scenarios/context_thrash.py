"""
Context Thrashing Scenario - Forgetting earlier context and reimplementing.

This demonstrates the tail-chasing pattern where an LLM:
1. Implements a function
2. Forgets it exists when context window fills
3. Reimplements the same logic with a different name
4. Creates multiple versions of the same functionality
5. Calls the wrong version or creates circular calls
"""

# First implementation (early in session)
class DataValidator:
    """Original validator implementation."""
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_phone(self, phone: str) -> bool:
        """Validate phone number."""
        import re
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        return len(digits) == 10 or len(digits) == 11
    
    def validate_data(self, data: dict) -> bool:
        """Validate a data dictionary."""
        if 'email' in data:
            if not self.validate_email(data['email']):
                return False
        if 'phone' in data:
            if not self.validate_phone(data['phone']):
                return False
        return True


# ... Many lines of code later (context window filling) ...

# LLM forgets DataValidator exists and creates a new one
class InputValidator:
    """Duplicate validator - LLM forgot the first one exists."""
    
    def check_email(self, email_address: str) -> bool:
        """Check if email is valid."""
        # Same logic, different implementation
        if '@' not in email_address:
            return False
        parts = email_address.split('@')
        if len(parts) != 2:
            return False
        if '.' not in parts[1]:
            return False
        return True
    
    def check_phone_number(self, number: str) -> bool:
        """Check if phone number is valid."""
        # Similar logic, slightly different
        cleaned = ''.join(c for c in number if c.isdigit())
        return len(cleaned) in [10, 11]
    
    def check_input(self, input_data: dict) -> bool:
        """Check input data validity."""
        # Same concept, different method names
        if 'email' in input_data:
            if not self.check_email(input_data['email']):
                return False
        if 'phone' in input_data:
            if not self.check_phone_number(input_data['phone']):
                return False
        return True


# ... More code (context window continues filling) ...

# LLM creates ANOTHER validator, forgetting both previous ones
def validate_user_input(user_data):
    """Standalone validation function - third implementation."""
    import re
    
    def is_valid_email(email):
        """Nested function - fourth implementation of email validation."""
        return '@' in email and '.' in email.split('@')[1] if '@' in email else False
    
    def is_valid_phone(phone):
        """Nested function - fourth implementation of phone validation."""
        digits = ''.join(filter(str.isdigit, phone))
        return 10 <= len(digits) <= 11
    
    # Validate email if present
    if 'email' in user_data and not is_valid_email(user_data['email']):
        return False, "Invalid email"
    
    # Validate phone if present
    if 'phone' in user_data and not is_valid_phone(user_data['phone']):
        return False, "Invalid phone"
    
    return True, "Valid"


# ... Even more code later ...

# LLM now creates a class that USES the validators inconsistently
class UserManager:
    """Class that uses multiple versions of the same validation."""
    
    def __init__(self):
        # LLM remembers one validator exists
        self.validator = DataValidator()
        # But also creates another
        self.input_checker = InputValidator()
    
    def create_user(self, user_info):
        """Create user with validation."""
        # Uses first validator
        if self.validator.validate_data(user_info):
            # Process user
            return self.process_user_creation(user_info)
        return None
    
    def update_user(self, user_id, update_data):
        """Update user with validation."""
        # Uses second validator (inconsistent!)
        if self.input_checker.check_input(update_data):
            # Process update
            return self.process_user_update(user_id, update_data)
        return None
    
    def import_user(self, import_data):
        """Import user with validation."""
        # Uses standalone function (third approach!)
        is_valid, message = validate_user_input(import_data)
        if is_valid:
            return self.process_user_import(import_data)
        return None
    
    def process_user_creation(self, data):
        """Process user creation - yet another validation!"""
        # LLM adds inline validation, forgetting all previous validators
        if 'email' in data:
            if not ('@' in data['email'] and '.' in data['email']):
                raise ValueError("Invalid email format")
        
        # Create user...
        return {"id": 123, "data": data}
    
    def process_user_update(self, user_id, data):
        """Process user update."""
        # Might even create another validator here
        class QuickValidator:
            @staticmethod
            def validate(d):
                return True  # Placeholder because LLM is confused
        
        if QuickValidator.validate(data):
            return {"id": user_id, "updated": data}
    
    def process_user_import(self, data):
        """Process user import."""
        # Calls wrong validator method (confusion from multiple versions)
        validator = DataValidator()
        # Tries to call method that doesn't exist (mixed up with InputValidator)
        # validator.check_input(data)  # <- This would error
        
        # So LLM "fixes" by creating another inline validation
        if 'email' in data and '@' in data['email']:
            return {"imported": True, "data": data}
        return None


# Utility functions that duplicate existing functionality

def email_is_valid(email_str):
    """Another email validator - lost count at this point."""
    import re
    email_pattern = re.compile(r'^[\w\.-]+@[\w\.-]+\.\w+$')
    return email_pattern.match(email_str) is not None


def phone_is_valid(phone_str):
    """Another phone validator."""
    numbers_only = ''.join(c for c in phone_str if c.isdigit())
    return len(numbers_only) in [10, 11]


def validate_contact_info(info):
    """Yet another validation function."""
    # Uses the utility functions (creating more confusion)
    valid = True
    if 'email' in info:
        valid = valid and email_is_valid(info['email'])
    if 'phone' in info:
        valid = valid and phone_is_valid(info['phone'])
    return valid


# Signs of context thrashing:
# 1. Multiple implementations of the same functionality
# 2. Inconsistent naming (validate vs check vs verify)
# 3. Different validation approaches in same codebase
# 4. Classes and functions that duplicate each other
# 5. Inconsistent usage - different parts use different versions
# 6. Inline reimplementation of existing methods
# 7. Methods trying to call non-existent methods from other similar classes

def main():
    """Demonstration of the confusion from context thrashing."""
    # Which validator to use? There are 5+ implementations!
    
    # Developer uses first one
    validator1 = DataValidator()
    result1 = validator1.validate_data({"email": "test@example.com"})
    
    # Or second one?
    validator2 = InputValidator()
    result2 = validator2.check_input({"email": "test@example.com"})
    
    # Or standalone?
    result3, message = validate_user_input({"email": "test@example.com"})
    
    # Or utility function?
    result4 = validate_contact_info({"email": "test@example.com"})
    
    # Or inline validation?
    result5 = email_is_valid("test@example.com")
    
    print(f"We have {5} different ways to validate the same data!")
    print("This is context thrashing - forgetting and reimplementing")


if __name__ == "__main__":
    main()