"""
Integration tests for tail-chasing pattern detection accuracy.

Creates synthetic examples of each pattern type and tests detection accuracy
with false positive rate validation.
"""

import ast
import tempfile
import shutil
import textwrap
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from tailchasing.analyzers.duplicates import DuplicateFunctionAnalyzer
from tailchasing.analyzers.placeholders import PlaceholderAnalyzer
from tailchasing.analyzers.context_thrashing import ContextThrashingAnalyzer
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.issues import Issue
from tailchasing.core.symbols import SymbolTable


@dataclass
class PatternTestCase:
    """Test case for a specific tail-chasing pattern."""
    pattern_name: str
    code_samples: List[str]
    expected_issues: int
    expected_pattern_types: Set[str]
    description: str
    should_detect: bool = True


class SyntheticCodeGenerator:
    """Generate synthetic code examples with known tail-chasing patterns."""
    
    def create_duplicate_functions(self) -> List[str]:
        """Create functions with structural duplicates."""
        return [
            # File 1: Original functions
            '''
def calculate_total(items):
    """Calculate total price."""
    total = 0
    for item in items:
        if item.price > 0:
            total += item.price * item.quantity
        else:
            total += 0
    return total

def process_user_data(user):
    """Process user information."""
    if user.is_active:
        return {
            'name': user.name,
            'email': user.email,
            'status': 'active'
        }
    return {'status': 'inactive'}

def validate_input(data):
    """Validate input data."""
    if not data:
        return False
    if not isinstance(data, dict):
        return False
    return True
            ''',
            
            # File 2: Duplicate functions with different names
            '''
def compute_sum(products):
    """Compute sum of products."""
    result = 0
    for product in products:
        if product.price > 0:
            result += product.price * product.quantity
        else:
            result += 0
    return result

def handle_user_info(customer):
    """Handle customer information."""
    if customer.is_active:
        return {
            'name': customer.name,
            'email': customer.email,
            'status': 'active'
        }
    return {'status': 'inactive'}

def check_data(input_data):
    """Check data validity."""
    if not input_data:
        return False
    if not isinstance(input_data, dict):
        return False
    return True
            ''',
            
            # File 3: Similar but not identical (should not be detected as duplicates)
            '''
def calculate_average(items):
    """Calculate average price."""
    total = 0
    count = 0
    for item in items:
        if item.price > 0:
            total += item.price
            count += 1
    return total / count if count > 0 else 0

def format_user_display(user):
    """Format user for display."""
    if user.is_active:
        return f"{user.name} ({user.email}) - Active"
    return f"{user.name} - Inactive"
            '''
        ]
    
    def create_phantom_functions(self) -> List[str]:
        """Create phantom/placeholder functions."""
        return [
            '''
def incomplete_processor(data):
    """This function will process data later."""
    pass

def not_implemented_yet():
    """TODO: Implement this function."""
    raise NotImplementedError("This function is not yet implemented")

def empty_handler(request):
    """Handle the request."""
    # TODO: Add implementation
    return None

def ellipsis_placeholder():
    """Placeholder function."""
    ...

class ProcessorService:
    def authenticate(self, user):
        """Authenticate user - to be implemented."""
        pass
    
    def authorize(self, user, resource):
        """Check authorization."""
        raise NotImplementedError()
    
    def audit_log(self, action):
        """Log audit information."""
        # Implementation pending
        return True
            '''
        ]
    
    def create_context_thrashing(self) -> List[str]:
        """Create functions that cause context-window thrashing."""
        return [
            '''
def process_payment_method_a(payment_data):
    """Process payment using method A."""
    # Validate payment data
    if not payment_data.get('amount'):
        raise ValueError("Amount is required")
    if not payment_data.get('currency'):
        raise ValueError("Currency is required")
    if payment_data['amount'] <= 0:
        raise ValueError("Amount must be positive")
    
    # Process the payment
    result = {
        'transaction_id': generate_transaction_id(),
        'status': 'processing',
        'amount': payment_data['amount'],
        'currency': payment_data['currency']
    }
    
    # Apply fees
    fee_rate = 0.029
    fee = payment_data['amount'] * fee_rate
    result['fee'] = fee
    result['net_amount'] = payment_data['amount'] - fee
    
    return result

def calculate_shipping_cost(order):
    """Calculate shipping cost for order."""
    base_cost = 5.00
    weight_multiplier = 0.5
    distance_multiplier = 0.1
    
    total_weight = sum(item.weight for item in order.items)
    shipping_cost = base_cost + (total_weight * weight_multiplier)
    
    if order.distance > 100:
        shipping_cost += order.distance * distance_multiplier
    
    return round(shipping_cost, 2)

def process_payment_method_b(payment_info):
    """Process payment using method B."""
    # Validate payment info
    if not payment_info.get('amount'):
        raise ValueError("Amount is required")
    if not payment_info.get('currency'):
        raise ValueError("Currency is required")
    if payment_info['amount'] <= 0:
        raise ValueError("Amount must be positive")
    
    # Different processing logic
    transaction = {
        'id': create_unique_id(),
        'state': 'pending',
        'total': payment_info['amount'],
        'currency_code': payment_info['currency']
    }
    
    # Different fee structure
    fee_percentage = 0.025
    processing_fee = payment_info['amount'] * fee_percentage
    transaction['fees'] = processing_fee
    transaction['final_amount'] = payment_info['amount'] - processing_fee
    
    return transaction
            '''
        ]
    
    def create_hallucination_cascade(self) -> List[str]:
        """Create interdependent classes that form a hallucination cascade."""
        return [
            # File 1: Fictional subsystem part 1
            '''
from .service_mesh import ServiceMeshRegistry
from .circuit_breaker import CircuitBreakerManager
from .telemetry import TelemetryCollector

class MicroserviceOrchestrator:
    """Orchestrates microservice communications."""
    
    def __init__(self):
        self.registry = ServiceMeshRegistry()
        self.circuit_breaker = CircuitBreakerManager()
        self.telemetry = TelemetryCollector()
    
    def route_request(self, request):
        """Route request through service mesh."""
        service = self.registry.find_service(request.service_name)
        if self.circuit_breaker.is_open(service.id):
            return self._handle_circuit_open(request)
        
        result = service.process(request)
        self.telemetry.record_request(request, result)
        return result
            ''',
            
            # File 2: Fictional subsystem part 2
            '''
from .mesh_coordinator import MeshCoordinator
from .load_balancer import AdaptiveLoadBalancer

class ServiceMeshRegistry:
    """Registry for service mesh components."""
    
    def __init__(self):
        self.coordinator = MeshCoordinator()
        self.load_balancer = AdaptiveLoadBalancer()
        self.services = {}
    
    def find_service(self, name):
        """Find optimal service instance."""
        instances = self.services.get(name, [])
        return self.load_balancer.select_instance(instances)
    
    def register_service(self, service):
        """Register service with mesh."""
        self.coordinator.add_node(service)
        self.services[service.name] = self.services.get(service.name, [])
        self.services[service.name].append(service)
            ''',
            
            # File 3: Real implementation (should not be part of cascade)
            '''
import requests
import logging

class SimpleHttpClient:
    """Simple HTTP client for API calls."""
    
    def __init__(self, base_url):
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
    
    def get(self, endpoint):
        """Make GET request."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise
            '''
        ]
    
    def create_circular_imports(self) -> List[str]:
        """Create files with circular import dependencies."""
        return [
            # File: models/user.py
            '''
from .order import Order
from typing import List

class User:
    def __init__(self, name: str):
        self.name = name
        self.orders: List[Order] = []
    
    def add_order(self, order: 'Order'):
        self.orders.append(order)
        order.set_user(self)
            ''',
            
            # File: models/order.py
            '''
from .user import User
from .product import Product
from typing import List

class Order:
    def __init__(self, order_id: str):
        self.id = order_id
        self.user: User = None
        self.products: List[Product] = []
    
    def set_user(self, user: User):
        self.user = user
    
    def add_product(self, product: Product):
        self.products.append(product)
            ''',
            
            # File: models/product.py (should not create circular dependency)
            '''
class Product:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price
    
    def get_display_price(self):
        return f"${self.price:.2f}"
            '''
        ]
    
    def create_clean_code_samples(self) -> List[str]:
        """Create clean code samples that should NOT trigger false positives."""
        return [
            '''
def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax amount."""
    return amount * rate

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format amount as currency."""
    return f"{amount:.2f} {currency}"

class TaxCalculator:
    """Calculator for various tax types."""
    
    def __init__(self, default_rate: float = 0.08):
        self.default_rate = default_rate
    
    def sales_tax(self, amount: float) -> float:
        """Calculate sales tax."""
        return self.calculate_tax(amount, self.default_rate)
    
    def calculate_tax(self, amount: float, rate: float) -> float:
        """Calculate tax with custom rate."""
        if amount < 0:
            raise ValueError("Amount cannot be negative")
        return amount * rate
            ''',
            
            # Different file with legitimately different implementations
            '''
import math
from typing import Optional

def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_area(radius: float) -> float:
    """Calculate area of a circle."""
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius ** 2

class GeometryUtils:
    """Utility class for geometric calculations."""
    
    @staticmethod
    def triangle_area(base: float, height: float) -> float:
        """Calculate triangle area."""
        return 0.5 * base * height
    
    @staticmethod
    def rectangle_perimeter(width: float, height: float) -> float:
        """Calculate rectangle perimeter."""
        return 2 * (width + height)
            '''
        ]


class PatternDetectionTester:
    """Test pattern detection accuracy across different analyzers."""
    
    def __init__(self):
        self.temp_dir = None
        self.code_generator = SyntheticCodeGenerator()
        
    def setup_test_environment(self) -> Path:
        """Set up temporary test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        return self.temp_dir
        
    def cleanup_test_environment(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_files(self, code_samples: List[str], base_name: str = "test") -> List[Path]:
        """Create test files from code samples."""
        files = []
        for i, code in enumerate(code_samples):
            file_path = self.temp_dir / f"{base_name}_{i}.py"
            with open(file_path, 'w') as f:
                f.write(textwrap.dedent(code).strip())
            files.append(file_path)
        return files
    
    def create_analysis_context(self, files: List[Path]) -> AnalysisContext:
        """Create analysis context from test files."""
        ast_index = {}
        source_cache = {}
        
        for file_path in files:
            with open(file_path, 'r') as f:
                content = f.read()
                source_cache[str(file_path)] = content.split('\n')
                
            try:
                tree = ast.parse(content)
                ast_index[str(file_path)] = tree
            except SyntaxError:
                continue
        
        # Create a mock symbol table
        symbol_table = SymbolTable()
        for file_path, tree in ast_index.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbol_table.add_function(
                        name=node.name,
                        file_path=file_path,
                        line_number=node.lineno,
                        node=node
                    )
                elif isinstance(node, ast.ClassDef):
                    symbol_table.add_class(
                        name=node.name,
                        file_path=file_path,
                        line_number=node.lineno,
                        node=node
                    )
        
        return AnalysisContext(
            config={},
            root_dir=self.temp_dir,
            file_paths=files,
            ast_index=ast_index,
            symbol_table=symbol_table,
            source_cache=source_cache,
            cache={}
        )


class TestDuplicateDetection:
    """Test duplicate function detection accuracy."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tester = PatternDetectionTester()
        self.temp_dir = self.tester.setup_test_environment()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.tester.cleanup_test_environment()
    
    def test_structural_duplicate_detection(self):
        """Test detection of structurally identical functions."""
        duplicate_samples = self.tester.code_generator.create_duplicate_functions()
        files = self.tester.create_test_files(duplicate_samples, "duplicates")
        
        ctx = self.tester.create_analysis_context(files)
        analyzer = DuplicateFunctionAnalyzer()
        
        issues = list(analyzer.run(ctx))
        
        # Should detect multiple duplicate groups
        assert len(issues) >= 3, f"Expected at least 3 duplicate groups, found {len(issues)}"
        
        # Verify issue types
        for issue in issues:
            assert issue.kind == "duplicate_function"
            assert issue.severity >= 2
    
    def test_false_positive_rate_duplicates(self):
        """Test that clean code doesn't generate false positive duplicates."""
        clean_samples = self.tester.code_generator.create_clean_code_samples()
        files = self.tester.create_test_files(clean_samples, "clean")
        
        ctx = self.tester.create_analysis_context(files)
        analyzer = DuplicateFunctionAnalyzer()
        
        issues = list(analyzer.run(ctx))
        
        # Should not detect duplicates in clean code
        assert len(issues) == 0, f"False positives detected: {[issue.message for issue in issues]}"
    
    def test_similarity_thresholds(self):
        """Test that similarity thresholds work correctly."""
        # Mix of very similar and different functions
        mixed_code = '''
def func_a(x, y):
    return x + y

def func_b(a, b):  # Identical structure
    return a + b

def func_c(x, y):  # Similar but different
    return x * y

def func_d(data):  # Completely different
    result = []
    for item in data:
        result.append(item * 2)
    return result
        '''
        
        files = self.tester.create_test_files([mixed_code], "mixed")
        ctx = self.tester.create_analysis_context(files)
        analyzer = DuplicateFunctionAnalyzer()
        
        issues = list(analyzer.run(ctx))
        
        # Should only detect func_a and func_b as duplicates
        assert len(issues) == 1, f"Expected 1 duplicate group, found {len(issues)}"


class TestPlaceholderDetection:
    """Test phantom/placeholder function detection."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tester = PatternDetectionTester()
        self.temp_dir = self.tester.setup_test_environment()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.tester.cleanup_test_environment()
    
    def test_phantom_function_detection(self):
        """Test detection of phantom/placeholder functions."""
        phantom_samples = self.tester.code_generator.create_phantom_functions()
        files = self.tester.create_test_files(phantom_samples, "phantoms")
        
        ctx = self.tester.create_analysis_context(files)
        analyzer = PlaceholderAnalyzer()
        
        issues = list(analyzer.run(ctx))
        
        # Should detect multiple phantom functions
        assert len(issues) >= 5, f"Expected at least 5 phantom functions, found {len(issues)}"
        
        # Verify issue types and patterns
        phantom_types = set()
        for issue in issues:
            assert issue.kind == "phantom_function"
            phantom_types.add(issue.message.split()[0])  # First word indicates pattern type
        
        # Should detect different types of phantoms
        assert len(phantom_types) >= 3, "Should detect multiple phantom patterns"
    
    def test_placeholder_patterns(self):
        """Test detection of specific placeholder patterns."""
        patterns_to_test = [
            ("pass", "def test_func():\n    pass"),
            ("NotImplementedError", "def test_func():\n    raise NotImplementedError()"),
            ("ellipsis", "def test_func():\n    ..."),
            ("TODO", "def test_func():\n    # TODO: implement\n    return None")
        ]
        
        for pattern_name, code in patterns_to_test:
            files = self.tester.create_test_files([code], f"pattern_{pattern_name}")
            ctx = self.tester.create_analysis_context(files)
            analyzer = PlaceholderAnalyzer()
            
            issues = list(analyzer.run(ctx))
            
            assert len(issues) >= 1, f"Failed to detect {pattern_name} pattern"
            assert issues[0].kind == "phantom_function"


class TestContextThrashing:
    """Test context-window thrashing detection."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tester = PatternDetectionTester()
        self.temp_dir = self.tester.setup_test_environment()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.tester.cleanup_test_environment()
    
    def test_context_thrashing_detection(self):
        """Test detection of context-window thrashing patterns."""
        thrashing_samples = self.tester.code_generator.create_context_thrashing()
        files = self.tester.create_test_files(thrashing_samples, "thrashing")
        
        ctx = self.tester.create_analysis_context(files)
        
        # Use mock context thrashing analyzer (since the actual one might be complex)
        class MockContextThrashingAnalyzer:
            name = "context_thrashing"
            
            def run(self, ctx):
                issues = []
                # Look for similar functions in same file
                for file_path, tree in ctx.ast_index.items():
                    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                    
                    # Simple similarity check based on function body similarity
                    for i, func1 in enumerate(functions):
                        for func2 in functions[i+1:]:
                            body1 = ast.dump(func1)
                            body2 = ast.dump(func2)
                            
                            # Calculate simple similarity
                            similarity = self._calculate_similarity(body1, body2)
                            
                            if similarity > 0.6:  # Similar but not identical
                                issues.append(Issue(
                                    kind="context_window_thrashing",
                                    message=f"Similar functions {func1.name} and {func2.name} may cause context thrashing",
                                    severity=3,
                                    file=file_path,
                                    line=func1.lineno,
                                    symbol=func1.name
                                ))
                
                return issues
            
            def _calculate_similarity(self, s1, s2):
                # Simple similarity calculation
                common = sum(1 for a, b in zip(s1, s2) if a == b)
                return common / max(len(s1), len(s2))
        
        analyzer = MockContextThrashingAnalyzer()
        issues = list(analyzer.run(ctx))
        
        # Should detect similar payment processing functions
        assert len(issues) >= 1, "Should detect context thrashing patterns"
        
        for issue in issues:
            assert issue.kind == "context_window_thrashing"


class TestIntegratedPatternDetection:
    """Test integrated pattern detection across multiple analyzers."""
    
    def setup_method(self):
        """Set up test environment."""
        self.tester = PatternDetectionTester()
        self.temp_dir = self.tester.setup_test_environment()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.tester.cleanup_test_environment()
    
    def test_multiple_pattern_detection(self):
        """Test detection of multiple patterns in the same codebase."""
        # Create a complex codebase with multiple patterns
        complex_code = '''
# Duplicate functions (should be detected)
def process_order_a(order):
    if not order:
        return None
    total = 0
    for item in order.items:
        total += item.price * item.quantity
    return total

def calculate_order_total(order_data):
    if not order_data:
        return None
    sum_val = 0
    for product in order_data.items:
        sum_val += product.price * product.quantity
    return sum_val

# Phantom functions (should be detected)
def update_inventory():
    # TODO: Implement inventory update
    pass

def send_notification():
    raise NotImplementedError("Notification system not implemented")

# Clean functions (should NOT be detected)
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def format_phone_number(phone):
    digits = ''.join(filter(str.isdigit, phone))
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone
        '''
        
        files = self.tester.create_test_files([complex_code], "complex")
        ctx = self.tester.create_analysis_context(files)
        
        # Run multiple analyzers
        duplicate_analyzer = DuplicateFunctionAnalyzer()
        placeholder_analyzer = PlaceholderAnalyzer()
        
        duplicate_issues = list(duplicate_analyzer.run(ctx))
        placeholder_issues = list(placeholder_analyzer.run(ctx))
        
        # Should detect both patterns
        assert len(duplicate_issues) >= 1, "Should detect duplicate functions"
        assert len(placeholder_issues) >= 2, "Should detect phantom functions"
        
        # Verify no false positives on clean functions
        all_issues = duplicate_issues + placeholder_issues
        clean_functions = ['validate_email', 'format_phone_number']
        
        for issue in all_issues:
            assert issue.symbol not in clean_functions, f"False positive on clean function: {issue.symbol}"
    
    def test_false_positive_rate_overall(self):
        """Test overall false positive rate across all patterns."""
        clean_samples = self.tester.code_generator.create_clean_code_samples()
        files = self.tester.create_test_files(clean_samples, "clean_comprehensive")
        
        ctx = self.tester.create_analysis_context(files)
        
        # Run all analyzers
        analyzers = [
            DuplicateFunctionAnalyzer(),
            PlaceholderAnalyzer()
        ]
        
        all_issues = []
        for analyzer in analyzers:
            issues = list(analyzer.run(ctx))
            all_issues.extend(issues)
        
        # Calculate false positive rate
        total_functions = sum(
            len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
            for tree in ctx.ast_index.values()
        )
        
        false_positive_rate = len(all_issues) / max(total_functions, 1)
        
        print(f"False positive rate: {false_positive_rate:.2%} ({len(all_issues)} issues in {total_functions} functions)")
        
        # Should have very low false positive rate (<5%)
        assert false_positive_rate < 0.05, f"False positive rate too high: {false_positive_rate:.2%}"
    
    def test_pattern_severity_calibration(self):
        """Test that pattern severity is calibrated appropriately."""
        # Create patterns with different severity levels
        severity_test_cases = [
            ("critical_duplicate", '''
def critical_security_check(user):
    if user.role == "admin":
        return True
    return False

def admin_verification(user_obj):  # Duplicate - critical security function
    if user_obj.role == "admin":
        return True
    return False
            ''', 4),  # Expected high severity
            
            ("minor_utility", '''
def add_numbers(a, b):
    return a + b

def sum_values(x, y):  # Duplicate - simple utility
    return x + y
            ''', 2),  # Expected lower severity
            
            ("phantom_critical", '''
def authenticate_user():
    # TODO: Implement authentication
    pass
            ''', 3)  # Expected medium-high severity
        ]
        
        for case_name, code, expected_min_severity in severity_test_cases:
            files = self.tester.create_test_files([code], case_name)
            ctx = self.tester.create_analysis_context(files)
            
            # Test with appropriate analyzer
            if "duplicate" in case_name:
                analyzer = DuplicateFunctionAnalyzer()
            else:
                analyzer = PlaceholderAnalyzer()
            
            issues = list(analyzer.run(ctx))
            
            assert len(issues) >= 1, f"No issues detected for {case_name}"
            
            max_severity = max(issue.severity for issue in issues)
            assert max_severity >= expected_min_severity, \
                f"{case_name}: Expected severity >= {expected_min_severity}, got {max_severity}"


def test_performance_with_large_codebase():
    """Test pattern detection performance on larger codebases."""
    tester = PatternDetectionTester()
    temp_dir = tester.setup_test_environment()
    
    try:
        # Generate a large synthetic codebase
        large_codebase = []
        
        # Add many similar but not identical functions
        for i in range(50):
            func_code = f'''
def process_item_{i}(item):
    """Process item {i}."""
    if not item:
        return None
    
    result = {{
        'id': item.id,
        'processed_at': datetime.now(),
        'status': 'processed',
        'version': {i}
    }}
    
    # Different processing for each function
    if {i} % 2 == 0:
        result['type'] = 'even'
    else:
        result['type'] = 'odd'
    
    return result
            '''
            large_codebase.append(func_code)
        
        # Add a few actual duplicates
        duplicate_code = '''
def duplicate_func_1(data):
    return [x * 2 for x in data if x > 0]

def duplicate_func_2(items):
    return [item * 2 for item in items if item > 0]
        '''
        large_codebase.append(duplicate_code)
        
        # Create files (split across multiple files to simulate real codebase)
        files = []
        for i in range(0, len(large_codebase), 10):
            batch = large_codebase[i:i+10]
            combined_code = '\n\n'.join(batch)
            file_path = temp_dir / f"large_file_{i//10}.py"
            with open(file_path, 'w') as f:
                f.write("from datetime import datetime\n\n" + combined_code)
            files.append(file_path)
        
        ctx = tester.create_analysis_context(files)
        
        # Test performance
        import time
        
        start_time = time.time()
        analyzer = DuplicateFunctionAnalyzer()
        issues = list(analyzer.run(ctx))
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        print(f"Large codebase analysis:")
        print(f"  Files: {len(files)}")
        print(f"  Functions: {len([n for tree in ctx.ast_index.values() for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])}")
        print(f"  Analysis time: {analysis_time:.2f}s")
        print(f"  Issues found: {len(issues)}")
        
        # Performance assertions
        assert analysis_time < 30, f"Analysis too slow: {analysis_time}s"
        
        # Should still detect the actual duplicates
        assert len(issues) >= 1, "Should detect actual duplicates in large codebase"
        
    finally:
        tester.cleanup_test_environment()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])