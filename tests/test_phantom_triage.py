"""Tests for phantom stub security triage system."""

import ast
import tempfile
from pathlib import Path
import pytest

from tailchasing.analyzers.phantom_triage import (
    PhantomTriageAnalyzer,
    PhantomStubClassifier,
    PhantomStubGenerator,
    PhantomStub,
    PhantomPriority
)
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.issues import Issue


class TestPhantomStubClassifier:
    """Test phantom stub classification."""
    
    def test_p0_security_classification(self):
        """Test P0 security classification."""
        classifier = PhantomStubClassifier()
        
        # Create a mock context
        context = type('MockContext', (), {
            'config': {
                'placeholders': {
                    'block': ['**/crypto.py::CryptoModule.verify_signature'],
                    'allow': []
                }
            }
        })()
        
        # Create security-critical stub
        stub = PhantomStub(
            function_name='verify_signature',
            class_name='CryptoModule',
            file_path='/path/to/crypto.py',
            line_number=100,
            signature='(self, data, signature)',
            body_content='pass',
            priority=PhantomPriority.P1_FUNCTIONAL,  # Will be reclassified
            risk_factors=[],
            suggested_action='Initial action'
        )
        
        priority = classifier.classify_stub(stub, context)
        
        assert priority == PhantomPriority.P0_SECURITY
        assert stub.is_blocked == True
        assert 'Security risk' in stub.risk_factors[0]
    
    def test_p1_functional_classification(self):
        """Test P1 functional classification.""" 
        classifier = PhantomStubClassifier()
        
        context = type('MockContext', (), {
            'config': {'placeholders': {'block': [], 'allow': []}}
        })()
        
        stub = PhantomStub(
            function_name='get_config_value',
            class_name=None,
            file_path='/app/config.py',
            line_number=50,
            signature='(key, default=None)',
            body_content='return None',
            priority=PhantomPriority.P1_FUNCTIONAL,
            risk_factors=[],
            suggested_action='Initial action'
        )
        
        priority = classifier.classify_stub(stub, context)
        
        assert priority == PhantomPriority.P1_FUNCTIONAL
        assert 'Functional requirement' in stub.risk_factors[0]
    
    def test_p3_experimental_classification(self):
        """Test P3 experimental classification."""
        classifier = PhantomStubClassifier()
        
        context = type('MockContext', (), {
            'config': {
                'placeholders': {
                    'block': [],
                    'allow': ['**/experimental.py::generate_fpga_verilog']
                }
            }
        })()
        
        stub = PhantomStub(
            function_name='generate_fpga_verilog',
            class_name=None,
            file_path='/experimental/lut.py',
            line_number=200,
            signature='(lut_data)',
            body_content='raise NotImplementedError',
            priority=PhantomPriority.P1_FUNCTIONAL,
            risk_factors=[],
            suggested_action='Initial action'
        )
        
        priority = classifier.classify_stub(stub, context)
        
        assert priority == PhantomPriority.P3_EXPERIMENTAL
        assert stub.is_allowed == True
    
    def test_security_file_path_detection(self):
        """Test security classification based on file path."""
        classifier = PhantomStubClassifier()
        
        context = type('MockContext', (), {
            'config': {'placeholders': {'block': [], 'allow': []}}
        })()
        
        stub = PhantomStub(
            function_name='some_function',
            class_name=None,
            file_path='/path/to/crypto/utils.py',
            line_number=10,
            signature='()',
            body_content='pass',
            priority=PhantomPriority.P1_FUNCTIONAL,
            risk_factors=[],
            suggested_action='Initial action'
        )
        
        priority = classifier.classify_stub(stub, context)
        
        assert priority == PhantomPriority.P0_SECURITY
        assert 'Security-related file path' in stub.risk_factors


class TestPhantomStubGenerator:
    """Test phantom stub implementation generation."""
    
    def test_p0_security_blocked_generation(self):
        """Test generation for blocked P0 security stubs."""
        generator = PhantomStubGenerator()
        
        stub = PhantomStub(
            function_name='verify_dilithium_signature',
            class_name='PostQuantumCrypto',
            file_path='/crypto/pq.py',
            line_number=100,
            signature='(self, message, signature, public_key)',
            body_content='pass',
            priority=PhantomPriority.P0_SECURITY,
            risk_factors=['Security risk: post_quantum'],
            suggested_action='BLOCK CI',
            is_blocked=True
        )
        
        implementation = generator.generate_p0_security_implementation(stub)
        
        assert 'SECURITY CRITICAL' in implementation
        assert 'NotImplementedError' in implementation
        assert 'verify_dilithium_signature' in implementation
    
    def test_p1_functional_config_generation(self):
        """Test generation for P1 config getter stubs."""
        generator = PhantomStubGenerator()
        
        stub = PhantomStub(
            function_name='get_database_config',
            class_name=None,
            file_path='/app/config.py',
            line_number=50,
            signature='()',
            body_content='return None',
            priority=PhantomPriority.P1_FUNCTIONAL,
            risk_factors=['Functional requirement: config_getters'],
            suggested_action='Provide minimal implementation'
        )
        
        implementation = generator.generate_p1_functional_implementation(stub)
        
        assert 'config' in implementation.lower()
        assert 'json.load' in implementation or 'os.environ' in implementation
        assert 'get_database_config' in implementation
    
    def test_p3_experimental_generation(self):
        """Test generation for P3 experimental stubs."""
        generator = PhantomStubGenerator()
        
        stub = PhantomStub(
            function_name='generate_verilog_lut',
            class_name=None,
            file_path='/experimental/fpga.py',
            line_number=25,
            signature='(lut_table)',
            body_content='pass',
            priority=PhantomPriority.P3_EXPERIMENTAL,
            risk_factors=['Experimental feature: code_generation'],
            suggested_action='Allowlist until implemented'
        )
        
        implementation = generator.generate_p3_experimental_implementation(stub)
        
        assert 'Experimental' in implementation
        assert 'logging' in implementation
        assert 'generate_verilog_lut' in implementation


class TestPhantomTriageAnalyzer:
    """Test the main phantom triage analyzer."""
    
    def create_test_file_with_stubs(self) -> str:
        """Create a temporary Python file with phantom stubs."""
        test_code = '''
def verify_hsm_signature(data, signature):
    """Verify signature using HSM."""
    pass

def get_app_config():
    """Get application configuration."""
    return None
    
class MockDilithium:
    def verify(self, message, signature):
        """Verify Dilithium signature."""
        raise NotImplementedError
    
    def experimental_benchmark(self):
        """Experimental benchmarking."""
        pass

def generate_fpga_verilog(lut_data):
    """Generate Verilog from LUT data."""
    # TODO: implement
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            return f.name
    
    def test_analyzer_detection(self):
        """Test that analyzer detects phantom stubs correctly."""
        config = {
            'placeholders': {
                'block': ['**MockDilithium.verify', '*verify_hsm*'],
                'allow': ['*generate_fpga_verilog*'],
                'triage_enabled': True
            }
        }
        
        analyzer = PhantomTriageAnalyzer(config)
        
        # Create test file
        test_file = self.create_test_file_with_stubs()
        
        try:
            # Parse the file
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            # Create mock context
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {test_file: tree},
                'should_ignore_issue': lambda kind: False
            })()
            
            # Run analyzer
            issues = analyzer.run(context)
            
            # Should detect multiple phantom stubs
            assert len(issues) >= 4
            
            # Check that we have different priority levels
            priorities = {issue.evidence['priority'] for issue in issues}
            assert 'P0_SECURITY' in priorities  # HSM and Dilithium functions
            assert 'P1_FUNCTIONAL' in priorities  # Config getter
            assert 'P3_EXPERIMENTAL' in priorities  # FPGA generator
            
            # Check that blocked and allowed stubs are handled
            blocked_issues = [i for i in issues if i.evidence.get('is_blocked')]
            allowed_issues = [i for i in issues if i.evidence.get('is_allowed')]
            
            assert len(blocked_issues) >= 1  # MockDilithium.verify should be blocked
            assert len(allowed_issues) >= 1  # generate_fpga_verilog should be allowed
        
        finally:
            Path(test_file).unlink()
    
    def test_triage_report_generation(self):
        """Test generation of triage report."""
        config = {
            'placeholders': {
                'block': ['**/crypto.py::verify'],
                'allow': ['**/experimental.py::generate_verilog'],
                'triage_enabled': True
            }
        }
        
        analyzer = PhantomTriageAnalyzer(config)
        
        # Create test file
        test_file = self.create_test_file_with_stubs()
        
        try:
            # Parse and analyze
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {test_file: tree},
                'should_ignore_issue': lambda kind: False
            })()
            
            analyzer.run(context)
            
            # Generate report
            report = analyzer.generate_triage_report()
            
            assert 'PHANTOM STUB TRIAGE REPORT' in report
            assert 'P0_SECURITY' in report
            assert 'P1_FUNCTIONAL' in report  
            assert 'P3_EXPERIMENTAL' in report
            assert 'BLOCKED' in report
            assert 'ALLOWED' in report
        
        finally:
            Path(test_file).unlink()
    
    def test_security_patterns_detection(self):
        """Test that security patterns are detected correctly."""
        config = {'placeholders': {'block': [], 'allow': [], 'triage_enabled': True}}
        analyzer = PhantomTriageAnalyzer(config)
        
        # Create file with various security patterns
        security_code = '''
def authenticate_user(username, password):
    pass

def validate_signature(data, sig):
    raise NotImplementedError
    
def verify_hmac(message, mac):
    return None

class HSMManager:
    def hardware_security_sign(self, data):
        pass

def stark_proof_verify(proof, public_inputs):
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(security_code)
            test_file = f.name
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {test_file: tree},
                'should_ignore_issue': lambda kind: False
            })()
            
            issues = analyzer.run(context)
            
            # All should be classified as P0_SECURITY
            security_issues = [i for i in issues if i.evidence['priority'] == 'P0_SECURITY']
            assert len(security_issues) >= 4  # At least the security functions
            
            # Check risk factors
            for issue in security_issues:
                assert len(issue.evidence['risk_factors']) > 0
                assert any('Security risk' in rf for rf in issue.evidence['risk_factors'])
        
        finally:
            Path(test_file).unlink()


class TestIntegration:
    """Integration tests for the phantom triage system."""
    
    def test_genomevault_example_patterns(self):
        """Test patterns matching genomevault examples."""
        config = {
            'placeholders': {
                'block': [
                    '**/post_quantum_crypto.py::MockDilithium.verify',
                    '**/pir_server.py::_verify_hsm'
                ],
                'allow': [
                    'genomevault/hypervector/operations/hamming_lut.py::generate_fpga_verilog'
                ],
                'triage_enabled': True
            }
        }
        
        analyzer = PhantomTriageAnalyzer(config)
        
        # Simulate genomevault-like code structure
        genomevault_code = '''
class MockDilithium:
    def verify(self, message, signature, public_key):
        """Mock Dilithium verification - MUST BE IMPLEMENTED."""
        pass

def _verify_hsm(data, signature):
    """Verify using HSM - security critical."""
    raise NotImplementedError

def generate_fpga_verilog(hamming_lut):
    """Generate FPGA Verilog from Hamming LUT."""
    # Experimental feature for genomevault
    pass
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(genomevault_code)
            test_file = f.name
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            # Rename file to match patterns
            post_quantum_file = str(Path(test_file).parent / "post_quantum_crypto.py")
            pir_server_file = str(Path(test_file).parent / "pir_server.py") 
            hamming_file = "genomevault/hypervector/operations/hamming_lut.py"
            
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {
                    post_quantum_file: tree,  # MockDilithium.verify
                    pir_server_file: tree,    # _verify_hsm  
                    hamming_file: tree        # generate_fpga_verilog
                },
                'should_ignore_issue': lambda kind: False
            })()
            
            # Test each file pattern
            for file_path in [post_quantum_file, pir_server_file, hamming_file]:
                context.ast_index = {file_path: tree}
                issues = analyzer.run(context)
                
                if 'post_quantum' in file_path:
                    # MockDilithium.verify should be blocked
                    blocked_issues = [i for i in issues if i.evidence.get('is_blocked')]
                    assert len(blocked_issues) >= 1
                
                elif 'pir_server' in file_path:
                    # _verify_hsm should be blocked
                    blocked_issues = [i for i in issues if i.evidence.get('is_blocked')]
                    assert len(blocked_issues) >= 1
                
                elif 'hamming_lut' in file_path:
                    # generate_fpga_verilog should be allowed
                    allowed_issues = [i for i in issues if i.evidence.get('is_allowed')]
                    assert len(allowed_issues) >= 1
        
        finally:
            Path(test_file).unlink()
    
    def test_ci_blocking_behavior(self):
        """Test that blocked stubs properly fail CI."""
        config = {
            'placeholders': {
                'block': ['**/critical_security.py::verify_signature'],
                'allow': [],
                'triage_enabled': True
            }
        }
        
        analyzer = PhantomTriageAnalyzer(config)
        
        critical_code = '''
def verify_signature(data, signature, public_key):
    """Critical security function."""
    pass  # This should block CI
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(critical_code)
            test_file = f.name
        
        try:
            # Rename to match blocking pattern
            critical_file = str(Path(test_file).parent / "critical_security.py")
            
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {critical_file: tree},
                'should_ignore_issue': lambda kind: False
            })()
            
            issues = analyzer.run(context)
            
            # Should have at least one blocked issue
            blocked_issues = [i for i in issues if i.evidence.get('is_blocked')]
            assert len(blocked_issues) >= 1
            
            # The blocked issue should be P0_SECURITY with high severity
            for issue in blocked_issues:
                assert issue.evidence['priority'] == 'P0_SECURITY'
                assert issue.severity >= 4  # Critical severity
                assert 'BLOCK CI' in issue.evidence['suggested_action'] or issue.evidence.get('is_blocked')
        
        finally:
            Path(test_file).unlink()