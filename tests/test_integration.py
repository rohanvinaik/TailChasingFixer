"""
End-to-end integration tests for TailChasingFixer.

Comprehensive integration tests covering the full workflow from
detection through analysis, fixing, and reporting.
"""

import unittest
import tempfile
import shutil
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
import textwrap

import pytest
from hypothesis import given, strategies as st, settings

from tailchasing.core.detector import TailChasingDetector
from tailchasing.orchestration.orchestrator import TailChasingOrchestrator
from tailchasing.ci.pipeline_analyzer import PipelineAnalyzer
from tailchasing.ci.github_integration import GitHubIntegration
from tailchasing.visualization.report_generator import ReportGenerator
from tailchasing.llm_integration.feedback_generator import FeedbackGenerator
from tailchasing.llm_integration.llm_adapters import UniversalAdapter


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end workflows."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        
        # Initialize git repo for CI tests
        subprocess.run(['git', 'init'], cwd=self.test_path, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'test@example.com'], 
                      cwd=self.test_path, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Test User'],
                      cwd=self.test_path, capture_output=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_complete_detection_and_fix_workflow(self):
        """Test complete workflow from detection to fixing."""
        # Create a codebase with various issues
        self.create_file("src/utils.py", """
            from typing import *
            import os, sys, json, csv
            
            def process_data(data):
                result = []
                for item in data:
                    if item > 0:
                        result.append(item * 2)
                return result
            
            def handle_data(input_data):
                # Duplicate of process_data
                output = []
                for element in input_data:
                    if element > 0:
                        output.append(element * 2)
                return output
            
            def placeholder_function():
                pass
            
            def another_stub():
                raise NotImplementedError()
        """)
        
        self.create_file("src/models.py", """
            from abc import ABC, abstractmethod
            
            class AbstractBase(ABC):
                @abstractmethod
                def process(self):
                    pass
            
            class MiddleLayer(AbstractBase):
                @abstractmethod
                def handle(self):
                    pass
                
                def process(self):
                    return self.handle()
            
            class ConcreteImpl(MiddleLayer):
                def handle(self):
                    return "result"
            
            class Manager:
                def __init__(self):
                    self.impl = ConcreteImpl()
        """)
        
        # Run detection
        detector = TailChasingDetector()
        issues = detector.detect(self.test_path)
        
        self.assertGreater(len(issues), 0)
        
        # Run orchestration with fixes
        orchestrator = TailChasingOrchestrator({
            'auto_fix': True,
            'dry_run': False,
            'validate_fixes': True
        })
        
        result = orchestrator.orchestrate(
            path=self.test_path,
            auto_fix=True
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(result['issues_found'], 0)
        
        # Generate report
        report_gen = ReportGenerator()
        report_gen.add_issues(issues)
        
        html_report = report_gen.generate_html_report()
        self.assertIn('<html>', html_report)
        self.assertIn('Tail-Chasing Analysis Report', html_report)
        
        json_report = report_gen.generate_json_report()
        report_data = json.loads(json_report)
        self.assertIn('issues', report_data)
        self.assertEqual(len(report_data['issues']), len(issues))
    
    def test_ci_pipeline_integration(self):
        """Test CI/CD pipeline integration."""
        # Create main branch with issues
        self.create_file("main.py", """
            def original_function():
                return 42
        """)
        
        subprocess.run(['git', 'add', '.'], cwd=self.test_path, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], 
                      cwd=self.test_path, capture_output=True)
        
        # Create feature branch with more issues
        subprocess.run(['git', 'checkout', '-b', 'feature'], 
                      cwd=self.test_path, capture_output=True)
        
        self.create_file("feature.py", """
            def original_function():
                # Duplicate from main.py
                return 42
            
            def another_original():
                # Another duplicate
                return 42
            
            def stub():
                pass
        """)
        
        subprocess.run(['git', 'add', '.'], cwd=self.test_path, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add feature'], 
                      cwd=self.test_path, capture_output=True)
        
        # Run pipeline analysis
        analyzer = PipelineAnalyzer()
        pr_analysis = analyzer.analyze_pr(
            pr_number=1,
            branch='feature',
            base_branch='main',
            repo_path=self.test_path
        )
        
        self.assertIsNotNone(pr_analysis)
        self.assertGreater(len(pr_analysis.new_issues), 0)
        
        # Check risk trajectory
        self.assertGreaterEqual(pr_analysis.risk_trajectory, 0)
        
        # Test GitHub integration (without actual API calls)
        github = GitHubIntegration({
            'auto_comment': False,  # Don't actually post
            'block_on_risk': True,
            'risk_threshold': 5.0
        })
        
        report = github._generate_pr_report(pr_analysis)
        self.assertIn('Tail-Chasing Detection Report', report)
        self.assertIn('New Issues', report)
        
        # Check if should block merge
        should_block = pr_analysis.should_block_merge(threshold=5.0)
        if pr_analysis.risk_trajectory > 5.0:
            self.assertTrue(should_block)
    
    def test_llm_feedback_generation(self):
        """Test LLM feedback generation."""
        # Create issues
        detector = TailChasingDetector()
        
        self.create_file("problems.py", """
            def func1():
                pass
            
            def func2():
                pass
            
            def func1_duplicate():
                pass
        """)
        
        issues = detector.detect(self.test_path)
        
        # Generate feedback
        feedback_gen = FeedbackGenerator()
        feedback = feedback_gen.generate_feedback(issues)
        
        self.assertIsNotNone(feedback)
        self.assertGreater(len(feedback.rules), 0)
        
        # Test formatting for different LLMs
        adapter = UniversalAdapter()
        
        # OpenAI format
        openai_prompt = adapter.format_feedback(feedback, 'openai')
        self.assertIsNotNone(openai_prompt)
        self.assertGreater(len(openai_prompt.messages), 0)
        
        # Anthropic format
        anthropic_prompt = adapter.format_feedback(feedback, 'anthropic')
        self.assertIsNotNone(anthropic_prompt)
        self.assertIsNotNone(anthropic_prompt.system_prompt)
        
        # Local LLM format
        local_prompt = adapter.format_feedback(feedback, 'local')
        self.assertIsNotNone(local_prompt)
        self.assertIsNotNone(local_prompt.system_prompt)
    
    def test_visualization_generation(self):
        """Test visualization and report generation."""
        # Create complex codebase
        self.create_file("module_a.py", """
            from .module_b import func_b
            
            def func_a():
                return func_b()
        """)
        
        self.create_file("module_b.py", """
            from .module_c import func_c
            
            def func_b():
                return func_c()
        """)
        
        self.create_file("module_c.py", """
            def func_c():
                return 42
            
            def func_c_duplicate():
                return 42
        """)
        
        # Detect issues
        detector = TailChasingDetector()
        issues = detector.detect(self.test_path)
        
        # Generate visualizations
        from tailchasing.visualization.tail_chase_visualizer import TailChaseVisualizer
        
        visualizer = TailChaseVisualizer()
        visualizer.add_issues(issues)
        
        # Generate dependency graph
        dep_graph = visualizer.generate_dependency_graph()
        self.assertIn('<svg', dep_graph)
        self.assertIn('module_a', dep_graph)
        
        # Generate similarity heatmap
        heatmap = visualizer.generate_similarity_heatmap()
        self.assertIn('<svg', heatmap)
        
        # Generate full HTML report
        report_gen = ReportGenerator()
        report_gen.add_issues(issues)
        
        html = report_gen.generate_html_report(
            include_visualizations=True,
            embed_data=True
        )
        
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('d3.js', html)  # D3 should be embedded
        self.assertIn('Risk Score', html)


class TestPerformanceIntegration(unittest.TestCase):
    """Test performance on larger codebases."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_large_codebase(self, num_files: int, functions_per_file: int):
        """Create a large synthetic codebase."""
        for i in range(num_files):
            lines = []
            for j in range(functions_per_file):
                lines.extend([
                    f"def function_{i}_{j}(param_{j}):",
                    f"    # Function in file {i}",
                    f"    result = param_{j} * {j}",
                    f"    if result > 100:",
                    f"        return result / 2",
                    f"    return result",
                    ""
                ])
                
                # Add some duplicates
                if j % 5 == 0 and j > 0:
                    lines.extend([
                        f"def function_{i}_{j}_dup(param_{j}):",
                        f"    # Duplicate function",
                        f"    result = param_{j} * {j}",
                        f"    if result > 100:",
                        f"        return result / 2",
                        f"    return result",
                        ""
                    ])
            
            (self.test_path / f"module_{i}.py").write_text("\n".join(lines))
    
    def test_performance_medium_codebase(self):
        """Test performance on medium-sized codebase."""
        # Create 50 files with 20 functions each (1000 functions total)
        self.create_large_codebase(num_files=50, functions_per_file=20)
        
        start_time = time.time()
        
        # Run detection
        detector = TailChasingDetector()
        issues = detector.detect(self.test_path)
        
        detection_time = time.time() - start_time
        
        print(f"\nMedium codebase performance:")
        print(f"  Files: 50")
        print(f"  Functions: ~1000")
        print(f"  Detection time: {detection_time:.2f}s")
        print(f"  Issues found: {len(issues)}")
        print(f"  Throughput: {1000/detection_time:.1f} functions/second")
        
        # Should complete in reasonable time
        self.assertLess(detection_time, 60)  # 1 minute max
        
        # Should find some issues
        self.assertGreater(len(issues), 50)  # At least some duplicates
    
    def test_performance_with_semantic_analysis(self):
        """Test performance with semantic analysis enabled."""
        # Create smaller codebase for semantic analysis
        self.create_large_codebase(num_files=10, functions_per_file=10)
        
        config = {
            'enable_semantic': True,
            'semantic_threshold': 0.8
        }
        
        start_time = time.time()
        
        detector = TailChasingDetector(config)
        issues = detector.detect(self.test_path)
        
        semantic_time = time.time() - start_time
        
        print(f"\nSemantic analysis performance:")
        print(f"  Files: 10")
        print(f"  Functions: ~100")
        print(f"  Analysis time: {semantic_time:.2f}s")
        print(f"  Issues found: {len(issues)}")
        
        # Semantic analysis is slower but should still be reasonable
        self.assertLess(semantic_time, 30)  # 30 seconds max
    
    def test_memory_usage(self):
        """Test memory usage on large codebase."""
        import psutil
        import os
        
        # Create large codebase
        self.create_large_codebase(num_files=100, functions_per_file=50)
        
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run detection
        detector = TailChasingDetector()
        issues = detector.detect(self.test_path)
        
        # Measure peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        print(f"\nMemory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        print(f"  Issues found: {len(issues)}")
        
        # Should not use excessive memory
        self.assertLess(memory_increase, 500)  # Less than 500MB increase


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world scenarios and edge cases."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_django_like_project(self):
        """Test on Django-like project structure."""
        # Create Django-like structure
        self.create_file("myapp/models.py", """
            from django.db import models
            
            class User(models.Model):
                name = models.CharField(max_length=100)
                email = models.EmailField()
                
                def save(self, *args, **kwargs):
                    # Custom save logic
                    super().save(*args, **kwargs)
            
            class Customer(models.Model):
                name = models.CharField(max_length=100)
                email = models.EmailField()
                
                def save(self, *args, **kwargs):
                    # Duplicate save logic
                    super().save(*args, **kwargs)
        """)
        
        self.create_file("myapp/views.py", """
            from django.shortcuts import render
            from .models import User, Customer
            
            def user_list(request):
                users = User.objects.all()
                return render(request, 'users.html', {'users': users})
            
            def customer_list(request):
                customers = Customer.objects.all()
                return render(request, 'customers.html', {'customers': customers})
        """)
        
        self.create_file("myapp/serializers.py", """
            from rest_framework import serializers
            from .models import User, Customer
            
            class UserSerializer(serializers.ModelSerializer):
                class Meta:
                    model = User
                    fields = '__all__'
            
            class CustomerSerializer(serializers.ModelSerializer):
                class Meta:
                    model = Customer
                    fields = '__all__'
        """)
        
        # Run detection
        detector = TailChasingDetector()
        issues = detector.detect(self.test_path)
        
        # Should detect structural duplicates but not flag framework patterns
        duplicate_issues = [i for i in issues if 'duplicate' in i.kind.lower()]
        
        # Should find the duplicate save methods
        self.assertGreater(len(duplicate_issues), 0)
        
        # But shouldn't flag standard Django patterns as high severity
        high_severity = [i for i in issues if i.severity >= 4]
        self.assertEqual(len(high_severity), 0)
    
    def test_flask_like_project(self):
        """Test on Flask-like project structure."""
        self.create_file("app.py", """
            from flask import Flask, request, jsonify
            
            app = Flask(__name__)
            
            @app.route('/users', methods=['GET'])
            def get_users():
                # Get all users
                return jsonify({'users': []})
            
            @app.route('/customers', methods=['GET'])
            def get_customers():
                # Duplicate logic
                return jsonify({'customers': []})
            
            @app.route('/products', methods=['GET'])
            def get_products():
                # Another duplicate
                return jsonify({'products': []})
            
            if __name__ == '__main__':
                app.run()
        """)
        
        self.create_file("models.py", """
            from dataclasses import dataclass
            
            @dataclass
            class User:
                id: int
                name: str
                email: str
            
            @dataclass
            class Customer:
                id: int
                name: str
                email: str
            
            @dataclass
            class Product:
                id: int
                name: str
                price: float
        """)
        
        # Run orchestrated analysis
        orchestrator = TailChasingOrchestrator()
        result = orchestrator.orchestrate(
            path=self.test_path,
            auto_fix=False
        )
        
        self.assertGreater(result['issues_found'], 0)
        
        # Should detect the duplicate route handlers
        issues = result.get('issues', [])
        route_duplicates = [i for i in issues if 'get_' in str(i)]
        self.assertGreater(len(route_duplicates), 0)
    
    def test_data_science_project(self):
        """Test on data science project with notebooks."""
        self.create_file("data_processing.py", """
            import pandas as pd
            import numpy as np
            
            def load_data(filepath):
                return pd.read_csv(filepath)
            
            def preprocess_data(df):
                # Remove nulls
                df = df.dropna()
                # Normalize
                df = (df - df.mean()) / df.std()
                return df
            
            def clean_data(dataframe):
                # Duplicate of preprocess_data
                dataframe = dataframe.dropna()
                dataframe = (dataframe - dataframe.mean()) / dataframe.std()
                return dataframe
            
            def transform_data(data):
                # Another duplicate
                data = data.dropna()
                data = (data - data.mean()) / data.std()
                return data
        """)
        
        self.create_file("model_training.py", """
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            
            def train_model(X, y):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                return model
            
            def build_model(features, target):
                # Duplicate training logic
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target, test_size=0.2, random_state=42
                )
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                return model
        """)
        
        # Run analysis with semantic detection
        detector = TailChasingDetector({'enable_semantic': True})
        issues = detector.detect(self.test_path)
        
        # Should detect the duplicate data processing functions
        processing_duplicates = [
            i for i in issues 
            if any(name in str(i) for name in ['preprocess', 'clean', 'transform'])
        ]
        self.assertGreater(len(processing_duplicates), 0)
        
        # Should detect duplicate model training
        training_duplicates = [
            i for i in issues
            if any(name in str(i) for name in ['train_model', 'build_model'])
        ]
        self.assertGreater(len(training_duplicates), 0)


class TestCLIIntegration(unittest.TestCase):
    """Test command-line interface integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_cli_basic_detection(self):
        """Test basic CLI detection."""
        # Create test files
        self.create_file("test.py", """
            def func1():
                return 1
            
            def func2():
                return 1
        """)
        
        # Run CLI command
        result = subprocess.run(
            ['python', '-m', 'tailchasing', str(self.test_path)],
            capture_output=True,
            text=True
        )
        
        # Should succeed
        self.assertEqual(result.returncode, 0)
        
        # Should detect issues
        self.assertIn('issue', result.stdout.lower())
    
    def test_cli_with_config(self):
        """Test CLI with configuration file."""
        # Create config
        config = {
            'severity_threshold': 3,
            'enable_semantic': True,
            'auto_fix': False,
            'output_format': 'json'
        }
        
        config_file = self.test_path / '.tailchasing.yml'
        import yaml
        config_file.write_text(yaml.dump(config))
        
        # Create test file
        self.create_file("test.py", """
            def duplicate1():
                pass
            
            def duplicate2():
                pass
        """)
        
        # Run with config
        result = subprocess.run(
            ['python', '-m', 'tailchasing', str(self.test_path), 
             '--config', str(config_file)],
            capture_output=True,
            text=True
        )
        
        # Should produce JSON output
        try:
            output = json.loads(result.stdout)
            self.assertIn('issues', output)
        except json.JSONDecodeError:
            self.fail("Expected JSON output")


if __name__ == '__main__':
    unittest.main()