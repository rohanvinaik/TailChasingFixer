"""
Test fixtures for hallucination cascade detection.

Contains interdependent classes that form fictional subsystems,
typically created when LLMs invent entire architectures to satisfy dependencies.
"""

# Fictional microservice architecture - Part 1

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime


class ServiceMeshNode(ABC):
    """Abstract base for service mesh nodes."""
    
    def __init__(self, node_id: str, registry: 'ServiceRegistry'):
        self.node_id = node_id
        self.registry = registry
        self.health_status = "unknown"
        self.load_factor = 0.0
        self.connections = {}
    
    @abstractmethod
    def process_request(self, request: 'MeshRequest') -> 'MeshResponse':
        """Process incoming mesh request."""
        pass
    
    def register_with_mesh(self):
        """Register this node with the service mesh."""
        self.registry.register_node(self)
        self.health_status = "healthy"
    
    def update_load_metrics(self, cpu_usage: float, memory_usage: float):
        """Update load metrics for load balancing."""
        self.load_factor = (cpu_usage + memory_usage) / 2
        self.registry.update_node_metrics(self.node_id, self.load_factor)


class DistributedLoadBalancer:
    """Advanced distributed load balancer with mesh integration."""
    
    def __init__(self, mesh_coordinator: 'MeshCoordinator'):
        self.coordinator = mesh_coordinator
        self.balancing_strategies = {
            'round_robin': RoundRobinStrategy(),
            'weighted': WeightedStrategy(),
            'adaptive': AdaptiveStrategy()
        }
        self.current_strategy = 'adaptive'
    
    def select_optimal_node(self, service_type: str, request_context: Dict) -> ServiceMeshNode:
        """Select optimal node using advanced algorithms."""
        available_nodes = self.coordinator.get_healthy_nodes(service_type)
        strategy = self.balancing_strategies[self.current_strategy]
        
        return strategy.select_node(available_nodes, request_context)
    
    def rebalance_mesh(self):
        """Trigger mesh rebalancing based on current load."""
        self.coordinator.trigger_rebalancing()


class MeshCoordinator:
    """Central coordinator for service mesh operations."""
    
    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.circuit_breaker = CircuitBreakerManager()
        self.telemetry_collector = TelemetryCollector()
        self.mesh_nodes = {}
        self.rebalancing_threshold = 0.8
    
    def get_healthy_nodes(self, service_type: str) -> List[ServiceMeshNode]:
        """Get healthy nodes for a service type."""
        all_nodes = self.service_registry.get_nodes_by_type(service_type)
        return [node for node in all_nodes if node.health_status == "healthy"]
    
    def trigger_rebalancing(self):
        """Trigger mesh-wide load rebalancing."""
        high_load_nodes = [
            node for node in self.mesh_nodes.values() 
            if node.load_factor > self.rebalancing_threshold
        ]
        
        if high_load_nodes:
            self.redistribute_load(high_load_nodes)
    
    def redistribute_load(self, overloaded_nodes: List[ServiceMeshNode]):
        """Redistribute load from overloaded nodes."""
        for node in overloaded_nodes:
            self.circuit_breaker.temporarily_disable(node.node_id)
            self.telemetry_collector.log_rebalancing_event(node.node_id)


class ServiceRegistry:
    """Registry for service mesh nodes and discovery."""
    
    def __init__(self):
        self.nodes_by_type = {}
        self.node_metrics = {}
        self.discovery_cache = DiscoveryCache()
    
    def register_node(self, node: ServiceMeshNode):
        """Register a new service mesh node."""
        service_type = node.__class__.__name__
        if service_type not in self.nodes_by_type:
            self.nodes_by_type[service_type] = []
        
        self.nodes_by_type[service_type].append(node)
        self.discovery_cache.invalidate_for_type(service_type)
    
    def get_nodes_by_type(self, service_type: str) -> List[ServiceMeshNode]:
        """Get all nodes of a specific service type."""
        cached_result = self.discovery_cache.get(service_type)
        if cached_result:
            return cached_result
        
        nodes = self.nodes_by_type.get(service_type, [])
        self.discovery_cache.set(service_type, nodes)
        return nodes
    
    def update_node_metrics(self, node_id: str, load_factor: float):
        """Update metrics for a node."""
        self.node_metrics[node_id] = {
            'load_factor': load_factor,
            'last_updated': datetime.now(),
            'status': 'active'
        }


class CircuitBreakerManager:
    """Manages circuit breakers for service mesh resilience."""
    
    def __init__(self):
        self.circuit_states = {}
        self.failure_thresholds = {}
        self.recovery_timeouts = {}
    
    def temporarily_disable(self, node_id: str):
        """Temporarily disable a node via circuit breaker."""
        self.circuit_states[node_id] = 'open'
        self.recovery_timeouts[node_id] = datetime.now().timestamp() + 60  # 1 minute
    
    def is_circuit_open(self, node_id: str) -> bool:
        """Check if circuit breaker is open for a node."""
        if node_id not in self.circuit_states:
            return False
        
        if self.circuit_states[node_id] == 'open':
            # Check if recovery timeout has passed
            if datetime.now().timestamp() > self.recovery_timeouts.get(node_id, 0):
                self.circuit_states[node_id] = 'half-open'
                return False
            return True
        
        return False


class TelemetryCollector:
    """Collects and aggregates telemetry from the service mesh."""
    
    def __init__(self):
        self.metrics_buffer = MetricsBuffer()
        self.aggregation_engine = AggregationEngine()
        self.alerting_system = AlertingSystem()
    
    def log_rebalancing_event(self, node_id: str):
        """Log a load rebalancing event."""
        event = {
            'event_type': 'rebalancing',
            'node_id': node_id,
            'timestamp': datetime.now().isoformat(),
            'severity': 'info'
        }
        
        self.metrics_buffer.add_event(event)
        self.check_alerting_conditions(event)
    
    def check_alerting_conditions(self, event: Dict):
        """Check if event should trigger alerts."""
        if event['event_type'] == 'rebalancing':
            recent_rebalances = self.count_recent_rebalances()
            if recent_rebalances > 5:  # Too many rebalances
                self.alerting_system.trigger_alert('high_rebalancing_frequency')


# Load balancing strategies (more fictional components)

class RoundRobinStrategy:
    """Round-robin load balancing strategy."""
    
    def __init__(self):
        self.current_index = 0
    
    def select_node(self, nodes: List[ServiceMeshNode], context: Dict) -> ServiceMeshNode:
        """Select next node in round-robin fashion."""
        if not nodes:
            return None
        
        selected = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return selected


class WeightedStrategy:
    """Weighted load balancing strategy."""
    
    def select_node(self, nodes: List[ServiceMeshNode], context: Dict) -> ServiceMeshNode:
        """Select node based on inverse load weighting."""
        if not nodes:
            return None
        
        # Select node with lowest load factor
        return min(nodes, key=lambda node: node.load_factor)


class AdaptiveStrategy:
    """Adaptive load balancing with ML predictions."""
    
    def __init__(self):
        self.predictor = LoadPredictor()
        self.adaptation_engine = AdaptationEngine()
    
    def select_node(self, nodes: List[ServiceMeshNode], context: Dict) -> ServiceMeshNode:
        """Select node using adaptive algorithms."""
        if not nodes:
            return None
        
        predictions = self.predictor.predict_loads(nodes, context)
        optimal_node = self.adaptation_engine.optimize_selection(nodes, predictions)
        
        return optimal_node


# More fictional supporting classes

class DiscoveryCache:
    """Cache for service discovery results."""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}
    
    def get(self, service_type: str) -> Optional[List]:
        """Get cached discovery result."""
        if service_type in self.cache:
            if datetime.now().timestamp() < self.cache_ttl.get(service_type, 0):
                return self.cache[service_type]
        return None
    
    def set(self, service_type: str, nodes: List):
        """Cache discovery result."""
        self.cache[service_type] = nodes
        self.cache_ttl[service_type] = datetime.now().timestamp() + 300  # 5 minutes
    
    def invalidate_for_type(self, service_type: str):
        """Invalidate cache for service type."""
        self.cache.pop(service_type, None)
        self.cache_ttl.pop(service_type, None)


class MetricsBuffer:
    """Buffer for collecting metrics before aggregation."""
    
    def __init__(self):
        self.events = []
        self.max_buffer_size = 1000
    
    def add_event(self, event: Dict):
        """Add event to buffer."""
        self.events.append(event)
        if len(self.events) > self.max_buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Flush buffer to aggregation engine."""
        # Simulate flushing
        self.events = []


class AggregationEngine:
    """Engine for aggregating telemetry data."""
    
    def aggregate_metrics(self, time_window: int) -> Dict:
        """Aggregate metrics over time window."""
        # Fictional aggregation logic
        return {
            'total_requests': 0,
            'average_latency': 0,
            'error_rate': 0
        }


class AlertingSystem:
    """System for triggering alerts based on metrics."""
    
    def __init__(self):
        self.alert_channels = []
    
    def trigger_alert(self, alert_type: str):
        """Trigger an alert."""
        # Simulate alert triggering
        pass


class LoadPredictor:
    """ML-based load prediction system."""
    
    def predict_loads(self, nodes: List[ServiceMeshNode], context: Dict) -> Dict[str, float]:
        """Predict future loads for nodes."""
        # Fictional ML prediction
        return {node.node_id: 0.5 for node in nodes}


class AdaptationEngine:
    """Engine for adaptive load balancing decisions."""
    
    def optimize_selection(self, nodes: List[ServiceMeshNode], predictions: Dict) -> ServiceMeshNode:
        """Optimize node selection based on predictions."""
        # Fictional optimization logic
        return nodes[0] if nodes else None


# Request/Response classes for the fictional mesh

class MeshRequest:
    """Request object for service mesh communication."""
    
    def __init__(self, request_id: str, service_type: str, payload: Dict):
        self.request_id = request_id
        self.service_type = service_type
        self.payload = payload
        self.timestamp = datetime.now()
        self.trace_id = f"trace_{request_id}"


class MeshResponse:
    """Response object for service mesh communication."""
    
    def __init__(self, request_id: str, status: str, data: Any = None, error: str = None):
        self.request_id = request_id
        self.status = status
        self.data = data
        self.error = error
        self.timestamp = datetime.now()
        self.processing_time_ms = 0


# Concrete service implementations

class DataProcessingService(ServiceMeshNode):
    """Concrete service for data processing."""
    
    def process_request(self, request: MeshRequest) -> MeshResponse:
        """Process data processing requests."""
        # Fictional data processing logic
        return MeshResponse(
            request_id=request.request_id,
            status="success",
            data={"processed": True, "items": 42}
        )


class AnalyticsService(ServiceMeshNode):
    """Concrete service for analytics."""
    
    def process_request(self, request: MeshRequest) -> MeshResponse:
        """Process analytics requests."""
        # Fictional analytics logic
        return MeshResponse(
            request_id=request.request_id,
            status="success",
            data={"metrics": {"views": 100, "clicks": 25}}
        )


# This is a REAL utility class that should NOT be part of the cascade
class ConfigurationManager:
    """Real configuration manager - properly implemented."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config_data = {}
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from file."""
        try:
            import json
            with open(self.config_file, 'r') as f:
                self.config_data = json.load(f)
        except FileNotFoundError:
            self.config_data = self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration values."""
        return {
            'database_url': 'sqlite:///app.db',
            'debug_mode': False,
            'log_level': 'INFO'
        }
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config_data.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        self.config_data[key] = value
    
    def save_configuration(self):
        """Save configuration to file."""
        import json
        with open(self.config_file, 'w') as f:
            json.dump(self.config_data, f, indent=2)