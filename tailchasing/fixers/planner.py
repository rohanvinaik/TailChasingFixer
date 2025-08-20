"""
Strategy planner for intelligent fix ordering and selection.

This module provides planning capabilities for selecting and ordering fix strategies
based on analyzer signals, cost estimates, and historical success rates.
"""

import json
import pathlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

from ..core.issues import Issue
from ..utils.logging_setup import get_logger


@dataclass
class StrategyScore:
    """Score and metadata for a strategy."""
    strategy_name: str
    base_score: float
    severity_weight: float
    frequency_weight: float
    cost_weight: float
    success_weight: float
    final_score: float
    estimated_time: float
    estimated_cost: float
    prior_success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningCounter:
    """Track success/failure counts for strategies."""
    strategy_name: str
    issue_signature: str
    success_count: int = 0
    failure_count: int = 0
    total_time: float = 0.0
    last_updated: str = ""
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Default to 50% for unknown
        return self.success_count / total
    
    @property
    def avg_time(self) -> float:
        """Calculate average time."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.total_time / total


class StrategyPlanner:
    """Plans and ranks fix strategies based on multiple signals."""
    
    # Default weight configuration
    DEFAULT_WEIGHTS = {
        "severity": 0.3,      # Issue severity impact
        "frequency": 0.2,     # How often this issue occurs
        "cost": 0.15,         # Computational/time cost
        "success": 0.25,      # Historical success rate
        "confidence": 0.1     # Strategy confidence
    }
    
    # Cost estimates per strategy type (in abstract units)
    STRATEGY_COSTS = {
        "simple_replacement": 1.0,
        "ast_transformation": 2.0,
        "semantic_analysis": 3.0,
        "llm_generation": 5.0,
        "multi_file_refactor": 8.0,
        "codebase_restructure": 10.0
    }
    
    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 learning_db_path: Optional[pathlib.Path] = None):
        """
        Initialize the planner.
        
        Args:
            weights: Custom weight configuration
            learning_db_path: Path to persist learning counters
        """
        self.logger = get_logger("strategy_planner")
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.learning_db_path = learning_db_path or pathlib.Path(".tailchasing_learning.json")
        self.learning_counters = self._load_learning_counters()
        
    def _load_learning_counters(self) -> Dict[str, LearningCounter]:
        """Load persisted learning counters from disk."""
        counters = {}
        
        if self.learning_db_path.exists():
            try:
                with open(self.learning_db_path, 'r') as f:
                    data = json.load(f)
                    for key, values in data.items():
                        counters[key] = LearningCounter(**values)
                self.logger.debug(f"Loaded {len(counters)} learning counters")
            except Exception as e:
                self.logger.error(f"Failed to load learning counters: {e}")
        
        return counters
    
    def _save_learning_counters(self) -> None:
        """Persist learning counters to disk."""
        try:
            data = {}
            for key, counter in self.learning_counters.items():
                data[key] = asdict(counter)
            
            with open(self.learning_db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.debug(f"Saved {len(data)} learning counters")
        except Exception as e:
            self.logger.error(f"Failed to save learning counters: {e}")
    
    def _get_issue_signature(self, issue: Issue) -> str:
        """Generate a stable signature for an issue type."""
        # Create signature from issue kind and key evidence
        signature_parts = [issue.kind]
        
        if issue.evidence:
            # Add key evidence patterns
            if "pattern" in issue.evidence:
                signature_parts.append(issue.evidence["pattern"])
            if "type" in issue.evidence:
                signature_parts.append(issue.evidence["type"])
            if "count" in issue.evidence:
                # Bucket the count to avoid over-specialization
                count = issue.evidence["count"]
                if count <= 2:
                    signature_parts.append("few")
                elif count <= 5:
                    signature_parts.append("several")
                else:
                    signature_parts.append("many")
        
        return ":".join(str(p) for p in signature_parts)
    
    def _get_learning_counter(self, strategy_name: str, issue: Issue) -> LearningCounter:
        """Get or create learning counter for strategy/issue pair."""
        signature = self._get_issue_signature(issue)
        key = f"{strategy_name}:{signature}"
        
        if key not in self.learning_counters:
            self.learning_counters[key] = LearningCounter(
                strategy_name=strategy_name,
                issue_signature=signature
            )
        
        return self.learning_counters[key]
    
    def _estimate_strategy_cost(self, strategy: Any, issue: Issue) -> Tuple[float, float]:
        """
        Estimate cost and time for a strategy.
        
        Returns:
            Tuple of (cost_units, time_seconds)
        """
        # Get base cost from strategy type
        strategy_type = getattr(strategy, "strategy_type", "simple_replacement")
        base_cost = self.STRATEGY_COSTS.get(strategy_type, 2.0)
        
        # Adjust based on issue complexity
        complexity_multiplier = 1.0
        if issue.evidence:
            if issue.evidence.get("count", 0) > 10:
                complexity_multiplier *= 1.5
            if issue.evidence.get("files_affected", 1) > 3:
                complexity_multiplier *= 2.0
            if issue.evidence.get("dependencies", 0) > 5:
                complexity_multiplier *= 1.3
        
        cost = base_cost * complexity_multiplier
        
        # Estimate time (rough approximation)
        time_seconds = cost * 0.5  # 0.5 seconds per cost unit
        
        return cost, time_seconds
    
    def _rank_strategies(self, 
                        strategies: List[Any],
                        issues: List[Issue],
                        context: Optional[Dict[str, Any]] = None) -> List[StrategyScore]:
        """
        Rank strategies based on multiple weighted factors.
        
        This scores strategies by:
        1. Analyzer signals (severity, frequency)
        2. Estimated cost/time
        3. Prior success on similar issues
        
        Args:
            strategies: List of available strategies
            issues: List of issues to consider
            context: Additional context (analyzer results, etc.)
        
        Returns:
            List of StrategyScore objects, sorted by final score (descending)
        """
        scores = []
        
        # Aggregate issue statistics
        issue_stats = self._analyze_issues(issues)
        
        for strategy in strategies:
            strategy_name = getattr(strategy, "name", str(strategy))
            
            # Calculate component scores
            severity_score = 0.0
            frequency_score = 0.0
            cost_score = 0.0
            success_score = 0.0
            confidence_score = 0.0
            
            # Aggregate across relevant issues
            relevant_issues = [
                issue for issue in issues
                if hasattr(strategy, "can_handle") and strategy.can_handle(issue)
            ]
            
            if not relevant_issues:
                continue
            
            # 1. Severity score (higher severity = higher score)
            avg_severity = sum(issue.severity for issue in relevant_issues) / len(relevant_issues)
            severity_score = avg_severity / 5.0  # Normalize to 0-1
            
            # 2. Frequency score (more frequent = higher score)
            frequency = len(relevant_issues) / max(1, len(issues))
            frequency_score = frequency
            
            # 3. Cost score (lower cost = higher score)
            total_cost = 0.0
            total_time = 0.0
            for issue in relevant_issues:
                cost, time = self._estimate_strategy_cost(strategy, issue)
                total_cost += cost
                total_time += time
            avg_cost = total_cost / len(relevant_issues)
            # Invert and normalize (assuming max cost of 10)
            cost_score = 1.0 - min(avg_cost / 10.0, 1.0)
            
            # 4. Success score (historical success rate)
            success_rates = []
            for issue in relevant_issues:
                counter = self._get_learning_counter(strategy_name, issue)
                success_rates.append(counter.success_rate)
            if success_rates:
                success_score = sum(success_rates) / len(success_rates)
            else:
                success_score = 0.5  # Default to neutral
            
            # 5. Confidence score (strategy's self-reported confidence)
            if hasattr(strategy, "get_confidence"):
                confidence_score = strategy.get_confidence(relevant_issues)
            else:
                confidence_score = 0.7  # Default confidence
            
            # Calculate weighted final score
            final_score = (
                self.weights["severity"] * severity_score +
                self.weights["frequency"] * frequency_score +
                self.weights["cost"] * cost_score +
                self.weights["success"] * success_score +
                self.weights["confidence"] * confidence_score
            )
            
            scores.append(StrategyScore(
                strategy_name=strategy_name,
                base_score=confidence_score,
                severity_weight=severity_score * self.weights["severity"],
                frequency_weight=frequency_score * self.weights["frequency"],
                cost_weight=cost_score * self.weights["cost"],
                success_weight=success_score * self.weights["success"],
                final_score=final_score,
                estimated_time=total_time,
                estimated_cost=total_cost,
                prior_success_rate=success_score,
                metadata={
                    "relevant_issues": len(relevant_issues),
                    "avg_severity": avg_severity,
                    "total_issues": len(issues)
                }
            ))
        
        # Sort by final score (descending)
        scores.sort(key=lambda x: x.final_score, reverse=True)
        
        # Log ranking results
        self.logger.debug("Strategy ranking results:")
        for i, score in enumerate(scores[:5]):  # Top 5
            self.logger.debug(
                f"  {i+1}. {score.strategy_name}: {score.final_score:.3f} "
                f"(sev={score.severity_weight:.2f}, freq={score.frequency_weight:.2f}, "
                f"cost={score.cost_weight:.2f}, success={score.success_weight:.2f})"
            )
        
        return scores
    
    def _analyze_issues(self, issues: List[Issue]) -> Dict[str, Any]:
        """Analyze issue statistics for scoring."""
        stats = {
            "total": len(issues),
            "by_kind": defaultdict(int),
            "by_severity": defaultdict(int),
            "avg_severity": 0.0,
            "unique_files": set(),
        }
        
        for issue in issues:
            stats["by_kind"][issue.kind] += 1
            stats["by_severity"][issue.severity] += 1
            if issue.file:
                stats["unique_files"].add(issue.file)
        
        if issues:
            stats["avg_severity"] = sum(i.severity for i in issues) / len(issues)
        
        stats["unique_files"] = len(stats["unique_files"])
        
        return stats
    
    def update_learning(self,
                       strategy_name: str,
                       issue: Issue,
                       success: bool,
                       time_taken: float) -> None:
        """
        Update learning counters after strategy execution.
        
        Args:
            strategy_name: Name of the strategy
            issue: The issue that was addressed
            success: Whether the strategy succeeded
            time_taken: Time taken in seconds
        """
        counter = self._get_learning_counter(strategy_name, issue)
        
        if success:
            counter.success_count += 1
        else:
            counter.failure_count += 1
        
        counter.total_time += time_taken
        counter.last_updated = datetime.now().isoformat()
        
        # Persist updates
        self._save_learning_counters()
        
        self.logger.info(
            f"Updated learning for {strategy_name} on {issue.kind}: "
            f"success={success}, rate={counter.success_rate:.2%}, "
            f"avg_time={counter.avg_time:.2f}s"
        )
    
    def get_recommended_order(self,
                            strategies: List[Any],
                            issues: List[Issue],
                            context: Optional[Dict[str, Any]] = None) -> List[Tuple[Any, List[Issue]]]:
        """
        Get recommended execution order for strategies.
        
        Returns:
            List of (strategy, applicable_issues) tuples in recommended order
        """
        # Rank strategies
        scores = self._rank_strategies(strategies, issues, context)
        
        # Build execution plan
        plan = []
        processed_issues = set()
        
        for score in scores:
            # Find the strategy object
            strategy = None
            for s in strategies:
                if getattr(s, "name", str(s)) == score.strategy_name:
                    strategy = s
                    break
            
            if not strategy:
                continue
            
            # Find unprocessed issues this strategy can handle
            applicable_issues = []
            for issue in issues:
                if id(issue) not in processed_issues:
                    if hasattr(strategy, "can_handle") and strategy.can_handle(issue):
                        applicable_issues.append(issue)
                        processed_issues.add(id(issue))
            
            if applicable_issues:
                plan.append((strategy, applicable_issues))
        
        return plan
    
    def suggest_parallelization(self,
                               plan: List[Tuple[Any, List[Issue]]]) -> List[List[Tuple[Any, List[Issue]]]]:
        """
        Suggest which strategies can be run in parallel.
        
        Returns:
            List of batches, where each batch contains strategies that can run in parallel
        """
        batches = []
        
        # Simple heuristic: strategies working on different files can parallelize
        current_batch = []
        used_files = set()
        
        for strategy, issues in plan:
            # Get files affected by this strategy
            strategy_files = set()
            for issue in issues:
                if issue.file:
                    strategy_files.add(issue.file)
            
            # Check if conflicts with current batch
            if strategy_files & used_files:
                # Conflict - start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [(strategy, issues)]
                used_files = strategy_files
            else:
                # No conflict - add to current batch
                current_batch.append((strategy, issues))
                used_files.update(strategy_files)
        
        if current_batch:
            batches.append(current_batch)
        
        return batches


def main():
    """CLI entry point for testing the planner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test strategy planner")
    parser.add_argument("--weights", type=json.loads, 
                       help="JSON string of custom weights")
    parser.add_argument("--learning-db", type=pathlib.Path,
                       help="Path to learning database")
    
    args = parser.parse_args()
    
    planner = StrategyPlanner(
        weights=args.weights,
        learning_db_path=args.learning_db
    )
    
    # Example usage
    print("Strategy Planner initialized")
    print(f"Weights: {planner.weights}")
    print(f"Learning DB: {planner.learning_db_path}")
    print(f"Loaded {len(planner.learning_counters)} learning counters")


if __name__ == "__main__":
    main()