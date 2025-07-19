"""Risk scoring system for tail-chasing issues."""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .issues import Issue


DEFAULT_WEIGHTS = {
    "circular_import": 3,
    "duplicate_function": 2,
    "phantom_function": 2,
    "missing_symbol": 2,
    "hallucinated_import": 3,
    "wrapper_abstraction": 1,
    "drift_entropy": 1,
    "tail_chasing_chain": 4,
}


class RiskScorer:
    """Calculate risk scores for tail-chasing issues."""
    
    def __init__(self, weights: Optional[Dict[str, int]] = None):
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        
    def calculate_scores(
        self, 
        issues: List[Issue]
    ) -> Tuple[Dict[str, float], float]:
        """Calculate module and global risk scores.
        
        Returns:
            Tuple of (module_scores, global_score)
        """
        module_scores = defaultdict(float)
        
        for issue in issues:
            # Get base weight for issue type
            weight = self.weights.get(issue.kind, 1)
            
            # Apply severity multiplier
            score = weight * issue.severity
            
            # Apply confidence modifier
            score *= issue.confidence
            
            # Assign to module
            module = issue.file or "<global>"
            module_scores[module] += score
            
        # Calculate global score
        if module_scores:
            # Average across modules, with penalty for number of affected modules
            avg_score = sum(module_scores.values()) / len(module_scores)
            module_penalty = min(len(module_scores) / 10, 2.0)  # Cap at 2x
            global_score = avg_score * module_penalty
        else:
            global_score = 0.0
            
        return dict(module_scores), round(global_score, 2)
        
    def calculate_module_score(self, issues: List[Issue], module: str) -> float:
        """Calculate risk score for a specific module."""
        score = 0.0
        
        for issue in issues:
            if issue.file == module:
                weight = self.weights.get(issue.kind, 1)
                score += weight * issue.severity * issue.confidence
                
        return round(score, 2)
        
    def get_risk_level(self, score: float, thresholds: Dict[str, int]) -> str:
        """Determine risk level based on score and thresholds."""
        fail_threshold = thresholds.get("fail", 30)
        warn_threshold = thresholds.get("warn", 15)
        
        if score >= fail_threshold:
            return "CRITICAL"
        elif score >= warn_threshold:
            return "WARNING"
        else:
            return "OK"
            
    def get_top_modules(
        self, 
        module_scores: Dict[str, float], 
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Get the top N modules by risk score."""
        sorted_modules = sorted(
            module_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_modules[:limit]
        
    def get_issue_distribution(self, issues: List[Issue]) -> Dict[str, int]:
        """Get count of issues by type."""
        distribution = defaultdict(int)
        
        for issue in issues:
            distribution[issue.kind] += 1
            
        return dict(distribution)
        
    def calculate_trend(
        self, 
        current_issues: List[Issue], 
        previous_issues: List[Issue]
    ) -> Dict[str, any]:
        """Calculate trend between two sets of issues."""
        current_scores, current_global = self.calculate_scores(current_issues)
        previous_scores, previous_global = self.calculate_scores(previous_issues)
        
        # Calculate changes
        score_change = current_global - previous_global
        score_change_pct = (
            (score_change / previous_global * 100) if previous_global > 0 else 0
        )
        
        # Count changes
        current_count = len(current_issues)
        previous_count = len(previous_issues)
        count_change = current_count - previous_count
        
        # New and resolved issues
        current_hashes = {
            (i.kind, i.file, i.line, i.symbol) for i in current_issues
        }
        previous_hashes = {
            (i.kind, i.file, i.line, i.symbol) for i in previous_issues
        }
        
        new_issues = current_hashes - previous_hashes
        resolved_issues = previous_hashes - current_hashes
        
        return {
            "score_change": round(score_change, 2),
            "score_change_pct": round(score_change_pct, 1),
            "count_change": count_change,
            "new_issues": len(new_issues),
            "resolved_issues": len(resolved_issues),
            "trend": "improving" if score_change < 0 else "worsening"
        }
