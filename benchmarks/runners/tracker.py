"""
Performance tracker for monitoring benchmark trends over time.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceTracker:
    """Track and analyze benchmark performance over time."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize performance tracker with database.
        
        Args:
            db_path: Path to SQLite database for storing results
        """
        self.db_path = db_path or Path("benchmark_results/performance.db")
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    scenario_name TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    steps_taken INTEGER,
                    expected_steps_min INTEGER,
                    expected_steps_max INTEGER,
                    time_elapsed REAL,
                    tokens_used INTEGER,
                    cost_estimate REAL,
                    efficiency_score REAL,
                    regressions_count INTEGER,
                    error_message TEXT,
                    git_commit TEXT,
                    code_version TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmark_runs(timestamp);
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scenario_model ON benchmark_runs(scenario_name, model_name);
            """)
    
    def record_run(self, 
                  scenario_name: str,
                  model_name: str,
                  result: Dict[str, Any],
                  git_commit: Optional[str] = None,
                  code_version: Optional[str] = None):
        """Record a benchmark run in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO benchmark_runs (
                    scenario_name, model_name, success, steps_taken,
                    expected_steps_min, expected_steps_max,
                    time_elapsed, tokens_used, cost_estimate,
                    efficiency_score, regressions_count, error_message,
                    git_commit, code_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scenario_name,
                model_name,
                result.get("success", False),
                result.get("steps_taken"),
                result.get("expected_steps", [0, 0])[0],
                result.get("expected_steps", [0, 0])[1],
                result.get("time_elapsed"),
                result.get("tokens_used"),
                result.get("cost_estimate"),
                result.get("efficiency_score"),
                len(result.get("regressions_detected", [])),
                result.get("error_message"),
                git_commit,
                code_version
            ))
    
    def get_performance_trends(self,
                              scenario_name: Optional[str] = None,
                              model_name: Optional[str] = None,
                              days_back: int = 30) -> pd.DataFrame:
        """Get performance trends over time."""
        query = """
            SELECT * FROM benchmark_runs
            WHERE timestamp >= datetime('now', '-{} days')
        """.format(days_back)
        
        params = []
        if scenario_name:
            query += " AND scenario_name = ?"
            params.append(scenario_name)
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY timestamp"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def identify_regressions(self, 
                           scenario_name: str,
                           model_name: str,
                           window_size: int = 10) -> List[Dict[str, Any]]:
        """Identify performance regressions using statistical analysis."""
        df = self.get_performance_trends(scenario_name, model_name)
        
        if len(df) < window_size:
            return []
        
        regressions = []
        
        # Calculate moving averages
        df['steps_ma'] = df['steps_taken'].rolling(window=window_size, min_periods=1).mean()
        df['time_ma'] = df['time_elapsed'].rolling(window=window_size, min_periods=1).mean()
        df['success_rate'] = df['success'].rolling(window=window_size, min_periods=1).mean()
        
        # Detect regressions (significant increases in steps/time or decreases in success rate)
        for i in range(window_size, len(df)):
            current = df.iloc[i]
            
            # Check for step count regression (>20% increase)
            if current['steps_taken'] > df['steps_ma'].iloc[i-1] * 1.2:
                regressions.append({
                    "timestamp": current['timestamp'],
                    "type": "step_count",
                    "value": current['steps_taken'],
                    "baseline": df['steps_ma'].iloc[i-1],
                    "git_commit": current.get('git_commit')
                })
            
            # Check for time regression (>30% increase)
            if current['time_elapsed'] > df['time_ma'].iloc[i-1] * 1.3:
                regressions.append({
                    "timestamp": current['timestamp'],
                    "type": "execution_time",
                    "value": current['time_elapsed'],
                    "baseline": df['time_ma'].iloc[i-1],
                    "git_commit": current.get('git_commit')
                })
            
            # Check for success rate drop (>10% decrease)
            current_window_success = df['success'].iloc[i-window_size+1:i+1].mean()
            prev_window_success = df['success'].iloc[i-window_size:i].mean()
            
            if prev_window_success > 0 and current_window_success < prev_window_success * 0.9:
                regressions.append({
                    "timestamp": current['timestamp'],
                    "type": "success_rate",
                    "value": current_window_success,
                    "baseline": prev_window_success,
                    "git_commit": current.get('git_commit')
                })
        
        return regressions
    
    def identify_problem_patterns(self) -> Dict[str, List[str]]:
        """Identify common problem patterns across all benchmarks."""
        with sqlite3.connect(self.db_path) as conn:
            # Find scenarios that frequently fail
            failing_scenarios = pd.read_sql_query("""
                SELECT scenario_name, model_name,
                       AVG(success) as success_rate,
                       COUNT(*) as run_count
                FROM benchmark_runs
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY scenario_name, model_name
                HAVING success_rate < 0.5 AND run_count >= 5
                ORDER BY success_rate
            """, conn)
            
            # Find scenarios with high regression counts
            regression_prone = pd.read_sql_query("""
                SELECT scenario_name, model_name,
                       AVG(regressions_count) as avg_regressions,
                       COUNT(*) as run_count
                FROM benchmark_runs
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY scenario_name, model_name
                HAVING avg_regressions > 1 AND run_count >= 5
                ORDER BY avg_regressions DESC
            """, conn)
            
            # Find scenarios taking too many steps
            inefficient = pd.read_sql_query("""
                SELECT scenario_name, model_name,
                       AVG(steps_taken) as avg_steps,
                       AVG(expected_steps_max) as expected_max,
                       COUNT(*) as run_count
                FROM benchmark_runs
                WHERE timestamp >= datetime('now', '-30 days') AND success = 1
                GROUP BY scenario_name, model_name
                HAVING avg_steps > expected_max * 1.5 AND run_count >= 5
                ORDER BY avg_steps DESC
            """, conn)
        
        patterns = {
            "frequently_failing": failing_scenarios.to_dict('records') if not failing_scenarios.empty else [],
            "regression_prone": regression_prone.to_dict('records') if not regression_prone.empty else [],
            "inefficient": inefficient.to_dict('records') if not inefficient.empty else []
        }
        
        return patterns
    
    def generate_performance_report(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {},
            "trends": {},
            "regressions": {},
            "problem_patterns": {}
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall summary
            summary = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_runs,
                    AVG(success) as overall_success_rate,
                    AVG(steps_taken) as avg_steps,
                    AVG(time_elapsed) as avg_time,
                    AVG(tokens_used) as avg_tokens,
                    AVG(cost_estimate) as avg_cost
                FROM benchmark_runs
                WHERE timestamp >= datetime('now', '-30 days')
            """, conn).iloc[0].to_dict()
            
            report["summary"] = summary
            
            # Per-model summary
            model_summary = pd.read_sql_query("""
                SELECT 
                    model_name,
                    COUNT(*) as run_count,
                    AVG(success) as success_rate,
                    AVG(steps_taken) as avg_steps,
                    AVG(time_elapsed) as avg_time,
                    AVG(cost_estimate) as avg_cost
                FROM benchmark_runs
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY model_name
                ORDER BY success_rate DESC
            """, conn)
            
            report["model_performance"] = model_summary.to_dict('records')
            
            # Scenario difficulty ranking
            scenario_difficulty = pd.read_sql_query("""
                SELECT 
                    scenario_name,
                    AVG(success) as success_rate,
                    AVG(steps_taken) as avg_steps,
                    AVG(steps_taken * 1.0 / expected_steps_max) as relative_difficulty
                FROM benchmark_runs
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY scenario_name
                ORDER BY relative_difficulty DESC
            """, conn)
            
            report["scenario_difficulty"] = scenario_difficulty.to_dict('records')
        
        # Identify problem patterns
        report["problem_patterns"] = self.identify_problem_patterns()
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def plot_performance_trends(self, 
                               scenario_name: Optional[str] = None,
                               model_name: Optional[str] = None,
                               save_path: Optional[Path] = None):
        """Plot performance trends over time."""
        df = self.get_performance_trends(scenario_name, model_name)
        
        if df.empty:
            print("No data available for plotting")
            return
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Trends: {scenario_name or "All Scenarios"} - {model_name or "All Models"}')
        
        # Success rate over time
        ax = axes[0, 0]
        df_grouped = df.groupby(df['timestamp'].dt.date)['success'].mean()
        ax.plot(df_grouped.index, df_grouped.values, marker='o')
        ax.set_title('Success Rate Over Time')
        ax.set_ylabel('Success Rate')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        # Steps taken over time
        ax = axes[0, 1]
        successful_runs = df[df['success'] == True]
        if not successful_runs.empty:
            ax.scatter(successful_runs['timestamp'], successful_runs['steps_taken'], alpha=0.6)
            # Add trend line
            z = np.polyfit(range(len(successful_runs)), successful_runs['steps_taken'], 1)
            p = np.poly1d(z)
            ax.plot(successful_runs['timestamp'], p(range(len(successful_runs))), "r--", alpha=0.8)
        ax.set_title('Steps to Convergence')
        ax.set_ylabel('Steps Taken')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        # Execution time over time
        ax = axes[1, 0]
        if not successful_runs.empty:
            ax.scatter(successful_runs['timestamp'], successful_runs['time_elapsed'], alpha=0.6)
        ax.set_title('Execution Time')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        # Cost over time
        ax = axes[1, 1]
        if not successful_runs.empty:
            ax.scatter(successful_runs['timestamp'], successful_runs['cost_estimate'], alpha=0.6)
        ax.set_title('Cost per Run')
        ax.set_ylabel('Cost (USD)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()