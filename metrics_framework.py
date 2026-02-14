"""
Routing Metrics Framework for Agent Memory Systems
===================================================

Metrics designed for:
1. Evaluating multi-store routing policies
2. LLMOps monitoring and observability
3. Benchmarking memory systems

These metrics can be adopted by the field for standardized evaluation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import json

#==============================================================================
# METRIC DEFINITIONS
#==============================================================================

@dataclass
class RoutingMetrics:
    """
    Comprehensive metrics for evaluating routing policies.
    
    Categories:
    1. Accuracy Metrics - Did we route correctly?
    2. Efficiency Metrics - Did we minimize cost?
    3. Coverage Metrics - Did we retrieve what was needed?
    4. Robustness Metrics - How does performance degrade?
    5. Operational Metrics - For production monitoring
    """
    
    # Accuracy Metrics
    routing_accuracy: float = 0.0          # Exact match of predicted vs ground truth stores
    store_precision: float = 0.0           # Precision across all stores
    store_recall: float = 0.0              # Recall across all stores
    store_f1: float = 0.0                  # F1 score for store selection
    
    # Per-store accuracy
    per_store_precision: Dict[str, float] = field(default_factory=dict)
    per_store_recall: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency Metrics
    retrieval_efficiency: float = 0.0      # Oracle_k / Actual_k (higher = more efficient)
    over_retrieval_rate: float = 0.0       # % queries that retrieved from unnecessary stores
    under_retrieval_rate: float = 0.0      # % queries that missed necessary stores
    cost_ratio: float = 0.0                # Actual_cost / Uniform_cost
    
    # Coverage Metrics
    critical_recall: float = 0.0           # Recall on critical queries
    optional_recall: float = 0.0           # Recall on optional queries
    answer_coverage: float = 0.0           # % queries where retrieved content contains answer
    
    # Robustness Metrics
    staleness_sensitivity: float = 0.0     # Accuracy drop per turn of staleness
    cold_start_accuracy: float = 0.0       # Accuracy in first N turns
    warm_accuracy: float = 0.0             # Accuracy after N turns
    
    # Operational Metrics (for LLMOps)
    avg_latency_ms: float = 0.0            # Average routing decision latency
    p99_latency_ms: float = 0.0            # 99th percentile latency
    store_utilization: Dict[str, float] = field(default_factory=dict)  # % queries hitting each store
    redundancy_exploitation: float = 0.0   # When info in multiple stores, % routed to cheapest
    
    # Composite Scores
    cost_adjusted_accuracy: float = 0.0    # Accuracy * Efficiency
    criticality_weighted_accuracy: float = 0.0  # Accuracy weighted by query importance


class MetricsCalculator:
    """Calculate all routing metrics from predictions and ground truth."""
    
    STORE_COSTS = {
        "stm": 1,
        "summary": 1,
        "ltm": 3,
        "episodic": 5,
    }
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulators."""
        self.predictions = []
        self.ground_truths = []
        self.queries = []
        self.latencies = []
        
        # Per-store tracking
        self.store_tp = defaultdict(int)
        self.store_fp = defaultdict(int)
        self.store_fn = defaultdict(int)
        
        # Criticality tracking
        self.critical_correct = 0
        self.critical_total = 0
        self.optional_correct = 0
        self.optional_total = 0
        
        # Efficiency tracking
        self.total_k_predicted = 0
        self.total_k_oracle = 0
        self.over_retrieved = 0
        self.under_retrieved = 0
        
        # Cold start tracking
        self.turn_results = defaultdict(list)
        
        # Redundancy tracking
        self.redundant_queries = 0
        self.redundant_optimal = 0
    
    def add_prediction(
        self,
        predicted_stores: Set[str],
        ground_truth_stores: Set[str],
        predicted_k: int,
        oracle_k: int,
        query: Dict,
        latency_ms: float = 0.0,
        turn_number: int = 0,
        redundant_stores: Optional[Set[str]] = None
    ):
        """Add a single prediction for metric calculation."""
        
        self.predictions.append(predicted_stores)
        self.ground_truths.append(ground_truth_stores)
        self.queries.append(query)
        self.latencies.append(latency_ms)
        
        # Store-level metrics
        for store in ["stm", "summary", "ltm", "episodic"]:
            pred_has = store in predicted_stores
            gt_has = store in ground_truth_stores
            
            if pred_has and gt_has:
                self.store_tp[store] += 1
            elif pred_has and not gt_has:
                self.store_fp[store] += 1
            elif not pred_has and gt_has:
                self.store_fn[store] += 1
        
        # Criticality tracking
        criticality = query.get("criticality", "MEDIUM")
        correct = ground_truth_stores.issubset(predicted_stores) if ground_truth_stores else len(predicted_stores) == 0
        
        if criticality == "HIGH":
            self.critical_total += 1
            if correct:
                self.critical_correct += 1
        else:
            self.optional_total += 1
            if correct:
                self.optional_correct += 1
        
        # Efficiency tracking
        self.total_k_predicted += predicted_k
        self.total_k_oracle += oracle_k
        
        extra_stores = predicted_stores - ground_truth_stores
        missing_stores = ground_truth_stores - predicted_stores
        
        if extra_stores:
            self.over_retrieved += 1
        if missing_stores:
            self.under_retrieved += 1
        
        # Turn tracking (for cold start)
        self.turn_results[turn_number].append(correct)
        
        # Redundancy tracking
        if redundant_stores and len(redundant_stores) > 1:
            self.redundant_queries += 1
            # Check if we picked the cheapest store
            predicted_in_redundant = predicted_stores & redundant_stores
            if predicted_in_redundant:
                cheapest = min(predicted_in_redundant, key=lambda s: self.STORE_COSTS.get(s, 10))
                cheapest_overall = min(redundant_stores, key=lambda s: self.STORE_COSTS.get(s, 10))
                if cheapest == cheapest_overall:
                    self.redundant_optimal += 1
    
    def compute_metrics(self) -> RoutingMetrics:
        """Compute all metrics from accumulated predictions."""
        
        n = len(self.predictions)
        if n == 0:
            return RoutingMetrics()
        
        metrics = RoutingMetrics()
        
        # === Accuracy Metrics ===
        exact_matches = sum(
            1 for pred, gt in zip(self.predictions, self.ground_truths)
            if pred == gt
        )
        metrics.routing_accuracy = exact_matches / n
        
        # Store-level precision/recall
        total_tp = sum(self.store_tp.values())
        total_fp = sum(self.store_fp.values())
        total_fn = sum(self.store_fn.values())
        
        metrics.store_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        metrics.store_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        metrics.store_f1 = (
            2 * metrics.store_precision * metrics.store_recall / 
            (metrics.store_precision + metrics.store_recall)
            if (metrics.store_precision + metrics.store_recall) > 0 else 0
        )
        
        # Per-store metrics
        for store in ["stm", "summary", "ltm", "episodic"]:
            tp, fp, fn = self.store_tp[store], self.store_fp[store], self.store_fn[store]
            metrics.per_store_precision[store] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics.per_store_recall[store] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # === Efficiency Metrics ===
        metrics.retrieval_efficiency = self.total_k_oracle / self.total_k_predicted if self.total_k_predicted > 0 else 0
        metrics.over_retrieval_rate = self.over_retrieved / n
        metrics.under_retrieval_rate = self.under_retrieved / n
        metrics.cost_ratio = self.total_k_predicted / (n * 10)
        
        # === Coverage Metrics ===
        metrics.critical_recall = self.critical_correct / self.critical_total if self.critical_total > 0 else 0
        metrics.optional_recall = self.optional_correct / self.optional_total if self.optional_total > 0 else 0
        
        # === Robustness Metrics ===
        cold_results = []
        warm_results = []
        for turn, results in self.turn_results.items():
            if turn < 10:
                cold_results.extend(results)
            else:
                warm_results.extend(results)
        
        metrics.cold_start_accuracy = np.mean(cold_results) if cold_results else 0
        metrics.warm_accuracy = np.mean(warm_results) if warm_results else 0
        
        # === Operational Metrics ===
        if self.latencies:
            metrics.avg_latency_ms = np.mean(self.latencies)
            metrics.p99_latency_ms = np.percentile(self.latencies, 99)
        
        store_counts = defaultdict(int)
        for pred in self.predictions:
            for store in pred:
                store_counts[store] += 1
        for store in ["stm", "summary", "ltm", "episodic"]:
            metrics.store_utilization[store] = store_counts[store] / n
        
        metrics.redundancy_exploitation = (
            self.redundant_optimal / self.redundant_queries 
            if self.redundant_queries > 0 else 0
        )
        
        # === Composite Scores ===
        metrics.cost_adjusted_accuracy = metrics.routing_accuracy * metrics.retrieval_efficiency
        
        if self.critical_total + self.optional_total > 0:
            weighted_correct = 2 * self.critical_correct + self.optional_correct
            weighted_total = 2 * self.critical_total + self.optional_total
            metrics.criticality_weighted_accuracy = weighted_correct / weighted_total
        
        return metrics
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for JSON serialization."""
        metrics = self.compute_metrics()
        return {
            "accuracy": {
                "routing_accuracy": metrics.routing_accuracy,
                "store_precision": metrics.store_precision,
                "store_recall": metrics.store_recall,
                "store_f1": metrics.store_f1,
            },
            "efficiency": {
                "retrieval_efficiency": metrics.retrieval_efficiency,
                "over_retrieval_rate": metrics.over_retrieval_rate,
                "under_retrieval_rate": metrics.under_retrieval_rate,
                "cost_ratio": metrics.cost_ratio,
            },
            "coverage": {
                "critical_recall": metrics.critical_recall,
                "optional_recall": metrics.optional_recall,
            },
            "robustness": {
                "cold_start_accuracy": metrics.cold_start_accuracy,
                "warm_accuracy": metrics.warm_accuracy,
            },
            "operational": {
                "avg_latency_ms": metrics.avg_latency_ms,
                "store_utilization": dict(metrics.store_utilization),
                "redundancy_exploitation": metrics.redundancy_exploitation,
            },
            "composite": {
                "cost_adjusted_accuracy": metrics.cost_adjusted_accuracy,
                "criticality_weighted_accuracy": metrics.criticality_weighted_accuracy,
            },
        }


def print_metrics_report(metrics: RoutingMetrics, name: str = "Policy"):
    """Print a formatted metrics report."""
    
    print(f"\n{'='*60}")
    print(f"METRICS REPORT: {name}")
    print(f"{'='*60}")
    
    print("\n📊 ACCURACY METRICS")
    print(f"  Routing Accuracy:     {metrics.routing_accuracy:.3f}")
    print(f"  Store Precision:      {metrics.store_precision:.3f}")
    print(f"  Store Recall:         {metrics.store_recall:.3f}")
    print(f"  Store F1:             {metrics.store_f1:.3f}")
    
    print("\n⚡ EFFICIENCY METRICS")
    print(f"  Retrieval Efficiency: {metrics.retrieval_efficiency:.3f}")
    print(f"  Over-retrieval Rate:  {metrics.over_retrieval_rate:.3f}")
    print(f"  Under-retrieval Rate: {metrics.under_retrieval_rate:.3f}")
    
    print("\n🎯 COVERAGE METRICS")
    print(f"  Critical Recall:      {metrics.critical_recall:.3f}")
    print(f"  Optional Recall:      {metrics.optional_recall:.3f}")
    
    print("\n🛡️ ROBUSTNESS METRICS")
    print(f"  Cold Start Accuracy:  {metrics.cold_start_accuracy:.3f}")
    print(f"  Warm Accuracy:        {metrics.warm_accuracy:.3f}")
    
    print("\n🏆 COMPOSITE SCORES")
    print(f"  Cost-Adjusted Acc:    {metrics.cost_adjusted_accuracy:.3f}")
    print(f"  Criticality-Weighted: {metrics.criticality_weighted_accuracy:.3f}")
