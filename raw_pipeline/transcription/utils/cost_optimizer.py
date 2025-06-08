#!/usr/bin/env python3
"""
Cost Optimizer

Tracks transcription costs, analyzes usage patterns, and optimizes
method selection for cost efficiency while maintaining performance.
"""

import time
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

from utils.logging_setup import setup_logger

logger = setup_logger("cost_optimizer", "cost_optimizer.log")

@dataclass
class TranscriptionRecord:
    """Record of a transcription operation"""
    timestamp: float
    method: str
    duration_seconds: float
    processing_time_seconds: float
    estimated_cost: float
    actual_cost: Optional[float] = None
    file_size_mb: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None

class CostOptimizer:
    """Optimizes transcription costs through intelligent method selection"""
    
    def __init__(self):
        self.records: List[TranscriptionRecord] = []
        self.cost_models = self._initialize_cost_models()
        self.performance_models = self._initialize_performance_models()
        self.monthly_budget = float(os.getenv("MONTHLY_TRANSCRIPTION_BUDGET", "1000.0"))
        self.cost_threshold_per_hour = float(os.getenv("COST_THRESHOLD_PER_HOUR", "0.10"))
        
        # Load existing records
        self._load_records()
        
        logger.info("CostOptimizer initialized")
    
    def _initialize_cost_models(self) -> Dict:
        """Initialize cost models for different transcription methods"""
        
        return {
            "deepgram": {
                "per_minute": 0.0045,  # $0.0045 per minute
                "setup_cost": 0.0,
                "description": "Deepgram API pricing"
            },
            "parakeet_gpu": {
                "per_gpu_hour": 0.45,  # $0.45 per GPU hour
                "processing_speed_ratio": 40.0,  # 40x real-time
                "setup_cost": 0.02,  # Model loading overhead
                "description": "Cloud Run GPU L4 pricing"
            },
            "parakeet_cpu": {
                "per_cpu_hour": 0.10,  # $0.10 per CPU hour
                "processing_speed_ratio": 2.0,  # 2x real-time
                "setup_cost": 0.01,  # Model loading overhead
                "description": "Cloud Run CPU pricing"
            },
            "hybrid": {
                "split_threshold_hours": 2.5,
                "description": "Hybrid Parakeet + Deepgram"
            }
        }
    
    def _initialize_performance_models(self) -> Dict:
        """Initialize performance models for method selection"""
        
        return {
            "deepgram": {
                "processing_time_seconds": 10.0,  # ~10 seconds regardless
                "reliability_score": 0.99,
                "quality_score": 0.95
            },
            "parakeet_gpu": {
                "processing_speed_ratio": 40.0,
                "reliability_score": 0.95,
                "quality_score": 0.90
            },
            "parakeet_cpu": {
                "processing_speed_ratio": 2.0,
                "reliability_score": 0.90,
                "quality_score": 0.90
            }
        }
    
    def calculate_transcription_cost(self, duration_seconds: float, method: str) -> float:
        """Calculate estimated cost for transcription"""
        
        duration_minutes = duration_seconds / 60.0
        duration_hours = duration_seconds / 3600.0
        
        if method == "deepgram":
            return duration_minutes * self.cost_models["deepgram"]["per_minute"]
        
        elif method == "parakeet_gpu":
            processing_time_hours = duration_hours / self.cost_models["parakeet_gpu"]["processing_speed_ratio"]
            gpu_cost = processing_time_hours * self.cost_models["parakeet_gpu"]["per_gpu_hour"]
            setup_cost = self.cost_models["parakeet_gpu"]["setup_cost"]
            return gpu_cost + setup_cost
        
        elif method == "parakeet_cpu":
            processing_time_hours = duration_hours / self.cost_models["parakeet_cpu"]["processing_speed_ratio"]
            cpu_cost = processing_time_hours * self.cost_models["parakeet_cpu"]["per_cpu_hour"]
            setup_cost = self.cost_models["parakeet_cpu"]["setup_cost"]
            return cpu_cost + setup_cost
        
        elif method == "hybrid":
            return self._calculate_hybrid_cost(duration_seconds)
        
        else:
            logger.warning(f"Unknown method for cost calculation: {method}")
            return 0.0
    
    def _calculate_hybrid_cost(self, duration_seconds: float) -> float:
        """Calculate cost for hybrid processing"""
        
        split_threshold_seconds = self.cost_models["hybrid"]["split_threshold_hours"] * 3600
        
        if duration_seconds <= split_threshold_seconds:
            # Use Parakeet GPU for entire file
            return self.calculate_transcription_cost(duration_seconds, "parakeet_gpu")
        else:
            # Hybrid: Parakeet for first part, Deepgram for remainder
            parakeet_cost = self.calculate_transcription_cost(split_threshold_seconds, "parakeet_gpu")
            deepgram_duration = duration_seconds - split_threshold_seconds
            deepgram_cost = self.calculate_transcription_cost(deepgram_duration, "deepgram")
            return parakeet_cost + deepgram_cost
    
    def get_optimal_method(self, duration_hours: float, gpu_available: bool, 
                          file_size_mb: float = 0) -> str:
        """Get optimal transcription method based on cost and performance"""
        
        duration_seconds = duration_hours * 3600
        
        # Available methods based on resources
        available_methods = ["deepgram"]
        if gpu_available:
            available_methods.extend(["parakeet_gpu", "hybrid"])
        else:
            available_methods.append("parakeet_cpu")
        
        # Calculate cost and performance for each method
        method_scores = {}
        
        for method in available_methods:
            cost = self.calculate_transcription_cost(duration_seconds, method)
            processing_time = self._estimate_processing_time(duration_seconds, method)
            
            # Combined score: cost + time penalty + reliability factor
            reliability = self.performance_models.get(method, {}).get("reliability_score", 0.9)
            time_penalty = processing_time * 0.001  # $0.001 per second time penalty
            reliability_bonus = (1 - reliability) * cost  # Penalty for unreliability
            
            score = cost + time_penalty + reliability_bonus
            
            method_scores[method] = {
                "cost": cost,
                "processing_time": processing_time,
                "score": score,
                "reliability": reliability
            }
        
        # Select method with lowest score
        optimal_method = min(method_scores.keys(), key=lambda k: method_scores[k]["score"])
        
        # Log decision
        logger.info(f"Cost optimization for {duration_hours:.2f}h audio:")
        for method, metrics in method_scores.items():
            logger.info(f"  {method}: ${metrics['cost']:.3f}, {metrics['processing_time']:.1f}s, score: {metrics['score']:.3f}")
        logger.info(f"  Selected: {optimal_method}")
        
        return optimal_method
    
    def _estimate_processing_time(self, duration_seconds: float, method: str) -> float:
        """Estimate processing time for given method"""
        
        if method == "deepgram":
            return self.performance_models["deepgram"]["processing_time_seconds"]
        elif method == "parakeet_gpu":
            return duration_seconds / self.performance_models["parakeet_gpu"]["processing_speed_ratio"]
        elif method == "parakeet_cpu":
            return duration_seconds / self.performance_models["parakeet_cpu"]["processing_speed_ratio"]
        elif method == "hybrid":
            return self._estimate_hybrid_processing_time(duration_seconds)
        else:
            return duration_seconds  # 1x real-time fallback
    
    def _estimate_hybrid_processing_time(self, duration_seconds: float) -> float:
        """Estimate processing time for hybrid method"""
        
        split_threshold_seconds = self.cost_models["hybrid"]["split_threshold_hours"] * 3600
        
        if duration_seconds <= split_threshold_seconds:
            return self._estimate_processing_time(duration_seconds, "parakeet_gpu")
        else:
            # Parallel processing: max of both parts
            parakeet_time = self._estimate_processing_time(split_threshold_seconds, "parakeet_gpu")
            deepgram_time = self._estimate_processing_time(
                duration_seconds - split_threshold_seconds, "deepgram"
            )
            return max(parakeet_time, deepgram_time)
    
    def record_transcription(self, method: str, duration_seconds: float, 
                           processing_time_seconds: float, estimated_cost: float,
                           actual_cost: Optional[float] = None, 
                           file_size_mb: Optional[float] = None,
                           success: bool = True, error_message: Optional[str] = None):
        """Record a transcription operation for analysis"""
        
        record = TranscriptionRecord(
            timestamp=time.time(),
            method=method,
            duration_seconds=duration_seconds,
            processing_time_seconds=processing_time_seconds,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            file_size_mb=file_size_mb,
            success=success,
            error_message=error_message
        )
        
        self.records.append(record)
        
        # Save records periodically
        if len(self.records) % 10 == 0:
            self._save_records()
        
        logger.info(f"Recorded transcription: {method}, {duration_seconds/60:.1f}min, ${estimated_cost:.3f}")
    
    def get_cost_analysis(self, days: int = 30) -> Dict:
        """Get cost analysis for specified period"""
        
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_records = [r for r in self.records if r.timestamp >= cutoff_time]
        
        if not recent_records:
            return {"error": "No records found for specified period"}
        
        # Aggregate by method
        method_stats = defaultdict(lambda: {
            "count": 0,
            "total_duration_hours": 0,
            "total_cost": 0,
            "total_processing_time": 0,
            "success_rate": 0
        })
        
        total_cost = 0
        total_duration_hours = 0
        
        for record in recent_records:
            method = record.method
            duration_hours = record.duration_seconds / 3600
            cost = record.actual_cost or record.estimated_cost
            
            method_stats[method]["count"] += 1
            method_stats[method]["total_duration_hours"] += duration_hours
            method_stats[method]["total_cost"] += cost
            method_stats[method]["total_processing_time"] += record.processing_time_seconds
            method_stats[method]["success_rate"] += 1 if record.success else 0
            
            total_cost += cost
            total_duration_hours += duration_hours
        
        # Calculate averages and rates
        for method, stats in method_stats.items():
            if stats["count"] > 0:
                stats["avg_cost_per_hour"] = stats["total_cost"] / max(stats["total_duration_hours"], 0.001)
                stats["avg_processing_time"] = stats["total_processing_time"] / stats["count"]
                stats["success_rate"] = stats["success_rate"] / stats["count"]
        
        # Calculate savings vs Deepgram baseline
        deepgram_baseline_cost = total_duration_hours * 60 * 0.0045  # $0.0045 per minute
        savings = deepgram_baseline_cost - total_cost
        savings_percentage = (savings / deepgram_baseline_cost) * 100 if deepgram_baseline_cost > 0 else 0
        
        return {
            "period_days": days,
            "total_transcriptions": len(recent_records),
            "total_duration_hours": total_duration_hours,
            "total_cost": total_cost,
            "avg_cost_per_hour": total_cost / max(total_duration_hours, 0.001),
            "deepgram_baseline_cost": deepgram_baseline_cost,
            "total_savings": savings,
            "savings_percentage": savings_percentage,
            "method_breakdown": dict(method_stats),
            "monthly_projection": total_cost * (30 / days) if days > 0 else 0,
            "budget_utilization": (total_cost * (30 / days)) / self.monthly_budget if days > 0 else 0
        }
    
    def get_recommendations(self) -> List[str]:
        """Get cost optimization recommendations"""
        
        recommendations = []
        analysis = self.get_cost_analysis(30)
        
        if "error" in analysis:
            return ["Insufficient data for recommendations"]
        
        # Budget utilization
        budget_util = analysis.get("budget_utilization", 0)
        if budget_util > 0.8:
            recommendations.append(f"High budget utilization ({budget_util:.1%}). Consider more aggressive cost optimization.")
        
        # Method efficiency
        method_breakdown = analysis.get("method_breakdown", {})
        if "deepgram" in method_breakdown and "parakeet_gpu" in method_breakdown:
            deepgram_cost_per_hour = method_breakdown["deepgram"]["avg_cost_per_hour"]
            parakeet_cost_per_hour = method_breakdown["parakeet_gpu"]["avg_cost_per_hour"]
            
            if deepgram_cost_per_hour > parakeet_cost_per_hour * 2:
                recommendations.append("Consider using Parakeet GPU for more files to increase savings.")
        
        # Success rates
        for method, stats in method_breakdown.items():
            if stats["success_rate"] < 0.95:
                recommendations.append(f"Low success rate for {method} ({stats['success_rate']:.1%}). Review error patterns.")
        
        # Savings potential
        savings_pct = analysis.get("savings_percentage", 0)
        if savings_pct < 80:
            recommendations.append(f"Current savings: {savings_pct:.1%}. Target 90%+ savings with more GPU usage.")
        
        return recommendations if recommendations else ["Cost optimization is performing well."]
    
    def _save_records(self):
        """Save transcription records to file"""
        
        try:
            records_data = [asdict(record) for record in self.records]
            
            os.makedirs("output/cost_tracking", exist_ok=True)
            with open("output/cost_tracking/transcription_records.json", "w") as f:
                json.dump(records_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save cost records: {e}")
    
    def _load_records(self):
        """Load existing transcription records"""
        
        try:
            records_file = "output/cost_tracking/transcription_records.json"
            if os.path.exists(records_file):
                with open(records_file, "r") as f:
                    records_data = json.load(f)
                
                self.records = [TranscriptionRecord(**record) for record in records_data]
                logger.info(f"Loaded {len(self.records)} existing cost records")
                
        except Exception as e:
            logger.warning(f"Failed to load existing cost records: {e}")
            self.records = []
    
    def export_cost_report(self, days: int = 30) -> str:
        """Export detailed cost report"""
        
        analysis = self.get_cost_analysis(days)
        recommendations = self.get_recommendations()
        
        report = {
            "report_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_period_days": days,
            "cost_analysis": analysis,
            "recommendations": recommendations,
            "cost_models": self.cost_models,
            "performance_models": self.performance_models
        }
        
        # Save report
        os.makedirs("output/cost_tracking", exist_ok=True)
        report_file = f"output/cost_tracking/cost_report_{int(time.time())}.json"
        
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Cost report exported to: {report_file}")
        return report_file
