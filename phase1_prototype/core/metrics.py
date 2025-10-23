"""
Metrics Collector for VigilAI
Handles performance metrics and monitoring
"""

import logging
import time
from typing import Dict, List, Optional
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages performance metrics for VigilAI"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 readings
        self.start_time = time.time()
        
        # Performance counters
        self.frame_count = 0
        self.detection_count = 0
        self.intervention_count = 0
        self.error_count = 0
        
        # Timing metrics
        self.processing_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.intervention_times = deque(maxlen=100)
        
        # Accuracy metrics
        self.fatigue_detections = 0
        self.stress_detections = 0
        self.false_positives = 0
        self.true_positives = 0
        
        # System metrics
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.fps_history = deque(maxlen=100)
        
    def update(self, detection_result: Dict):
        """Update metrics with new detection result"""
        try:
            current_time = time.time()
            
            # Extract metrics from detection result
            metrics = {
                "timestamp": current_time,
                "fatigue_score": detection_result.get("fatigue_score", 0.0),
                "stress_score": detection_result.get("stress_score", 0.0),
                "combined_score": detection_result.get("combined_score", 0.0),
                "confidence": detection_result.get("confidence", 0.0),
                "processing_time": detection_result.get("processing_time", 0.0),
                "frame_count": self.frame_count,
                "detection_count": self.detection_count
            }
            
            # Add system metrics
            metrics.update(self._get_system_metrics())
            
            # Store metrics
            self.metrics_history.append(metrics)
            
            # Update counters
            self.detection_count += 1
            self.processing_times.append(metrics["processing_time"])
            
            # Update accuracy metrics
            self._update_accuracy_metrics(detection_result)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            self.error_count += 1
    
    def _get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            self.cpu_usage.append(cpu_percent)
            self.memory_usage.append(memory.percent)
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available,
                "memory_total": memory.total
            }
            
        except ImportError:
            # psutil not available, return default values
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "memory_available": 0,
                "memory_total": 0
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "memory_available": 0,
                "memory_total": 0
            }
    
    def _update_accuracy_metrics(self, detection_result: Dict):
        """Update accuracy-related metrics"""
        fatigue_score = detection_result.get("fatigue_score", 0.0)
        stress_score = detection_result.get("stress_score", 0.0)
        
        # Count detections
        if fatigue_score > 0.5:
            self.fatigue_detections += 1
        if stress_score > 0.5:
            self.stress_detections += 1
        
        # In a real implementation, you would compare against ground truth
        # For now, we'll use heuristics
        if fatigue_score > 0.8 or stress_score > 0.8:
            # Assume high scores are true positives
            self.true_positives += 1
        elif fatigue_score > 0.3 or stress_score > 0.3:
            # Medium scores might be false positives
            self.false_positives += 1
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate averages
        avg_processing_time = self._calculate_average(self.processing_times)
        avg_cpu_usage = self._calculate_average(self.cpu_usage)
        avg_memory_usage = self._calculate_average(self.memory_usage)
        
        # Calculate accuracy metrics
        total_detections = self.fatigue_detections + self.stress_detections
        accuracy = self._calculate_accuracy()
        
        # Calculate FPS
        fps = self._calculate_fps()
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_uptime(uptime),
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "intervention_count": self.intervention_count,
            "error_count": self.error_count,
            "avg_processing_time": avg_processing_time,
            "avg_cpu_usage": avg_cpu_usage,
            "avg_memory_usage": avg_memory_usage,
            "current_fps": fps,
            "fatigue_detections": self.fatigue_detections,
            "stress_detections": self.stress_detections,
            "accuracy": accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "total_detections": total_detections
        }
    
    def get_latest(self) -> Optional[Dict]:
        """Get latest metrics"""
        if not self.metrics_history:
            return None
        
        return self.metrics_history[-1]
    
    def get_timestamp(self) -> float:
        """Get current timestamp"""
        return time.time()
    
    def _calculate_average(self, values: deque) -> float:
        """Calculate average of deque values"""
        if not values:
            return 0.0
        
        try:
            return sum(values) / len(values)
        except:
            return 0.0
    
    def _calculate_accuracy(self) -> float:
        """Calculate detection accuracy"""
        total_detections = self.true_positives + self.false_positives
        if total_detections == 0:
            return 0.0
        
        return self.true_positives / total_detections
    
    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        if not self.fps_history:
            return 0.0
        
        try:
            return sum(self.fps_history) / len(self.fps_history)
        except:
            return 0.0
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as human-readable string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_performance_report(self) -> Dict:
        """Get detailed performance report"""
        summary = self.get_summary()
        
        # Add performance analysis
        performance_analysis = {
            "processing_time_stats": self._get_stats(self.processing_times),
            "cpu_usage_stats": self._get_stats(self.cpu_usage),
            "memory_usage_stats": self._get_stats(self.memory_usage),
            "fps_stats": self._get_stats(self.fps_history)
        }
        
        return {
            **summary,
            "performance_analysis": performance_analysis
        }
    
    def _get_stats(self, values: deque) -> Dict:
        """Get statistical summary of values"""
        if not values:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        
        try:
            values_list = list(values)
            return {
                "count": len(values_list),
                "mean": statistics.mean(values_list),
                "min": min(values_list),
                "max": max(values_list),
                "std": statistics.stdev(values_list) if len(values_list) > 1 else 0.0
            }
        except:
            return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    
    def record_intervention(self, intervention_type: str, severity: float):
        """Record intervention metrics"""
        self.intervention_count += 1
        self.intervention_times.append(time.time())
        
        logger.info(f"Intervention recorded: {intervention_type} (severity: {severity:.2f})")
    
    def record_error(self, error_type: str, error_message: str):
        """Record error metrics"""
        self.error_count += 1
        
        logger.error(f"Error recorded: {error_type} - {error_message}")
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics_history.clear()
        self.processing_times.clear()
        self.detection_times.clear()
        self.intervention_times.clear()
        self.cpu_usage.clear()
        self.memory_usage.clear()
        self.fps_history.clear()
        
        self.frame_count = 0
        self.detection_count = 0
        self.intervention_count = 0
        self.error_count = 0
        self.fatigue_detections = 0
        self.stress_detections = 0
        self.false_positives = 0
        self.true_positives = 0
        
        self.start_time = time.time()
        
        logger.info("Metrics reset")
    
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to file"""
        try:
            import json
            
            metrics_data = {
                "summary": self.get_summary(),
                "history": list(self.metrics_history),
                "performance_report": self.get_performance_report()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False
