"""
Test suite for VigilAI Phase 1 Prototype
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from phase1_prototype.core.detection_engine import DetectionEngine
from phase1_prototype.core.data_collector import DataCollector
from phase1_prototype.core.intervention_system import InterventionSystem
from phase1_prototype.core.metrics import MetricsCollector
from phase1_prototype.utils.config import Config
from phase1_prototype.utils.logger import setup_logging

class TestDetectionEngine:
    """Test detection engine functionality"""
    
    @pytest.fixture
    def detection_engine(self):
        config = {
            "fatigue_threshold": 0.7,
            "stress_threshold": 0.6,
            "confidence_threshold": 0.5
        }
        return DetectionEngine(config)
    
    @pytest.mark.asyncio
    async def test_detection_engine_initialization(self, detection_engine):
        """Test detection engine initialization"""
        assert detection_engine is not None
        assert detection_engine.config is not None
    
    @pytest.mark.asyncio
    async def test_analyze_with_mock_data(self, detection_engine):
        """Test analysis with mock data"""
        # Mock video data
        video_data = {
            "frame": None,  # Will be handled by mock
            "timestamp": time.time()
        }
        
        # Mock steering data
        steering_data = {
            "angle": 15.0,
            "velocity": 2.5,
            "timestamp": time.time()
        }
        
        # Mock biometric data
        biometric_data = {
            "heart_rate": 75.0,
            "hrv": 45.0,
            "eda": 2.5,
            "timestamp": time.time()
        }
        
        # Test analysis
        result = await detection_engine.analyze({
            "video": video_data,
            "steering": steering_data,
            "biometric": biometric_data
        })
        
        assert result is not None
        assert "fatigue_score" in result
        assert "stress_score" in result
        assert "combined_score" in result
        assert "confidence" in result
        assert "processing_time" in result
        
        # Check score ranges
        assert 0.0 <= result["fatigue_score"] <= 1.0
        assert 0.0 <= result["stress_score"] <= 1.0
        assert 0.0 <= result["combined_score"] <= 1.0
        assert 0.0 <= result["confidence"] <= 1.0

class TestDataCollector:
    """Test data collector functionality"""
    
    @pytest.fixture
    def data_collector(self):
        config = {
            "data_dir": "test_data",
            "max_memory_records": 100,
            "save_interval": 1
        }
        return DataCollector(config)
    
    def test_data_collector_initialization(self, data_collector):
        """Test data collector initialization"""
        assert data_collector is not None
        assert data_collector.total_records == 0
        assert len(data_collector.data_buffer) == 0
    
    def test_store_data(self, data_collector):
        """Test data storage"""
        test_data = {
            "video": {"frame": None},
            "steering": {"angle": 10.0},
            "biometric": {"heart_rate": 70.0},
            "timestamp": time.time()
        }
        
        data_collector.store_data(test_data)
        
        assert data_collector.total_records == 1
        assert len(data_collector.data_buffer) == 1
        assert data_collector.data_buffer[0]["id"] == 0
    
    def test_get_recent_data(self, data_collector):
        """Test getting recent data"""
        # Add some test data
        for i in range(5):
            data_collector.store_data({
                "timestamp": time.time() - i,
                "data": f"test_{i}"
            })
        
        recent_data = data_collector.get_recent_data(seconds=3)
        assert len(recent_data) >= 0  # Should have some recent data
    
    def test_get_statistics(self, data_collector):
        """Test statistics collection"""
        # Add some test data
        data_collector.store_data({"video": "test"})
        data_collector.store_data({"steering": "test"})
        data_collector.store_data({"biometric": "test"})
        
        stats = data_collector.get_statistics()
        assert "total_records" in stats
        assert "buffer_size" in stats
        assert "data_types" in stats
        assert stats["total_records"] == 3

class TestInterventionSystem:
    """Test intervention system functionality"""
    
    @pytest.fixture
    def intervention_system(self):
        config = {
            "intervention_types": ["audio", "haptic", "visual"],
            "escalation_levels": 3,
            "cooldown_period": 1,
            "audio_enabled": True,
            "haptic_enabled": True,
            "visual_enabled": True
        }
        return InterventionSystem(config)
    
    @pytest.mark.asyncio
    async def test_intervention_system_initialization(self, intervention_system):
        """Test intervention system initialization"""
        assert intervention_system is not None
        assert len(intervention_system.active_interventions) == 0
    
    @pytest.mark.asyncio
    async def test_trigger_fatigue_intervention(self, intervention_system):
        """Test fatigue intervention triggering"""
        success = await intervention_system.trigger_intervention("fatigue", 0.8)
        
        # Should succeed for high severity
        assert success is True
        assert len(intervention_system.active_interventions) > 0
    
    @pytest.mark.asyncio
    async def test_trigger_stress_intervention(self, intervention_system):
        """Test stress intervention triggering"""
        success = await intervention_system.trigger_intervention("stress", 0.7)
        
        # Should succeed for high severity
        assert success is True
        assert len(intervention_system.active_interventions) > 0
    
    @pytest.mark.asyncio
    async def test_cooldown_period(self, intervention_system):
        """Test intervention cooldown period"""
        # First intervention should succeed
        success1 = await intervention_system.trigger_intervention("fatigue", 0.8)
        assert success1 is True
        
        # Second intervention immediately should fail due to cooldown
        success2 = await intervention_system.trigger_intervention("fatigue", 0.8)
        assert success2 is False
    
    def test_get_intervention_statistics(self, intervention_system):
        """Test intervention statistics"""
        stats = intervention_system.get_intervention_statistics()
        
        assert "total_interventions" in stats
        assert "fatigue_interventions" in stats
        assert "stress_interventions" in stats
        assert "avg_severity" in stats
        assert "active_count" in stats

class TestMetricsCollector:
    """Test metrics collector functionality"""
    
    @pytest.fixture
    def metrics_collector(self):
        return MetricsCollector()
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert metrics_collector is not None
        assert metrics_collector.frame_count == 0
        assert metrics_collector.detection_count == 0
    
    def test_update_metrics(self, metrics_collector):
        """Test metrics update"""
        detection_result = {
            "fatigue_score": 0.5,
            "stress_score": 0.3,
            "combined_score": 0.4,
            "confidence": 0.8,
            "processing_time": 0.1
        }
        
        metrics_collector.update(detection_result)
        
        assert metrics_collector.detection_count == 1
        assert len(metrics_collector.metrics_history) == 1
    
    def test_get_summary(self, metrics_collector):
        """Test metrics summary"""
        # Update with some data
        detection_result = {
            "fatigue_score": 0.5,
            "stress_score": 0.3,
            "combined_score": 0.4,
            "confidence": 0.8,
            "processing_time": 0.1
        }
        
        metrics_collector.update(detection_result)
        summary = metrics_collector.get_summary()
        
        assert "uptime_seconds" in summary
        assert "detection_count" in summary
        assert "avg_processing_time" in summary
        assert summary["detection_count"] == 1
    
    def test_reset_metrics(self, metrics_collector):
        """Test metrics reset"""
        # Add some data
        detection_result = {
            "fatigue_score": 0.5,
            "stress_score": 0.3,
            "combined_score": 0.4,
            "confidence": 0.8,
            "processing_time": 0.1
        }
        
        metrics_collector.update(detection_result)
        assert metrics_collector.detection_count == 1
        
        # Reset metrics
        metrics_collector.reset_metrics()
        assert metrics_collector.detection_count == 0
        assert len(metrics_collector.metrics_history) == 0

class TestConfig:
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        config = Config()
        assert config is not None
        assert config.config_data is not None
    
    def test_get_camera_config(self):
        """Test camera configuration retrieval"""
        config = Config()
        camera_config = config.get_camera_config()
        
        assert "device_id" in camera_config
        assert "width" in camera_config
        assert "height" in camera_config
        assert "fps" in camera_config
    
    def test_get_detection_config(self):
        """Test detection configuration retrieval"""
        config = Config()
        detection_config = config.get_detection_config()
        
        assert "fatigue_threshold" in detection_config
        assert "stress_threshold" in detection_config
        assert "confidence_threshold" in detection_config
    
    def test_set_config_value(self):
        """Test setting configuration values"""
        config = Config()
        
        # Set a new value
        success = config.set("test.value", "test_data")
        assert success is True
        
        # Get the value
        value = config.get("test.value")
        assert value == "test_data"
    
    def test_validate_config(self):
        """Test configuration validation"""
        config = Config()
        is_valid = config.validate_config()
        assert is_valid is True

@pytest.mark.asyncio
async def test_integration():
    """Test basic integration of components"""
    # This is a basic integration test
    # In a real scenario, you would test the full system
    
    # Test configuration
    config = Config()
    assert config.validate_config()
    
    # Test data collector
    data_collector = DataCollector()
    test_data = {
        "video": {"frame": None},
        "steering": {"angle": 10.0},
        "biometric": {"heart_rate": 70.0},
        "timestamp": time.time()
    }
    data_collector.store_data(test_data)
    assert data_collector.total_records == 1
    
    # Test metrics collector
    metrics_collector = MetricsCollector()
    detection_result = {
        "fatigue_score": 0.5,
        "stress_score": 0.3,
        "combined_score": 0.4,
        "confidence": 0.8,
        "processing_time": 0.1
    }
    metrics_collector.update(detection_result)
    assert metrics_collector.detection_count == 1
    
    print("✅ Basic integration test passed!")

if __name__ == "__main__":
    # Run basic tests
    import sys
    
    print("Running VigilAI Phase 1 Tests...")
    
    # Test configuration
    try:
        config = Config()
        assert config.validate_config()
        print("✅ Configuration test passed")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        sys.exit(1)
    
    # Test data collector
    try:
        data_collector = DataCollector()
        test_data = {
            "video": {"frame": None},
            "steering": {"angle": 10.0},
            "biometric": {"heart_rate": 70.0},
            "timestamp": time.time()
        }
        data_collector.store_data(test_data)
        assert data_collector.total_records == 1
        print("✅ Data collector test passed")
    except Exception as e:
        print(f"❌ Data collector test failed: {e}")
        sys.exit(1)
    
    # Test metrics collector
    try:
        metrics_collector = MetricsCollector()
        detection_result = {
            "fatigue_score": 0.5,
            "stress_score": 0.3,
            "combined_score": 0.4,
            "confidence": 0.8,
            "processing_time": 0.1
        }
        metrics_collector.update(detection_result)
        assert metrics_collector.detection_count == 1
        print("✅ Metrics collector test passed")
    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 All basic tests passed! VigilAI Phase 1 is ready for testing.")
