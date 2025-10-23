"""
Basic tests for VigilAI Phase 1 Prototype
Tests core functionality without heavy dependencies
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_config_management():
    """Test configuration management"""
    print("Testing configuration management...")
    
    try:
        from phase1_prototype.utils.config import Config
        
        # Test configuration loading
        config = Config()
        assert config is not None
        
        # Test getting camera config
        camera_config = config.get_camera_config()
        assert "device_id" in camera_config
        assert "width" in camera_config
        assert "height" in camera_config
        
        # Test getting detection config
        detection_config = config.get_detection_config()
        assert "fatigue_threshold" in detection_config
        assert "stress_threshold" in detection_config
        
        # Test configuration validation
        is_valid = config.validate_config()
        assert is_valid is True
        
        print("✅ Configuration management test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration management test failed: {e}")
        return False

def test_data_collector():
    """Test data collector functionality"""
    print("Testing data collector...")
    
    try:
        from phase1_prototype.core.data_collector import DataCollector
        
        # Test initialization
        data_collector = DataCollector()
        assert data_collector is not None
        assert data_collector.total_records == 0
        
        # Test data storage
        test_data = {
            "video": {"frame": None},
            "steering": {"angle": 10.0},
            "biometric": {"heart_rate": 70.0},
            "timestamp": time.time()
        }
        
        data_collector.store_data(test_data)
        assert data_collector.total_records == 1
        assert len(data_collector.data_buffer) == 1
        
        # Test statistics
        stats = data_collector.get_statistics()
        assert "total_records" in stats
        assert "buffer_size" in stats
        assert stats["total_records"] == 1
        
        print("✅ Data collector test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data collector test failed: {e}")
        return False

def test_metrics_collector():
    """Test metrics collector functionality"""
    print("Testing metrics collector...")
    
    try:
        from phase1_prototype.core.metrics import MetricsCollector
        
        # Test initialization
        metrics_collector = MetricsCollector()
        assert metrics_collector is not None
        assert metrics_collector.detection_count == 0
        
        # Test metrics update
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
        
        # Test summary
        summary = metrics_collector.get_summary()
        assert "uptime_seconds" in summary
        assert "detection_count" in summary
        assert summary["detection_count"] == 1
        
        print("✅ Metrics collector test passed")
        return True
        
    except Exception as e:
        print(f"❌ Metrics collector test failed: {e}")
        return False

def test_intervention_system():
    """Test intervention system functionality"""
    print("Testing intervention system...")
    
    try:
        from phase1_prototype.core.intervention_system import InterventionSystem
        
        # Test initialization
        config = {
            "intervention_types": ["audio", "haptic", "visual"],
            "escalation_levels": 3,
            "cooldown_period": 1,
            "audio_enabled": False,  # Disable audio for testing
            "haptic_enabled": True,
            "visual_enabled": True
        }
        
        intervention_system = InterventionSystem(config)
        assert intervention_system is not None
        assert len(intervention_system.active_interventions) == 0
        
        # Test statistics
        stats = intervention_system.get_intervention_statistics()
        assert "total_interventions" in stats
        assert "fatigue_interventions" in stats
        assert "stress_interventions" in stats
        
        print("✅ Intervention system test passed")
        return True
        
    except Exception as e:
        print(f"❌ Intervention system test failed: {e}")
        return False

def test_logging_setup():
    """Test logging setup"""
    print("Testing logging setup...")
    
    try:
        from phase1_prototype.utils.logger import setup_logging
        
        # Test logger creation
        logger = setup_logging("test_module")
        assert logger is not None
        assert logger.name == "test_module"
        
        # Test logging
        logger.info("Test log message")
        
        print("✅ Logging setup test passed")
        return True
        
    except Exception as e:
        print(f"❌ Logging setup test failed: {e}")
        return False

def run_all_tests():
    """Run all basic tests"""
    print("🚀 Running VigilAI Phase 1 Basic Tests...\n")
    
    tests = [
        test_config_management,
        test_data_collector,
        test_metrics_collector,
        test_intervention_system,
        test_logging_setup
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
            failed += 1
        print()  # Add spacing between tests
    
    print(f"📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All basic tests passed! VigilAI Phase 1 core components are working.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
