"""
Basic tests for VigilAI Phase 2 Core Development
Tests core functionality without heavy dependencies
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_video_fatigue_detector_structure():
    """Test video fatigue detector structure"""
    print("Testing video fatigue detector structure...")
    
    try:
        # Test basic structure without TensorFlow
        from phase2_core.models.video.fatigue_detector import VideoFatigueDetector
        
        config = {
            "input_shape": (64, 64, 3),
            "sequence_length": 30,
            "num_classes": 3
        }
        
        detector = VideoFatigueDetector(config)
        assert detector is not None
        assert detector.config == config
        assert detector.input_shape == (64, 64, 3)
        assert detector.sequence_length == 30
        assert detector.num_classes == 3
        
        print("✅ Video fatigue detector structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Video fatigue detector structure test failed: {e}")
        return False

def test_steering_analyzer_structure():
    """Test steering analyzer structure"""
    print("Testing steering analyzer structure...")
    
    try:
        from phase2_core.models.steering.steering_analyzer import SteeringAnalyzer
        
        config = {
            "sequence_length": 100,
            "feature_dim": 6,
            "num_classes": 3
        }
        
        analyzer = SteeringAnalyzer(config)
        assert analyzer is not None
        assert analyzer.config == config
        assert analyzer.sequence_length == 100
        assert analyzer.feature_dim == 6
        assert analyzer.num_classes == 3
        
        print("✅ Steering analyzer structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Steering analyzer structure test failed: {e}")
        return False

def test_biometric_stress_detector_structure():
    """Test biometric stress detector structure"""
    print("Testing biometric stress detector structure...")
    
    try:
        from phase2_core.models.biometric.stress_detector import BiometricStressDetector
        
        config = {
            "sequence_length": 60,
            "feature_dim": 8,
            "num_classes": 3
        }
        
        detector = BiometricStressDetector(config)
        assert detector is not None
        assert detector.config == config
        assert detector.sequence_length == 60
        assert detector.feature_dim == 8
        assert detector.num_classes == 3
        
        print("✅ Biometric stress detector structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Biometric stress detector structure test failed: {e}")
        return False

def test_multimodal_fusion_structure():
    """Test multi-modal fusion structure"""
    print("Testing multi-modal fusion structure...")
    
    try:
        from phase2_core.models.fusion.multimodal_fusion import MultiModalFusion
        
        config = {
            "video_dim": 128,
            "steering_dim": 64,
            "biometric_dim": 64,
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "num_classes": 3
        }
        
        fusion = MultiModalFusion(config)
        assert fusion is not None
        assert fusion.config == config
        assert fusion.video_dim == 128
        assert fusion.steering_dim == 64
        assert fusion.biometric_dim == 64
        assert fusion.hidden_dim == 256
        assert fusion.num_heads == 8
        assert fusion.num_layers == 6
        assert fusion.num_classes == 3
        
        print("✅ Multi-modal fusion structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Multi-modal fusion structure test failed: {e}")
        return False

def test_inference_engine_structure():
    """Test real-time inference engine structure"""
    print("Testing real-time inference engine structure...")
    
    try:
        from phase2_core.inference.real_time_inference import RealTimeInferenceEngine
        
        config = {
            "video": {"input_shape": (64, 64, 3), "sequence_length": 30},
            "steering": {"sequence_length": 100, "feature_dim": 6},
            "biometric": {"sequence_length": 60, "feature_dim": 8},
            "fusion": {"video_dim": 128, "steering_dim": 64, "biometric_dim": 64},
            "inference_interval": 0.1
        }
        
        engine = RealTimeInferenceEngine(config)
        assert engine is not None
        assert engine.config == config
        assert engine.video_detector is not None
        assert engine.steering_analyzer is not None
        assert engine.biometric_detector is not None
        assert engine.fusion_model is not None
        assert engine.inference_interval == 0.1
        
        print("✅ Real-time inference engine structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Real-time inference engine structure test failed: {e}")
        return False

def test_feature_extraction():
    """Test basic feature extraction without TensorFlow"""
    print("Testing feature extraction...")
    
    try:
        from phase2_core.models.steering.steering_analyzer import SteeringAnalyzer
        
        config = {"sequence_length": 100, "feature_dim": 6, "num_classes": 3}
        analyzer = SteeringAnalyzer(config)
        
        # Test steering feature extraction
        steering_data = {
            "angle": 15.0,
            "velocity": 2.5,
            "speed": 60.0,
            "timestamp": time.time()
        }
        
        features = analyzer.extract_steering_features(steering_data)
        assert len(features) == 6
        assert features[0] == 15.0  # angle
        assert features[1] == 2.5    # velocity
        assert features[2] == 0.0    # acceleration (first reading)
        assert features[3] == 0.0    # jerk (first reading)
        assert features[4] == 0.0    # entropy (no history)
        assert features[5] == 0.0    # variance (no history)
        
        print("✅ Feature extraction test passed")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_fusion_logic():
    """Test fusion logic without TensorFlow"""
    print("Testing fusion logic...")
    
    try:
        from phase2_core.models.fusion.multimodal_fusion import MultiModalFusion
        
        config = {
            "video_dim": 128,
            "steering_dim": 64,
            "biometric_dim": 64,
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "num_classes": 3
        }
        
        fusion = MultiModalFusion(config)
        
        # Test prediction fusion
        video_pred = {"fatigue_level": 1, "confidence": 0.8}
        steering_pred = {"fatigue_level": 0, "confidence": 0.6}
        biometric_pred = {"stress_level": 1, "confidence": 0.7}
        
        fused = fusion.fuse_predictions(video_pred, steering_pred, biometric_pred)
        
        assert "fatigue_level" in fused
        assert "confidence" in fused
        assert "weighted_fatigue_score" in fused
        assert "modality_contributions" in fused
        
        # Check value ranges
        assert 0 <= fused["fatigue_level"] <= 2
        assert 0.0 <= fused["confidence"] <= 1.0
        assert 0.0 <= fused["weighted_fatigue_score"] <= 2.0
        
        print("✅ Fusion logic test passed")
        return True
        
    except Exception as e:
        print(f"❌ Fusion logic test failed: {e}")
        return False

def test_inference_engine_buffers():
    """Test inference engine buffer management"""
    print("Testing inference engine buffers...")
    
    try:
        from phase2_core.inference.real_time_inference import RealTimeInferenceEngine
        
        config = {
            "video": {"input_shape": (64, 64, 3), "sequence_length": 30},
            "steering": {"sequence_length": 100, "feature_dim": 6},
            "biometric": {"sequence_length": 60, "feature_dim": 8},
            "fusion": {"video_dim": 128, "steering_dim": 64, "biometric_dim": 64},
            "inference_interval": 0.1
        }
        
        engine = RealTimeInferenceEngine(config)
        
        # Test buffer status
        status = engine.get_buffer_status()
        assert "video_buffer_size" in status
        assert "steering_buffer_size" in status
        assert "biometric_buffer_size" in status
        assert "has_enough_data" in status
        
        # Initially should be empty
        assert status["video_buffer_size"] == 0
        assert status["steering_buffer_size"] == 0
        assert status["biometric_buffer_size"] == 0
        assert not status["has_enough_data"]
        
        # Test buffer updates
        video_data = {"frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
        steering_data = {"angle": 15.0, "velocity": 2.5, "speed": 60.0}
        biometric_data = {"heart_rate": 75.0, "hrv": 45.0, "eda": 2.5}
        
        engine._update_buffers(video_data, steering_data, biometric_data)
        
        assert len(engine.video_buffer) == 1
        assert len(engine.steering_buffer) == 1
        assert len(engine.biometric_buffer) == 1
        
        # Test buffer clearing
        engine.clear_buffers()
        assert len(engine.video_buffer) == 0
        assert len(engine.steering_buffer) == 0
        assert len(engine.biometric_buffer) == 0
        
        print("✅ Inference engine buffers test passed")
        return True
        
    except Exception as e:
        print(f"❌ Inference engine buffers test failed: {e}")
        return False

def run_all_phase2_tests():
    """Run all Phase 2 basic tests"""
    print("🚀 Running VigilAI Phase 2 Basic Tests...\n")
    
    tests = [
        test_video_fatigue_detector_structure,
        test_steering_analyzer_structure,
        test_biometric_stress_detector_structure,
        test_multimodal_fusion_structure,
        test_inference_engine_structure,
        test_feature_extraction,
        test_fusion_logic,
        test_inference_engine_buffers
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
        print("\n🎉 All Phase 2 basic tests passed! VigilAI core components are working.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_phase2_tests()
    sys.exit(0 if success else 1)
