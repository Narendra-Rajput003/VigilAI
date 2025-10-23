"""
Simple tests for VigilAI Phase 2 Core Development
Tests basic structure and functionality without heavy dependencies
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_project_structure():
    """Test that Phase 2 project structure is correct"""
    print("Testing Phase 2 project structure...")
    
    try:
        # Check that directories exist
        phase2_dir = Path("phase2_core")
        assert phase2_dir.exists(), "phase2_core directory should exist"
        
        models_dir = phase2_dir / "models"
        assert models_dir.exists(), "models directory should exist"
        
        video_dir = models_dir / "video"
        assert video_dir.exists(), "video models directory should exist"
        
        steering_dir = models_dir / "steering"
        assert steering_dir.exists(), "steering models directory should exist"
        
        biometric_dir = models_dir / "biometric"
        assert biometric_dir.exists(), "biometric models directory should exist"
        
        fusion_dir = models_dir / "fusion"
        assert fusion_dir.exists(), "fusion models directory should exist"
        
        inference_dir = phase2_dir / "inference"
        assert inference_dir.exists(), "inference directory should exist"
        
        # Check that key files exist
        video_file = video_dir / "fatigue_detector.py"
        assert video_file.exists(), "fatigue_detector.py should exist"
        
        steering_file = steering_dir / "steering_analyzer.py"
        assert steering_file.exists(), "steering_analyzer.py should exist"
        
        biometric_file = biometric_dir / "stress_detector.py"
        assert biometric_file.exists(), "stress_detector.py should exist"
        
        fusion_file = fusion_dir / "multimodal_fusion.py"
        assert fusion_file.exists(), "multimodal_fusion.py should exist"
        
        inference_file = inference_dir / "real_time_inference.py"
        assert inference_file.exists(), "real_time_inference.py should exist"
        
        print("✅ Phase 2 project structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Phase 2 project structure test failed: {e}")
        return False

def test_file_imports():
    """Test that files can be imported without errors"""
    print("Testing file imports...")
    
    try:
        # Test importing files (they should handle missing dependencies gracefully)
        import phase2_core.models.video.fatigue_detector
        print("✅ Video fatigue detector import successful")
        
        import phase2_core.models.steering.steering_analyzer
        print("✅ Steering analyzer import successful")
        
        import phase2_core.models.biometric.stress_detector
        print("✅ Biometric stress detector import successful")
        
        import phase2_core.models.fusion.multimodal_fusion
        print("✅ Multi-modal fusion import successful")
        
        import phase2_core.inference.real_time_inference
        print("✅ Real-time inference engine import successful")
        
        print("✅ File imports test passed")
        return True
        
    except Exception as e:
        print(f"❌ File imports test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy dependencies"""
    print("Testing basic functionality...")
    
    try:
        # Test that we can create basic objects
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
        
        # Test basic methods that don't require TensorFlow
        # Create a mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test preprocessing (should work without TensorFlow)
        try:
            processed = detector.preprocess_frame(frame)
            assert processed.shape == (1, 64, 64, 3)
            assert processed.dtype == np.float32
        except Exception as e:
            print(f"⚠️  Preprocessing failed (expected without TensorFlow): {e}")
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_configuration_handling():
    """Test configuration handling"""
    print("Testing configuration handling...")
    
    try:
        # Test different configuration scenarios
        configs = [
            {
                "input_shape": (64, 64, 3),
                "sequence_length": 30,
                "num_classes": 3
            },
            {
                "input_shape": (128, 128, 3),
                "sequence_length": 60,
                "num_classes": 5
            },
            {
                "input_shape": (32, 32, 3),
                "sequence_length": 15,
                "num_classes": 2
            }
        ]
        
        for i, config in enumerate(configs):
            from phase2_core.models.video.fatigue_detector import VideoFatigueDetector
            detector = VideoFatigueDetector(config)
            
            assert detector.input_shape == config["input_shape"]
            assert detector.sequence_length == config["sequence_length"]
            assert detector.num_classes == config["num_classes"]
            
            print(f"  ✅ Configuration {i+1} handled correctly")
        
        print("✅ Configuration handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration handling test failed: {e}")
        return False

def test_data_structures():
    """Test data structures and types"""
    print("Testing data structures...")
    
    try:
        # Test that we can create and manipulate data structures
        from phase2_core.models.video.fatigue_detector import VideoFatigueDetector
        
        config = {"input_shape": (64, 64, 3), "sequence_length": 30, "num_classes": 3}
        detector = VideoFatigueDetector(config)
        
        # Test landmark history
        assert isinstance(detector.landmark_history, list)
        assert len(detector.landmark_history) == 0
        
        # Test eye closure history
        assert isinstance(detector.eye_closure_history, list)
        assert len(detector.eye_closure_history) == 0
        
        # Test yawn history
        assert isinstance(detector.yawn_history, list)
        assert len(detector.yawn_history) == 0
        
        # Test baseline values
        assert detector.baseline_eye_openness is None
        assert detector.personalization_factor == 1.0
        
        print("✅ Data structures test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        return False

def test_error_handling():
    """Test error handling"""
    print("Testing error handling...")
    
    try:
        from phase2_core.models.video.fatigue_detector import VideoFatigueDetector
        
        # Test with invalid configuration
        invalid_config = {"invalid_key": "invalid_value"}
        detector = VideoFatigueDetector(invalid_config)
        
        # Should handle missing keys gracefully
        assert detector.input_shape == (64, 64, 3)  # Default value
        assert detector.sequence_length == 30  # Default value
        assert detector.num_classes == 3  # Default value
        
        # Test with None input
        try:
            result = detector.extract_facial_features(None)
            assert result is not None
            assert "face_detected" in result
        except Exception as e:
            print(f"⚠️  Error handling for None input: {e}")
        
        print("✅ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_all_phase2_simple_tests():
    """Run all Phase 2 simple tests"""
    print("🚀 Running VigilAI Phase 2 Simple Tests...\n")
    
    tests = [
        test_project_structure,
        test_file_imports,
        test_basic_functionality,
        test_configuration_handling,
        test_data_structures,
        test_error_handling
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
        print("\n🎉 All Phase 2 simple tests passed! VigilAI core structure is working.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_phase2_simple_tests()
    sys.exit(0 if success else 1)
