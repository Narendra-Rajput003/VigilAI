"""
Test suite for VigilAI Phase 2 Core Development
Tests multi-modal AI fusion and detection models
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Phase 2 components
from phase2_core.models.video.fatigue_detector import VideoFatigueDetector
from phase2_core.models.steering.steering_analyzer import SteeringAnalyzer
from phase2_core.models.biometric.stress_detector import BiometricStressDetector
from phase2_core.models.fusion.multimodal_fusion import MultiModalFusion
from phase2_core.inference.real_time_inference import RealTimeInferenceEngine

class TestVideoFatigueDetector:
    """Test video fatigue detection model"""
    
    @pytest.fixture
    def video_detector(self):
        config = {
            "input_shape": (64, 64, 3),
            "sequence_length": 30,
            "num_classes": 3
        }
        return VideoFatigueDetector(config)
    
    def test_video_detector_initialization(self, video_detector):
        """Test video detector initialization"""
        assert video_detector is not None
        assert video_detector.config is not None
        assert video_detector.input_shape == (64, 64, 3)
        assert video_detector.sequence_length == 30
    
    def test_build_model(self, video_detector):
        """Test model building"""
        model = video_detector.build_model()
        assert model is not None
        assert model.name == "VideoFatigueDetector"
    
    def test_extract_facial_features(self, video_detector):
        """Test facial feature extraction"""
        # Create mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        features = video_detector.extract_facial_features(frame)
        
        assert "landmarks" in features
        assert "eye_openness" in features
        assert "yawn_detected" in features
        assert "head_pose" in features
        assert "face_detected" in features
        
        # Check value ranges
        assert 0.0 <= features["eye_openness"] <= 1.0
        assert isinstance(features["yawn_detected"], bool)
        assert isinstance(features["face_detected"], bool)
    
    def test_preprocess_frame(self, video_detector):
        """Test frame preprocessing"""
        # Create mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        processed = video_detector.preprocess_frame(frame)
        
        assert processed.shape == (1, 64, 64, 3)
        assert processed.dtype == np.float32
        assert 0.0 <= np.min(processed) <= 1.0
        assert 0.0 <= np.max(processed) <= 1.0
    
    def test_create_sequence(self, video_detector):
        """Test sequence creation"""
        # Create mock frames
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
        
        sequence = video_detector.create_sequence(frames)
        
        assert sequence.shape == (1, 30, 64, 64, 3)
        assert sequence.dtype == np.float32

class TestSteeringAnalyzer:
    """Test steering analysis model"""
    
    @pytest.fixture
    def steering_analyzer(self):
        config = {
            "sequence_length": 100,
            "feature_dim": 6,
            "num_classes": 3
        }
        return SteeringAnalyzer(config)
    
    def test_steering_analyzer_initialization(self, steering_analyzer):
        """Test steering analyzer initialization"""
        assert steering_analyzer is not None
        assert steering_analyzer.config is not None
        assert steering_analyzer.sequence_length == 100
        assert steering_analyzer.feature_dim == 6
    
    def test_build_model(self, steering_analyzer):
        """Test model building"""
        model = steering_analyzer.build_model()
        assert model is not None
        assert model.name == "SteeringAnalyzer"
    
    def test_extract_steering_features(self, steering_analyzer):
        """Test steering feature extraction"""
        steering_data = {
            "angle": 15.0,
            "velocity": 2.5,
            "speed": 60.0,
            "timestamp": time.time()
        }
        
        features = steering_analyzer.extract_steering_features(steering_data)
        
        assert len(features) == 6
        assert features[0] == 15.0  # angle
        assert features[1] == 2.5    # velocity
        assert features[2] == 0.0    # acceleration (first reading)
        assert features[3] == 0.0    # jerk (first reading)
        assert features[4] == 0.0    # entropy (no history)
        assert features[5] == 0.0    # variance (no history)
    
    def test_detect_anomalies(self, steering_analyzer):
        """Test anomaly detection"""
        steering_data = {
            "angle": 15.0,
            "velocity": 2.5,
            "speed": 60.0,
            "timestamp": time.time()
        }
        
        anomalies = steering_analyzer.detect_anomalies(steering_data)
        
        assert "anomalies" in anomalies
        assert "anomaly_score" in anomalies
        assert "features" in anomalies
        
        # Check anomaly score range
        assert 0.0 <= anomalies["anomaly_score"] <= 1.0

class TestBiometricStressDetector:
    """Test biometric stress detection model"""
    
    @pytest.fixture
    def biometric_detector(self):
        config = {
            "sequence_length": 60,
            "feature_dim": 8,
            "num_classes": 3
        }
        return BiometricStressDetector(config)
    
    def test_biometric_detector_initialization(self, biometric_detector):
        """Test biometric detector initialization"""
        assert biometric_detector is not None
        assert biometric_detector.config is not None
        assert biometric_detector.sequence_length == 60
        assert biometric_detector.feature_dim == 8
    
    def test_build_model(self, biometric_detector):
        """Test model building"""
        model = biometric_detector.build_model()
        assert model is not None
        assert model.name == "BiometricStressDetector"
    
    def test_extract_biometric_features(self, biometric_detector):
        """Test biometric feature extraction"""
        biometric_data = {
            "heart_rate": 75.0,
            "hrv": 45.0,
            "eda": 2.5,
            "temperature": 36.5,
            "timestamp": time.time()
        }
        
        features = biometric_detector.extract_biometric_features(biometric_data)
        
        assert len(features) == 8
        assert features[0] == 75.0   # heart_rate
        assert features[1] == 45.0   # hrv
        assert features[2] == 2.5    # eda
        assert features[3] == 36.5   # temperature
        assert features[4] == 0.0    # hr_variability (no history)
        assert features[5] == 0.0    # hrv_rmssd (no history)
        assert features[6] == 0.0    # eda_derivative (no history)
        assert 0.0 <= features[7] <= 1.0  # stress_index
    
    def test_detect_stress_patterns(self, biometric_detector):
        """Test stress pattern detection"""
        biometric_data = {
            "heart_rate": 75.0,
            "hrv": 45.0,
            "eda": 2.5,
            "temperature": 36.5,
            "timestamp": time.time()
        }
        
        patterns = biometric_detector.detect_stress_patterns(biometric_data)
        
        assert "stress_indicators" in patterns
        assert "stress_score" in patterns
        assert "features" in patterns
        
        # Check stress score range
        assert 0.0 <= patterns["stress_score"] <= 1.0

class TestMultiModalFusion:
    """Test multi-modal fusion model"""
    
    @pytest.fixture
    def fusion_model(self):
        config = {
            "video_dim": 128,
            "steering_dim": 64,
            "biometric_dim": 64,
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "num_classes": 3
        }
        return MultiModalFusion(config)
    
    def test_fusion_model_initialization(self, fusion_model):
        """Test fusion model initialization"""
        assert fusion_model is not None
        assert fusion_model.config is not None
        assert fusion_model.video_dim == 128
        assert fusion_model.steering_dim == 64
        assert fusion_model.biometric_dim == 64
    
    def test_build_model(self, fusion_model):
        """Test model building"""
        model = fusion_model.build_model()
        assert model is not None
        assert model.name == "MultiModalFusion"
    
    def test_fuse_predictions(self, fusion_model):
        """Test prediction fusion"""
        video_pred = {"fatigue_level": 1, "confidence": 0.8}
        steering_pred = {"fatigue_level": 0, "confidence": 0.6}
        biometric_pred = {"stress_level": 1, "confidence": 0.7}
        
        fused = fusion_model.fuse_predictions(video_pred, steering_pred, biometric_pred)
        
        assert "fatigue_level" in fused
        assert "confidence" in fused
        assert "weighted_fatigue_score" in fused
        assert "modality_contributions" in fused
        
        # Check value ranges
        assert 0 <= fused["fatigue_level"] <= 2
        assert 0.0 <= fused["confidence"] <= 1.0
        assert 0.0 <= fused["weighted_fatigue_score"] <= 2.0
    
    def test_predict_fatigue(self, fusion_model):
        """Test fatigue prediction"""
        # Create mock features
        video_features = np.random.randn(128)
        steering_features = np.random.randn(64)
        biometric_features = np.random.randn(64)
        
        # Build model first
        fusion_model.model = fusion_model.build_model()
        
        prediction = fusion_model.predict_fatigue(video_features, steering_features, biometric_features)
        
        assert "fatigue_level" in prediction
        assert "confidence" in prediction
        assert "probabilities" in prediction
        
        # Check value ranges
        assert 0 <= prediction["fatigue_level"] <= 2
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert len(prediction["probabilities"]) == 3

class TestRealTimeInferenceEngine:
    """Test real-time inference engine"""
    
    @pytest.fixture
    def inference_engine(self):
        config = {
            "video": {"input_shape": (64, 64, 3), "sequence_length": 30},
            "steering": {"sequence_length": 100, "feature_dim": 6},
            "biometric": {"sequence_length": 60, "feature_dim": 8},
            "fusion": {"video_dim": 128, "steering_dim": 64, "biometric_dim": 64},
            "inference_interval": 0.1
        }
        return RealTimeInferenceEngine(config)
    
    def test_inference_engine_initialization(self, inference_engine):
        """Test inference engine initialization"""
        assert inference_engine is not None
        assert inference_engine.config is not None
        assert inference_engine.video_detector is not None
        assert inference_engine.steering_analyzer is not None
        assert inference_engine.biometric_detector is not None
        assert inference_engine.fusion_model is not None
    
    def test_update_buffers(self, inference_engine):
        """Test buffer updates"""
        video_data = {"frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
        steering_data = {"angle": 15.0, "velocity": 2.5, "speed": 60.0}
        biometric_data = {"heart_rate": 75.0, "hrv": 45.0, "eda": 2.5}
        
        inference_engine._update_buffers(video_data, steering_data, biometric_data)
        
        assert len(inference_engine.video_buffer) == 1
        assert len(inference_engine.steering_buffer) == 1
        assert len(inference_engine.biometric_buffer) == 1
    
    def test_has_enough_data(self, inference_engine):
        """Test data sufficiency check"""
        # Initially should not have enough data
        assert not inference_engine._has_enough_data()
        
        # Add enough data
        for i in range(15):
            video_data = {"frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
            steering_data = {"angle": 15.0, "velocity": 2.5, "speed": 60.0}
            biometric_data = {"heart_rate": 75.0, "hrv": 45.0, "eda": 2.5}
            inference_engine._update_buffers(video_data, steering_data, biometric_data)
        
        # Now should have enough data
        assert inference_engine._has_enough_data()
    
    def test_get_buffer_status(self, inference_engine):
        """Test buffer status"""
        status = inference_engine.get_buffer_status()
        
        assert "video_buffer_size" in status
        assert "steering_buffer_size" in status
        assert "biometric_buffer_size" in status
        assert "has_enough_data" in status
        
        # Initially should be empty
        assert status["video_buffer_size"] == 0
        assert status["steering_buffer_size"] == 0
        assert status["biometric_buffer_size"] == 0
        assert not status["has_enough_data"]
    
    def test_clear_buffers(self, inference_engine):
        """Test buffer clearing"""
        # Add some data
        video_data = {"frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)}
        steering_data = {"angle": 15.0, "velocity": 2.5, "speed": 60.0}
        biometric_data = {"heart_rate": 75.0, "hrv": 45.0, "eda": 2.5}
        inference_engine._update_buffers(video_data, steering_data, biometric_data)
        
        # Clear buffers
        inference_engine.clear_buffers()
        
        # Check that buffers are empty
        assert len(inference_engine.video_buffer) == 0
        assert len(inference_engine.steering_buffer) == 0
        assert len(inference_engine.biometric_buffer) == 0

def test_integration():
    """Test basic integration of Phase 2 components"""
    print("Testing VigilAI Phase 2 Integration...")
    
    # Test video detector
    try:
        video_config = {"input_shape": (64, 64, 3), "sequence_length": 30, "num_classes": 3}
        video_detector = VideoFatigueDetector(video_config)
        model = video_detector.build_model()
        assert model is not None
        print("✅ Video fatigue detector test passed")
    except Exception as e:
        print(f"❌ Video fatigue detector test failed: {e}")
        return False
    
    # Test steering analyzer
    try:
        steering_config = {"sequence_length": 100, "feature_dim": 6, "num_classes": 3}
        steering_analyzer = SteeringAnalyzer(steering_config)
        model = steering_analyzer.build_model()
        assert model is not None
        print("✅ Steering analyzer test passed")
    except Exception as e:
        print(f"❌ Steering analyzer test failed: {e}")
        return False
    
    # Test biometric detector
    try:
        biometric_config = {"sequence_length": 60, "feature_dim": 8, "num_classes": 3}
        biometric_detector = BiometricStressDetector(biometric_config)
        model = biometric_detector.build_model()
        assert model is not None
        print("✅ Biometric stress detector test passed")
    except Exception as e:
        print(f"❌ Biometric stress detector test failed: {e}")
        return False
    
    # Test fusion model
    try:
        fusion_config = {
            "video_dim": 128, "steering_dim": 64, "biometric_dim": 64,
            "hidden_dim": 256, "num_heads": 8, "num_layers": 6, "num_classes": 3
        }
        fusion_model = MultiModalFusion(fusion_config)
        model = fusion_model.build_model()
        assert model is not None
        print("✅ Multi-modal fusion test passed")
    except Exception as e:
        print(f"❌ Multi-modal fusion test failed: {e}")
        return False
    
    # Test inference engine
    try:
        inference_config = {
            "video": {"input_shape": (64, 64, 3), "sequence_length": 30},
            "steering": {"sequence_length": 100, "feature_dim": 6},
            "biometric": {"sequence_length": 60, "feature_dim": 8},
            "fusion": {"video_dim": 128, "steering_dim": 64, "biometric_dim": 64},
            "inference_interval": 0.1
        }
        inference_engine = RealTimeInferenceEngine(inference_config)
        assert inference_engine is not None
        print("✅ Real-time inference engine test passed")
    except Exception as e:
        print(f"❌ Real-time inference engine test failed: {e}")
        return False
    
    print("\n🎉 All Phase 2 integration tests passed!")
    return True

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
