"""
Real-time Inference Engine for VigilAI
Coordinates multi-modal inference for real-time fatigue detection
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

# Import models
from ..models.video.fatigue_detector import VideoFatigueDetector
from ..models.steering.steering_analyzer import SteeringAnalyzer
from ..models.biometric.stress_detector import BiometricStressDetector
from ..models.fusion.multimodal_fusion import MultiModalFusion

logger = logging.getLogger(__name__)

class RealTimeInferenceEngine:
    """Real-time inference engine for VigilAI"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize models
        self.video_detector = VideoFatigueDetector(config.get("video", {}))
        self.steering_analyzer = SteeringAnalyzer(config.get("steering", {}))
        self.biometric_detector = BiometricStressDetector(config.get("biometric", {}))
        self.fusion_model = MultiModalFusion(config.get("fusion", {}))
        
        # Data buffers
        self.video_buffer = deque(maxlen=30)  # 1 second at 30fps
        self.steering_buffer = deque(maxlen=100)  # 10 seconds at 10Hz
        self.biometric_buffer = deque(maxlen=60)  # 1 minute at 1Hz
        
        # Inference state
        self.is_initialized = False
        self.last_inference_time = 0
        self.inference_interval = config.get("inference_interval", 0.1)  # 10Hz
        
        # Performance metrics
        self.inference_times = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        
    async def initialize(self) -> bool:
        """Initialize the inference engine"""
        try:
            logger.info("Initializing real-time inference engine...")
            
            # Load models
            video_model_path = self.config.get("video", {}).get("model_path")
            if video_model_path:
                if not self.video_detector.load_model(video_model_path):
                    logger.warning("Failed to load video model, using default")
                    self.video_detector.model = self.video_detector.build_model()
            
            steering_model_path = self.config.get("steering", {}).get("model_path")
            if steering_model_path:
                if not self.steering_analyzer.load_model(steering_model_path):
                    logger.warning("Failed to load steering model, using default")
                    self.steering_analyzer.model = self.steering_analyzer.build_model()
            
            biometric_model_path = self.config.get("biometric", {}).get("model_path")
            if biometric_model_path:
                if not self.biometric_detector.load_model(biometric_model_path):
                    logger.warning("Failed to load biometric model, using default")
                    self.biometric_detector.model = self.biometric_detector.build_model()
            
            fusion_model_path = self.config.get("fusion", {}).get("model_path")
            if fusion_model_path:
                if not self.fusion_model.load_model(fusion_model_path):
                    logger.warning("Failed to load fusion model, using default")
                    self.fusion_model.model = self.fusion_model.build_model()
            
            self.is_initialized = True
            logger.info("Real-time inference engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing inference engine: {e}")
            return False
    
    async def process_frame(self, video_data: Dict, steering_data: Dict, biometric_data: Dict) -> Dict:
        """Process a single frame of multi-modal data"""
        if not self.is_initialized:
            return {"error": "Inference engine not initialized"}
        
        start_time = time.time()
        
        try:
            # Update buffers
            self._update_buffers(video_data, steering_data, biometric_data)
            
            # Check if enough data for inference
            if not self._has_enough_data():
                return {"error": "Insufficient data for inference"}
            
            # Run individual modality predictions
            video_pred = await self._predict_video_fatigue()
            steering_pred = await self._predict_steering_fatigue()
            biometric_pred = await self._predict_biometric_stress()
            
            # Fuse predictions
            fused_pred = self._fuse_predictions(video_pred, steering_pred, biometric_pred)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.inference_times.append(processing_time)
            
            # Update performance metrics
            self._update_performance_metrics(fused_pred)
            
            return {
                "fatigue_level": fused_pred["fatigue_level"],
                "confidence": fused_pred["confidence"],
                "processing_time": processing_time,
                "modality_predictions": {
                    "video": video_pred,
                    "steering": steering_pred,
                    "biometric": biometric_pred
                },
                "fused_prediction": fused_pred,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return {"error": str(e)}
    
    def _update_buffers(self, video_data: Dict, steering_data: Dict, biometric_data: Dict):
        """Update data buffers"""
        # Update video buffer
        if video_data and "frame" in video_data:
            self.video_buffer.append(video_data)
        
        # Update steering buffer
        if steering_data:
            self.steering_analyzer.update_history(steering_data)
            self.steering_buffer.append(steering_data)
        
        # Update biometric buffer
        if biometric_data:
            self.biometric_detector.update_history(biometric_data)
            self.biometric_buffer.append(biometric_data)
    
    def _has_enough_data(self) -> bool:
        """Check if there's enough data for inference"""
        return (len(self.video_buffer) >= 10 and  # At least 10 frames
                len(self.steering_buffer) >= 10 and  # At least 10 steering readings
                len(self.biometric_buffer) >= 5)  # At least 5 biometric readings
    
    async def _predict_video_fatigue(self) -> Dict:
        """Predict fatigue from video data"""
        try:
            if not self.video_detector.model:
                return {"fatigue_level": 0, "confidence": 0.0, "error": "Model not loaded"}
            
            # Get recent frames
            recent_frames = [data["frame"] for data in list(self.video_buffer)[-10:]]
            
            # Predict fatigue
            prediction = self.video_detector.predict_fatigue(recent_frames)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting video fatigue: {e}")
            return {"fatigue_level": 0, "confidence": 0.0, "error": str(e)}
    
    async def _predict_steering_fatigue(self) -> Dict:
        """Predict fatigue from steering data"""
        try:
            if not self.steering_analyzer.model:
                return {"fatigue_level": 0, "confidence": 0.0, "error": "Model not loaded"}
            
            # Get recent steering data
            recent_steering = list(self.steering_buffer)[-20:]
            
            # Predict fatigue
            prediction = self.steering_analyzer.predict_fatigue(recent_steering)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting steering fatigue: {e}")
            return {"fatigue_level": 0, "confidence": 0.0, "error": str(e)}
    
    async def _predict_biometric_stress(self) -> Dict:
        """Predict stress from biometric data"""
        try:
            if not self.biometric_detector.model:
                return {"stress_level": 0, "confidence": 0.0, "error": "Model not loaded"}
            
            # Get recent biometric data
            recent_biometric = list(self.biometric_buffer)[-10:]
            
            # Predict stress
            prediction = self.biometric_detector.predict_stress(recent_biometric)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting biometric stress: {e}")
            return {"stress_level": 0, "confidence": 0.0, "error": str(e)}
    
    def _fuse_predictions(self, video_pred: Dict, steering_pred: Dict, biometric_pred: Dict) -> Dict:
        """Fuse predictions from all modalities"""
        try:
            # Use fusion model if available
            if self.fusion_model.model:
                # Extract features for fusion
                video_features = self._extract_video_features()
                steering_features = self._extract_steering_features()
                biometric_features = self._extract_biometric_features()
                
                # Predict using fusion model
                fusion_pred = self.fusion_model.predict_fatigue(
                    video_features, steering_features, biometric_features
                )
                
                return fusion_pred
            else:
                # Use simple weighted fusion
                return self.fusion_model.fuse_predictions(video_pred, steering_pred, biometric_pred)
                
        except Exception as e:
            logger.error(f"Error fusing predictions: {e}")
            return {"fatigue_level": 0, "confidence": 0.0, "error": str(e)}
    
    def _extract_video_features(self) -> np.ndarray:
        """Extract video features for fusion"""
        try:
            # Get recent frames
            recent_frames = [data["frame"] for data in list(self.video_buffer)[-10:]]
            
            # Extract features from each frame
            features = []
            for frame in recent_frames:
                if frame is not None:
                    frame_features = self.video_detector.extract_facial_features(frame)
                    # Convert to feature vector
                    feature_vector = np.array([
                        frame_features["eye_openness"],
                        float(frame_features["yawn_detected"]),
                        frame_features["head_pose"]["pitch"] if frame_features["head_pose"] else 0.0,
                        frame_features["head_pose"]["yaw"] if frame_features["head_pose"] else 0.0,
                        frame_features["head_pose"]["roll"] if frame_features["head_pose"] else 0.0
                    ])
                    features.append(feature_vector)
            
            if features:
                # Average features across frames
                return np.mean(features, axis=0)
            else:
                return np.zeros(5)
                
        except Exception as e:
            logger.error(f"Error extracting video features: {e}")
            return np.zeros(5)
    
    def _extract_steering_features(self) -> np.ndarray:
        """Extract steering features for fusion"""
        try:
            # Get recent steering data
            recent_steering = list(self.steering_buffer)[-10:]
            
            # Extract features
            features = []
            for data in recent_steering:
                feature_vector = self.steering_analyzer.extract_steering_features(data)
                features.append(feature_vector)
            
            if features:
                # Average features across time
                return np.mean(features, axis=0)
            else:
                return np.zeros(6)
                
        except Exception as e:
            logger.error(f"Error extracting steering features: {e}")
            return np.zeros(6)
    
    def _extract_biometric_features(self) -> np.ndarray:
        """Extract biometric features for fusion"""
        try:
            # Get recent biometric data
            recent_biometric = list(self.biometric_buffer)[-10:]
            
            # Extract features
            features = []
            for data in recent_biometric:
                feature_vector = self.biometric_detector.extract_biometric_features(data)
                features.append(feature_vector)
            
            if features:
                # Average features across time
                return np.mean(features, axis=0)
            else:
                return np.zeros(8)
                
        except Exception as e:
            logger.error(f"Error extracting biometric features: {e}")
            return np.zeros(8)
    
    def _update_performance_metrics(self, prediction: Dict):
        """Update performance metrics"""
        try:
            # Calculate accuracy (simplified)
            confidence = prediction.get("confidence", 0.0)
            self.accuracy_history.append(confidence)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        try:
            if not self.inference_times:
                return {}
            
            return {
                "avg_inference_time": float(np.mean(self.inference_times)),
                "max_inference_time": float(np.max(self.inference_times)),
                "min_inference_time": float(np.min(self.inference_times)),
                "avg_confidence": float(np.mean(self.accuracy_history)) if self.accuracy_history else 0.0,
                "total_inferences": len(self.inference_times)
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_buffer_status(self) -> Dict:
        """Get buffer status"""
        return {
            "video_buffer_size": len(self.video_buffer),
            "steering_buffer_size": len(self.steering_buffer),
            "biometric_buffer_size": len(self.biometric_buffer),
            "has_enough_data": self._has_enough_data()
        }
    
    def clear_buffers(self):
        """Clear all data buffers"""
        self.video_buffer.clear()
        self.steering_buffer.clear()
        self.biometric_buffer.clear()
        logger.info("All buffers cleared")
    
    def update_fusion_weights(self, video_weight: float, steering_weight: float, biometric_weight: float):
        """Update fusion weights"""
        self.fusion_model.update_fusion_weights(video_weight, steering_weight, biometric_weight)
    
    async def cleanup(self):
        """Cleanup inference engine"""
        try:
            self.clear_buffers()
            self.is_initialized = False
            logger.info("Inference engine cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
