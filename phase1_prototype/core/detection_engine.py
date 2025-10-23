"""
VigilAI Detection Engine
Core AI/ML components for fatigue and stress detection
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp
from scipy import signal
from scipy.stats import entropy

logger = logging.getLogger(__name__)

class DetectionEngine:
    """Main detection engine for fatigue and stress analysis"""
    
    def __init__(self, config):
        self.config = config
        self.mediapipe_face = mp.solutions.face_mesh
        self.face_mesh = self.mediapipe_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Detection state
        self.eye_closure_history = []
        self.yawn_history = []
        self.head_pose_history = []
        self.steering_history = []
        self.biometric_history = []
        
        # Calibration data
        self.baseline_eye_openness = None
        self.baseline_hrv = None
        self.personalization_factor = 1.0
        
    async def analyze(self, data: Dict) -> Dict:
        """
        Analyze multi-modal data for fatigue and stress detection
        
        Args:
            data: Dictionary containing video, steering, and biometric data
            
        Returns:
            Dictionary with detection results and scores
        """
        start_time = time.time()
        
        try:
            # Extract features from each modality
            video_features = await self._extract_video_features(data.get("video"))
            steering_features = await self._extract_steering_features(data.get("steering"))
            biometric_features = await self._extract_biometric_features(data.get("biometric"))
            
            # Calculate individual scores
            fatigue_score = await self._calculate_fatigue_score(
                video_features, steering_features, biometric_features
            )
            stress_score = await self._calculate_stress_score(
                video_features, steering_features, biometric_features
            )
            
            # Combine scores with weighted fusion
            combined_score = self._fuse_scores(fatigue_score, stress_score)
            
            # Update history for temporal analysis
            self._update_history(video_features, steering_features, biometric_features)
            
            processing_time = time.time() - start_time
            
            return {
                "fatigue_score": fatigue_score,
                "stress_score": stress_score,
                "combined_score": combined_score,
                "confidence": self._calculate_confidence(),
                "processing_time": processing_time,
                "features": {
                    "video": video_features,
                    "steering": steering_features,
                    "biometric": biometric_features
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in detection analysis: {e}")
            return {
                "fatigue_score": 0.0,
                "stress_score": 0.0,
                "combined_score": 0.0,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def _extract_video_features(self, video_data: Optional[Dict]) -> Dict:
        """Extract features from video data"""
        if not video_data or "frame" not in video_data:
            return {"eye_openness": 0.5, "yawn_detected": False, "head_pose": None}
        
        frame = video_data["frame"]
        if frame is None:
            return {"eye_openness": 0.5, "yawn_detected": False, "head_pose": None}
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        features = {
            "eye_openness": 0.5,  # Default neutral
            "yawn_detected": False,
            "head_pose": None,
            "face_detected": False
        }
        
        if results.multi_face_landmarks:
            features["face_detected"] = True
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract eye openness (PERCLOS calculation)
            eye_openness = self._calculate_eye_openness(face_landmarks)
            features["eye_openness"] = eye_openness
            
            # Detect yawn
            yawn_detected = self._detect_yawn(face_landmarks)
            features["yawn_detected"] = yawn_detected
            
            # Calculate head pose
            head_pose = self._calculate_head_pose(face_landmarks)
            features["head_pose"] = head_pose
        
        return features
    
    async def _extract_steering_features(self, steering_data: Optional[Dict]) -> Dict:
        """Extract features from steering data"""
        if not steering_data:
            return {"entropy": 0.0, "variance": 0.0, "anomaly_score": 0.0}
        
        # Calculate steering entropy and variance
        steering_angle = steering_data.get("angle", 0.0)
        steering_velocity = steering_data.get("velocity", 0.0)
        
        # Add to history
        self.steering_history.append({
            "angle": steering_angle,
            "velocity": steering_velocity,
            "timestamp": time.time()
        })
        
        # Keep only recent history (last 30 seconds)
        cutoff_time = time.time() - 30
        self.steering_history = [
            s for s in self.steering_history if s["timestamp"] > cutoff_time
        ]
        
        if len(self.steering_history) < 10:
            return {"entropy": 0.0, "variance": 0.0, "anomaly_score": 0.0}
        
        # Calculate features
        angles = [s["angle"] for s in self.steering_history]
        velocities = [s["velocity"] for s in self.steering_history]
        
        entropy_val = entropy(np.histogram(angles, bins=20)[0] + 1e-10)
        variance = np.var(angles)
        
        # Anomaly detection using statistical methods
        anomaly_score = self._detect_steering_anomaly(angles, velocities)
        
        return {
            "entropy": float(entropy_val),
            "variance": float(variance),
            "anomaly_score": float(anomaly_score)
        }
    
    async def _extract_biometric_features(self, biometric_data: Optional[Dict]) -> Dict:
        """Extract features from biometric data"""
        if not biometric_data:
            return {"hrv": 0.0, "eda": 0.0, "stress_indicator": 0.0}
        
        hrv = biometric_data.get("hrv", 0.0)
        eda = biometric_data.get("eda", 0.0)
        heart_rate = biometric_data.get("heart_rate", 0.0)
        
        # Add to history
        self.biometric_history.append({
            "hrv": hrv,
            "eda": eda,
            "heart_rate": heart_rate,
            "timestamp": time.time()
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 60  # 1 minute
        self.biometric_history = [
            b for b in self.biometric_history if b["timestamp"] > cutoff_time
        ]
        
        if len(self.biometric_history) < 5:
            return {"hrv": 0.0, "eda": 0.0, "stress_indicator": 0.0}
        
        # Calculate stress indicator
        stress_indicator = self._calculate_stress_indicator()
        
        return {
            "hrv": float(hrv),
            "eda": float(eda),
            "stress_indicator": float(stress_indicator)
        }
    
    def _calculate_eye_openness(self, face_landmarks) -> float:
        """Calculate eye openness using PERCLOS method"""
        # Get eye landmark indices (simplified)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Calculate eye aspect ratio (simplified)
        def get_eye_ratio(landmarks, indices):
            points = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y] 
                              for i in indices])
            
            # Vertical distance
            vertical_1 = np.linalg.norm(points[1] - points[5])
            vertical_2 = np.linalg.norm(points[2] - points[4])
            
            # Horizontal distance
            horizontal = np.linalg.norm(points[0] - points[3])
            
            if horizontal == 0:
                return 0.5
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        
        left_ear = get_eye_ratio(face_landmarks, left_eye_indices)
        right_ear = get_eye_ratio(face_landmarks, right_eye_indices)
        
        # Average of both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, ear))
    
    def _detect_yawn(self, face_landmarks) -> bool:
        """Detect yawn based on mouth opening"""
        # Mouth landmark indices (simplified)
        mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        
        points = np.array([[face_landmarks.landmark[i].x, face_landmarks.landmark[i].y] 
                          for i in mouth_indices])
        
        # Calculate mouth opening
        vertical_distance = np.linalg.norm(points[2] - points[8])  # Top to bottom
        horizontal_distance = np.linalg.norm(points[0] - points[6])  # Left to right
        
        if horizontal_distance == 0:
            return False
        
        mouth_ratio = vertical_distance / horizontal_distance
        
        # Threshold for yawn detection
        return mouth_ratio > 0.3
    
    def _calculate_head_pose(self, face_landmarks) -> Dict:
        """Calculate head pose (pitch, yaw, roll)"""
        # Simplified head pose calculation
        # In a real implementation, you'd use more sophisticated methods
        
        # Get key facial points
        nose_tip = face_landmarks.landmark[1]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[362]
        
        # Calculate basic pose angles
        eye_vector = np.array([right_eye.x - left_eye.x, right_eye.y - left_eye.y])
        eye_angle = np.arctan2(eye_vector[1], eye_vector[0])
        
        return {
            "pitch": 0.0,  # Simplified
            "yaw": float(eye_angle),
            "roll": 0.0    # Simplified
        }
    
    def _detect_steering_anomaly(self, angles: List[float], velocities: List[float]) -> float:
        """Detect steering anomalies using statistical methods"""
        if len(angles) < 5:
            return 0.0
        
        # Calculate z-scores for anomaly detection
        angles_array = np.array(angles)
        velocities_array = np.array(velocities)
        
        # Z-score for angles
        angle_mean = np.mean(angles_array)
        angle_std = np.std(angles_array)
        if angle_std > 0:
            angle_z_scores = np.abs((angles_array - angle_mean) / angle_std)
        else:
            angle_z_scores = np.zeros_like(angles_array)
        
        # Z-score for velocities
        velocity_mean = np.mean(velocities_array)
        velocity_std = np.std(velocities_array)
        if velocity_std > 0:
            velocity_z_scores = np.abs((velocities_array - velocity_mean) / velocity_std)
        else:
            velocity_z_scores = np.zeros_like(velocities_array)
        
        # Combine z-scores
        combined_z_scores = (angle_z_scores + velocity_z_scores) / 2
        
        # Return maximum anomaly score
        return float(np.max(combined_z_scores))
    
    def _calculate_stress_indicator(self) -> float:
        """Calculate stress indicator from biometric history"""
        if len(self.biometric_history) < 3:
            return 0.0
        
        # Extract recent HRV and EDA values
        recent_data = self.biometric_history[-10:]  # Last 10 readings
        
        hrv_values = [d["hrv"] for d in recent_data if d["hrv"] > 0]
        eda_values = [d["eda"] for d in recent_data if d["eda"] > 0]
        
        if not hrv_values and not eda_values:
            return 0.0
        
        stress_score = 0.0
        
        # HRV-based stress (lower HRV = higher stress)
        if hrv_values:
            avg_hrv = np.mean(hrv_values)
            if self.baseline_hrv is None:
                self.baseline_hrv = avg_hrv
            else:
                hrv_ratio = avg_hrv / self.baseline_hrv if self.baseline_hrv > 0 else 1.0
                stress_score += (1.0 - hrv_ratio) * 0.6  # 60% weight
        
        # EDA-based stress (higher EDA = higher stress)
        if eda_values:
            avg_eda = np.mean(eda_values)
            # Normalize EDA (simplified)
            eda_normalized = min(1.0, avg_eda / 10.0)  # Assume max EDA of 10
            stress_score += eda_normalized * 0.4  # 40% weight
        
        return min(1.0, stress_score)
    
    async def _calculate_fatigue_score(self, video_features: Dict, 
                                     steering_features: Dict, 
                                     biometric_features: Dict) -> float:
        """Calculate fatigue score from all modalities"""
        fatigue_score = 0.0
        
        # Video-based fatigue (60% weight)
        eye_openness = video_features.get("eye_openness", 0.5)
        yawn_detected = video_features.get("yawn_detected", False)
        
        # PERCLOS calculation
        if self.baseline_eye_openness is None:
            self.baseline_eye_openness = eye_openness
        else:
            # Update baseline with exponential moving average
            self.baseline_eye_openness = 0.9 * self.baseline_eye_openness + 0.1 * eye_openness
        
        # Eye closure contribution
        if self.baseline_eye_openness > 0:
            eye_closure_ratio = 1.0 - (eye_openness / self.baseline_eye_openness)
            fatigue_score += max(0.0, eye_closure_ratio) * 0.4
        
        # Yawn contribution
        if yawn_detected:
            fatigue_score += 0.2
        
        # Steering-based fatigue (30% weight)
        steering_entropy = steering_features.get("entropy", 0.0)
        steering_anomaly = steering_features.get("anomaly_score", 0.0)
        
        # Higher entropy and anomalies indicate fatigue
        fatigue_score += min(0.3, steering_entropy * 0.1)
        fatigue_score += min(0.3, steering_anomaly * 0.1)
        
        # Biometric-based fatigue (10% weight)
        stress_indicator = biometric_features.get("stress_indicator", 0.0)
        fatigue_score += stress_indicator * 0.1
        
        # Apply personalization factor
        fatigue_score *= self.personalization_factor
        
        return min(1.0, fatigue_score)
    
    async def _calculate_stress_score(self, video_features: Dict,
                                    steering_features: Dict,
                                    biometric_features: Dict) -> float:
        """Calculate stress score from all modalities"""
        stress_score = 0.0
        
        # Biometric-based stress (70% weight)
        stress_indicator = biometric_features.get("stress_indicator", 0.0)
        stress_score += stress_indicator * 0.7
        
        # Video-based stress (20% weight)
        head_pose = video_features.get("head_pose")
        if head_pose:
            # Head movement indicates stress
            head_movement = abs(head_pose.get("yaw", 0.0))
            stress_score += min(0.2, head_movement * 0.1)
        
        # Steering-based stress (10% weight)
        steering_variance = steering_features.get("variance", 0.0)
        stress_score += min(0.1, steering_variance * 0.01)
        
        return min(1.0, stress_score)
    
    def _fuse_scores(self, fatigue_score: float, stress_score: float) -> float:
        """Fuse fatigue and stress scores into combined score"""
        # Weighted combination
        combined = (fatigue_score * 0.7) + (stress_score * 0.3)
        return min(1.0, combined)
    
    def _calculate_confidence(self) -> float:
        """Calculate confidence in detection results"""
        # Simple confidence based on data availability
        confidence = 0.5  # Base confidence
        
        if len(self.eye_closure_history) > 0:
            confidence += 0.2
        
        if len(self.steering_history) > 0:
            confidence += 0.2
        
        if len(self.biometric_history) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _update_history(self, video_features: Dict, steering_features: Dict, biometric_features: Dict):
        """Update detection history for temporal analysis"""
        current_time = time.time()
        
        # Update eye closure history
        self.eye_closure_history.append({
            "openness": video_features.get("eye_openness", 0.5),
            "timestamp": current_time
        })
        
        # Keep only recent history (last 5 minutes)
        cutoff_time = current_time - 300
        self.eye_closure_history = [
            h for h in self.eye_closure_history if h["timestamp"] > cutoff_time
        ]
    
    def get_current_score(self) -> float:
        """Get current fatigue score for real-time monitoring"""
        # This would be called by the web interface
        return getattr(self, '_current_fatigue_score', 0.0)
    
    def calibrate(self, duration_seconds: int = 300):
        """Calibrate the system for the current user"""
        logger.info(f"Starting {duration_seconds}s calibration...")
        
        # Reset baselines
        self.baseline_eye_openness = None
        self.baseline_hrv = None
        self.personalization_factor = 1.0
        
        # The actual calibration would happen during normal operation
        # This is just a placeholder for the calibration process
        logger.info("Calibration complete")
