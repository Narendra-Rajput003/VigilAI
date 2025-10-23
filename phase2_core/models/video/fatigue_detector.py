"""
Video-based Fatigue Detection Model
CNN-LSTM architecture for real-time fatigue detection from video frames
"""

import logging
import numpy as np
# Optional TensorFlow import
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
from typing import Dict, List, Optional, Tuple
import cv2

# Optional MediaPipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None

logger = logging.getLogger(__name__)

class VideoFatigueDetector:
    """CNN-LSTM model for video-based fatigue detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        if MEDIAPIPE_AVAILABLE:
            self.mediapipe_face = mp.solutions.face_mesh
            self.face_mesh = self.mediapipe_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.mediapipe_face = None
            self.face_mesh = None
        
        # Model parameters
        self.input_shape = config.get("input_shape", (64, 64, 3))
        self.sequence_length = config.get("sequence_length", 30)  # 30 frames = 1 second at 30fps
        self.num_classes = config.get("num_classes", 3)  # No fatigue, mild, severe
        
        # Feature extraction
        self.landmark_history = []
        self.eye_closure_history = []
        self.yawn_history = []
        
        # Calibration data
        self.baseline_eye_openness = None
        self.baseline_hrv = None
        self.personalization_factor = 1.0
        
    def build_model(self):
        """Build the CNN-LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, returning None")
            return None
        
        # Input layer
        input_layer = keras.Input(
            shape=(self.sequence_length, *self.input_shape),
            name="video_sequence"
        )
        
        # CNN feature extractor
        cnn_features = self._build_cnn_backbone(input_layer)
        
        # LSTM for temporal modeling
        lstm_output = self._build_lstm_layers(cnn_features)
        
        # Classification head
        classification_head = self._build_classification_head(lstm_output)
        
        # Create model
        model = keras.Model(
            inputs=input_layer,
            outputs=classification_head,
            name="VideoFatigueDetector"
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
        
        return model
    
    def _build_cnn_backbone(self, input_layer):
        """Build CNN backbone for feature extraction"""
        # Reshape for CNN processing
        x = layers.TimeDistributed(
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            name="conv2d_1"
        )(input_layer)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name="maxpool_1"
        )(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            name="conv2d_2"
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name="maxpool_2"
        )(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            name="conv2d_3"
        )(x)
        x = layers.TimeDistributed(
            layers.MaxPooling2D((2, 2)),
            name="maxpool_3"
        )(x)
        
        x = layers.TimeDistributed(
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            name="conv2d_4"
        )(x)
        x = layers.TimeDistributed(
            layers.GlobalAveragePooling2D(),
            name="global_avg_pool"
        )(x)
        
        return x
    
    def _build_lstm_layers(self, cnn_features):
        """Build LSTM layers for temporal modeling"""
        # First LSTM layer
        lstm_1 = layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name="lstm_1"
        )(cnn_features)
        
        # Second LSTM layer
        lstm_2 = layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2,
            name="lstm_2"
        )(lstm_1)
        
        return lstm_2
    
    def _build_classification_head(self, lstm_output):
        """Build classification head"""
        # Dense layers
        x = layers.Dense(128, activation="relu", name="dense_1")(lstm_output)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
        
        # Output layer
        output = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="fatigue_classification"
        )(x)
        
        return output
    
    def extract_facial_features(self, frame: np.ndarray) -> Dict:
        """Extract facial features using MediaPipe"""
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            return {
                "landmarks": None,
                "eye_openness": 0.5,
                "yawn_detected": False,
                "head_pose": None,
                "face_detected": False
            }
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            features = {
                "landmarks": None,
                "eye_openness": 0.5,
                "yawn_detected": False,
                "head_pose": None,
                "face_detected": False
            }
            
            if results.multi_face_landmarks:
                features["face_detected"] = True
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract landmarks
                landmarks = np.array([
                    [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
                ])
                features["landmarks"] = landmarks
                
                # Calculate eye openness (PERCLOS)
                eye_openness = self._calculate_eye_openness(landmarks)
                features["eye_openness"] = eye_openness
                
                # Detect yawn
                yawn_detected = self._detect_yawn(landmarks)
                features["yawn_detected"] = yawn_detected
                
                # Calculate head pose
                head_pose = self._calculate_head_pose(landmarks)
                features["head_pose"] = head_pose
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {e}")
            return {
                "landmarks": None,
                "eye_openness": 0.5,
                "yawn_detected": False,
                "head_pose": None,
                "face_detected": False
            }
    
    def _calculate_eye_openness(self, landmarks: np.ndarray) -> float:
        """Calculate eye openness using PERCLOS method"""
        # Eye landmark indices (MediaPipe face mesh)
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        def get_eye_ratio(eye_indices):
            eye_points = landmarks[eye_indices]
            
            # Calculate eye aspect ratio (EAR)
            vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
            vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
            horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
            
            if horizontal == 0:
                return 0.5
            
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        
        left_ear = get_eye_ratio(left_eye_indices)
        right_ear = get_eye_ratio(right_eye_indices)
        
        # Average of both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, ear))
    
    def _detect_yawn(self, landmarks: np.ndarray) -> bool:
        """Detect yawn based on mouth opening"""
        # Mouth landmark indices
        mouth_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        mouth_points = landmarks[mouth_indices]
        
        # Calculate mouth opening ratio
        vertical_distance = np.linalg.norm(mouth_points[2] - mouth_points[8])
        horizontal_distance = np.linalg.norm(mouth_points[0] - mouth_points[6])
        
        if horizontal_distance == 0:
            return False
        
        mouth_ratio = vertical_distance / horizontal_distance
        
        # Threshold for yawn detection
        return mouth_ratio > 0.3
    
    def _calculate_head_pose(self, landmarks: np.ndarray) -> Dict:
        """Calculate head pose (pitch, yaw, roll)"""
        # Key facial points for pose estimation
        nose_tip = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[362]
        chin = landmarks[175]
        
        # Calculate basic pose angles
        eye_vector = right_eye - left_eye
        eye_angle = np.arctan2(eye_vector[1], eye_vector[0])
        
        # Calculate pitch from nose to chin
        nose_chin_vector = chin - nose_tip
        pitch = np.arctan2(nose_chin_vector[2], nose_chin_vector[1])
        
        # Calculate roll from eye line
        roll = np.arctan2(eye_vector[1], eye_vector[0])
        
        return {
            "pitch": float(pitch),
            "yaw": float(eye_angle),
            "roll": float(roll)
        }
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input"""
        try:
            # Resize to model input size
            resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Ensure correct shape
            if len(normalized.shape) == 3:
                normalized = np.expand_dims(normalized, axis=0)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return np.zeros((1, *self.input_shape), dtype=np.float32)
    
    def create_sequence(self, frames: List[np.ndarray]) -> np.ndarray:
        """Create sequence from frames for LSTM input"""
        if len(frames) < self.sequence_length:
            # Pad with zeros if not enough frames
            padded_frames = frames + [np.zeros(self.input_shape) for _ in range(self.sequence_length - len(frames))]
        else:
            # Take the last sequence_length frames
            padded_frames = frames[-self.sequence_length:]
        
        # Stack frames into sequence
        sequence = np.stack(padded_frames, axis=0)
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)
        
        return sequence
    
    def predict_fatigue(self, frames: List[np.ndarray]) -> Dict:
        """Predict fatigue level from video sequence"""
        if not self.model:
            logger.error("Model not loaded")
            return {"fatigue_level": 0, "confidence": 0.0, "features": {}}
        
        try:
            # Preprocess frames
            processed_frames = [self.preprocess_frame(frame) for frame in frames]
            
            # Create sequence
            sequence = self.create_sequence(processed_frames)
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Extract features from last frame
            features = self.extract_facial_features(frames[-1])
            
            return {
                "fatigue_level": int(predicted_class),
                "confidence": confidence,
                "probabilities": prediction[0].tolist(),
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Error predicting fatigue: {e}")
            return {"fatigue_level": 0, "confidence": 0.0, "features": {}}
    
    def load_model(self, model_path: str) -> bool:
        """Load pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """Save model"""
        if not self.model:
            logger.error("No model to save")
            return False
        
        try:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if not self.model:
            return "Model not built"
        
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            self.model.summary()
        
        return f.getvalue()
