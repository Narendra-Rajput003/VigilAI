"""
Steering Analysis Model
Time-series analysis for steering pattern detection and anomaly detection
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Tuple
from scipy import signal
from scipy.stats import entropy
import pandas as pd

logger = logging.getLogger(__name__)

class SteeringAnalyzer:
    """Advanced steering pattern analysis for fatigue detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Model parameters
        self.sequence_length = config.get("sequence_length", 100)  # 10 seconds at 10Hz
        self.feature_dim = config.get("feature_dim", 6)  # angle, velocity, acceleration, etc.
        self.num_classes = config.get("num_classes", 3)  # Normal, mild fatigue, severe fatigue
        
        # Steering data history
        self.steering_history = []
        self.max_history_size = 1000
        
        # Feature extraction parameters
        self.sampling_rate = config.get("sampling_rate", 10.0)  # Hz
        self.window_size = config.get("window_size", 50)  # 5 seconds
        
        # Anomaly detection thresholds
        self.entropy_threshold = config.get("entropy_threshold", 2.0)
        self.variance_threshold = config.get("variance_threshold", 100.0)
        self.velocity_threshold = config.get("velocity_threshold", 50.0)
        
    def build_model(self) -> keras.Model:
        """Build the steering analysis model"""
        # Input layer
        input_layer = keras.Input(
            shape=(self.sequence_length, self.feature_dim),
            name="steering_sequence"
        )
        
        # LSTM layers for temporal modeling
        lstm_output = self._build_lstm_layers(input_layer)
        
        # Attention mechanism
        attention_output = self._build_attention_layer(lstm_output)
        
        # Classification head
        classification_head = self._build_classification_head(attention_output)
        
        # Create model
        model = keras.Model(
            inputs=input_layer,
            outputs=classification_head,
            name="SteeringAnalyzer"
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
        
        return model
    
    def _build_lstm_layers(self, input_layer):
        """Build LSTM layers for temporal modeling"""
        # First LSTM layer
        lstm_1 = layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name="lstm_1"
        )(input_layer)
        
        # Second LSTM layer
        lstm_2 = layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            name="lstm_2"
        )(lstm_1)
        
        return lstm_2
    
    def _build_attention_layer(self, lstm_output):
        """Build attention mechanism"""
        # Attention weights
        attention_weights = layers.Dense(1, activation="tanh", name="attention_weights")(lstm_output)
        attention_weights = layers.Softmax(axis=1, name="attention_softmax")(attention_weights)
        
        # Apply attention
        attention_output = layers.Multiply(name="attention_apply")([lstm_output, attention_weights])
        attention_output = layers.GlobalAveragePooling1D(name="attention_pool")(attention_output)
        
        return attention_output
    
    def _build_classification_head(self, attention_output):
        """Build classification head"""
        # Dense layers
        x = layers.Dense(128, activation="relu", name="dense_1")(attention_output)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)
        
        # Output layer
        output = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="steering_classification"
        )(x)
        
        return output
    
    def extract_steering_features(self, steering_data: Dict) -> np.ndarray:
        """Extract comprehensive steering features"""
        try:
            # Basic features
            angle = steering_data.get("angle", 0.0)
            velocity = steering_data.get("velocity", 0.0)
            speed = steering_data.get("speed", 0.0)
            
            # Calculate additional features
            acceleration = self._calculate_acceleration(velocity)
            jerk = self._calculate_jerk(acceleration)
            steering_entropy = self._calculate_steering_entropy()
            steering_variance = self._calculate_steering_variance()
            
            # Create feature vector
            features = np.array([
                angle,
                velocity,
                acceleration,
                jerk,
                steering_entropy,
                steering_variance
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting steering features: {e}")
            return np.zeros(self.feature_dim)
    
    def _calculate_acceleration(self, velocity: float) -> float:
        """Calculate steering acceleration"""
        if len(self.steering_history) < 2:
            return 0.0
        
        # Get previous velocity
        prev_velocity = self.steering_history[-1].get("velocity", 0.0)
        prev_time = self.steering_history[-1].get("timestamp", 0.0)
        current_time = time.time()
        
        if current_time - prev_time > 0:
            acceleration = (velocity - prev_velocity) / (current_time - prev_time)
            return acceleration
        
        return 0.0
    
    def _calculate_jerk(self, acceleration: float) -> float:
        """Calculate steering jerk (rate of change of acceleration)"""
        if len(self.steering_history) < 2:
            return 0.0
        
        # Get previous acceleration
        prev_data = self.steering_history[-1]
        prev_acceleration = prev_data.get("acceleration", 0.0)
        prev_time = prev_data.get("timestamp", 0.0)
        current_time = time.time()
        
        if current_time - prev_time > 0:
            jerk = (acceleration - prev_acceleration) / (current_time - prev_time)
            return jerk
        
        return 0.0
    
    def _calculate_steering_entropy(self) -> float:
        """Calculate steering entropy as measure of irregularity"""
        if len(self.steering_history) < 10:
            return 0.0
        
        # Get recent steering angles
        recent_angles = [data["angle"] for data in self.steering_history[-20:]]
        
        try:
            # Calculate entropy using histogram
            hist, _ = np.histogram(recent_angles, bins=10)
            hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            hist_normalized = hist_normalized + 1e-10  # Avoid log(0)
            
            return float(entropy(hist_normalized))
        except:
            return 0.0
    
    def _calculate_steering_variance(self) -> float:
        """Calculate steering variance"""
        if len(self.steering_history) < 5:
            return 0.0
        
        # Get recent steering angles
        recent_angles = [data["angle"] for data in self.steering_history[-10:]]
        
        return float(np.var(recent_angles))
    
    def detect_anomalies(self, steering_data: Dict) -> Dict:
        """Detect steering anomalies"""
        try:
            # Extract features
            features = self.extract_steering_features(steering_data)
            
            # Check for anomalies
            anomalies = {
                "high_entropy": features[4] > self.entropy_threshold,
                "high_variance": features[5] > self.variance_threshold,
                "high_velocity": abs(features[1]) > self.velocity_threshold,
                "sudden_changes": self._detect_sudden_changes(),
                "oscillation": self._detect_oscillation()
            }
            
            # Calculate anomaly score
            anomaly_score = sum(anomalies.values()) / len(anomalies)
            
            return {
                "anomalies": anomalies,
                "anomaly_score": anomaly_score,
                "features": features.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {
                "anomalies": {},
                "anomaly_score": 0.0,
                "features": []
            }
    
    def _detect_sudden_changes(self) -> bool:
        """Detect sudden steering changes"""
        if len(self.steering_history) < 3:
            return False
        
        # Get recent angles
        recent_angles = [data["angle"] for data in self.steering_history[-3:]]
        
        # Calculate differences
        differences = [abs(recent_angles[i] - recent_angles[i-1]) for i in range(1, len(recent_angles))]
        
        # Check for sudden changes
        return any(diff > 30.0 for diff in differences)  # 30 degrees threshold
    
    def _detect_oscillation(self) -> bool:
        """Detect steering oscillation patterns"""
        if len(self.steering_history) < 10:
            return False
        
        # Get recent angles
        recent_angles = [data["angle"] for data in self.steering_history[-10:]]
        
        # Calculate frequency domain features
        try:
            # FFT analysis
            fft = np.fft.fft(recent_angles)
            frequencies = np.fft.fftfreq(len(recent_angles))
            
            # Check for high-frequency components
            high_freq_power = np.sum(np.abs(fft[frequencies > 0.1])**2)
            total_power = np.sum(np.abs(fft)**2)
            
            if total_power > 0:
                high_freq_ratio = high_freq_power / total_power
                return high_freq_ratio > 0.3  # 30% threshold
            else:
                return False
                
        except:
            return False
    
    def create_sequence(self, steering_data: List[Dict]) -> np.ndarray:
        """Create sequence from steering data for model input"""
        if len(steering_data) < self.sequence_length:
            # Pad with zeros if not enough data
            padded_data = steering_data + [{"angle": 0.0, "velocity": 0.0, "speed": 0.0, "timestamp": 0.0} 
                                         for _ in range(self.sequence_length - len(steering_data))]
        else:
            # Take the last sequence_length data points
            padded_data = steering_data[-self.sequence_length:]
        
        # Extract features for each data point
        sequences = []
        for data in padded_data:
            features = self.extract_steering_features(data)
            sequences.append(features)
        
        # Convert to numpy array
        sequence = np.array(sequences)
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)
        
        return sequence
    
    def predict_fatigue(self, steering_data: List[Dict]) -> Dict:
        """Predict fatigue level from steering data"""
        if not self.model:
            logger.error("Model not loaded")
            return {"fatigue_level": 0, "confidence": 0.0, "anomalies": {}}
        
        try:
            # Create sequence
            sequence = self.create_sequence(steering_data)
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Detect anomalies
            anomalies = self.detect_anomalies(steering_data[-1]) if steering_data else {}
            
            return {
                "fatigue_level": int(predicted_class),
                "confidence": confidence,
                "probabilities": prediction[0].tolist(),
                "anomalies": anomalies
            }
            
        except Exception as e:
            logger.error(f"Error predicting fatigue: {e}")
            return {"fatigue_level": 0, "confidence": 0.0, "anomalies": {}}
    
    def update_history(self, steering_data: Dict):
        """Update steering history"""
        # Add timestamp if not present
        if "timestamp" not in steering_data:
            steering_data["timestamp"] = time.time()
        
        # Add to history
        self.steering_history.append(steering_data)
        
        # Keep only recent history
        if len(self.steering_history) > self.max_history_size:
            self.steering_history = self.steering_history[-self.max_history_size:]
    
    def get_steering_statistics(self) -> Dict:
        """Get steering statistics"""
        if not self.steering_history:
            return {}
        
        # Extract recent data
        recent_data = self.steering_history[-100:]  # Last 100 readings
        
        angles = [data["angle"] for data in recent_data]
        velocities = [data["velocity"] for data in recent_data]
        
        return {
            "mean_angle": float(np.mean(angles)),
            "std_angle": float(np.std(angles)),
            "mean_velocity": float(np.mean(velocities)),
            "std_velocity": float(np.std(velocities)),
            "data_points": len(recent_data),
            "entropy": self._calculate_steering_entropy(),
            "variance": self._calculate_steering_variance()
        }
    
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
