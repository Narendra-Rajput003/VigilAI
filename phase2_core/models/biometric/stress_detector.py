"""
Biometric Stress Detection Model
Advanced stress detection using HRV, EDA, and other biometric signals
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

class BiometricStressDetector:
    """Advanced stress detection using biometric signals"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Model parameters
        self.sequence_length = config.get("sequence_length", 60)  # 1 minute at 1Hz
        self.feature_dim = config.get("feature_dim", 8)  # HR, HRV, EDA, etc.
        self.num_classes = config.get("num_classes", 3)  # No stress, mild, severe
        
        # Biometric data history
        self.biometric_history = []
        self.max_history_size = 1000
        
        # Feature extraction parameters
        self.sampling_rate = config.get("sampling_rate", 1.0)  # Hz
        self.window_size = config.get("window_size", 30)  # 30 seconds
        
        # Stress detection thresholds
        self.hrv_threshold = config.get("hrv_threshold", 30.0)  # ms
        self.eda_threshold = config.get("eda_threshold", 5.0)  # μS
        self.hr_variability_threshold = config.get("hr_variability_threshold", 20.0)  # BPM
        
    def build_model(self) -> keras.Model:
        """Build the biometric stress detection model"""
        # Input layer
        input_layer = keras.Input(
            shape=(self.sequence_length, self.feature_dim),
            name="biometric_sequence"
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
            name="BiometricStressDetector"
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
            name="stress_classification"
        )(x)
        
        return output
    
    def extract_biometric_features(self, biometric_data: Dict) -> np.ndarray:
        """Extract comprehensive biometric features"""
        try:
            # Basic features
            heart_rate = biometric_data.get("heart_rate", 0.0)
            hrv = biometric_data.get("hrv", 0.0)
            eda = biometric_data.get("eda", 0.0)
            temperature = biometric_data.get("temperature", 0.0)
            
            # Calculate additional features
            hr_variability = self._calculate_hr_variability(heart_rate)
            hrv_rmssd = self._calculate_hrv_rmssd()
            eda_derivative = self._calculate_eda_derivative(eda)
            stress_index = self._calculate_stress_index(hrv, eda)
            
            # Create feature vector
            features = np.array([
                heart_rate,
                hrv,
                eda,
                temperature,
                hr_variability,
                hrv_rmssd,
                eda_derivative,
                stress_index
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting biometric features: {e}")
            return np.zeros(self.feature_dim)
    
    def _calculate_hr_variability(self, heart_rate: float) -> float:
        """Calculate heart rate variability"""
        if len(self.biometric_history) < 5:
            return 0.0
        
        # Get recent heart rates
        recent_hr = [data["heart_rate"] for data in self.biometric_history[-10:]]
        recent_hr.append(heart_rate)
        
        # Calculate standard deviation
        return float(np.std(recent_hr))
    
    def _calculate_hrv_rmssd(self) -> float:
        """Calculate HRV RMSSD (Root Mean Square of Successive Differences)"""
        if len(self.biometric_history) < 3:
            return 0.0
        
        # Get recent HRV values
        recent_hrv = [data["hrv"] for data in self.biometric_history[-10:]]
        
        if len(recent_hrv) < 2:
            return 0.0
        
        # Calculate RMSSD
        differences = [recent_hrv[i] - recent_hrv[i-1] for i in range(1, len(recent_hrv))]
        rmssd = np.sqrt(np.mean([d**2 for d in differences]))
        
        return float(rmssd)
    
    def _calculate_eda_derivative(self, eda: float) -> float:
        """Calculate EDA derivative (rate of change)"""
        if len(self.biometric_history) < 2:
            return 0.0
        
        # Get previous EDA
        prev_eda = self.biometric_history[-1].get("eda", 0.0)
        prev_time = self.biometric_history[-1].get("timestamp", 0.0)
        current_time = time.time()
        
        if current_time - prev_time > 0:
            derivative = (eda - prev_eda) / (current_time - prev_time)
            return derivative
        
        return 0.0
    
    def _calculate_stress_index(self, hrv: float, eda: float) -> float:
        """Calculate composite stress index"""
        # Normalize HRV (lower HRV = higher stress)
        hrv_normalized = max(0.0, min(1.0, (100.0 - hrv) / 100.0))
        
        # Normalize EDA (higher EDA = higher stress)
        eda_normalized = max(0.0, min(1.0, eda / 10.0))
        
        # Combine with weights
        stress_index = (hrv_normalized * 0.6) + (eda_normalized * 0.4)
        
        return stress_index
    
    def detect_stress_patterns(self, biometric_data: Dict) -> Dict:
        """Detect stress patterns in biometric data"""
        try:
            # Extract features
            features = self.extract_biometric_features(biometric_data)
            
            # Check for stress indicators
            stress_indicators = {
                "low_hrv": features[1] < self.hrv_threshold,
                "high_eda": features[2] > self.eda_threshold,
                "high_hr_variability": features[4] > self.hr_variability_threshold,
                "high_stress_index": features[7] > 0.7,
                "sudden_changes": self._detect_sudden_biometric_changes(),
                "trend_analysis": self._analyze_stress_trend()
            }
            
            # Calculate stress score
            stress_score = sum(stress_indicators.values()) / len(stress_indicators)
            
            return {
                "stress_indicators": stress_indicators,
                "stress_score": stress_score,
                "features": features.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error detecting stress patterns: {e}")
            return {
                "stress_indicators": {},
                "stress_score": 0.0,
                "features": []
            }
    
    def _detect_sudden_biometric_changes(self) -> bool:
        """Detect sudden changes in biometric signals"""
        if len(self.biometric_history) < 3:
            return False
        
        # Get recent data
        recent_data = self.biometric_history[-3:]
        
        # Check for sudden changes in HR
        hr_changes = []
        for i in range(1, len(recent_data)):
            hr_change = abs(recent_data[i]["heart_rate"] - recent_data[i-1]["heart_rate"])
            hr_changes.append(hr_change)
        
        # Check for sudden changes in EDA
        eda_changes = []
        for i in range(1, len(recent_data)):
            eda_change = abs(recent_data[i]["eda"] - recent_data[i-1]["eda"])
            eda_changes.append(eda_change)
        
        # Thresholds for sudden changes
        hr_threshold = 20.0  # BPM
        eda_threshold = 2.0   # μS
        
        return (any(change > hr_threshold for change in hr_changes) or
                any(change > eda_threshold for change in eda_changes))
    
    def _analyze_stress_trend(self) -> bool:
        """Analyze stress trend over time"""
        if len(self.biometric_history) < 10:
            return False
        
        # Get recent stress indices
        recent_stress = [data.get("stress_index", 0.0) for data in self.biometric_history[-10:]]
        
        # Calculate trend
        x = np.arange(len(recent_stress))
        slope, _ = np.polyfit(x, recent_stress, 1)
        
        # Positive slope indicates increasing stress
        return slope > 0.01
    
    def create_sequence(self, biometric_data: List[Dict]) -> np.ndarray:
        """Create sequence from biometric data for model input"""
        if len(biometric_data) < self.sequence_length:
            # Pad with zeros if not enough data
            padded_data = biometric_data + [{"heart_rate": 0.0, "hrv": 0.0, "eda": 0.0, "temperature": 0.0, "timestamp": 0.0} 
                                         for _ in range(self.sequence_length - len(biometric_data))]
        else:
            # Take the last sequence_length data points
            padded_data = biometric_data[-self.sequence_length:]
        
        # Extract features for each data point
        sequences = []
        for data in padded_data:
            features = self.extract_biometric_features(data)
            sequences.append(features)
        
        # Convert to numpy array
        sequence = np.array(sequences)
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)
        
        return sequence
    
    def predict_stress(self, biometric_data: List[Dict]) -> Dict:
        """Predict stress level from biometric data"""
        if not self.model:
            logger.error("Model not loaded")
            return {"stress_level": 0, "confidence": 0.0, "patterns": {}}
        
        try:
            # Create sequence
            sequence = self.create_sequence(biometric_data)
            
            # Make prediction
            prediction = self.model.predict(sequence, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            # Detect stress patterns
            patterns = self.detect_stress_patterns(biometric_data[-1]) if biometric_data else {}
            
            return {
                "stress_level": int(predicted_class),
                "confidence": confidence,
                "probabilities": prediction[0].tolist(),
                "patterns": patterns
            }
            
        except Exception as e:
            logger.error(f"Error predicting stress: {e}")
            return {"stress_level": 0, "confidence": 0.0, "patterns": {}}
    
    def update_history(self, biometric_data: Dict):
        """Update biometric history"""
        # Add timestamp if not present
        if "timestamp" not in biometric_data:
            biometric_data["timestamp"] = time.time()
        
        # Add to history
        self.biometric_history.append(biometric_data)
        
        # Keep only recent history
        if len(self.biometric_history) > self.max_history_size:
            self.biometric_history = self.biometric_history[-self.max_history_size:]
    
    def get_biometric_statistics(self) -> Dict:
        """Get biometric statistics"""
        if not self.biometric_history:
            return {}
        
        # Extract recent data
        recent_data = self.biometric_history[-100:]  # Last 100 readings
        
        heart_rates = [data["heart_rate"] for data in recent_data]
        hrv_values = [data["hrv"] for data in recent_data]
        eda_values = [data["eda"] for data in recent_data]
        
        return {
            "mean_heart_rate": float(np.mean(heart_rates)),
            "std_heart_rate": float(np.std(heart_rates)),
            "mean_hrv": float(np.mean(hrv_values)),
            "std_hrv": float(np.std(hrv_values)),
            "mean_eda": float(np.mean(eda_values)),
            "std_eda": float(np.std(eda_values)),
            "data_points": len(recent_data)
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
