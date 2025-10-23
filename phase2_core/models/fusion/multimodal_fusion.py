"""
Multi-modal Fusion Model
Transformer-based fusion of video, steering, and biometric data
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)

class MultiModalFusion:
    """Transformer-based multi-modal fusion for VigilAI"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Model parameters
        self.video_dim = config.get("video_dim", 128)
        self.steering_dim = config.get("steering_dim", 64)
        self.biometric_dim = config.get("biometric_dim", 64)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        self.num_classes = config.get("num_classes", 3)  # No fatigue, mild, severe
        
        # Fusion weights
        self.video_weight = config.get("video_weight", 0.5)
        self.steering_weight = config.get("steering_weight", 0.3)
        self.biometric_weight = config.get("biometric_weight", 0.2)
        
        # Attention mechanisms
        self.attention_weights = None
        
    def build_model(self) -> keras.Model:
        """Build the multi-modal fusion model"""
        # Input layers
        video_input = keras.Input(shape=(self.video_dim,), name="video_features")
        steering_input = keras.Input(shape=(self.steering_dim,), name="steering_features")
        biometric_input = keras.Input(shape=(self.biometric_dim,), name="biometric_features")
        
        # Project inputs to common dimension
        video_projected = self._project_input(video_input, self.video_dim, self.hidden_dim, "video_projection")
        steering_projected = self._project_input(steering_input, self.steering_dim, self.hidden_dim, "steering_projection")
        biometric_projected = self._project_input(biometric_input, self.biometric_dim, self.hidden_dim, "biometric_projection")
        
        # Create multi-modal sequence
        multimodal_sequence = self._create_multimodal_sequence(
            video_projected, steering_projected, biometric_projected
        )
        
        # Transformer encoder
        transformer_output = self._build_transformer_encoder(multimodal_sequence)
        
        # Fusion layer
        fusion_output = self._build_fusion_layer(transformer_output)
        
        # Classification head
        classification_head = self._build_classification_head(fusion_output)
        
        # Create model
        model = keras.Model(
            inputs=[video_input, steering_input, biometric_input],
            outputs=classification_head,
            name="MultiModalFusion"
        )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
        
        return model
    
    def _project_input(self, input_tensor, input_dim, output_dim, name):
        """Project input to common dimension"""
        return layers.Dense(
            output_dim,
            activation="relu",
            name=f"{name}_dense"
        )(input_tensor)
    
    def _create_multimodal_sequence(self, video_projected, steering_projected, biometric_projected):
        """Create multi-modal sequence for transformer"""
        # Stack projections
        sequence = layers.Concatenate(axis=0, name="multimodal_concat")([
            video_projected, steering_projected, biometric_projected
        ])
        
        # Add positional encoding
        sequence = self._add_positional_encoding(sequence)
        
        return sequence
    
    def _add_positional_encoding(self, sequence):
        """Add positional encoding to sequence"""
        seq_len = tf.shape(sequence)[0]
        hidden_dim = tf.shape(sequence)[1]
        
        # Create positional encoding
        pos_encoding = self._get_positional_encoding(seq_len, hidden_dim)
        
        # Add to sequence
        return layers.Add(name="add_positional_encoding")([sequence, pos_encoding])
    
    def _get_positional_encoding(self, seq_len, hidden_dim):
        """Get positional encoding matrix"""
        pos_encoding = np.zeros((seq_len, hidden_dim))
        
        for pos in range(seq_len):
            for i in range(0, hidden_dim, 2):
                pos_encoding[pos, i] = math.sin(pos / (10000 ** (2 * i / hidden_dim)))
                if i + 1 < hidden_dim:
                    pos_encoding[pos, i + 1] = math.cos(pos / (10000 ** (2 * (i + 1) / hidden_dim)))
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def _build_transformer_encoder(self, sequence):
        """Build transformer encoder layers"""
        x = sequence
        
        for i in range(self.num_layers):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.hidden_dim // self.num_heads,
                name=f"attention_{i}"
            )(x, x)
            
            # Add & Norm
            x = layers.Add(name=f"add_{i}")([x, attention_output])
            x = layers.LayerNormalization(name=f"norm_1_{i}")(x)
            
            # Feed forward
            ff_output = layers.Dense(
                self.hidden_dim * 4,
                activation="relu",
                name=f"ff_dense_1_{i}"
            )(x)
            ff_output = layers.Dense(
                self.hidden_dim,
                name=f"ff_dense_2_{i}"
            )(ff_output)
            
            # Add & Norm
            x = layers.Add(name=f"add_ff_{i}")([x, ff_output])
            x = layers.LayerNormalization(name=f"norm_2_{i}")(x)
        
        return x
    
    def _build_fusion_layer(self, transformer_output):
        """Build fusion layer"""
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D(name="global_avg_pool")(transformer_output)
        
        # Dense layers
        x = layers.Dense(512, activation="relu", name="fusion_dense_1")(pooled)
        x = layers.Dropout(0.3, name="fusion_dropout_1")(x)
        
        x = layers.Dense(256, activation="relu", name="fusion_dense_2")(x)
        x = layers.Dropout(0.3, name="fusion_dropout_2")(x)
        
        return x
    
    def _build_classification_head(self, fusion_output):
        """Build classification head"""
        # Dense layers
        x = layers.Dense(128, activation="relu", name="classifier_dense_1")(fusion_output)
        x = layers.Dropout(0.3, name="classifier_dropout_1")(x)
        
        x = layers.Dense(64, activation="relu", name="classifier_dense_2")(x)
        x = layers.Dropout(0.3, name="classifier_dropout_2")(x)
        
        # Output layer
        output = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="fatigue_classification"
        )(x)
        
        return output
    
    def fuse_predictions(self, video_pred: Dict, steering_pred: Dict, biometric_pred: Dict) -> Dict:
        """Fuse predictions from individual modalities"""
        try:
            # Extract confidence scores
            video_confidence = video_pred.get("confidence", 0.0)
            steering_confidence = steering_pred.get("confidence", 0.0)
            biometric_confidence = biometric_pred.get("confidence", 0.0)
            
            # Extract fatigue levels
            video_fatigue = video_pred.get("fatigue_level", 0)
            steering_fatigue = steering_pred.get("fatigue_level", 0)
            biometric_stress = biometric_pred.get("stress_level", 0)
            
            # Weighted fusion
            total_weight = self.video_weight + self.steering_weight + self.biometric_weight
            
            # Normalize weights
            video_weight_norm = self.video_weight / total_weight
            steering_weight_norm = self.steering_weight / total_weight
            biometric_weight_norm = self.biometric_weight / total_weight
            
            # Calculate weighted fatigue score
            weighted_fatigue = (
                video_fatigue * video_weight_norm +
                steering_fatigue * steering_weight_norm +
                biometric_stress * biometric_weight_norm
            )
            
            # Calculate weighted confidence
            weighted_confidence = (
                video_confidence * video_weight_norm +
                steering_confidence * steering_weight_norm +
                biometric_confidence * biometric_weight_norm
            )
            
            # Determine final fatigue level
            if weighted_fatigue < 0.5:
                final_fatigue_level = 0  # No fatigue
            elif weighted_fatigue < 1.5:
                final_fatigue_level = 1  # Mild fatigue
            else:
                final_fatigue_level = 2  # Severe fatigue
            
            return {
                "fatigue_level": final_fatigue_level,
                "confidence": weighted_confidence,
                "weighted_fatigue_score": weighted_fatigue,
                "modality_contributions": {
                    "video": {
                        "fatigue_level": video_fatigue,
                        "confidence": video_confidence,
                        "weight": video_weight_norm
                    },
                    "steering": {
                        "fatigue_level": steering_fatigue,
                        "confidence": steering_confidence,
                        "weight": steering_weight_norm
                    },
                    "biometric": {
                        "stress_level": biometric_stress,
                        "confidence": biometric_confidence,
                        "weight": biometric_weight_norm
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error fusing predictions: {e}")
            return {
                "fatigue_level": 0,
                "confidence": 0.0,
                "weighted_fatigue_score": 0.0,
                "modality_contributions": {}
            }
    
    def predict_fatigue(self, video_features: np.ndarray, steering_features: np.ndarray, biometric_features: np.ndarray) -> Dict:
        """Predict fatigue using multi-modal fusion"""
        if not self.model:
            logger.error("Model not loaded")
            return {"fatigue_level": 0, "confidence": 0.0}
        
        try:
            # Prepare inputs
            video_input = np.expand_dims(video_features, axis=0)
            steering_input = np.expand_dims(steering_features, axis=0)
            biometric_input = np.expand_dims(biometric_features, axis=0)
            
            # Make prediction
            prediction = self.model.predict(
                [video_input, steering_input, biometric_input],
                verbose=0
            )
            
            # Get predicted class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            return {
                "fatigue_level": int(predicted_class),
                "confidence": confidence,
                "probabilities": prediction[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Error predicting fatigue: {e}")
            return {"fatigue_level": 0, "confidence": 0.0}
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get attention weights from the model"""
        return self.attention_weights
    
    def update_fusion_weights(self, video_weight: float, steering_weight: float, biometric_weight: float):
        """Update fusion weights based on performance"""
        total_weight = video_weight + steering_weight + biometric_weight
        
        if total_weight > 0:
            self.video_weight = video_weight / total_weight
            self.steering_weight = steering_weight / total_weight
            self.biometric_weight = biometric_weight / total_weight
            
            logger.info(f"Updated fusion weights - Video: {self.video_weight:.2f}, "
                       f"Steering: {self.steering_weight:.2f}, Biometric: {self.biometric_weight:.2f}")
    
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
