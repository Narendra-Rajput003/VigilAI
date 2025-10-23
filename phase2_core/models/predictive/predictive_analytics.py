"""
Predictive Analytics Engine for VigilAI
Advanced AI-powered predictions for driver behavior, fatigue patterns, and risk assessment
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    """Result of a prediction"""
    prediction_type: str
    predicted_value: float
    confidence: float
    timestamp: datetime
    features_used: List[str]
    model_version: str
    prediction_horizon: int  # minutes ahead

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    risk_level: str  # low, medium, high, critical
    risk_score: float  # 0-1
    risk_factors: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: datetime

class PredictiveAnalyticsEngine:
    """Advanced predictive analytics engine for VigilAI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.is_initialized = False
        
        # Model configurations
        self.model_configs = {
            'fatigue_prediction': {
                'type': 'regression',
                'horizon_minutes': [5, 15, 30, 60],
                'features': ['eye_openness', 'yawn_frequency', 'head_movement', 'steering_variance']
            },
            'stress_prediction': {
                'type': 'regression', 
                'horizon_minutes': [5, 15, 30, 60],
                'features': ['hrv', 'eda', 'heart_rate', 'temperature']
            },
            'accident_risk': {
                'type': 'classification',
                'horizon_minutes': [5, 15, 30],
                'features': ['fatigue_level', 'stress_level', 'steering_irregularity', 'speed_variance']
            },
            'intervention_effectiveness': {
                'type': 'regression',
                'horizon_minutes': [1, 5, 15],
                'features': ['intervention_type', 'fatigue_before', 'stress_before', 'time_of_day']
            }
        }
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
    
    async def initialize(self) -> bool:
        """Initialize the predictive analytics engine"""
        try:
            logger.info("Initializing predictive analytics engine...")
            
            # Load or create models for each prediction type
            for model_name, config in self.model_configs.items():
                await self._initialize_model(model_name, config)
            
            self.is_initialized = True
            logger.info("Predictive analytics engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing predictive analytics: {e}")
            return False
    
    async def _initialize_model(self, model_name: str, config: Dict[str, Any]):
        """Initialize a specific model"""
        try:
            model_path = self.config.get('model_paths', {}).get(model_name)
            
            if model_path and self._model_exists(model_path):
                # Load existing model
                self.models[model_name] = joblib.load(model_path)
                self.scalers[model_name] = joblib.load(f"{model_path}_scaler")
                logger.info(f"Loaded existing model: {model_name}")
            else:
                # Create new model
                if config['type'] == 'regression':
                    self.models[model_name] = self._create_regression_model()
                else:
                    self.models[model_name] = self._create_classification_model()
                
                self.scalers[model_name] = StandardScaler()
                logger.info(f"Created new model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {e}")
            # Create fallback model
            self.models[model_name] = self._create_fallback_model()
            self.scalers[model_name] = StandardScaler()
    
    def _create_regression_model(self):
        """Create regression model ensemble"""
        from sklearn.ensemble import VotingRegressor
        
        models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42))
        ]
        
        return VotingRegressor(models)
    
    def _create_classification_model(self):
        """Create classification model ensemble"""
        from sklearn.ensemble import VotingClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]
        
        return VotingClassifier(models)
    
    def _create_fallback_model(self):
        """Create simple fallback model"""
        return RandomForestRegressor(n_estimators=10, random_state=42)
    
    def _model_exists(self, model_path: str) -> bool:
        """Check if model file exists"""
        import os
        return os.path.exists(model_path)
    
    async def predict_fatigue(self, historical_data: List[Dict], horizon_minutes: int = 15) -> PredictionResult:
        """Predict future fatigue levels"""
        try:
            if not self.is_initialized:
                return self._create_error_prediction("Engine not initialized")
            
            # Extract features
            features = self._extract_fatigue_features(historical_data)
            
            if len(features) < 10:  # Need minimum data
                return self._create_error_prediction("Insufficient historical data")
            
            # Prepare data for prediction
            X = self._prepare_prediction_data(features, 'fatigue_prediction')
            
            # Make prediction
            prediction = self.models['fatigue_prediction'].predict(X.reshape(1, -1))[0]
            confidence = self._calculate_prediction_confidence(X, 'fatigue_prediction')
            
            return PredictionResult(
                prediction_type='fatigue',
                predicted_value=float(prediction),
                confidence=confidence,
                timestamp=datetime.utcnow(),
                features_used=self.model_configs['fatigue_prediction']['features'],
                model_version='1.0.0',
                prediction_horizon=horizon_minutes
            )
            
        except Exception as e:
            logger.error(f"Error predicting fatigue: {e}")
            return self._create_error_prediction(str(e))
    
    async def predict_stress(self, historical_data: List[Dict], horizon_minutes: int = 15) -> PredictionResult:
        """Predict future stress levels"""
        try:
            if not self.is_initialized:
                return self._create_error_prediction("Engine not initialized")
            
            # Extract features
            features = self._extract_stress_features(historical_data)
            
            if len(features) < 10:
                return self._create_error_prediction("Insufficient historical data")
            
            # Prepare data for prediction
            X = self._prepare_prediction_data(features, 'stress_prediction')
            
            # Make prediction
            prediction = self.models['stress_prediction'].predict(X.reshape(1, -1))[0]
            confidence = self._calculate_prediction_confidence(X, 'stress_prediction')
            
            return PredictionResult(
                prediction_type='stress',
                predicted_value=float(prediction),
                confidence=confidence,
                timestamp=datetime.utcnow(),
                features_used=self.model_configs['stress_prediction']['features'],
                model_version='1.0.0',
                prediction_horizon=horizon_minutes
            )
            
        except Exception as e:
            logger.error(f"Error predicting stress: {e}")
            return self._create_error_prediction(str(e))
    
    async def assess_accident_risk(self, current_data: Dict, historical_data: List[Dict]) -> RiskAssessment:
        """Assess accident risk based on current and historical data"""
        try:
            if not self.is_initialized:
                return self._create_error_risk_assessment("Engine not initialized")
            
            # Extract risk features
            risk_features = self._extract_risk_features(current_data, historical_data)
            
            # Prepare data for prediction
            X = self._prepare_prediction_data(risk_features, 'accident_risk')
            
            # Predict risk probability
            risk_probability = self.models['accident_risk'].predict_proba(X.reshape(1, -1))[0][1]
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_probability)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(current_data, risk_probability)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(risk_level, risk_factors)
            
            return RiskAssessment(
                risk_level=risk_level,
                risk_score=float(risk_probability),
                risk_factors=risk_factors,
                recommendations=recommendations,
                confidence=0.85,  # Simplified confidence calculation
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error assessing accident risk: {e}")
            return self._create_error_risk_assessment(str(e))
    
    async def predict_intervention_effectiveness(self, intervention_data: Dict, user_profile: Dict) -> PredictionResult:
        """Predict effectiveness of interventions"""
        try:
            if not self.is_initialized:
                return self._create_error_prediction("Engine not initialized")
            
            # Extract intervention features
            features = self._extract_intervention_features(intervention_data, user_profile)
            
            # Prepare data for prediction
            X = self._prepare_prediction_data(features, 'intervention_effectiveness')
            
            # Make prediction
            prediction = self.models['intervention_effectiveness'].predict(X.reshape(1, -1))[0]
            confidence = self._calculate_prediction_confidence(X, 'intervention_effectiveness')
            
            return PredictionResult(
                prediction_type='intervention_effectiveness',
                predicted_value=float(prediction),
                confidence=confidence,
                timestamp=datetime.utcnow(),
                features_used=self.model_configs['intervention_effectiveness']['features'],
                model_version='1.0.0',
                prediction_horizon=5
            )
            
        except Exception as e:
            logger.error(f"Error predicting intervention effectiveness: {e}")
            return self._create_error_prediction(str(e))
    
    def _extract_fatigue_features(self, historical_data: List[Dict]) -> List[float]:
        """Extract features for fatigue prediction"""
        features = []
        
        for data in historical_data[-20:]:  # Last 20 data points
            video_data = data.get('video', {})
            steering_data = data.get('steering', {})
            
            # Eye openness (average over time)
            eye_openness = video_data.get('eye_openness', 0.5)
            
            # Yawn frequency (count per minute)
            yawn_frequency = video_data.get('yawn_frequency', 0)
            
            # Head movement variance
            head_movement = video_data.get('head_movement_variance', 0)
            
            # Steering variance
            steering_variance = steering_data.get('variance', 0)
            
            features.extend([eye_openness, yawn_frequency, head_movement, steering_variance])
        
        return features
    
    def _extract_stress_features(self, historical_data: List[Dict]) -> List[float]:
        """Extract features for stress prediction"""
        features = []
        
        for data in historical_data[-20:]:
            biometric_data = data.get('biometric', {})
            
            # HRV (Heart Rate Variability)
            hrv = biometric_data.get('hrv', 0.05)
            
            # EDA (Electrodermal Activity)
            eda = biometric_data.get('eda', 2.0)
            
            # Heart rate
            heart_rate = biometric_data.get('heart_rate', 70)
            
            # Temperature
            temperature = biometric_data.get('temperature', 36.5)
            
            features.extend([hrv, eda, heart_rate, temperature])
        
        return features
    
    def _extract_risk_features(self, current_data: Dict, historical_data: List[Dict]) -> List[float]:
        """Extract features for risk assessment"""
        features = []
        
        # Current fatigue and stress levels
        current_fatigue = current_data.get('fatigue_level', 0)
        current_stress = current_data.get('stress_level', 0)
        
        # Steering irregularity (variance in steering patterns)
        steering_data = current_data.get('steering', {})
        steering_irregularity = steering_data.get('irregularity_score', 0)
        
        # Speed variance
        speed_variance = current_data.get('speed_variance', 0)
        
        # Historical patterns
        fatigue_trend = self._calculate_trend([d.get('fatigue_level', 0) for d in historical_data[-10:]])
        stress_trend = self._calculate_trend([d.get('stress_level', 0) for d in historical_data[-10:]])
        
        features = [
            current_fatigue, current_stress, steering_irregularity, 
            speed_variance, fatigue_trend, stress_trend
        ]
        
        return features
    
    def _extract_intervention_features(self, intervention_data: Dict, user_profile: Dict) -> List[float]:
        """Extract features for intervention effectiveness prediction"""
        features = []
        
        # Intervention type (encoded)
        intervention_type = intervention_data.get('type', 'audio')
        type_encoding = {'audio': 0, 'visual': 1, 'haptic': 2, 'combined': 3}.get(intervention_type, 0)
        
        # Current state before intervention
        fatigue_before = intervention_data.get('fatigue_before', 0)
        stress_before = intervention_data.get('stress_before', 0)
        
        # Time of day (circadian rhythm factor)
        hour = datetime.now().hour
        time_factor = np.sin(2 * np.pi * hour / 24)  # Circadian rhythm
        
        # User profile factors
        age = user_profile.get('age', 35)
        experience = user_profile.get('driving_experience', 10)
        
        features = [type_encoding, fatigue_before, stress_before, time_factor, age, experience]
        
        return features
    
    def _prepare_prediction_data(self, features: List[float], model_name: str) -> np.ndarray:
        """Prepare data for model prediction"""
        # Pad or truncate to expected feature count
        expected_features = len(self.model_configs[model_name]['features']) * 5  # Simplified
        
        if len(features) < expected_features:
            features.extend([0] * (expected_features - len(features)))
        elif len(features) > expected_features:
            features = features[:expected_features]
        
        # Scale features
        X = np.array(features).reshape(1, -1)
        if hasattr(self.scalers[model_name], 'fit'):
            X = self.scalers[model_name].transform(X)
        
        return X
    
    def _calculate_prediction_confidence(self, X: np.ndarray, model_name: str) -> float:
        """Calculate prediction confidence"""
        # Simplified confidence calculation
        # In practice, this would use prediction intervals or ensemble variance
        base_confidence = 0.8
        
        # Adjust based on feature quality
        feature_quality = np.mean(np.abs(X))
        quality_factor = min(1.0, feature_quality / 2.0)
        
        return base_confidence * quality_factor
    
    def _determine_risk_level(self, risk_probability: float) -> str:
        """Determine risk level from probability"""
        if risk_probability >= self.risk_thresholds['critical']:
            return 'critical'
        elif risk_probability >= self.risk_thresholds['high']:
            return 'high'
        elif risk_probability >= self.risk_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _identify_risk_factors(self, current_data: Dict, risk_probability: float) -> List[str]:
        """Identify contributing risk factors"""
        risk_factors = []
        
        if current_data.get('fatigue_level', 0) > 0.7:
            risk_factors.append('High fatigue level')
        
        if current_data.get('stress_level', 0) > 0.7:
            risk_factors.append('High stress level')
        
        if current_data.get('steering', {}).get('irregularity_score', 0) > 0.5:
            risk_factors.append('Irregular steering patterns')
        
        if current_data.get('speed_variance', 0) > 0.3:
            risk_factors.append('Inconsistent speed')
        
        if not risk_factors:
            risk_factors.append('Normal driving conditions')
        
        return risk_factors
    
    def _generate_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        if risk_level in ['high', 'critical']:
            recommendations.extend([
                'Take a break immediately',
                'Find a safe place to rest',
                'Consider calling for assistance'
            ])
        elif risk_level == 'medium':
            recommendations.extend([
                'Take a short break soon',
                'Stay alert and focused',
                'Consider reducing driving time'
            ])
        else:
            recommendations.extend([
                'Continue monitoring',
                'Maintain current driving habits'
            ])
        
        # Add specific recommendations based on risk factors
        if 'High fatigue level' in risk_factors:
            recommendations.append('Get adequate sleep before driving')
        
        if 'High stress level' in risk_factors:
            recommendations.append('Practice stress management techniques')
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = increasing, negative = decreasing)"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _create_error_prediction(self, error_message: str) -> PredictionResult:
        """Create error prediction result"""
        return PredictionResult(
            prediction_type='error',
            predicted_value=0.0,
            confidence=0.0,
            timestamp=datetime.utcnow(),
            features_used=[],
            model_version='error',
            prediction_horizon=0
        )
    
    def _create_error_risk_assessment(self, error_message: str) -> RiskAssessment:
        """Create error risk assessment"""
        return RiskAssessment(
            risk_level='unknown',
            risk_score=0.0,
            risk_factors=['System error'],
            recommendations=['Contact support'],
            confidence=0.0,
            timestamp=datetime.utcnow()
        )
    
    async def train_models(self, training_data: List[Dict]) -> Dict[str, float]:
        """Train models with new data"""
        try:
            results = {}
            
            for model_name in self.model_configs.keys():
                # Extract training features and labels
                X, y = self._prepare_training_data(training_data, model_name)
                
                if len(X) < 10:  # Need minimum training data
                    results[model_name] = 0.0
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
                
                # Train model
                self.models[model_name].fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = self.models[model_name].predict(X_test_scaled)
                score = r2_score(y_test, y_pred) if self.model_configs[model_name]['type'] == 'regression' else self.models[model_name].score(X_test_scaled, y_test)
                
                results[model_name] = float(score)
                
                # Save models
                model_path = self.config.get('model_paths', {}).get(model_name)
                if model_path:
                    joblib.dump(self.models[model_name], model_path)
                    joblib.dump(self.scalers[model_name], f"{model_path}_scaler")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def _prepare_training_data(self, training_data: List[Dict], model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for a specific model"""
        X = []
        y = []
        
        for i, data in enumerate(training_data):
            # Extract features based on model type
            if model_name == 'fatigue_prediction':
                features = self._extract_fatigue_features(training_data[max(0, i-20):i+1])
                target = data.get('fatigue_level', 0)
            elif model_name == 'stress_prediction':
                features = self._extract_stress_features(training_data[max(0, i-20):i+1])
                target = data.get('stress_level', 0)
            elif model_name == 'accident_risk':
                features = self._extract_risk_features(data, training_data[max(0, i-10):i])
                target = 1 if data.get('accident_occurred', False) else 0
            elif model_name == 'intervention_effectiveness':
                features = self._extract_intervention_features(data, data.get('user_profile', {}))
                target = data.get('intervention_effectiveness', 0)
            else:
                continue
            
            if len(features) > 0:
                X.append(features)
                y.append(target)
        
        return np.array(X), np.array(y)
    
    def get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all models"""
        performance = {}
        
        for model_name in self.model_configs.keys():
            if model_name in self.models:
                # This would be calculated from validation data
                performance[model_name] = {
                    'accuracy': 0.85,  # Placeholder
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85
                }
        
        return performance
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.models.clear()
            self.scalers.clear()
            self.is_initialized = False
            logger.info("Predictive analytics engine cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
