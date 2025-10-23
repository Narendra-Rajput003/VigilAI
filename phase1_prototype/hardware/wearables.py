"""
Wearables Manager for VigilAI
Handles biometric data collection from wearables
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List
import json
import random

logger = logging.getLogger(__name__)

class WearablesManager:
    """Manages wearable device integration for biometric data"""
    
    def __init__(self, config):
        self.config = config
        self.is_initialized = False
        
        # Wearable device settings
        self.device_type = config.get("device_type", "mock")  # mock, fitbit, apple_watch, etc.
        self.connection_timeout = config.get("connection_timeout", 10.0)
        self.data_interval = config.get("data_interval", 1.0)  # 1 Hz
        
        # Biometric data history
        self.biometric_history: List[Dict] = []
        self.max_history_size = 300  # 5 minutes at 1 Hz
        
        # Mock data settings (for testing without real devices)
        self.mock_data_enabled = config.get("mock_data", True)
        self.mock_baseline_hr = 70.0
        self.mock_baseline_hrv = 50.0
        self.mock_baseline_eda = 2.0
        
        # Data collection state
        self.last_collection_time = 0
        self.connection_attempts = 0
        self.max_connection_attempts = 3
        
    async def initialize(self) -> bool:
        """Initialize wearable device connection"""
        try:
            logger.info(f"Initializing wearables manager (type: {self.device_type})...")
            
            if self.device_type == "mock":
                return await self._initialize_mock_device()
            elif self.device_type == "fitbit":
                return await self._initialize_fitbit()
            elif self.device_type == "apple_watch":
                return await self._initialize_apple_watch()
            else:
                logger.warning(f"Unknown device type: {self.device_type}, using mock")
                return await self._initialize_mock_device()
                
        except Exception as e:
            logger.error(f"Error initializing wearables: {e}")
            return False
    
    async def _initialize_mock_device(self) -> bool:
        """Initialize mock wearable device for testing"""
        logger.info("Initializing mock wearable device...")
        
        # Simulate connection delay
        await asyncio.sleep(1.0)
        
        self.is_initialized = True
        logger.info("Mock wearable device initialized")
        return True
    
    async def _initialize_fitbit(self) -> bool:
        """Initialize Fitbit device connection"""
        try:
            logger.info("Attempting to connect to Fitbit device...")
            
            # In a real implementation, you would:
            # 1. Use Fitbit API to authenticate
            # 2. Establish OAuth connection
            # 3. Subscribe to real-time data streams
            
            # For now, simulate connection
            await asyncio.sleep(2.0)
            
            # Simulate connection success/failure
            if self.connection_attempts < self.max_connection_attempts:
                self.connection_attempts += 1
                logger.warning(f"Fitbit connection attempt {self.connection_attempts} failed, using mock data")
                return await self._initialize_mock_device()
            
            self.is_initialized = True
            logger.info("Fitbit device connected")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Fitbit: {e}")
            return await self._initialize_mock_device()
    
    async def _initialize_apple_watch(self) -> bool:
        """Initialize Apple Watch connection"""
        try:
            logger.info("Attempting to connect to Apple Watch...")
            
            # In a real implementation, you would:
            # 1. Use HealthKit framework
            # 2. Request health data permissions
            # 3. Set up real-time data streaming
            
            # For now, simulate connection
            await asyncio.sleep(2.0)
            
            # Simulate connection success/failure
            if self.connection_attempts < self.max_connection_attempts:
                self.connection_attempts += 1
                logger.warning(f"Apple Watch connection attempt {self.connection_attempts} failed, using mock data")
                return await self._initialize_mock_device()
            
            self.is_initialized = True
            logger.info("Apple Watch connected")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Apple Watch: {e}")
            return await self._initialize_mock_device()
    
    async def get_biometric_data(self) -> Optional[Dict]:
        """Get current biometric data from wearables"""
        if not self.is_initialized:
            return None
        
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_collection_time < self.data_interval:
            return self._get_latest_data()
        
        try:
            if self.device_type == "mock":
                biometric_data = await self._get_mock_data()
            elif self.device_type == "fitbit":
                biometric_data = await self._get_fitbit_data()
            elif self.device_type == "apple_watch":
                biometric_data = await self._get_apple_watch_data()
            else:
                biometric_data = await self._get_mock_data()
            
            if biometric_data:
                # Store in history
                self._store_biometric_data(biometric_data)
                self.last_collection_time = current_time
            
            return biometric_data
            
        except Exception as e:
            logger.error(f"Error getting biometric data: {e}")
            return self._get_latest_data()
    
    async def _get_mock_data(self) -> Dict:
        """Generate mock biometric data for testing"""
        current_time = time.time()
        
        # Simulate realistic biometric variations
        # Heart rate with some variation
        hr_variation = random.gauss(0, 5)  # ±5 BPM variation
        heart_rate = max(50, min(120, self.mock_baseline_hr + hr_variation))
        
        # HRV (Heart Rate Variability) - lower values indicate stress
        hrv_variation = random.gauss(0, 10)
        hrv = max(20, min(100, self.mock_baseline_hrv + hrv_variation))
        
        # EDA (Electrodermal Activity) - higher values indicate stress
        eda_variation = random.gauss(0, 0.5)
        eda = max(0.5, min(10.0, self.mock_baseline_eda + eda_variation))
        
        # Simulate occasional stress events
        if random.random() < 0.1:  # 10% chance of stress event
            heart_rate += random.uniform(10, 20)
            hrv -= random.uniform(5, 15)
            eda += random.uniform(1, 3)
        
        # Simulate occasional fatigue events
        if random.random() < 0.05:  # 5% chance of fatigue event
            heart_rate -= random.uniform(5, 15)
            hrv += random.uniform(5, 15)
            eda -= random.uniform(0.5, 1.5)
        
        return {
            "heart_rate": float(heart_rate),
            "hrv": float(hrv),
            "eda": float(eda),
            "timestamp": current_time,
            "device_type": "mock",
            "quality": random.uniform(0.8, 1.0)  # Data quality indicator
        }
    
    async def _get_fitbit_data(self) -> Dict:
        """Get real biometric data from Fitbit (placeholder)"""
        # In a real implementation, you would:
        # 1. Make API calls to Fitbit
        # 2. Parse real-time heart rate data
        # 3. Calculate HRV from heart rate intervals
        # 4. Get EDA data if available
        
        # For now, return mock data
        return await self._get_mock_data()
    
    async def _get_apple_watch_data(self) -> Dict:
        """Get real biometric data from Apple Watch (placeholder)"""
        # In a real implementation, you would:
        # 1. Use HealthKit to access health data
        # 2. Get real-time heart rate
        # 3. Calculate HRV from heart rate data
        # 4. Access EDA data from Apple Watch sensors
        
        # For now, return mock data
        return await self._get_mock_data()
    
    def _store_biometric_data(self, data: Dict):
        """Store biometric data in history"""
        self.biometric_history.append(data)
        
        # Keep only recent history
        if len(self.biometric_history) > self.max_history_size:
            self.biometric_history = self.biometric_history[-self.max_history_size:]
    
    def _get_latest_data(self) -> Optional[Dict]:
        """Get the most recent biometric data"""
        if not self.biometric_history:
            return None
        
        return self.biometric_history[-1]
    
    def is_connected(self) -> bool:
        """Check if wearable device is connected"""
        return self.is_initialized
    
    def get_device_info(self) -> Dict:
        """Get wearable device information"""
        return {
            "device_type": self.device_type,
            "is_connected": self.is_connected(),
            "history_size": len(self.biometric_history),
            "data_interval": self.data_interval
        }
    
    def get_biometric_summary(self) -> Dict:
        """Get summary of recent biometric data"""
        if not self.biometric_history:
            return {}
        
        recent_data = self.biometric_history[-10:]  # Last 10 readings
        
        heart_rates = [d["heart_rate"] for d in recent_data if "heart_rate" in d]
        hrv_values = [d["hrv"] for d in recent_data if "hrv" in d]
        eda_values = [d["eda"] for d in recent_data if "eda" in d]
        
        summary = {
            "avg_heart_rate": sum(heart_rates) / len(heart_rates) if heart_rates else 0,
            "avg_hrv": sum(hrv_values) / len(hrv_values) if hrv_values else 0,
            "avg_eda": sum(eda_values) / len(eda_values) if eda_values else 0,
            "data_points": len(recent_data),
            "latest_timestamp": recent_data[-1]["timestamp"] if recent_data else 0
        }
        
        return summary
    
    async def calibrate_baseline(self, duration_seconds: int = 60):
        """Calibrate baseline biometric values"""
        logger.info(f"Starting {duration_seconds}s biometric calibration...")
        
        if not self.is_initialized:
            logger.error("Wearables not initialized")
            return False
        
        try:
            # Collect data for specified duration
            start_time = time.time()
            calibration_data = []
            
            while time.time() - start_time < duration_seconds:
                data = await self.get_biometric_data()
                if data:
                    calibration_data.append(data)
                await asyncio.sleep(1.0)  # Collect every second
            
            if calibration_data:
                # Calculate baseline values
                heart_rates = [d["heart_rate"] for d in calibration_data]
                hrv_values = [d["hrv"] for d in calibration_data]
                eda_values = [d["eda"] for d in calibration_data]
                
                self.mock_baseline_hr = sum(heart_rates) / len(heart_rates)
                self.mock_baseline_hrv = sum(hrv_values) / len(hrv_values)
                self.mock_baseline_eda = sum(eda_values) / len(eda_values)
                
                logger.info(f"Calibration complete - HR: {self.mock_baseline_hr:.1f}, "
                          f"HRV: {self.mock_baseline_hrv:.1f}, EDA: {self.mock_baseline_eda:.1f}")
                return True
            else:
                logger.error("No data collected during calibration")
                return False
                
        except Exception as e:
            logger.error(f"Error during calibration: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup wearable device connection"""
        if self.is_initialized:
            # In a real implementation, you would:
            # 1. Close API connections
            # 2. Unsubscribe from data streams
            # 3. Release resources
            
            self.is_initialized = False
            logger.info("Wearables cleanup complete")
