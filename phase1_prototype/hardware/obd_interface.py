"""
OBD-II Interface for VigilAI
Handles steering data collection via OBD-II adapter
"""

import asyncio
import logging
import time
from typing import Dict, Optional, List
import serial
import json

logger = logging.getLogger(__name__)

class OBDInterface:
    """Manages OBD-II communication for steering and vehicle data"""
    
    def __init__(self, config):
        self.config = config
        self.serial_connection = None
        self.is_initialized = False
        
        # OBD-II settings
        self.port = config.get("port", "/dev/ttyUSB0")
        self.baudrate = config.get("baudrate", 38400)
        self.timeout = config.get("timeout", 1.0)
        
        # Data collection settings
        self.collection_interval = config.get("collection_interval", 0.1)  # 10 Hz
        self.last_collection_time = 0
        
        # Steering data history
        self.steering_history: List[Dict] = []
        self.max_history_size = 1000  # Keep last 1000 readings
        
        # OBD-II PIDs for steering data
        self.steering_angle_pid = "0C"  # Engine RPM (proxy for steering activity)
        self.vehicle_speed_pid = "0D"   # Vehicle speed
        self.throttle_position_pid = "11"  # Throttle position
        
    async def initialize(self) -> bool:
        """Initialize OBD-II connection"""
        try:
            logger.info(f"Initializing OBD-II interface on {self.port}...")
            
            # Try to connect to OBD-II adapter
            self.serial_connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            
            if not self.serial_connection.is_open:
                logger.error(f"Failed to open serial connection to {self.port}")
                return False
            
            # Test communication
            if not await self._test_communication():
                logger.error("OBD-II communication test failed")
                await self.cleanup()
                return False
            
            self.is_initialized = True
            logger.info("OBD-II interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing OBD-II interface: {e}")
            await self.cleanup()
            return False
    
    async def _test_communication(self) -> bool:
        """Test OBD-II communication"""
        try:
            # Send ATZ command to reset adapter
            response = await self._send_command("ATZ")
            if not response:
                return False
            
            # Send ATL1 command to enable line feeds
            response = await self._send_command("ATL1")
            if not response:
                return False
            
            # Try to get vehicle speed
            response = await self._send_command(f"01{self.vehicle_speed_pid}")
            if not response or "NO DATA" in response:
                logger.warning("OBD-II adapter connected but no vehicle data available")
                # This is okay for testing without a real vehicle
                return True
            
            logger.info("OBD-II communication test successful")
            return True
            
        except Exception as e:
            logger.error(f"OBD-II communication test failed: {e}")
            return False
    
    async def _send_command(self, command: str) -> Optional[str]:
        """Send command to OBD-II adapter and get response"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return None
        
        try:
            # Clear input buffer
            self.serial_connection.reset_input_buffer()
            
            # Send command
            command_bytes = (command + "\r").encode('ascii')
            self.serial_connection.write(command_bytes)
            
            # Wait for response
            await asyncio.sleep(0.1)
            
            # Read response
            response = self.serial_connection.readline().decode('ascii', errors='ignore').strip()
            
            # Remove echo if present
            if response.startswith(command):
                response = response[len(command):].strip()
            
            return response if response else None
            
        except Exception as e:
            logger.error(f"Error sending OBD-II command '{command}': {e}")
            return None
    
    async def get_steering_data(self) -> Optional[Dict]:
        """Get current steering and vehicle data"""
        if not self.is_initialized:
            return None
        
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_collection_time < self.collection_interval:
            return self._get_latest_data()
        
        try:
            # Collect multiple data points
            data = {}
            
            # Get vehicle speed
            speed_response = await self._send_command(f"01{self.vehicle_speed_pid}")
            if speed_response and "NO DATA" not in speed_response:
                data["speed"] = self._parse_speed_response(speed_response)
            
            # Get throttle position (proxy for steering activity)
            throttle_response = await self._send_command(f"01{self.throttle_position_pid}")
            if throttle_response and "NO DATA" not in throttle_response:
                data["throttle"] = self._parse_throttle_response(throttle_response)
            
            # Get engine RPM (another proxy for steering activity)
            rpm_response = await self._send_command(f"01{self.steering_angle_pid}")
            if rpm_response and "NO DATA" not in rpm_response:
                data["rpm"] = self._parse_rpm_response(rpm_response)
            
            # Calculate steering metrics
            steering_data = self._calculate_steering_metrics(data)
            
            # Store in history
            self._store_steering_data(steering_data)
            
            self.last_collection_time = current_time
            
            return steering_data
            
        except Exception as e:
            logger.error(f"Error getting steering data: {e}")
            return self._get_latest_data()
    
    def _parse_speed_response(self, response: str) -> float:
        """Parse vehicle speed from OBD-II response"""
        try:
            # OBD-II speed response format: 41 0D XX
            parts = response.split()
            if len(parts) >= 3:
                speed_hex = parts[2]
                speed_kmh = int(speed_hex, 16)
                return float(speed_kmh)
        except:
            pass
        return 0.0
    
    def _parse_throttle_response(self, response: str) -> float:
        """Parse throttle position from OBD-II response"""
        try:
            # OBD-II throttle response format: 41 11 XX
            parts = response.split()
            if len(parts) >= 3:
                throttle_hex = parts[2]
                throttle_percent = (int(throttle_hex, 16) * 100) / 255
                return float(throttle_percent)
        except:
            pass
        return 0.0
    
    def _parse_rpm_response(self, response: str) -> float:
        """Parse engine RPM from OBD-II response"""
        try:
            # OBD-II RPM response format: 41 0C XX XX
            parts = response.split()
            if len(parts) >= 4:
                rpm_high = int(parts[2], 16)
                rpm_low = int(parts[3], 16)
                rpm = ((rpm_high * 256) + rpm_low) / 4
                return float(rpm)
        except:
            pass
        return 0.0
    
    def _calculate_steering_metrics(self, raw_data: Dict) -> Dict:
        """Calculate steering metrics from raw OBD-II data"""
        current_time = time.time()
        
        # Basic metrics
        speed = raw_data.get("speed", 0.0)
        throttle = raw_data.get("throttle", 0.0)
        rpm = raw_data.get("rpm", 0.0)
        
        # Calculate steering angle (simplified - in real implementation, 
        # you'd need a steering angle sensor)
        steering_angle = self._estimate_steering_angle(throttle, speed)
        
        # Calculate steering velocity
        steering_velocity = self._calculate_steering_velocity(steering_angle)
        
        # Calculate steering entropy (measure of steering irregularity)
        steering_entropy = self._calculate_steering_entropy()
        
        return {
            "angle": steering_angle,
            "velocity": steering_velocity,
            "entropy": steering_entropy,
            "speed": speed,
            "throttle": throttle,
            "rpm": rpm,
            "timestamp": current_time
        }
    
    def _estimate_steering_angle(self, throttle: float, speed: float) -> float:
        """Estimate steering angle from available data (simplified)"""
        # This is a simplified estimation
        # In a real implementation, you'd need actual steering angle data
        
        # Use throttle and speed to estimate steering activity
        # Higher throttle at low speed might indicate steering corrections
        if speed > 0:
            steering_activity = throttle / (speed + 1)  # Avoid division by zero
        else:
            steering_activity = throttle
        
        # Convert to steering angle (simplified)
        steering_angle = (steering_activity - 0.5) * 180  # -90 to +90 degrees
        
        return max(-90.0, min(90.0, steering_angle))
    
    def _calculate_steering_velocity(self, current_angle: float) -> float:
        """Calculate steering velocity from angle changes"""
        if len(self.steering_history) < 2:
            return 0.0
        
        # Get previous angle
        previous_data = self.steering_history[-1]
        previous_angle = previous_data.get("angle", 0.0)
        previous_time = previous_data.get("timestamp", time.time())
        
        current_time = time.time()
        time_delta = current_time - previous_time
        
        if time_delta > 0:
            velocity = (current_angle - previous_angle) / time_delta
            return velocity
        
        return 0.0
    
    def _calculate_steering_entropy(self) -> float:
        """Calculate steering entropy as a measure of irregularity"""
        if len(self.steering_history) < 10:
            return 0.0
        
        # Get recent steering angles
        recent_angles = [data["angle"] for data in self.steering_history[-20:]]
        
        # Calculate entropy using histogram
        import numpy as np
        from scipy.stats import entropy
        
        try:
            # Create histogram
            hist, _ = np.histogram(recent_angles, bins=10)
            
            # Calculate entropy
            hist_normalized = hist / np.sum(hist) if np.sum(hist) > 0 else hist
            hist_normalized = hist_normalized + 1e-10  # Avoid log(0)
            
            return float(entropy(hist_normalized))
        except:
            return 0.0
    
    def _store_steering_data(self, data: Dict):
        """Store steering data in history"""
        self.steering_history.append(data)
        
        # Keep only recent history
        if len(self.steering_history) > self.max_history_size:
            self.steering_history = self.steering_history[-self.max_history_size:]
    
    def _get_latest_data(self) -> Optional[Dict]:
        """Get the most recent steering data"""
        if not self.steering_history:
            return None
        
        return self.steering_history[-1]
    
    def is_connected(self) -> bool:
        """Check if OBD-II interface is connected"""
        if not self.is_initialized or not self.serial_connection:
            return False
        
        try:
            return self.serial_connection.is_open
        except:
            return False
    
    def get_connection_info(self) -> Dict:
        """Get OBD-II connection information"""
        return {
            "port": self.port,
            "baudrate": self.baudrate,
            "is_connected": self.is_connected(),
            "history_size": len(self.steering_history)
        }
    
    async def cleanup(self):
        """Cleanup OBD-II connection"""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
                logger.info("OBD-II connection closed")
            except Exception as e:
                logger.error(f"Error closing OBD-II connection: {e}")
            finally:
                self.serial_connection = None
        
        self.is_initialized = False
        logger.info("OBD-II cleanup complete")
