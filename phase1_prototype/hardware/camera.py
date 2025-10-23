"""
Camera Manager for VigilAI
Handles video capture and processing for fatigue detection
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages camera operations for video capture and processing"""
    
    def __init__(self, config):
        self.config = config
        self.camera = None
        self.is_initialized = False
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Camera settings
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self.fps = config.get("fps", 30)
        self.device_id = config.get("device_id", 0)
        
        # Image processing settings
        self.flip_horizontal = config.get("flip_horizontal", True)
        self.ir_mode = config.get("ir_mode", False)
        
    async def initialize(self) -> bool:
        """Initialize the camera"""
        try:
            logger.info(f"Initializing camera (device {self.device_id})...")
            
            # Try to open camera
            self.camera = cv2.VideoCapture(self.device_id)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open camera {self.device_id}")
                return False
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Additional settings for better performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture test frame")
                await self.cleanup()
                return False
            
            self.is_initialized = True
            logger.info(f"Camera initialized successfully ({self.width}x{self.height} @ {self.fps}fps)")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            await self.cleanup()
            return False
    
    async def capture_frame(self) -> Optional[Dict]:
        """Capture a single frame from the camera"""
        if not self.is_initialized or self.camera is None:
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("Failed to capture frame")
                return None
            
            # Flip frame horizontally if configured
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)
            
            # Convert to different color spaces if needed
            processed_frame = self._process_frame(frame)
            
            # Update FPS counter
            self._update_fps()
            
            return {
                "frame": processed_frame,
                "timestamp": time.time(),
                "frame_count": self.frame_count,
                "fps": self.current_fps
            }
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the captured frame"""
        processed = frame.copy()
        
        # Apply IR mode processing if enabled
        if self.ir_mode:
            # Convert to grayscale for IR processing
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            # Convert back to 3-channel for consistency
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        
        # Apply any additional processing here
        # (e.g., noise reduction, contrast enhancement)
        
        return processed
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        self.fps_counter += 1
        
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def is_connected(self) -> bool:
        """Check if camera is connected and working"""
        if not self.is_initialized or self.camera is None:
            return False
        
        try:
            return self.camera.isOpened()
        except:
            return False
    
    def get_camera_info(self) -> Dict:
        """Get camera information"""
        if not self.is_initialized:
            return {}
        
        return {
            "device_id": self.device_id,
            "width": self.width,
            "height": self.height,
            "fps": self.current_fps,
            "frame_count": self.frame_count,
            "is_connected": self.is_connected()
        }
    
    async def cleanup(self):
        """Cleanup camera resources"""
        if self.camera is not None:
            try:
                self.camera.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.camera = None
        
        self.is_initialized = False
        logger.info("Camera cleanup complete")
    
    async def test_camera(self) -> bool:
        """Test camera functionality"""
        if not self.is_initialized:
            return False
        
        try:
            # Capture a few frames to test
            for i in range(5):
                frame_data = await self.capture_frame()
                if frame_data is None:
                    return False
                await asyncio.sleep(0.1)  # Small delay between captures
            
            logger.info("Camera test successful")
            return True
            
        except Exception as e:
            logger.error(f"Camera test failed: {e}")
            return False
