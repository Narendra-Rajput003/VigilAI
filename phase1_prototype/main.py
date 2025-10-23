#!/usr/bin/env python3
"""
VigilAI Phase 1: MVP Prototype
Main entry point for the driver fatigue monitoring system
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import uvicorn

from .core.detection_engine import DetectionEngine
from .core.data_collector import DataCollector
from .core.intervention_system import InterventionSystem
from .core.metrics import MetricsCollector
from .hardware.camera import CameraManager
from .hardware.obd_interface import OBDInterface
from .hardware.wearables import WearablesManager
from .utils.config import Config
from .utils.logger import setup_logging

# Configure logging
logger = setup_logging(__name__)

class VigilAIPrototype:
    """Main VigilAI prototype controller"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
        self.running = False
        
        # Initialize components
        self.camera = CameraManager(self.config.camera)
        self.obd = OBDInterface(self.config.obd)
        self.wearables = WearablesManager(self.config.wearables)
        self.data_collector = DataCollector()
        self.detection_engine = DetectionEngine(self.config.detection)
        self.intervention = InterventionSystem(self.config.intervention)
        self.metrics = MetricsCollector()
        
        # FastAPI app for web interface
        self.app = FastAPI(title="VigilAI Prototype", version="0.1.0")
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "VigilAI Prototype API", "status": "running"}
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "components": {
                    "camera": self.camera.is_connected(),
                    "obd": self.obd.is_connected(),
                    "wearables": self.wearables.is_connected()
                }
            }
        
        @self.app.get("/metrics")
        async def get_metrics():
            return self.metrics.get_summary()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while self.running:
                    # Send real-time data
                    data = {
                        "timestamp": self.metrics.get_timestamp(),
                        "fatigue_score": self.detection_engine.get_current_score(),
                        "interventions": self.intervention.get_active_interventions(),
                        "metrics": self.metrics.get_latest()
                    }
                    await websocket.send_json(data)
                    await asyncio.sleep(0.1)  # 10 FPS updates
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                await websocket.close()
    
    async def start(self):
        """Start the VigilAI prototype"""
        logger.info("Starting VigilAI Prototype...")
        
        try:
            # Initialize hardware
            await self._initialize_hardware()
            
            # Start detection loop
            self.running = True
            detection_task = asyncio.create_task(self._detection_loop())
            
            # Start web server
            server_task = asyncio.create_task(
                uvicorn.run(self.app, host="0.0.0.0", port=8000, log_level="info")
            )
            
            # Wait for tasks
            await asyncio.gather(detection_task, server_task)
            
        except Exception as e:
            logger.error(f"Error starting VigilAI: {e}")
            await self.stop()
    
    async def _initialize_hardware(self):
        """Initialize all hardware components"""
        logger.info("Initializing hardware components...")
        
        # Initialize camera
        if not await self.camera.initialize():
            raise RuntimeError("Failed to initialize camera")
        
        # Initialize OBD-II
        if not await self.obd.initialize():
            logger.warning("OBD-II not connected, continuing without steering data")
        
        # Initialize wearables
        if not await self.wearables.initialize():
            logger.warning("Wearables not connected, continuing without biometric data")
        
        logger.info("Hardware initialization complete")
    
    async def _detection_loop(self):
        """Main detection and intervention loop"""
        logger.info("Starting detection loop...")
        
        while self.running:
            try:
                # Collect data from all sources
                video_data = await self.camera.capture_frame()
                steering_data = await self.obd.get_steering_data()
                biometric_data = await self.wearables.get_biometric_data()
                
                # Store data
                self.data_collector.store_data({
                    "video": video_data,
                    "steering": steering_data,
                    "biometric": biometric_data,
                    "timestamp": self.metrics.get_timestamp()
                })
                
                # Run detection
                detection_result = await self.detection_engine.analyze({
                    "video": video_data,
                    "steering": steering_data,
                    "biometric": biometric_data
                })
                
                # Update metrics
                self.metrics.update(detection_result)
                
                # Trigger interventions if needed
                if detection_result["fatigue_score"] > self.config.detection.fatigue_threshold:
                    await self.intervention.trigger_intervention(
                        "fatigue", detection_result["fatigue_score"]
                    )
                
                if detection_result["stress_score"] > self.config.detection.stress_threshold:
                    await self.intervention.trigger_intervention(
                        "stress", detection_result["stress_score"]
                    )
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def stop(self):
        """Stop the VigilAI prototype"""
        logger.info("Stopping VigilAI Prototype...")
        self.running = False
        
        # Cleanup hardware
        await self.camera.cleanup()
        await self.obd.cleanup()
        await self.wearables.cleanup()
        
        logger.info("VigilAI Prototype stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start prototype
    prototype = VigilAIPrototype()
    
    try:
        await prototype.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await prototype.stop()

if __name__ == "__main__":
    asyncio.run(main())
