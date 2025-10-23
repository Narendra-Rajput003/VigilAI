"""
Apache Kafka Producer for VigilAI
Handles real-time data streaming from edge devices to cloud
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from kafka import KafkaProducer
from kafka.errors import KafkaError
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VigilAIMessage:
    """Standard message format for VigilAI data"""
    message_id: str
    device_id: str
    user_id: str
    timestamp: datetime
    message_type: str  # video_frame, steering_data, biometric_data, health_data, alert
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class VigilAIKafkaProducer:
    """Kafka producer for VigilAI data streaming"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.topics = {
            "video_data": "vigilai.video",
            "steering_data": "vigilai.steering", 
            "biometric_data": "vigilai.biometric",
            "health_data": "vigilai.health",
            "alerts": "vigilai.alerts",
            "analytics": "vigilai.analytics"
        }
        self._connect()
    
    def _connect(self):
        """Connect to Kafka cluster"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # Wait for all replicas to acknowledge
                retries=3,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                max_block_ms=10000
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def _get_topic(self, message_type: str) -> str:
        """Get topic name for message type"""
        topic_mapping = {
            "video_frame": "video_data",
            "steering_data": "steering_data",
            "biometric_data": "biometric_data", 
            "health_data": "health_data",
            "alert": "alerts",
            "analytics": "analytics"
        }
        
        topic_key = topic_mapping.get(message_type, "analytics")
        return self.topics[topic_key]
    
    def send_message(self, message: VigilAIMessage) -> bool:
        """Send message to Kafka topic"""
        try:
            if not self.producer:
                self._connect()
            
            topic = self._get_topic(message.message_type)
            key = f"{message.device_id}:{message.timestamp.isoformat()}"
            
            # Convert message to dict
            message_dict = asdict(message)
            message_dict['timestamp'] = message.timestamp.isoformat()
            
            # Send message
            future = self.producer.send(
                topic,
                key=key,
                value=message_dict
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            logger.debug(f"Message sent to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}")
            
            return True
            
        except KafkaError as e:
            logger.error(f"Failed to send message: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            return False
    
    async def send_video_frame(self, device_id: str, user_id: str, frame_data: Dict[str, Any]) -> bool:
        """Send video frame data"""
        message = VigilAIMessage(
            message_id=str(uuid.uuid4()),
            device_id=device_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            message_type="video_frame",
            data=frame_data,
            metadata={"source": "camera", "format": "h264"}
        )
        
        return self.send_message(message)
    
    async def send_steering_data(self, device_id: str, user_id: str, steering_data: Dict[str, Any]) -> bool:
        """Send steering data"""
        message = VigilAIMessage(
            message_id=str(uuid.uuid4()),
            device_id=device_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            message_type="steering_data",
            data=steering_data,
            metadata={"source": "obd", "protocol": "can"}
        )
        
        return self.send_message(message)
    
    async def send_biometric_data(self, device_id: str, user_id: str, biometric_data: Dict[str, Any]) -> bool:
        """Send biometric data"""
        message = VigilAIMessage(
            message_id=str(uuid.uuid4()),
            device_id=device_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            message_type="biometric_data",
            data=biometric_data,
            metadata={"source": "wearables", "sensors": ["hr", "eda", "temperature"]}
        )
        
        return self.send_message(message)
    
    async def send_health_data(self, device_id: str, user_id: str, health_data: Dict[str, Any]) -> bool:
        """Send device health data"""
        message = VigilAIMessage(
            message_id=str(uuid.uuid4()),
            device_id=device_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            message_type="health_data",
            data=health_data,
            metadata={"source": "device_monitor", "metrics": ["cpu", "memory", "network"]}
        )
        
        return self.send_message(message)
    
    async def send_alert(self, device_id: str, user_id: str, alert_data: Dict[str, Any]) -> bool:
        """Send alert message"""
        message = VigilAIMessage(
            message_id=str(uuid.uuid4()),
            device_id=device_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            message_type="alert",
            data=alert_data,
            metadata={"priority": "high", "source": "detection_engine"}
        )
        
        return self.send_message(message)
    
    async def send_analytics(self, device_id: str, user_id: str, analytics_data: Dict[str, Any]) -> bool:
        """Send analytics data"""
        message = VigilAIMessage(
            message_id=str(uuid.uuid4()),
            device_id=device_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            message_type="analytics",
            data=analytics_data,
            metadata={"source": "analytics_engine", "type": "insights"}
        )
        
        return self.send_message(message)
    
    def close(self):
        """Close producer connection"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")

class VigilAIStreamingService:
    """High-level streaming service for VigilAI"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.producer = VigilAIKafkaProducer(bootstrap_servers)
        self.is_running = False
    
    async def start(self):
        """Start streaming service"""
        self.is_running = True
        logger.info("VigilAI streaming service started")
    
    async def stop(self):
        """Stop streaming service"""
        self.is_running = False
        self.producer.close()
        logger.info("VigilAI streaming service stopped")
    
    async def stream_video_data(self, device_id: str, user_id: str, video_stream):
        """Stream video data from edge device"""
        try:
            while self.is_running:
                # Get frame from video stream
                frame_data = await self._get_video_frame(video_stream)
                
                if frame_data:
                    success = await self.producer.send_video_frame(
                        device_id, user_id, frame_data
                    )
                    
                    if not success:
                        logger.warning("Failed to send video frame")
                
                await asyncio.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            logger.error(f"Error streaming video data: {e}")
    
    async def stream_steering_data(self, device_id: str, user_id: str, steering_stream):
        """Stream steering data from OBD interface"""
        try:
            while self.is_running:
                # Get steering data from OBD
                steering_data = await self._get_steering_data(steering_stream)
                
                if steering_data:
                    success = await self.producer.send_steering_data(
                        device_id, user_id, steering_data
                    )
                    
                    if not success:
                        logger.warning("Failed to send steering data")
                
                await asyncio.sleep(0.1)  # 10 Hz
                
        except Exception as e:
            logger.error(f"Error streaming steering data: {e}")
    
    async def stream_biometric_data(self, device_id: str, user_id: str, biometric_stream):
        """Stream biometric data from wearables"""
        try:
            while self.is_running:
                # Get biometric data from wearables
                biometric_data = await self._get_biometric_data(biometric_stream)
                
                if biometric_data:
                    success = await self.producer.send_biometric_data(
                        device_id, user_id, biometric_data
                    )
                    
                    if not success:
                        logger.warning("Failed to send biometric data")
                
                await asyncio.sleep(1.0)  # 1 Hz
                
        except Exception as e:
            logger.error(f"Error streaming biometric data: {e}")
    
    async def _get_video_frame(self, video_stream) -> Optional[Dict[str, Any]]:
        """Get video frame from stream (mock implementation)"""
        # TODO: Implement actual video frame capture
        return {
            "frame_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "width": 1920,
            "height": 1080,
            "format": "h264",
            "size_bytes": 50000,
            "landmarks": [],  # Facial landmarks
            "features": {}    # Extracted features
        }
    
    async def _get_steering_data(self, steering_stream) -> Optional[Dict[str, Any]]:
        """Get steering data from OBD (mock implementation)"""
        # TODO: Implement actual OBD data capture
        return {
            "steering_angle": 0.0,
            "steering_velocity": 0.0,
            "steering_acceleration": 0.0,
            "timestamp": datetime.utcnow().isoformat(),
            "quality": 1.0
        }
    
    async def _get_biometric_data(self, biometric_stream) -> Optional[Dict[str, Any]]:
        """Get biometric data from wearables (mock implementation)"""
        # TODO: Implement actual biometric data capture
        return {
            "heart_rate": 75.0,
            "hrv": 0.05,
            "eda": 2.5,
            "temperature": 36.5,
            "timestamp": datetime.utcnow().isoformat(),
            "quality": 0.9
        }

# Global streaming service instance
streaming_service = None

async def get_streaming_service() -> VigilAIStreamingService:
    """Get or create streaming service instance"""
    global streaming_service
    if streaming_service is None:
        streaming_service = VigilAIStreamingService()
        await streaming_service.start()
    return streaming_service

async def main():
    """Main entry point for streaming service"""
    service = VigilAIStreamingService()
    await service.start()
    
    try:
        # Keep service running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
