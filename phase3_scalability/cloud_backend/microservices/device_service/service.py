"""
Device Service for VigilAI Cloud Backend
Handles device management, fleet operations, and device health monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, Text, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import redis
import uvicorn
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = "postgresql://vigilai:password@localhost:5432/vigilai"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Security
security = HTTPBearer()
JWT_SECRET = "vigilai-secret-key"
JWT_ALGORITHM = "HS256"

# Enums
class DeviceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"

class DeviceType(str, Enum):
    VEHICLE = "vehicle"
    FLEET_MANAGER = "fleet_manager"
    EDGE_DEVICE = "edge_device"

# Database Models
class Device(Base):
    __tablename__ = "devices"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    device_id = Column(String, unique=True, index=True)  # Hardware device ID
    name = Column(String)
    device_type = Column(String)
    status = Column(String, default=DeviceStatus.ACTIVE)
    owner_id = Column(String, index=True)  # User ID who owns this device
    fleet_id = Column(String, index=True)  # Fleet ID if part of a fleet
    location = Column(Text)  # JSON string for location data
    capabilities = Column(Text)  # JSON string for device capabilities
    firmware_version = Column(String)
    last_seen = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(Text)  # JSON string for additional metadata

class DeviceHealth(Base):
    __tablename__ = "device_health"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    device_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    disk_usage = Column(Float)
    network_latency = Column(Float)
    temperature = Column(Float)
    battery_level = Column(Float)
    signal_strength = Column(Float)
    is_online = Column(Boolean, default=True)
    error_count = Column(Integer, default=0)
    last_error = Column(Text)

class Fleet(Base):
    __tablename__ = "fleets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String)
    description = Column(Text)
    owner_id = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    settings = Column(Text)  # JSON string for fleet settings

# Pydantic Models
class DeviceCreate(BaseModel):
    device_id: str
    name: str
    device_type: DeviceType
    location: Optional[Dict[str, Any]] = {}
    capabilities: Optional[Dict[str, Any]] = {}
    firmware_version: Optional[str] = "1.0.0"
    metadata: Optional[Dict[str, Any]] = {}

class DeviceUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    capabilities: Optional[Dict[str, Any]] = None
    firmware_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DeviceResponse(BaseModel):
    id: str
    device_id: str
    name: str
    device_type: str
    status: str
    owner_id: str
    fleet_id: Optional[str]
    location: Dict[str, Any]
    capabilities: Dict[str, Any]
    firmware_version: str
    last_seen: datetime
    created_at: datetime
    metadata: Dict[str, Any]

class DeviceHealthResponse(BaseModel):
    device_id: str
    timestamp: datetime
    cpu_usage: Optional[float]
    memory_usage: Optional[float]
    disk_usage: Optional[float]
    network_latency: Optional[float]
    temperature: Optional[float]
    battery_level: Optional[float]
    signal_strength: Optional[float]
    is_online: bool
    error_count: int
    last_error: Optional[str]

class FleetCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    settings: Optional[Dict[str, Any]] = {}

class FleetResponse(BaseModel):
    id: str
    name: str
    description: str
    owner_id: str
    created_at: datetime
    settings: Dict[str, Any]
    device_count: int

class DeviceService:
    """Device service implementation"""
    
    def __init__(self):
        self.app = FastAPI(title="Device Service", version="1.0.0")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/devices", response_model=DeviceResponse)
        async def register_device(
            device_data: DeviceCreate,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Register a new device"""
            try:
                # Verify token and get user ID
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                db = SessionLocal()
                try:
                    # Check if device already exists
                    existing_device = db.query(Device).filter(
                        Device.device_id == device_data.device_id
                    ).first()
                    
                    if existing_device:
                        raise HTTPException(
                            status_code=400,
                            detail="Device with this ID already exists"
                        )
                    
                    # Create device
                    device = Device(
                        device_id=device_data.device_id,
                        name=device_data.name,
                        device_type=device_data.device_type,
                        owner_id=user_id,
                        location=device_data.location,
                        capabilities=device_data.capabilities,
                        firmware_version=device_data.firmware_version,
                        metadata=device_data.metadata
                    )
                    
                    db.add(device)
                    db.commit()
                    db.refresh(device)
                    
                    return DeviceResponse(
                        id=device.id,
                        device_id=device.device_id,
                        name=device.name,
                        device_type=device.device_type,
                        status=device.status,
                        owner_id=device.owner_id,
                        fleet_id=device.fleet_id,
                        location=device.location or {},
                        capabilities=device.capabilities or {},
                        firmware_version=device.firmware_version,
                        last_seen=device.last_seen,
                        created_at=device.created_at,
                        metadata=device.metadata or {}
                    )
                    
                finally:
                    db.close()
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Device registration error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/devices", response_model=List[DeviceResponse])
        async def get_devices(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            fleet_id: Optional[str] = None,
            status: Optional[DeviceStatus] = None
        ):
            """Get devices for current user"""
            try:
                # Verify token and get user ID
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                db = SessionLocal()
                try:
                    query = db.query(Device).filter(Device.owner_id == user_id)
                    
                    if fleet_id:
                        query = query.filter(Device.fleet_id == fleet_id)
                    
                    if status:
                        query = query.filter(Device.status == status)
                    
                    devices = query.all()
                    
                    return [
                        DeviceResponse(
                            id=device.id,
                            device_id=device.device_id,
                            name=device.name,
                            device_type=device.device_type,
                            status=device.status,
                            owner_id=device.owner_id,
                            fleet_id=device.fleet_id,
                            location=device.location or {},
                            capabilities=device.capabilities or {},
                            firmware_version=device.firmware_version,
                            last_seen=device.last_seen,
                            created_at=device.created_at,
                            metadata=device.metadata or {}
                        )
                        for device in devices
                    ]
                    
                finally:
                    db.close()
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get devices error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/devices/{device_id}", response_model=DeviceResponse)
        async def get_device(
            device_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Get specific device"""
            try:
                # Verify token and get user ID
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                db = SessionLocal()
                try:
                    device = db.query(Device).filter(
                        Device.id == device_id,
                        Device.owner_id == user_id
                    ).first()
                    
                    if not device:
                        raise HTTPException(status_code=404, detail="Device not found")
                    
                    return DeviceResponse(
                        id=device.id,
                        device_id=device.device_id,
                        name=device.name,
                        device_type=device.device_type,
                        status=device.status,
                        owner_id=device.owner_id,
                        fleet_id=device.fleet_id,
                        location=device.location or {},
                        capabilities=device.capabilities or {},
                        firmware_version=device.firmware_version,
                        last_seen=device.last_seen,
                        created_at=device.created_at,
                        metadata=device.metadata or {}
                    )
                    
                finally:
                    db.close()
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get device error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/devices/{device_id}/health")
        async def update_device_health(
            device_id: str,
            health_data: DeviceHealthResponse,
            credentials: HTTPAuthorizationCredentials = Depends(security)
        ):
            """Update device health metrics"""
            try:
                # Verify token and get user ID
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                db = SessionLocal()
                try:
                    # Verify device ownership
                    device = db.query(Device).filter(
                        Device.id == device_id,
                        Device.owner_id == user_id
                    ).first()
                    
                    if not device:
                        raise HTTPException(status_code=404, detail="Device not found")
                    
                    # Create health record
                    health = DeviceHealth(
                        device_id=device_id,
                        timestamp=health_data.timestamp,
                        cpu_usage=health_data.cpu_usage,
                        memory_usage=health_data.memory_usage,
                        disk_usage=health_data.disk_usage,
                        network_latency=health_data.network_latency,
                        temperature=health_data.temperature,
                        battery_level=health_data.battery_level,
                        signal_strength=health_data.signal_strength,
                        is_online=health_data.is_online,
                        error_count=health_data.error_count,
                        last_error=health_data.last_error
                    )
                    
                    db.add(health)
                    
                    # Update device last_seen
                    device.last_seen = datetime.utcnow()
                    if not health_data.is_online:
                        device.status = DeviceStatus.OFFLINE
                    elif health_data.error_count > 0:
                        device.status = DeviceStatus.ERROR
                    else:
                        device.status = DeviceStatus.ACTIVE
                    
                    db.commit()
                    
                    return {"message": "Health data updated successfully"}
                    
                finally:
                    db.close()
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Update health error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/devices/{device_id}/health", response_model=List[DeviceHealthResponse])
        async def get_device_health(
            device_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security),
            hours: int = 24
        ):
            """Get device health history"""
            try:
                # Verify token and get user ID
                payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                user_id = payload.get("user_id")
                
                if not user_id:
                    raise HTTPException(status_code=401, detail="Invalid token")
                
                db = SessionLocal()
                try:
                    # Verify device ownership
                    device = db.query(Device).filter(
                        Device.id == device_id,
                        Device.owner_id == user_id
                    ).first()
                    
                    if not device:
                        raise HTTPException(status_code=404, detail="Device not found")
                    
                    # Get health data
                    since = datetime.utcnow() - timedelta(hours=hours)
                    health_records = db.query(DeviceHealth).filter(
                        DeviceHealth.device_id == device_id,
                        DeviceHealth.timestamp >= since
                    ).order_by(DeviceHealth.timestamp.desc()).all()
                    
                    return [
                        DeviceHealthResponse(
                            device_id=health.device_id,
                            timestamp=health.timestamp,
                            cpu_usage=health.cpu_usage,
                            memory_usage=health.memory_usage,
                            disk_usage=health.disk_usage,
                            network_latency=health.network_latency,
                            temperature=health.temperature,
                            battery_level=health.battery_level,
                            signal_strength=health.signal_strength,
                            is_online=health.is_online,
                            error_count=health.error_count,
                            last_error=health.last_error
                        )
                        for health in health_records
                    ]
                    
                finally:
                    db.close()
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Get health error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "device_service"}

def create_device_service() -> DeviceService:
    """Create device service instance"""
    return DeviceService()

def main():
    """Main entry point for device service"""
    service = create_device_service()
    
    uvicorn.run(
        service.app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )

if __name__ == "__main__":
    main()
