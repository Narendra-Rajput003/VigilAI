"""
Edge Gateway for VigilAI Edge-Cloud Hybrid Architecture
Handles local processing, cloud synchronization, and offline operation
"""

import asyncio
import logging
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
import jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EdgeDevice:
    """Edge device configuration"""
    device_id: str
    device_type: str
    capabilities: Dict[str, Any]
    location: Dict[str, Any]
    is_online: bool = True
    last_sync: Optional[datetime] = None

@dataclass
class ProcessingResult:
    """Result from edge processing"""
    device_id: str
    timestamp: datetime
    result_type: str  # fatigue_detection, stress_detection, anomaly_detection
    confidence: float
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class EdgeGateway:
    """Edge Gateway for local processing and cloud sync"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_id = config.get("device_id")
        self.cloud_endpoint = config.get("cloud_endpoint")
        self.sync_interval = config.get("sync_interval", 30)  # seconds
        self.offline_mode = config.get("offline_mode", False)
        
        # Local storage
        self.local_db = sqlite3.connect("edge_data.db")
        self._init_local_db()
        
        # Redis for caching
        self.redis = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Processing modules
        self.processing_modules = {}
        self._init_processing_modules()
        
        # Sync status
        self.is_syncing = False
        self.last_sync_time = None
        
        # FastAPI app
        self.app = FastAPI(title="Edge Gateway", version="1.0.0")
        self._setup_routes()
    
    def _init_local_db(self):
        """Initialize local SQLite database"""
        cursor = self.local_db.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                timestamp TEXT,
                result_type TEXT,
                confidence REAL,
                data TEXT,
                metadata TEXT,
                synced INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT,
                data TEXT,
                timestamp TEXT,
                retry_count INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS device_status (
                device_id TEXT PRIMARY KEY,
                status TEXT,
                last_seen TEXT,
                capabilities TEXT
            )
        """)
        
        self.local_db.commit()
    
    def _init_processing_modules(self):
        """Initialize local processing modules"""
        # Import processing modules
        try:
            from phase2_core.inference.real_time_inference import RealTimeInferenceEngine
            from phase2_core.models.fusion.multimodal_fusion import MultimodalFusionModel
            
            self.processing_modules["inference"] = RealTimeInferenceEngine()
            self.processing_modules["fusion"] = MultimodalFusionModel()
            
            logger.info("Processing modules initialized")
        except ImportError as e:
            logger.warning(f"Could not import processing modules: {e}")
            # Use mock modules for development
            self.processing_modules["inference"] = MockInferenceEngine()
            self.processing_modules["fusion"] = MockFusionModel()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/process")
        async def process_data(data: Dict[str, Any], background_tasks: BackgroundTasks):
            """Process incoming data"""
            try:
                result = await self._process_data(data)
                
                # Store result locally
                await self._store_result(result)
                
                # Queue for sync if online
                if not self.offline_mode:
                    background_tasks.add_task(self._sync_to_cloud, result)
                
                return {"status": "processed", "result_id": result.device_id}
                
            except Exception as e:
                logger.error(f"Processing error: {e}")
                raise HTTPException(status_code=500, detail="Processing failed")
        
        @self.app.get("/status")
        async def get_status():
            """Get edge gateway status"""
            return {
                "device_id": self.device_id,
                "is_online": not self.offline_mode,
                "last_sync": self.last_sync_time,
                "processing_modules": list(self.processing_modules.keys()),
                "local_queue_size": await self._get_queue_size()
            }
        
        @self.app.post("/sync")
        async def force_sync():
            """Force synchronization with cloud"""
            try:
                await self._sync_to_cloud()
                return {"status": "synced", "timestamp": datetime.utcnow()}
            except Exception as e:
                logger.error(f"Sync error: {e}")
                raise HTTPException(status_code=500, detail="Sync failed")
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "service": "edge_gateway"}
    
    async def _process_data(self, data: Dict[str, Any]) -> ProcessingResult:
        """Process data using local AI models"""
        try:
            # Extract data components
            video_data = data.get("video", {})
            steering_data = data.get("steering", {})
            biometric_data = data.get("biometric", {})
            
            # Process with inference engine
            if "inference" in self.processing_modules:
                result = await self.processing_modules["inference"].process(
                    video_data, steering_data, biometric_data
                )
            else:
                # Mock processing
                result = {
                    "fatigue_detected": False,
                    "stress_level": 0.3,
                    "confidence": 0.85,
                    "anomalies": []
                }
            
            # Create processing result
            processing_result = ProcessingResult(
                device_id=self.device_id,
                timestamp=datetime.utcnow(),
                result_type="fatigue_detection",
                confidence=result.get("confidence", 0.0),
                data=result,
                metadata={
                    "processing_time": 0.05,
                    "model_version": "1.0.0",
                    "edge_processing": True
                }
            )
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            raise
    
    async def _store_result(self, result: ProcessingResult):
        """Store processing result locally"""
        cursor = self.local_db.cursor()
        
        cursor.execute("""
            INSERT INTO processing_results 
            (device_id, timestamp, result_type, confidence, data, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            result.device_id,
            result.timestamp.isoformat(),
            result.result_type,
            result.confidence,
            json.dumps(result.data),
            json.dumps(result.metadata) if result.metadata else None
        ))
        
        self.local_db.commit()
    
    async def _sync_to_cloud(self, result: Optional[ProcessingResult] = None):
        """Synchronize data with cloud"""
        if self.is_syncing:
            return
        
        self.is_syncing = True
        
        try:
            if result:
                # Sync specific result
                await self._sync_result(result)
            else:
                # Sync all pending results
                await self._sync_pending_results()
            
            self.last_sync_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Cloud sync error: {e}")
            # Queue for retry
            await self._queue_for_retry(result)
        finally:
            self.is_syncing = False
    
    async def _sync_result(self, result: ProcessingResult):
        """Sync single result to cloud"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "device_id": result.device_id,
                    "timestamp": result.timestamp.isoformat(),
                    "result_type": result.result_type,
                    "confidence": result.confidence,
                    "data": result.data,
                    "metadata": result.metadata
                }
                
                async with session.post(
                    f"{self.cloud_endpoint}/api/v1/edge/sync",
                    json=payload,
                    headers={"Authorization": f"Bearer {self._get_auth_token()}"}
                ) as response:
                    if response.status == 200:
                        logger.info("Result synced to cloud")
                    else:
                        raise Exception(f"Sync failed with status {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to sync result: {e}")
            raise
    
    async def _sync_pending_results(self):
        """Sync all pending results to cloud"""
        cursor = self.local_db.cursor()
        
        # Get unsynced results
        cursor.execute("""
            SELECT * FROM processing_results 
            WHERE synced = 0 
            ORDER BY timestamp ASC 
            LIMIT 100
        """)
        
        results = cursor.fetchall()
        
        for result in results:
            try:
                # Convert to ProcessingResult
                processing_result = ProcessingResult(
                    device_id=result[1],
                    timestamp=datetime.fromisoformat(result[2]),
                    result_type=result[3],
                    confidence=result[4],
                    data=json.loads(result[5]),
                    metadata=json.loads(result[6]) if result[6] else None
                )
                
                await self._sync_result(processing_result)
                
                # Mark as synced
                cursor.execute("""
                    UPDATE processing_results 
                    SET synced = 1 
                    WHERE id = ?
                """, (result[0],))
                
            except Exception as e:
                logger.error(f"Failed to sync result {result[0]}: {e}")
        
        self.local_db.commit()
    
    async def _queue_for_retry(self, result: Optional[ProcessingResult]):
        """Queue result for retry"""
        cursor = self.local_db.cursor()
        
        if result:
            cursor.execute("""
                INSERT INTO sync_queue (data_type, data, timestamp)
                VALUES (?, ?, ?)
            """, (
                "processing_result",
                json.dumps(asdict(result)),
                datetime.utcnow().isoformat()
            ))
        
        self.local_db.commit()
    
    async def _get_queue_size(self) -> int:
        """Get size of sync queue"""
        cursor = self.local_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM sync_queue")
        return cursor.fetchone()[0]
    
    def _get_auth_token(self) -> str:
        """Get authentication token for cloud API"""
        # TODO: Implement proper authentication
        return "dummy_token"
    
    async def start_sync_scheduler(self):
        """Start periodic sync scheduler"""
        while True:
            try:
                if not self.offline_mode and not self.is_syncing:
                    await self._sync_to_cloud()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Sync scheduler error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def start(self):
        """Start edge gateway"""
        logger.info(f"Starting Edge Gateway for device {self.device_id}")
        
        # Start sync scheduler
        asyncio.create_task(self.start_sync_scheduler())
        
        logger.info("Edge Gateway started")

class MockInferenceEngine:
    """Mock inference engine for development"""
    
    async def process(self, video_data, steering_data, biometric_data):
        """Mock processing"""
        return {
            "fatigue_detected": False,
            "stress_level": 0.3,
            "confidence": 0.85,
            "anomalies": []
        }

class MockFusionModel:
    """Mock fusion model for development"""
    
    async def fuse(self, modalities):
        """Mock fusion"""
        return {
            "combined_confidence": 0.9,
            "risk_level": "low",
            "recommendations": []
        }

def create_edge_gateway(config_path: str = "edge_config.yaml") -> EdgeGateway:
    """Create edge gateway instance"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return EdgeGateway(config)

def main():
    """Main entry point for edge gateway"""
    config = {
        "device_id": "edge-device-001",
        "cloud_endpoint": "https://api.vigilai.com",
        "sync_interval": 30,
        "offline_mode": False
    }
    
    gateway = EdgeGateway(config)
    
    # Start gateway
    asyncio.run(gateway.start())
    
    # Run FastAPI server
    uvicorn.run(
        gateway.app,
        host="0.0.0.0",
        port=8006,
        log_level="info"
    )

if __name__ == "__main__":
    main()
