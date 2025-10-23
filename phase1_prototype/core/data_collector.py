"""
Data Collector for VigilAI
Handles data collection, storage, and management
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataCollector:
    """Manages data collection and storage for VigilAI"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Data storage settings
        self.data_dir = Path(self.config.get("data_dir", "data"))
        self.max_memory_records = self.config.get("max_memory_records", 1000)
        self.save_interval = self.config.get("save_interval", 60)  # seconds
        
        # In-memory data storage
        self.data_buffer: List[Dict] = []
        self.last_save_time = time.time()
        
        # Data statistics
        self.total_records = 0
        self.data_types = {
            "video": 0,
            "steering": 0,
            "biometric": 0,
            "detection": 0
        }
        
        # Create data directory
        self.data_dir.mkdir(exist_ok=True)
        
    def store_data(self, data: Dict):
        """Store data in memory buffer"""
        try:
            # Add metadata
            data["id"] = self.total_records
            data["stored_at"] = time.time()
            
            # Store in buffer
            self.data_buffer.append(data)
            self.total_records += 1
            
            # Update statistics
            self._update_statistics(data)
            
            # Check if we need to save to disk
            if self._should_save():
                asyncio.create_task(self._save_to_disk())
            
            # Manage buffer size
            if len(self.data_buffer) > self.max_memory_records:
                self.data_buffer = self.data_buffer[-self.max_memory_records:]
                
        except Exception as e:
            logger.error(f"Error storing data: {e}")
    
    def _update_statistics(self, data: Dict):
        """Update data collection statistics"""
        if "video" in data:
            self.data_types["video"] += 1
        if "steering" in data:
            self.data_types["steering"] += 1
        if "biometric" in data:
            self.data_types["biometric"] += 1
        if "detection" in data:
            self.data_types["detection"] += 1
    
    def _should_save(self) -> bool:
        """Check if data should be saved to disk"""
        current_time = time.time()
        return (current_time - self.last_save_time) >= self.save_interval
    
    async def _save_to_disk(self):
        """Save data buffer to disk"""
        if not self.data_buffer:
            return
        
        try:
            current_time = time.time()
            filename = f"vigilai_data_{int(current_time)}.json"
            filepath = self.data_dir / filename
            
            # Save data as JSON
            with open(filepath, 'w') as f:
                json.dump(self.data_buffer, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.data_buffer)} records to {filepath}")
            
            # Clear buffer
            self.data_buffer = []
            self.last_save_time = current_time
            
        except Exception as e:
            logger.error(f"Error saving data to disk: {e}")
    
    def get_recent_data(self, seconds: int = 60) -> List[Dict]:
        """Get recent data from the last N seconds"""
        cutoff_time = time.time() - seconds
        return [data for data in self.data_buffer if data.get("timestamp", 0) > cutoff_time]
    
    def get_data_by_type(self, data_type: str, limit: int = 100) -> List[Dict]:
        """Get data of a specific type"""
        filtered_data = []
        for data in self.data_buffer:
            if data_type in data:
                filtered_data.append(data)
                if len(filtered_data) >= limit:
                    break
        return filtered_data
    
    def get_statistics(self) -> Dict:
        """Get data collection statistics"""
        return {
            "total_records": self.total_records,
            "buffer_size": len(self.data_buffer),
            "data_types": self.data_types.copy(),
            "last_save_time": self.last_save_time,
            "data_dir": str(self.data_dir)
        }
    
    def export_data(self, filepath: str, format: str = "json") -> bool:
        """Export data to file"""
        try:
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.data_buffer, f, indent=2, default=str)
            elif format.lower() == "csv":
                # Convert to DataFrame and save as CSV
                df = pd.DataFrame(self.data_buffer)
                df.to_csv(filepath, index=False)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Data exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False
    
    def clear_data(self):
        """Clear all stored data"""
        self.data_buffer = []
        self.total_records = 0
        self.data_types = {
            "video": 0,
            "steering": 0,
            "biometric": 0,
            "detection": 0
        }
        logger.info("Data cleared")
    
    async def cleanup(self):
        """Cleanup data collector"""
        # Save any remaining data
        if self.data_buffer:
            await self._save_to_disk()
        
        logger.info("Data collector cleanup complete")
