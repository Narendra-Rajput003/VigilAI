"""
Configuration Manager for VigilAI
Handles configuration loading and management
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for VigilAI"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.config_data = {}
        
        # Default configuration
        self.defaults = {
            "camera": {
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
                "flip_horizontal": True,
                "ir_mode": False
            },
            "obd": {
                "port": "/dev/ttyUSB0",
                "baudrate": 38400,
                "timeout": 1.0,
                "collection_interval": 0.1
            },
            "wearables": {
                "device_type": "mock",
                "connection_timeout": 10.0,
                "data_interval": 1.0,
                "mock_data": True
            },
            "detection": {
                "fatigue_threshold": 0.7,
                "stress_threshold": 0.6,
                "confidence_threshold": 0.5,
                "processing_interval": 0.033  # ~30 FPS
            },
            "intervention": {
                "intervention_types": ["audio", "haptic", "visual"],
                "escalation_levels": 3,
                "cooldown_period": 30,
                "audio_enabled": True,
                "audio_volume": 0.7,
                "haptic_enabled": True,
                "haptic_intensity": 0.5,
                "visual_enabled": True
            },
            "data_collection": {
                "data_dir": "data",
                "max_memory_records": 1000,
                "save_interval": 60
            },
            "logging": {
                "level": "INFO",
                "file": "vigilai.log",
                "max_size": 10485760,  # 10MB
                "backup_count": 5
            },
            "web_interface": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            }
        }
        
        # Load configuration
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    self.config_data = json.load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.info("No configuration file found, using defaults")
                self.config_data = self.defaults.copy()
                self.save_config()  # Save defaults
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config_data = self.defaults.copy()
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            keys = key.split('.')
            config = self.config_data
            
            # Navigate to the parent of the target key
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Set the value
            config[keys[-1]] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting configuration {key}: {e}")
            return False
    
    def get_camera_config(self) -> Dict:
        """Get camera configuration"""
        return self.get("camera", self.defaults["camera"])
    
    def get_obd_config(self) -> Dict:
        """Get OBD-II configuration"""
        return self.get("obd", self.defaults["obd"])
    
    def get_wearables_config(self) -> Dict:
        """Get wearables configuration"""
        return self.get("wearables", self.defaults["wearables"])
    
    def get_detection_config(self) -> Dict:
        """Get detection configuration"""
        return self.get("detection", self.defaults["detection"])
    
    def get_intervention_config(self) -> Dict:
        """Get intervention configuration"""
        return self.get("intervention", self.defaults["intervention"])
    
    def get_data_collection_config(self) -> Dict:
        """Get data collection configuration"""
        return self.get("data_collection", self.defaults["data_collection"])
    
    def get_logging_config(self) -> Dict:
        """Get logging configuration"""
        return self.get("logging", self.defaults["logging"])
    
    def get_web_interface_config(self) -> Dict:
        """Get web interface configuration"""
        return self.get("web_interface", self.defaults["web_interface"])
    
    def update_config(self, updates: Dict) -> bool:
        """Update configuration with new values"""
        try:
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_update(self.config_data, updates)
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Validate configuration values"""
        try:
            # Validate camera settings
            camera_config = self.get_camera_config()
            if not isinstance(camera_config.get("width"), int) or camera_config["width"] <= 0:
                logger.error("Invalid camera width")
                return False
            
            if not isinstance(camera_config.get("height"), int) or camera_config["height"] <= 0:
                logger.error("Invalid camera height")
                return False
            
            # Validate detection thresholds
            detection_config = self.get_detection_config()
            fatigue_threshold = detection_config.get("fatigue_threshold", 0.7)
            if not 0.0 <= fatigue_threshold <= 1.0:
                logger.error("Invalid fatigue threshold")
                return False
            
            stress_threshold = detection_config.get("stress_threshold", 0.6)
            if not 0.0 <= stress_threshold <= 1.0:
                logger.error("Invalid stress threshold")
                return False
            
            # Validate intervention settings
            intervention_config = self.get_intervention_config()
            escalation_levels = intervention_config.get("escalation_levels", 3)
            if not isinstance(escalation_levels, int) or escalation_levels < 1:
                logger.error("Invalid escalation levels")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults"""
        try:
            self.config_data = self.defaults.copy()
            logger.info("Configuration reset to defaults")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting configuration: {e}")
            return False
    
    def export_config(self, filepath: str) -> bool:
        """Export configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config_data, f, indent=2)
            
            logger.info(f"Configuration exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, filepath: str) -> bool:
        """Import configuration from file"""
        try:
            with open(filepath, 'r') as f:
                imported_config = json.load(f)
            
            self.config_data = imported_config
            logger.info(f"Configuration imported from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
