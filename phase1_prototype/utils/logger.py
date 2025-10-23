"""
Logging utilities for VigilAI
Handles logging configuration and setup
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

def setup_logging(module_name: str, config: Optional[dict] = None) -> logging.Logger:
    """
    Setup logging for a module
    
    Args:
        module_name: Name of the module
        config: Optional logging configuration
        
    Returns:
        Configured logger instance
    """
    # Default logging configuration
    default_config = {
        "level": "INFO",
        "file": "vigilai.log",
        "max_size": 10485760,  # 10MB
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "console_output": True
    }
    
    if config:
        default_config.update(config)
    
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(getattr(logging, default_config["level"].upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(default_config["format"])
    
    # Console handler
    if default_config.get("console_output", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    log_file = Path(default_config["file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=default_config["max_size"],
        backupCount=default_config["backup_count"]
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def get_logger(module_name: str) -> logging.Logger:
    """Get a logger for a module"""
    return logging.getLogger(module_name)

def setup_root_logging(config: Optional[dict] = None):
    """Setup root logging configuration"""
    # Default configuration
    default_config = {
        "level": "INFO",
        "file": "vigilai.log",
        "max_size": 10485760,  # 10MB
        "backup_count": 5,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
    
    if config:
        default_config.update(config)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, default_config["level"].upper()),
        format=default_config["format"],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.handlers.RotatingFileHandler(
                default_config["file"],
                maxBytes=default_config["max_size"],
                backupCount=default_config["backup_count"]
            )
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
