import logging
import sys
from typing import Optional, List


def setup_logging(log_level=logging.INFO, log_file=None):
    logger = logging.getLogger('intrusion_detection')
    logger.setLevel(log_level)
    
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (IOError, OSError) as e:
            logger.warning(f"Failed to create log file {log_file}: {e}")
    
    return logger


def validate_config():
    errors = []
    
    try:
        from config import (
            VIDEO_PATH, ZONES_FILE, YOLO_MODEL, YOLO_CONFIDENCE,
            YOLO_DEVICE, PROCESS_EVERY_N_FRAMES, ALARM_DELAY_SECONDS
        )
        
        import os
        if not os.path.exists(VIDEO_PATH):
            errors.append(f"Video file not found: {VIDEO_PATH}")
        
        # zones file is optional (can be created later)
        # if not os.path.exists(ZONES_FILE):
        #     errors.append(f"Zones file not found: {ZONES_FILE}")
        
        if not 0.0 < YOLO_CONFIDENCE <= 1.0:
            errors.append(f"YOLO_CONFIDENCE must be in range (0.0, 1.0], got: {YOLO_CONFIDENCE}")
        
        if YOLO_DEVICE not in ["cpu", "cuda", "mps"]:
            errors.append(f"Unknown YOLO device: {YOLO_DEVICE}")
        
        if PROCESS_EVERY_N_FRAMES < 1:
            errors.append(f"PROCESS_EVERY_N_FRAMES must be >= 1, got: {PROCESS_EVERY_N_FRAMES}")
        
        if ALARM_DELAY_SECONDS < 0:
            errors.append(f"ALARM_DELAY_SECONDS must be >= 0, got: {ALARM_DELAY_SECONDS}")
        
    except ImportError as e:
        errors.append(f"Configuration import error: {e}")
    except Exception as e:
        errors.append(f"Unexpected validation error: {e}")
    
    return len(errors) == 0, errors

