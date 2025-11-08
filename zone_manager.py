"""
Module for managing restricted zones
"""
import json
import os
import logging
from config import ZONES_FILE, MIN_ZONE_POINTS

logger = logging.getLogger('intrusion_detection')


def load_zones():
    if not os.path.exists(ZONES_FILE):
        return []
    
    try:
        with open(ZONES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # handle both list format and dict format
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'zones' in data:
                return data['zones']
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error when loading zones: {e}")
        return []
    except IOError as e:
        logger.error(f"I/O error when loading zones: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error when loading zones: {e}", exc_info=True)
        return []


def save_zones(zones):
    try:
        with open(ZONES_FILE, 'w', encoding='utf-8') as f:
            json.dump(zones, f, indent=2, ensure_ascii=False)
        logger.info(f"Zones saved to {ZONES_FILE}")
    except IOError as e:
        logger.error(f"I/O error when saving zones: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving zones: {e}", exc_info=True)
        raise


def validate_zone(zone):
    if not isinstance(zone, list):
        return False
    if len(zone) < MIN_ZONE_POINTS:
        return False
    for point in zone:
        if not isinstance(point, list) or len(point) != 2:
            return False
        if not all(isinstance(coord, (int, float)) for coord in point):
            return False
    return True

