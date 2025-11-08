from zone_manager import load_zones
from config import (
    MIN_TRACK_AGE_FOR_INTRUSION, MIN_INTRUSION_FRAMES, INTRUSION_CONFIRMATION_RATIO
)


def point_in_polygon(point, polygon):
    # ray casting algorithm - standard point in polygon test
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        xinters = p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


class IntrusionDetector:
    def __init__(self):
        self.zones = load_zones()
        self.track_history = {}
        self.track_frame_count = {}
    
    def reload_zones(self):
        self.zones = load_zones()
    
    def check_intrusion(self, center_point):
        for zone in self.zones:
            if point_in_polygon(center_point, zone):
                return True
        return False
    
    def check_detections(self, detections):
        intruding_count = 0
        intruding_detections = []
        
        for det in detections:
            center = det.get('center')
            if not center:
                continue
            
            if self.check_intrusion(center):
                intruding_count += 1
                intruding_detections.append(det)
        
        return {
            'count': intruding_count,
            'detections': intruding_detections,
            'zones': self.zones
        }
    
    def check_tracks(self, tracks):
        intrusions = {}
        current_track_ids = set()
        
        for track in tracks:
            track_id = track.get('track_id')
            center = track.get('center')
            
            if not center or track_id is None:
                continue
            
            current_track_ids.add(track_id)
            
            if track_id not in self.track_frame_count:
                self.track_frame_count[track_id] = 0
                self.track_history[track_id] = []
            
            self.track_frame_count[track_id] += 1
            is_in_zone = self.check_intrusion(center)
            self.track_history[track_id].append(is_in_zone)
            
            # keep history limited to avoid memory issues
            if len(self.track_history[track_id]) > 20:
                self.track_history[track_id].pop(0)  # remove oldest
            
            track_age = self.track_frame_count[track_id]
            
            if track_age < MIN_TRACK_AGE_FOR_INTRUSION:
                intrusions[track_id] = False
                continue
            
            recent_history = self.track_history[track_id][-MIN_INTRUSION_FRAMES:]
            if len(recent_history) < MIN_INTRUSION_FRAMES:
                intrusions[track_id] = False
                continue
            
            intrusion_ratio = sum(recent_history) / len(recent_history)
            
            if intrusion_ratio >= INTRUSION_CONFIRMATION_RATIO:
                intrusions[track_id] = True
            else:
                intrusions[track_id] = False
        
        for track_id in list(self.track_history.keys()):
            if track_id not in current_track_ids:
                del self.track_history[track_id]
                del self.track_frame_count[track_id]
        
        return intrusions

