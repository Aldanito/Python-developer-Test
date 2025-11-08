from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from config import (
    USE_DEEPSORT, TRACK_EVERY_N_FRAMES, DEEPSORT_RESIZE_FACTOR, 
    DEEPSORT_MAX_AGE, DEEPSORT_N_INIT, MAX_TRACK_DISTANCE
)


class PersonTracker:
    def __init__(self):
        # only init DeepSort if enabled in config
        self.tracker = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT) if USE_DEEPSORT else None
        self.frame_count = 0
        self.last_tracks = []
        self.track_id_map = {}  # maps deepsort IDs to stable IDs
        self.stable_id_counter = 1
        self.stable_tracks = {}
        self.track_history = {}  # not really used but keeping for potential future use
    
    def update(self, detections, frame=None, force_update=False):
        self.frame_count += 1
        
        if not USE_DEEPSORT or self.tracker is None:
            return [{'track_id': None, **det} for det in detections]
        
        if not detections and not force_update:
            return []
        
        should_update = force_update or (self.frame_count % TRACK_EVERY_N_FRAMES == 0)
        
        if not should_update and self.last_tracks:
            if not detections:
                return self.last_tracks
            
            updated_tracks = []
            used_tracks = set()
            
            for det in detections:
                det_center = det.get('center')
                if det_center is None:
                    continue
                
                best_match = None
                min_distance = float('inf')
                
                for idx, track in enumerate(self.last_tracks):
                    if idx in used_tracks:
                        continue
                    track_center = track.get('center')
                    if track_center is None:
                        continue
                    
                    dx = det_center[0] - track_center[0]
                    dy = det_center[1] - track_center[1]
                    distance = (dx * dx + dy * dy) ** 0.5
                    
                    if distance < min_distance and distance < MAX_TRACK_DISTANCE:
                        min_distance = distance
                        best_match = (idx, track)
                
                if best_match:
                    idx, track = best_match
                    used_tracks.add(idx)
                    updated_tracks.append({
                        'track_id': track['track_id'],
                        'bbox': det['bbox'],
                        'center': det['center']
                    })
                else:
                    updated_tracks.append({
                        'track_id': None,
                        'bbox': det['bbox'],
                        'center': det['center']
                    })
            
            return updated_tracks if updated_tracks else []
        
        if not detections:
            if force_update and frame is not None:
                _ = self.tracker.update_tracks([], frame=frame)
                self.last_tracks = []
            return []
        
        if frame is None:
            return [{'track_id': None, 'bbox': det.get('bbox'), 'center': det.get('center')} for det in detections]
        
        scale_x = 1.0
        scale_y = 1.0
        if DEEPSORT_RESIZE_FACTOR < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * DEEPSORT_RESIZE_FACTOR)
            new_h = int(h * DEEPSORT_RESIZE_FACTOR)
            small_frame = cv2.resize(frame, (new_w, new_h))
            scale_x = new_w / w
            scale_y = new_h / h
        else:
            small_frame = frame
        
        detections_list = []
        for det in detections:
            bbox = det.get('bbox')
            if bbox is None:
                continue
                
            x1 = bbox[0] * scale_x
            y1 = bbox[1] * scale_y
            x2 = bbox[2] * scale_x
            y2 = bbox[3] * scale_y
            
            left = max(0, int(x1))
            top = max(0, int(y1))
            width = max(1, int(x2 - x1))
            height = max(1, int(y2 - y1))
            
            if small_frame is not None:
                small_h, small_w = small_frame.shape[:2]
                if left >= small_w or top >= small_h:
                    continue
                if left + width > small_w:
                    width = small_w - left
                if top + height > small_h:
                    height = small_h - top
            
            confidence = det.get('confidence', 0.5)
            detections_list.append(([left, top, width, height], confidence, 'person'))
        
        if not detections_list:
            if force_update and small_frame is not None:
                _ = self.tracker.update_tracks([], frame=small_frame)
            self.last_tracks = []
            return []
        
        if small_frame is None:
            return []
            
        tracks = self.tracker.update_tracks(detections_list, frame=small_frame)
        
        current_deepsort_ids = set()
        result = []
        
        for track in tracks:
            if track.is_confirmed():
                ltrb = track.to_ltrb()
                deepsort_id = track.track_id
                current_deepsort_ids.add(deepsort_id)
                
                if scale_x != 1.0 or scale_y != 1.0:
                    x1 = ltrb[0] / scale_x
                    y1 = ltrb[1] / scale_y
                    x2 = ltrb[2] / scale_x
                    y2 = ltrb[3] / scale_y
                else:
                    x1, y1, x2, y2 = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                center = (float(center_x), float(center_y))
                
                stable_id = self._get_stable_id(deepsort_id, center, [float(x1), float(y1), float(x2), float(y2)])
                
                result.append({
                    'track_id': stable_id,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': center
                })
        
        self._cleanup_old_tracks(current_deepsort_ids)
        self.last_tracks = result
        return result
    
    def _get_stable_id(self, deepsort_id, center, bbox):
        if deepsort_id in self.track_id_map:
            stable_id = self.track_id_map[deepsort_id]
            if stable_id in self.stable_tracks:
                self.stable_tracks[stable_id]['last_center'] = center
                self.stable_tracks[stable_id]['last_bbox'] = bbox
                self.stable_tracks[stable_id]['last_seen'] = self.frame_count
                self.stable_tracks[stable_id]['deepsort_id'] = deepsort_id
            return stable_id
        
        best_match_id = None
        best_distance = float('inf')
        max_match_distance = MAX_TRACK_DISTANCE
        
        for stable_id, track_info in self.stable_tracks.items():
            if track_info['last_seen'] < self.frame_count - 10:
                continue
            
            last_center = track_info.get('last_center')
            if last_center is None:
                continue
            
            dx = center[0] - last_center[0]
            dy = center[1] - last_center[1]
            distance = (dx * dx + dy * dy) ** 0.5
            
            if distance < best_distance and distance < max_match_distance:
                best_distance = distance
                best_match_id = stable_id
        
        if best_match_id is not None:
            stable_id = best_match_id
            self.track_id_map[deepsort_id] = stable_id
            self.stable_tracks[stable_id]['last_center'] = center
            self.stable_tracks[stable_id]['last_bbox'] = bbox
            self.stable_tracks[stable_id]['last_seen'] = self.frame_count
            self.stable_tracks[stable_id]['deepsort_id'] = deepsort_id
        else:
            stable_id = self.stable_id_counter
            self.stable_id_counter += 1
            self.track_id_map[deepsort_id] = stable_id
            self.stable_tracks[stable_id] = {
                'last_center': center,
                'last_bbox': bbox,
                'last_seen': self.frame_count,
                'deepsort_id': deepsort_id
            }
        
        return stable_id
    
    def _cleanup_old_tracks(self, current_deepsort_ids):
        # remove tracks that haven't been seen in a while
        frames_since_seen_threshold = 30  # arbitrary but works
        
        tracks_to_remove = []
        for stable_id, track_info in self.stable_tracks.items():
            if self.frame_count - track_info['last_seen'] > frames_since_seen_threshold:
                tracks_to_remove.append(stable_id)
        
        for stable_id in tracks_to_remove:
            deepsort_id = self.stable_tracks[stable_id].get('deepsort_id')
            if deepsort_id in self.track_id_map:
                del self.track_id_map[deepsort_id]
            del self.stable_tracks[stable_id]
        
        deepsort_ids_to_remove = []
        for deepsort_id in self.track_id_map.keys():
            if deepsort_id not in current_deepsort_ids:
                deepsort_ids_to_remove.append(deepsort_id)
        
        for deepsort_id in deepsort_ids_to_remove:
            del self.track_id_map[deepsort_id]

