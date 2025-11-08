import cv2
import time
import numpy as np

import sys
import os
from detector import PersonDetector
from tracker import PersonTracker
from intrusion_detector import IntrusionDetector
from alarm_manager import AlarmManager
from zone_manager import load_zones
from utils import setup_logging, validate_config
from config import (
    VIDEO_PATH, ZONE_COLOR, ZONE_THICKNESS, DETECTION_COLOR, 
    DETECTION_THICKNESS, ALARM_COLOR, ALARM_FONT_SCALE, 
    ALARM_FONT_THICKNESS, SHOW_FPS, FPS_POSITION, PROCESS_EVERY_N_FRAMES,
    WINDOW_NAME,
    SHOW_ZONE_FILL, ZONE_FILL_ALPHA,
    KEY_QUIT, TRACK_IOU_THRESHOLD, TRACK_CENTER_DISTANCE_THRESHOLD,
    MIN_TRACK_DISTANCE
)

logger = setup_logging()


def initialize_components():
    logger.info("Initializing system...")
    
    # TODO: maybe add retry logic here if model download fails
    try:
        detector = PersonDetector()
        logger.info("Detector initialized")
    except Exception as e:
        logger.error(f"Error initializing detector: {e}", exc_info=True)
        raise
    
    tracker = PersonTracker()
    logger.info("Tracker initialized")
    
    try:
        intrusion_detector = IntrusionDetector()
        logger.info("Intrusion detector initialized")
    except Exception as e:
        logger.error(f"Error initializing intrusion detector: {e}", exc_info=True)
        raise
    
    alarm_manager = AlarmManager()
    logger.info("Alarm manager initialized")
    
    return detector, tracker, intrusion_detector, alarm_manager


def process_frame(frame, frame_count, detector, tracker, intrusion_detector, alarm_manager, last_detections, last_tracks):
    should_process = (frame_count % PROCESS_EVERY_N_FRAMES == 0)
    
    if should_process:
        try:
            detections = detector.detect(frame)
            last_detections = detections
        except Exception as e:
            logger.error(f"Detection error on frame {frame_count}: {e}", exc_info=True)
            detections = last_detections  # use previous detections if current fails
    else:
        detections = last_detections
    
    if detections:
        intrusion_info = intrusion_detector.check_detections(detections)
        intruding_count = intrusion_info.get('count', 0)
        # passing track IDs as range is a bit hacky but works for now
        alarm_manager.update(list(range(intruding_count)))
    else:
        intrusion_info = {'count': 0, 'detections': [], 'zones': intrusion_detector.zones}
        alarm_manager.update([])
    
    alarm_status = alarm_manager.get_alarm_status()
    tracks = [{'bbox': d.get('bbox'), 'center': d.get('center')} for d in detections]
    
    return last_detections, tracks, intrusion_info, alarm_status


def draw_zones(frame, zones, show_fill=SHOW_ZONE_FILL, alpha=ZONE_FILL_ALPHA):
    overlay = frame.copy() if show_fill else None
    
    for zone in zones:
        if len(zone) >= 2:
            pts = np.array(zone, np.int32)
            cv2.polylines(frame, [pts], True, ZONE_COLOR, ZONE_THICKNESS)

            if show_fill and overlay is not None:
                cv2.fillPoly(overlay, [pts], ZONE_COLOR)
    
    if show_fill and overlay is not None:
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def calculate_iou_tracks(track1, track2):
    bbox1 = track1.get('bbox', [])
    bbox2 = track2.get('bbox', [])
    
    if len(bbox1) != 4 or len(bbox2) != 4:
        return 0.0
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def filter_duplicate_tracks(tracks, iou_threshold=TRACK_IOU_THRESHOLD, center_distance_threshold=TRACK_CENTER_DISTANCE_THRESHOLD, min_track_distance=MIN_TRACK_DISTANCE):
    # quick exit if nothing to filter
    if len(tracks) <= 1:
        return tracks
    
    filtered = []
    used_indices = set()
    used_track_ids = set()
    
    for i, track1 in enumerate(tracks):
        if i in used_indices:
            continue
        
        track_id1 = track1.get('track_id')
        if track_id1 is None:
            continue
        
        if track_id1 in used_track_ids:
            continue
        
        best_track = track1
        best_idx = i
        best_score = 0
        
        center1 = track1.get('center', (0, 0))
        bbox1 = track1.get('bbox', [])
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) if len(bbox1) == 4 else 0
        
        is_duplicate_found = False
        
        for j, track2 in enumerate(tracks[i+1:], start=i+1):
            if j in used_indices:
                continue
            
            track_id2 = track2.get('track_id')
            if track_id2 is None:
                continue
            
            if track_id1 == track_id2:
                is_duplicate_found = True
                center2 = track2.get('center', (0, 0))
                dx = center1[0] - center2[0]
                dy = center1[1] - center2[1]
                center_distance = (dx * dx + dy * dy) ** 0.5
                
                if center_distance > min_track_distance:
                    bbox2 = track2.get('bbox', [])
                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) if len(bbox2) == 4 else 0
                    
                    used_indices.add(j)
                    score = area2
                    if score > best_score:
                        best_track = track2
                        best_idx = j
                        best_score = score
                else:
                    used_indices.add(j)
            else:
                iou = calculate_iou_tracks(track1, track2)
                center2 = track2.get('center', (0, 0))
                
                dx = center1[0] - center2[0]
                dy = center1[1] - center2[1]
                center_distance = (dx * dx + dy * dy) ** 0.5
                
                bbox2 = track2.get('bbox', [])
                area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) if len(bbox2) == 4 else 0
                
                is_duplicate = False
                if iou > iou_threshold:
                    is_duplicate = True
                elif center_distance < center_distance_threshold and iou > 0.1:
                    is_duplicate = True
                
                if is_duplicate:
                    used_indices.add(j)
                    score = area2
                    if score > best_score:
                        best_track = track2
                        best_idx = j
                        best_score = score
        
        if best_idx not in used_indices:
            filtered.append(best_track)
            used_indices.add(best_idx)
            used_track_ids.add(track_id1)
    
    return filtered




def draw_detections(frame, detections, intrusion_info=None):
    if intrusion_info is None:
        intrusion_info = {'detections': []}
    
    intruding_centers = set()
    for det in intrusion_info.get('detections', []):
        center = det.get('center')
        if center:
            intruding_centers.add((round(center[0]), round(center[1])))
    
    for det in detections:
        bbox = det.get('bbox')
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        
        if x2 <= x1 or y2 <= y1:
            continue  # skip invalid boxes
        
        center = det.get('center')
        if center:
            c = (round(center[0]), round(center[1]))
            if c in intruding_centers:
                color = ALARM_COLOR
            else:
                color = DETECTION_COLOR
        else:
            color = DETECTION_COLOR
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, DETECTION_THICKNESS)
        

def draw_alarm(frame, is_alarm_active):
    if is_alarm_active:
        text = "ALARM!"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                    ALARM_FONT_SCALE, ALARM_FONT_THICKNESS)[0]
        h, w = frame.shape[:2]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, ALARM_FONT_SCALE, 
                   ALARM_COLOR, ALARM_FONT_THICKNESS)


def main():
    try:
        logger.info("Checking configuration...")
        is_valid, errors = validate_config()
        if not is_valid:
            logger.error("Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            logger.error("Fix errors and restart the application.")
            return
        
        try:
            detector, tracker, intrusion_detector, alarm_manager = initialize_components()
        except Exception:
            return
        
        try:
            zones = load_zones()
            if not zones:
                logger.warning("Zones not found. Run zone_marker.py to mark zones.")
                return
            logger.info(f"Loaded {len(zones)} zones")
        except Exception as e:
            logger.error(f"Error loading zones: {e}", exc_info=True)
            return
        
        try:
            cap = cv2.VideoCapture(VIDEO_PATH)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {VIDEO_PATH}")
                return
        except Exception as e:
            logger.error(f"Error opening video: {e}", exc_info=True)
            return
    
        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_delay = 1.0 / video_fps if video_fps > 0 else 0.04
            
            if video_fps <= 0:
                logger.warning(f"Failed to determine video FPS, using default: {1.0/frame_delay:.2f}")
        except Exception as e:
            logger.error(f"Error getting video parameters: {e}", exc_info=True)
            cap.release()
            return
        
        from config import YOLO_DEVICE
        if YOLO_DEVICE == "mps":
            device_info = "MPS (Apple Silicon)"
        else:
            device_info = YOLO_DEVICE.upper()
        
        logger.info(f"Video: {width}x{height}, FPS: {video_fps:.2f}")
        logger.info(f"Processing device: {device_info}")
        logger.info("Press 'q' to exit")
        
        frame_count = 0
        start_time = time.time()  # not really used but keeping for now
        last_detections = []
        last_tracks = []
        target_frame_time = frame_delay
        processing_times = []
        fps_history_size = 30  # seems like a good number
        
        logger.info("Starting video processing...")
        logger.info("Press 'q' to exit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Reached end of video")
                    break
                
                if frame is None:
                    continue
                
                frame_count += 1
                frame_start_time = time.time()
                
                try:
                    last_detections, last_tracks, intrusion_info, alarm_status = process_frame(
                        frame, frame_count, detector, tracker, 
                        intrusion_detector, alarm_manager,
                        last_detections, last_tracks
                    )
                    
                    processing_time = time.time() - frame_start_time
                    processing_times.append(processing_time)
                    if len(processing_times) > fps_history_size:
                        processing_times.pop(0)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}", exc_info=True)
                    continue
                
                try:
                    draw_zones(frame, zones, SHOW_ZONE_FILL, ZONE_FILL_ALPHA)
                    
                    if last_detections:
                        draw_detections(frame, last_detections, intrusion_info)
                    
                    draw_alarm(frame, alarm_status)
                    
                    if SHOW_FPS and processing_times:
                        avg_time = sum(processing_times) / len(processing_times)
                        fps = 1.0 / avg_time if avg_time > 0 else 0
                        cv2.putText(frame, f"FPS: {fps:.1f}", FPS_POSITION, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Detections: {len(last_detections)}", 
                                   (FPS_POSITION[0], FPS_POSITION[1] + 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except Exception as e:
                    logger.error(f"Error drawing on frame {frame_count}: {e}", exc_info=True)
                
                cv2.imshow(WINDOW_NAME, frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == KEY_QUIT:
                    logger.info("Exit requested by user")
                    break
                
                # try to maintain video fps
                if processing_time > 0:
                    if processing_time < target_frame_time * 0.8:
                        sleep_time = target_frame_time - processing_time
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
        except Exception as e:
            logger.error(f"Critical error on frame {frame_count}: {e}", exc_info=True)
        
        finally:
            logger.info("Releasing resources...")
            
            try:
                cap.release()
            except Exception as e:
                logger.error(f"Error releasing video: {e}")
            
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logger.error(f"Error closing windows: {e}")
            
            logger.info(f"Processing completed. Processed {frame_count} frames")
    
    except Exception as e:
        logger.critical(f"Critical error in main(): {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

