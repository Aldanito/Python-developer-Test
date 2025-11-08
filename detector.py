from ultralytics import YOLO
import cv2
import numpy as np
from config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_DEVICE, YOLO_IMGSZ, RESIZE_FACTOR,
    MIN_DETECTION_AREA_RATIO, MIN_DETECTION_HEIGHT_RATIO,
    MAX_ASPECT_RATIO, MIN_ASPECT_RATIO, IOU_THRESHOLD,
    YOLO_IOU_THRESHOLD, YOLO_MAX_DETECTIONS, MIN_CONFIDENCE_AFTER_FILTER
)


def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def filter_overlapping_detections(detections, iou_threshold=IOU_THRESHOLD):
    if len(detections) <= 1:
        return detections
    
    sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    filtered = []
    for det in sorted_detections:
        is_overlapping = False
        for existing in filtered:
            iou = calculate_iou(det['bbox'], existing['bbox'])
            if iou > iou_threshold:
                is_overlapping = True
                break
        if not is_overlapping:
            filtered.append(det)
    return filtered


class PersonDetector:
    def __init__(self):
        import os
        import logging
        logger = logging.getLogger('intrusion_detection')
        model_path = YOLO_MODEL
        
        # check if model exists, if not try to download or use fallback
        if not os.path.exists(model_path):
            logger.info(f"Model {YOLO_MODEL} not found locally. YOLO will try to download it automatically.")
            logger.info("If download fails, will use fallback to available model.")
            
            fallback_models = []
            if "m.pt" in model_path:
                fallback_models = [
                    model_path.replace("m.pt", "s.pt"),
                    model_path.replace("m.pt", "n.pt"),
                ]
            elif "s.pt" in model_path:
                fallback_models = [
                    model_path.replace("s.pt", "n.pt"),
                ]
            elif "n.pt" in model_path:
                fallback_models = [
                    model_path.replace("n.pt", "s.pt"),
                ]
            elif "l.pt" in model_path or "x.pt" in model_path:
                fallback_models = [
                    model_path.replace("l.pt", "m.pt").replace("x.pt", "m.pt"),
                    model_path.replace("l.pt", "s.pt").replace("x.pt", "s.pt"),
                    model_path.replace("l.pt", "n.pt").replace("x.pt", "n.pt"),
                ]
            
            self.fallback_models = fallback_models
        else:
            self.fallback_models = []
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model {model_path}: {e}")
            if hasattr(self, 'fallback_models') and self.fallback_models:
                found_fallback = False
                for fallback in self.fallback_models:
                    if os.path.exists(fallback):
                        try:
                            self.model = YOLO(fallback)
                            logger.warning(f"Using fallback model: {fallback}")
                            found_fallback = True
                            break
                        except Exception:
                            continue
                
                if not found_fallback:
                    standard_models = ["yolov8s.pt", "yolov8n.pt"]
                    for std_model in standard_models:
                        if os.path.exists(std_model):
                            try:
                                self.model = YOLO(std_model)
                                logger.warning(f"Using available model: {std_model}")
                                break
                            except Exception:
                                continue
                    else:
                        raise Exception(f"Failed to load YOLO model and no available fallback models found")
            else:
                raise Exception(f"Error loading YOLO model: {e}")
        
        self.confidence = YOLO_CONFIDENCE
        
        # handle MPS device (Apple Silicon)
        if YOLO_DEVICE == "mps":
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except:
                self.device = "cpu"  # fallback to cpu if torch not available
        else:
            self.device = YOLO_DEVICE
            
        self.imgsz = YOLO_IMGSZ
        self.resize_factor = RESIZE_FACTOR
        self.detection_history = []
        self.history_size = 0
    
    def detect(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Empty or invalid frame for detection")
        
        original_h, original_w = frame.shape[:2]
        
        if self.resize_factor != 1.0:
            new_w = int(original_w * self.resize_factor)
            new_h = int(original_h * self.resize_factor)
            process_frame = cv2.resize(frame, (new_w, new_h))
        else:
            process_frame = frame
        
        # preprocess dark frames to improve detection
        frame_mean = process_frame.mean()
        
        if frame_mean < 100:
            # use CLAHE for dark frames - helps a lot with detection
            lab = cv2.cvtColor(process_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            darkness_factor = max(0, (100 - frame_mean) / 100.0)
            clip_limit = 2.0 + (darkness_factor * 3.0)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            process_frame = cv2.merge([l, a, b])
            process_frame = cv2.cvtColor(process_frame, cv2.COLOR_LAB2BGR)
            
            # extra boost for very dark frames
            if frame_mean < 70:
                brightness_boost = 1.2 + (darkness_factor * 0.2)
                contrast_boost = 8 + int(darkness_factor * 12)
                process_frame = cv2.convertScaleAbs(process_frame, alpha=brightness_boost, beta=contrast_boost)
        
        half = (self.device == "cuda")
        results = self.model(
            process_frame, 
            conf=self.confidence, 
            device=self.device, 
            imgsz=self.imgsz, 
            verbose=False, 
            iou=YOLO_IOU_THRESHOLD, 
            max_det=YOLO_MAX_DETECTIONS,
            half=half,
            agnostic_nms=False,
            retina_masks=False
        )
        
        detections = []
        h, w = process_frame.shape[:2]
        min_area = (w * h) * MIN_DETECTION_AREA_RATIO
        min_height = h * MIN_DETECTION_HEIGHT_RATIO
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls) == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    if self.resize_factor != 1.0:
                        scale_x = original_w / w
                        scale_y = original_h / h
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                    
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    if area < min_area or height < min_height:
                        continue
                    
                    aspect_ratio = width / height if height > 0 else 0
                    if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
                        continue
                    
                    if confidence < MIN_CONFIDENCE_AFTER_FILTER:
                        continue
                    
                    # filter out boxes that are too wide (probably not a person)
                    if width >= height:
                        continue
                    
                    # aspect ratio check - people are taller than wide
                    if width / height > 0.65:
                        continue
                    
                    # some edge cases for weird detections
                    if height < width * 1.15:
                        if height < 40:
                            continue
                    
                    # minimum size check
                    if height < 25 or width < 12:
                        continue
                    
                    # maximum size check - probably false positive if too big
                    if height > h * 0.65 or width > w * 0.65:
                        continue
                    
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': confidence,
                        'center': (float(center_x), float(center_y))
                    })
        
        detections = filter_overlapping_detections(detections, iou_threshold=IOU_THRESHOLD)
        
        # print(f"Found {len(detections)} detections")  # debug
        return detections

