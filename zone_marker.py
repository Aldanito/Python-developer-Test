import cv2
import numpy as np
from zone_manager import load_zones, save_zones, validate_zone
from config import VIDEO_PATH, ZONE_COLOR, ZONE_THICKNESS, ZONE_MARKER_WINDOW_NAME


class ZoneMarker:
    def __init__(self):
        self.zones = load_zones()
        self.current_zone = []
        self.drawing = False
        self.window_name = ZONE_MARKER_WINDOW_NAME
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_zone.append([x, y])
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            pass  # could add preview line here later
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def draw_zones(self, frame):
        for zone in self.zones:
            if len(zone) >= 2:
                pts = np.array(zone, np.int32)
                cv2.polylines(frame, [pts], True, ZONE_COLOR, ZONE_THICKNESS)
        
        if len(self.current_zone) >= 2:
            pts = np.array(self.current_zone, np.int32)
            cv2.polylines(frame, [pts], False, ZONE_COLOR, ZONE_THICKNESS)
        
        for point in self.current_zone:
            cv2.circle(frame, tuple(point), 5, ZONE_COLOR, -1)
    
    def run(self):
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: failed to open video {VIDEO_PATH}")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: failed to read frame from video")
            return
        
        display_frame = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\nInstructions:")
        print("- Left click: add point to current zone")
        print("- 's': save current zone and add to list")
        print("- 'r': reset current zone")
        print("- 'q': exit and save all zones")
        print("- 'c': clear all zones")
        print()
        
        while True:
            display_frame = frame.copy()
            self.draw_zones(display_frame)
            
            cv2.putText(display_frame, "Left click: add point, 's': save zone, 'r': reset, 'q': quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Zones: {len(self.zones)}, Current points: {len(self.current_zone)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                if self.zones:
                    save_zones(self.zones)
                break
            elif key == ord('s'):
                if validate_zone(self.current_zone):
                    self.zones.append(self.current_zone)
                    self.current_zone = []
                    print(f"Zone saved. Total zones: {len(self.zones)}")
                else:
                    print("Error: zone must contain at least 3 points")
            elif key == ord('r'):
                self.current_zone = []
                print("Current zone reset")
            elif key == ord('c'):
                self.zones = []
                self.current_zone = []
                print("All zones cleared")
        
        cv2.destroyAllWindows()
        print(f"Marking completed. Saved {len(self.zones)} zones")


if __name__ == "__main__":
    marker = ZoneMarker()
    marker.run()

