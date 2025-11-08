# Architecture Overview

Simple modular structure - each file handles one responsibility. Makes it easier to understand and modify.

## Main Components

### config.py

All settings in one place. Makes it easy to change things without digging through code.

### detector.py

Uses YOLOv8 for person detection. Chose it because:

- Fast and accurate
- Simple ultralytics API
- Works on CPU/GPU/Apple Silicon
- Multiple model sizes available

What it does:

- Detects people only (class 0)
- Preprocesses dark frames (CLAHE)
- Filters small/wrong aspect ratio detections
- Returns bboxes and confidence scores

### tracker.py

DeepSORT tracking for stable person IDs across frames. Optional - can disable in config.

### zone_manager.py

Loads/saves zones from JSON. Simple format, easy to edit manually if needed.

### zone_marker.py

Interactive tool to mark zones on the video. You click points and it draws polygons.

### intrusion_detector.py

Checks if person center is inside restricted zones. Uses ray casting algorithm (simple and reliable).

### alarm_manager.py

Manages alarm state. Activates on intrusion, deactivates 3 seconds after person leaves (prevents flickering).

### main.py

Ties everything together. Main loop that:

1. Reads video frames
2. Detects people
3. Checks if they're in zones
4. Updates alarm
5. Draws everything on screen

## How It Works

1. YOLO finds people in each frame
2. If tracking is enabled, DeepSORT assigns IDs
3. For each person, check if their center is in a restricted zone
4. If yes, trigger alarm
5. Draw zones, boxes, and alarm text on frame
6. Show the frame

## Reducing False Positives

The system has several ways to avoid false alarms:

1. **Confidence threshold** - YOLO must be confident it's a person
2. **Size filtering** - Ignores detections that are too small or wrong aspect ratio
3. **Track age** - Only checks people who've been tracked for a few frames
4. **Alarm delay** - Alarm stays on for 3 seconds after person leaves (prevents flickering)

## Libraries Used

- **ultralytics** - YOLO models, simple API
- **opencv-python** - Video processing and drawing
- **numpy** - Array operations
- **deep-sort-realtime** - Object tracking (optional)

## Design Principles

- DRY - avoid repetition, use functions
- KISS - keep it simple
- Modular - one file, one responsibility

Pretty straightforward code. Should be easy to follow if you know Python and have used YOLO.

## Useful Resources

Here are some links that helped me understand and implement this project:

### YOLO and Object Detection

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/) - Official docs for YOLOv8, includes installation, usage examples, and API reference
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics) - Source code and examples
- [YOLO: Real-Time Object Detection Paper](https://arxiv.org/abs/1506.02640) - Original YOLO paper (v1)
- [YOLOv8 Tutorial by Roboflow](https://blog.roboflow.com/yolov8/) - Good introduction to YOLOv8
- [COCO Dataset Classes](https://cocodataset.org/#explore) - YOLO uses COCO classes, person is class 0

### OpenCV

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) - Official OpenCV Python tutorials
- [OpenCV Video Processing](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html) - Working with video files
- [OpenCV Drawing Functions](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html) - Drawing shapes, text, etc.
- [CLAHE (Contrast Limited Adaptive Histogram Equalization)](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html) - Image enhancement technique used for dark frames

### Object Tracking

- [DeepSORT Paper](https://arxiv.org/abs/1703.07402) - Simple Online and Realtime Tracking with a Deep Association Metric
- [DeepSORT GitHub](https://github.com/nwojke/deep_sort) - Original DeepSORT implementation
- [deep-sort-realtime PyPI](https://pypi.org/project/deep-sort-realtime/) - Python package used in this project
- [Object Tracking Tutorial](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/) - Basic tracking concepts

### Point-in-Polygon Algorithm

- [Ray Casting Algorithm](https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm) - Wikipedia explanation of the algorithm used
- [Point in Polygon Test](https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/) - GeeksforGeeks tutorial with code examples
- [Winding Number Algorithm](https://en.wikipedia.org/wiki/Point_in_polygon#Winding_number_algorithm) - Alternative algorithm (not used here)

### Video Processing

- [Video Processing with OpenCV](https://opencv-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html) - Reading and writing video files
- [FPS and Frame Timing](https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/) - Understanding video frame rates

### Computer Vision Basics

- [Computer Vision Course (Stanford)](http://cs231n.stanford.edu/) - CS231n course materials
- [PyImageSearch Blog](https://www.pyimagesearch.com/) - Lots of practical computer vision tutorials
- [OpenCV Image Processing](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) - Image preprocessing techniques

### Python and Best Practices

- [Python Style Guide (PEP 8)](https://pep8.org/) - Code style guidelines
- [Real Python](https://realpython.com/) - Python tutorials and best practices
- [Python Logging](https://docs.python.org/3/library/logging.html) - Official logging documentation

### JSON and Data Handling

- [Python JSON Documentation](https://docs.python.org/3/library/json.html) - Working with JSON files
- [JSON Format Specification](https://www.json.org/) - Understanding JSON structure

These resources helped me to understand the concepts/techniques and apply in this project.
