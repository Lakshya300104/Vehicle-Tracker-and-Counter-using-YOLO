# Vehicle-Tracker-and-Counter-using-YOLO

This project is a **Vehicle Tracker and Counter** built with **Python** and **YOLO** (You Only Look Once) for real-time vehicle detection and counting. The system tracks and counts vehicles such as cars, trucks, and buses moving in a specific direction within a video stream. 

## Features

- **Real-time vehicle detection** using YOLO (v8).
- **Accurate vehicle counting** by detecting vehicles crossing a specific line in the video.
- **GPU Acceleration** with CUDA & cuDNN for faster processing.
- **Tracking** using the SORT (Simple Online and Realtime Tracking) algorithm.
- **Masking** for focusing on a specific area of the video to improve accuracy.
- **Simple visualization** with bounding boxes, counters, and unique ID labels for each vehicle.

## Prerequisites

To run this project, you'll need:

- Python 3.x
- OpenCV
- YOLOv8 weights file
- NumPy
- **CUDA** & **cuDNN** for GPU acceleration (optional but recommended for faster performance)
- **Sort** tracking library for vehicle tracking

## Setup

1. Clone this repository to your local machine:
   
   ```bash
   git clone https://github.com/yourusername/vehicle-tracker.git
   cd vehicle-tracker
   ```

2. Download YOLOv8 weights from the official [Ultralytics repository](https://github.com/ultralytics/yolov8) or other available sources, and place the weights file (`yolov8l.pt`) in the `Yolo-Weights/` directory.

3. Prepare your input video file. Place your video in the `Demo Videos/` folder (or modify the path in the code accordingly).

4. Place your **mask image** (`mask.png`) in the same directory as the script to focus the detection on a specific region.

5. Run the script:

   ```bash
   python vehicle_tracker.py
   ```

The program will process the video and display the real-time vehicle tracking and counting on your screen.

## Code Walkthrough

### Imports and Initialization
```python
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
```
- **YOLO**: Used for vehicle detection.
- **OpenCV (cv2)**: Used for image and video processing.
- **cvzone**: Used for overlaying graphics and adding text on images.
- **math**: For numerical operations.
- **Sort**: Used for vehicle tracking.

### Model and Video Setup
```python
cap = cv2.VideoCapture("../Demo Videos/cars.mp4")  # For Video
model = YOLO("../Yolo-Weights/yolov8l.pt")  # Load YOLOv8 model
```
- The video file (`cars.mp4`) is loaded and the YOLO model is initialized.

### Masking and Tracking Setup
```python
mask = cv2.imread("mask.png")  # Mask image for focusing on area of interest
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # SORT Tracker
```
- The mask image is loaded to specify the area where tracking will occur.
- SORT (Simple Online and Realtime Tracking) is initialized for vehicle tracking.

### Vehicle Detection & Tracking Loop
```python
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)  # Apply mask to focus on region
    results = model(imgRegion, stream=True)  # YOLO detection
    detections = np.empty((0, 5))  # Array to store detection data
```
- The video is read frame by frame.
- YOLO is used to detect vehicles in the masked region.

### Bounding Box and Confidence Filtering
```python
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = math.ceil((box.conf[0] * 100)) / 100
        cls = int(box.cls[0])
        currentClass = classNames[cls]
        if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))
```
- Bounding boxes are drawn around detected vehicles.
- Only vehicles with a confidence greater than 30% are considered.
- The detection results are stored in an array.

### Tracking and Counting
```python
resultsTracker = tracker.update(detections)  # Update tracker with current detections
for result in resultsTracker:
    x1, y1, x2, y2, ID = result
    cx, cy = x1 + w//2, y1 + h//2  # Calculate center of the vehicle
    if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
        if totalCount.count(ID) == 0:
            totalCount.append(ID)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)  # Green line
```
- **SORT** is used to track vehicles across frames. If a vehicle crosses the defined line (limits), its ID is added to the count.

### Final Output
```python
cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
cv2.imshow("Image", img)
```
- The total count is displayed on the screen.
- The final image with vehicle IDs and count is shown.

## Future Improvements
- Add more vehicle types for detection.
- Improve vehicle classification and tracking under occlusion.
- Integrate with cloud storage for real-time data monitoring.
- Use a more optimized tracking algorithm for faster performance.

## Contact
For collaboration or inquiries, feel free to reach out through my [LinkedIn]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/lakshya-arora-76a567259/)).

