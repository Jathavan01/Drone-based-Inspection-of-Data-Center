import cv2
import numpy as np
import time
from djitellopy import Tello, TelloException
from ultralytics import YOLO  # YOLOv8 for object detection

# Initialize Tello (No Flying)
tello = Tello()
tello.connect()
tello.send_control_command("command")  # Ensures SDK mode is active
tello.streamon()  # Start video stream

# Load YOLOv8 Model (Pre-trained on COCO dataset)
model = YOLO("best.pt")  # Use "yolov8n.pt" (nano) or "yolov8s.pt" (small)

# ArUco dictionary and parameters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Camera calibration values
FOCAL_LENGTH = 780.26  # User's focal length
MARKER_SIZE = 10  # cm

# Target marker ID
TARGET_MARKER = 200  # ArUco Marker ID to track

# Function to detect ArUco markers
def detect_marker(frame, target_marker):
    frame_height, frame_width, _ = frame.shape
    frame_center_x = frame_width // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_height), (255, 0, 0), 2)  # Draw centerline

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == target_marker:
                marker_corners = corners[i][0]
                center_x = int((marker_corners[0][0] + marker_corners[2][0]) / 2)

                # Draw bounding box around marker
                cv2.polylines(frame, [np.int32(marker_corners)], True, (0, 255, 0), 2)
                cv2.putText(frame, f"ID {target_marker}", (center_x - 20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Distance estimation
                marker_width_pixels = marker_corners[1][0] - marker_corners[0][0]
                distance = (MARKER_SIZE * FOCAL_LENGTH) / marker_width_pixels

                return frame, center_x, distance, frame_center_x  # Also return frame center for alignment

    return frame, None, None, None

# Function to detect objects using YOLO
def detect_object(frame):
    # Run YOLOv8 object detection
    results = model(frame)  # Predict the objects in the frame

    # Extract detection results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding box coordinates (x1, y1, x2, y2)
    confidences = results[0].boxes.conf.cpu().numpy()  # Extract confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Extract class indices

    # Get the class names from results.names
    class_names = results[0].names  # This maps the class ID to the class name

    # Initialize tracking variables
    target_detected = False
    target_center = (0, 0)
    target_bounding_box = None
    target_confidence = 0

    # Loop through all detections and decide drone actions
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        conf = confidences[i]
        class_id = class_ids[i]
        class_name = class_names[class_id]  # Get the name of the detected object

        # Draw the bounding box for every detected object
        color = (0, 255, 0)  # Green color for bounding boxes (you pip uninstcan change this)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw the bounding box

        # Label with class name and confidence
        label = f'{class_name} {conf:.2f}'  # Label with class name and confidence
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # If the target person is detected (or any other object of interest), update the tracking
        if class_name == 'bottle' and conf > 0.5:  # Only act on 'person' class for now (can be modified)
            target_detected = True
            target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            target_bounding_box = (x1, y1, x2, y2)

    return frame

# Function to process video stream (No Drone Movement)
def process_video():
    print("🔍 Starting ArUco & YOLO Detection (No Flight)...")

    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect ArUco marker
        frame, marker_x, distance, frame_center_x = detect_marker(frame, TARGET_MARKER)

        if marker_x is not None:
            print(f"📏 ArUco Marker {TARGET_MARKER} detected! Distance: {distance:.2f} cm")

        # Detect objects in the frame
        frame = detect_object(frame)

        # Display the live feed
        cv2.imshow("Tello Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🚪 Exiting...")
            break

    # Cleanup
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()

if __name__ == "__main__":
    process_video()
