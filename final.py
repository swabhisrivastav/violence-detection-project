import cv2
import numpy as np
import time
import os
import csv
import json
import logging
from collections import deque
from datetime import datetime
from imutils import resize
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet
from model import Model  # Importing the violence detection model

# Initialize logging
log_filename = 'detection_log.txt'
if os.path.exists(log_filename):
    os.remove(log_filename)
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# Constants for violence detection
DETECTION_THRESHOLD = 40
TIME_WINDOW = 3
ALERT_COOLDOWN = 10
BATCH_SIZE = 9
MOTION_THRESHOLD = 500

# YOLO Configuration
YOLO_WEIGHTS = "YOLOv4-tiny/yolov4-tiny.weights"
YOLO_CONFIG = "YOLOv4-tiny/yolov4-tiny.cfg"
FRAME_SIZE = 416  # Resolution for YOLO processing
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Deep SORT configuration
MAX_COSINE_DISTANCE = 0.7
NN_BUDGET = None
TRACK_MAX_AGE = 30
MODEL_FILENAME = 'model_data/mars-small128.pb'

# Initialize the crowd detection model
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

encoder = gdet.create_box_encoder(MODEL_FILENAME, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
tracker = Tracker(metric, max_age=TRACK_MAX_AGE)

# Create output directory
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
crowd_data_file = open('processed_data/crowd_data.csv', 'w')
crowd_data_writer = csv.writer(crowd_data_file)
if os.path.getsize('processed_data/crowd_data.csv') == 0:
    crowd_data_writer.writerow(['Time', 'Human Count', 'Violence Alert'])

# Function to detect violence
def detect_violence(model, frames, detection_times, last_alert_time):
    labels = model.predict_batch(frames)
    current_time = time.time()
    for label_dict in labels:
        label = label_dict['label']
        if 'violence' in label.lower():
            detection_times.append(current_time)
            if len(detection_times) >= DETECTION_THRESHOLD and current_time - last_alert_time >= ALERT_COOLDOWN:
                logging.info("Alert: Violence detected!")
                return "Violence Detected!", current_time
        else:
            detection_times.clear()
    return "No Violence", last_alert_time

# Real-time processing for crowd and violence detection
def process_real_time():
    print("Starting real-time processing...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access the webcam")
        return

    model = Model()  # Initialize violence detection model
    detection_times = deque()
    last_alert_time = 0
    frame_batch = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame from the webcam")
            break

        frame = resize(frame, width=FRAME_SIZE)

        # YOLO crowd detection
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (FRAME_SIZE, FRAME_SIZE), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        boxes, confidences, class_ids = [], [], []
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONFIDENCE_THRESHOLD and class_id == 0:
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=NMS_THRESHOLD)
        detections = []
        if len(indices) > 0:
            boxes_for_encoding = [boxes[i] for i in indices.flatten()]
            x, y, w, h = box  # Extract the bounding box dimensions
            centroid = (x + w / 2, y + h / 2)
            features = encoder(frame, boxes_for_encoding)
            for i, box in enumerate(boxes_for_encoding):
                detections.append(
                    Detection(tlwh=box, confidence=confidences[indices.flatten()[i]], feature=features[i],centroid=centroid)
                )

        tracker.predict()
        tracker.update(detections, time=frame_count)

        human_count = sum(1 for track in tracker.tracks if track.is_confirmed() and track.time_since_update <= 1)

        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        # Batch violence detection
        violence_status = ""
        frame_batch.append(frame)
        if len(frame_batch) == BATCH_SIZE:
            violence_status, last_alert_time = detect_violence(model, frame_batch, detection_times, last_alert_time)
            frame_batch = []

        # Display crowd count and violence detection status
        cv2.putText(frame, f"Human Count: {human_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, violence_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Log to CSV
        current_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        crowd_data_writer.writerow([current_time, human_count, violence_status])

        # Display the frame
        cv2.imshow("Crowd and Violence Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time processing ended.")

# Run the integrated program
if __name__ == "__main__":
    process_real_time()
    crowd_data_file.close()
