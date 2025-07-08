import cv2
import os
import torch
from ultralytics import YOLO
import mediapipe as mp

# Paths
input_folder = "pose-img"
output_folder = "pose-output"
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 model (use yolov8n or yolov8s for speed)
yolo_model = YOLO("yolov8n.pt")

# Init MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# MediaPipe pose instance
pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                          enable_segmentation=True, min_detection_confidence=0.5)

# Process images
for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f" Couldn't load: {file}")
            continue

        h, w = image.shape[:2]
        results = yolo_model(image)[0]
        person_boxes = []

        for r in results.boxes:
            cls = int(r.cls[0])
            if cls == 0:  # Class 0 = person in COCO
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))

        if not person_boxes:
            print(f" No person detected in {file}")
            continue

        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            crop = image[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_result = pose_model.process(rgb_crop)

            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    crop,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            image[y1:y2, x1:x2] = crop

        # Save and show
        save_path = os.path.join(output_folder, f"pose_{file}")
        cv2.imwrite(save_path, image)
        print(f" Saved: {save_path}")

        # Resize if too big
        if image.shape[1] > 1000:
            scale = 1000 / image.shape[1]
            image = cv2.resize(image, (int(image.shape[1]*scale), int(image.shape[0]*scale)))

        cv2.imshow("Pose Result", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()
