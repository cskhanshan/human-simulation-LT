import cv2
import os
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
from deepface import DeepFace

# === CONFIGURATION ===
image_path = r"C:\Users\LT\Desktop\human-simulation-LT\motion\pose-img\pose22.jpg"
output_excel = "pose-video-output/image_emotion_dataset_detailed.xlsx"

# Check image exists
if not os.path.exists(image_path):
    print(f"❌ Image not found at: {image_path}")
    exit()

os.makedirs(os.path.dirname(output_excel), exist_ok=True)

# Load models
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                          enable_segmentation=False, min_detection_confidence=0.5)

# Load image
image = cv2.imread(image_path)
if image is None:
    print("❌ Failed to load image")
    exit()

h, w = image.shape[:2]
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pose detection on full image
pose_result = pose_model.process(rgb_image)
if not pose_result.pose_landmarks:
    print("❌ No pose landmarks detected.")
else:
    print(f"✅ Landmarks detected: {len(pose_result.pose_landmarks.landmark)}")

# Emotion detection on face
try:
    analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    print("🎭 Emotion detected:", analysis[0])
    emotion = analysis[0]['dominant_emotion']
    emotion_score = round(analysis[0]['emotion'][emotion], 2)
except:
    emotion = "Unknown"
    emotion_score = 0.0

# Head direction
head_dir = "Unknown"
if pose_result.pose_landmarks:
    lm = pose_result.pose_landmarks.landmark
    if lm[7].visibility > 0.5 and lm[8].visibility > 0.5:
        dx = lm[7].x - lm[8].x
        head_dir = "Left" if dx > 0.05 else "Right" if dx < -0.05 else "Forward"

# Save all landmark data
pose_data = []
if pose_result.pose_landmarks:
    for idx, lm in enumerate(pose_result.pose_landmarks.landmark):
        pose_data.append({
            "Image_Name": os.path.basename(image_path),
            "Landmark_ID": idx,
            "X": lm.x,
            "Y": lm.y,
            "Z": lm.z,
            "Visibility": lm.visibility,
            "Emotion": emotion,
            "Emotion_Score": emotion_score,
            "Head_Direction": head_dir
        })

# Save to Excel
if pose_data:
    df = pd.DataFrame(pose_data)
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Dataset saved: {output_excel}")
else:
    print("⚠️ No data collected.")
