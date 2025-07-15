import cv2
import os
import pandas as pd
from ultralytics import YOLO
import mediapipe as mp
from deepface import DeepFace

# === CONFIGURATION ===
video_path = r"C:\Users\LT\Desktop\human-simulation-LT\motion\pose-img\angry.mp4"
output_folder = "pose-video-output"
os.makedirs(output_folder, exist_ok=True)
excel_path = os.path.join(output_folder, "emotion_dataset.xlsx")

# === PATH CHECK ===
if not os.path.exists(video_path):
    print(f"❌ Video not found at: {video_path}")
    exit()

# === Load Models ===
yolo_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                          enable_segmentation=False, min_detection_confidence=0.5)

# === Open Video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(total_frames / fps)
pose_data = []

for sec in range(duration):
    frame_number = sec * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if not success:
        continue

    h, w = frame.shape[:2]
    person_boxes = []

    try:
        results = yolo_model(frame)[0]
        for r in results.boxes:
            if int(r.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                margin = 20
                x1 = max(x1 - margin, 0)
                y1 = max(y1 - margin, 0)
                x2 = min(x2 + margin, w)
                y2 = min(y2 + margin, h)
                person_boxes.append((x1, y1, x2, y2))

        if not person_boxes:
            person_boxes = [(0, 0, w, h)]

        for i, (x1, y1, x2, y2) in enumerate(person_boxes[:1]):  # Only first person
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            resized_crop = cv2.resize(crop, (256, 256))
            rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)

            # Pose detection
            try:
                pose_result = pose_model.process(rgb_crop)
            except:
                continue

            # Emotion detection
            try:
                emotion_data = DeepFace.analyze(resized_crop, actions=['emotion'], enforce_detection=False)
                emotion = emotion_data[0]['dominant_emotion']
                emotion_score = round(emotion_data[0]['emotion'][emotion], 2)
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

            # Store summary row per second
            pose_data.append({
                "Second": sec,
                "Face_X": x1,
                "Face_Y": y1,
                "Face_W": x2 - x1,
                "Face_H": y2 - y1,
                "Emotion": emotion,
                "Emotion_Score": emotion_score,
                "Head_Direction": head_dir,
                "Total_Landmarks": len(pose_result.pose_landmarks.landmark) if pose_result.pose_landmarks else 0
            })

            print(f"✅ Second {sec} | Emotion: {emotion} ({emotion_score}) | Head: {head_dir}")
            break

    except Exception as e:
        print(f"⚠️ Skipping second {sec} due to error: {e}")

cap.release()
cv2.destroyAllWindows()

# === Save Excel ===
df = pd.DataFrame(pose_data)
df.to_excel(excel_path, index=False)
print(f"\n🎯 Final dataset saved to: {excel_path}")
