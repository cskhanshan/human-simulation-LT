import cv2
import os
from deepface import DeepFace
import mediapipe as mp

# Folder path
img_folder = "images"
image_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Init mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=15, min_detection_confidence=0.5)

# Define facial connections (face only)
FACIAL_CONNECTIONS = mp_face_mesh.FACEMESH_TESSELATION  # You can also use FACEMESH_CONTOURS for outer face

# Emotion colors
emotion_colors = {
    'happy': (0, 255, 0),
    'neutral': (255, 255, 0),
    'sad': (0, 0, 255),
    'surprise': (255, 0, 255),
    'angry': (0, 165, 255),
    'fear': (128, 0, 128),
    'disgust': (0, 128, 128)
}

def detect_emotion_on_crop(face_img):
    try:
        result = DeepFace.analyze(face_img, actions=['emotion'], detector_backend='retinaface', enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emo = result["dominant_emotion"]
        score = result["emotion"][emo]
        return emo, score
    except:
        return "neutral", 0.0

print("🔎 Processing face skeleton with emotion")

for file in image_files:
    image_path = os.path.join(img_folder, file)
    img = cv2.imread(image_path)

    if img is None:
        print(f"❌ Could not load {file}")
        continue

    h, w = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)

    if results.multi_face_landmarks:
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            # Get bounding box from landmarks
            xs = [int(pt.x * w) for pt in face_landmarks.landmark]
            ys = [int(pt.y * h) for pt in face_landmarks.landmark]
            x_min, y_min, x_max, y_max = max(min(xs) - 20, 0), max(min(ys) - 20, 0), min(max(xs) + 20, w), min(max(ys) + 20, h)

            face_crop = img[y_min:y_max, x_min:x_max]
            emotion, conf = detect_emotion_on_crop(face_crop)
            color = emotion_colors.get(emotion, (255, 255, 255))

            # Draw landmarks and connections
            for connection in FACIAL_CONNECTIONS:
                start_idx, end_idx = connection
                start = face_landmarks.landmark[start_idx]
                end = face_landmarks.landmark[end_idx]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(img, (x1, y1), (x2, y2), color, 1)

            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (x, y), 1, color, -1)

            # Emotion label
            cv2.rectangle(img, (x_min, y_min - 25), (x_min + 150, y_min), color, -1)
            cv2.putText(img, f"{emotion} ({conf:.0f}%)", (x_min + 5, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Resize if too big
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))

    cv2.imshow("Facial Skeleton + Emotion", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break
    cv2.destroyAllWindows()

cv2.destroyAllWindows()
