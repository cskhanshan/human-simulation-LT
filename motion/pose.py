import cv2
import os
from ultralytics import YOLO
import mediapipe as mp

# Paths
input_folder = "pose-img"
output_folder = "pose-output"
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

# Init MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True, model_complexity=2,
                          enable_segmentation=True, min_detection_confidence=0.5)

# Process each image
for file in os.listdir(input_folder):
    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_folder, file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Couldn't load: {file}")
        continue

    h, w = image.shape[:2]
    results = yolo_model(image)[0]
    print(f"🧠 YOLO detected {len(results.boxes)} objects in {file}")
    person_boxes = []

    for r in results.boxes:
        cls = int(r.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            print(f"👤 Person detected at box: {x1},{y1},{x2},{y2}")
            # Add margin around person
            margin = 20
            x1 = max(x1 - margin, 0)
            y1 = max(y1 - margin, 0)
            x2 = min(x2 + margin, w)
            y2 = min(y2 + margin, h)
            person_boxes.append((x1, y1, x2, y2))

    if not person_boxes:
        print(f"⚠️ YOLO missed person in {file} — trying full image for pose")
        rgb_full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = pose_model.process(rgb_full)

        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        else:
            print(f"❌ No pose in full image of {file}")
            continue
    else:
        for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
            crop = image[y1:y2, x1:x2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = pose_model.process(rgb_crop)

            if result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    crop,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                image[y1:y2, x1:x2] = crop
            else:
                print(f"⚠️ No pose on cropped person {idx+1} in {file}")

    # Save & show
    save_path = os.path.join(output_folder, f"pose_{file}")
    cv2.imwrite(save_path, image)
    print(f"✅ Saved: {save_path}")

    # Resize if large
    if image.shape[1] > 1000:
        scale = 1000 / image.shape[1]
        image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

    cv2.imshow("Pose Detection", image)
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
