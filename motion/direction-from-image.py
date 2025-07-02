import cv2
import mediapipe as mp
import numpy as np
import os

folder_path = "images"  # Folder path inside motion folder

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10)

valid_extensions = ['.jpg', '.jpeg', '.png']

for filename in os.listdir(folder_path):
    if not any(filename.lower().endswith(ext) for ext in valid_extensions):
        continue

    image_path = os.path.join(folder_path, filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {filename}")
        continue

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    print(f"\nAnalyzing {filename}...")

    if not results.multi_face_landmarks:
        print("No face detected.")
        continue

    h, w, _ = image.shape
    for face_id, landmarks in enumerate(results.multi_face_landmarks, start=1):
        try:
            # Get key face points
            nose = np.array([landmarks.landmark[1].x * w, landmarks.landmark[1].y * h])
            eye_left = np.array([landmarks.landmark[33].x * w, landmarks.landmark[33].y * h])
            eye_right = np.array([landmarks.landmark[263].x * w, landmarks.landmark[263].y * h])
            chin = np.array([landmarks.landmark[152].x * w, landmarks.landmark[152].y * h])
            forehead = np.array([landmarks.landmark[10].x * w, landmarks.landmark[10].y * h])

            # Centers
            eye_center = (eye_left + eye_right) / 2
            vertical_center = (forehead + chin) / 2

            # Differences
            horizontal_diff = nose[0] - eye_center[0]
            vertical_diff = nose[1] - vertical_center[1]

            # Direction logic
            direction = ""
            if horizontal_diff > 15:
                direction += "Right"
            elif horizontal_diff < -15:
                direction += "Left"
            else:
                direction += "Straight"

            if vertical_diff > 15:
                direction += " & Down"
            elif vertical_diff < -15:
                direction += " & Up"

            print(f"Face {face_id} → Looking {direction}")

            # Draw face box
            points = [nose, eye_left, eye_right, chin, forehead]
            x_coords = [pt[0] for pt in points]
            y_coords = [pt[1] for pt in points]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(image, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 255), 2)

            # Text label
            cv2.putText(image, f"{direction}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error with face {face_id}: {e}")

    # Show image
    cv2.imshow(f"Result: {filename}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
