import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Key facial points (simplified)
KEY_POINTS = {
    'nose_tip': 1,
    'left_eye': 33,
    'right_eye': 263,
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_center': 13,
    'chin': 175
}

# Emotions to track
allowed_emotions = ['happy', 'neutral', 'surprise', 'sad', 'angry', 'fear']

# Webcam initialization
cap = cv2.VideoCapture(0)

# Emotion stability
previous_emotion = ""
stable_count = 0
STABLE_THRESHOLD = 3

def draw_key_points(frame, landmarks, h, w):
    """Draw only key facial points"""
    key_coordinates = {}
    
    for point_name, point_id in KEY_POINTS.items():
        if point_id < len(landmarks.landmark):
            lm = landmarks.landmark[point_id]
            x = int(lm.x * w)
            y = int(lm.y * h)
            key_coordinates[point_name] = (x, y)
            
            # Color coding for different features
            if 'eye' in point_name:
                color = (255, 100, 100)  # Light blue
            elif 'nose' in point_name:
                color = (100, 255, 100)  # Light green
            elif 'mouth' in point_name:
                color = (100, 100, 255)  # Light red
            else:
                color = (255, 255, 100)  # Light cyan
            
            # Draw point
            cv2.circle(frame, (x, y), 4, color, -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)
    
    return key_coordinates

def draw_face_connections(frame, coordinates):
    """Draw lines between key points"""
    if len(coordinates) >= 4:
        # Draw eye line
        if 'left_eye' in coordinates and 'right_eye' in coordinates:
            cv2.line(frame, coordinates['left_eye'], coordinates['right_eye'], (255, 255, 255), 1)
        
        # Draw mouth line
        if 'mouth_left' in coordinates and 'mouth_right' in coordinates:
            cv2.line(frame, coordinates['mouth_left'], coordinates['mouth_right'], (255, 255, 255), 1)
        
        # Draw nose to mouth line
        if 'nose_tip' in coordinates and 'mouth_center' in coordinates:
            cv2.line(frame, coordinates['nose_tip'], coordinates['mouth_center'], (255, 255, 255), 1)

print("Starting emotion detection with key points...")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    try:
        # Emotion detection using DeepFace
        emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        if isinstance(emotion_result, list) and len(emotion_result) > 0:
            emotion_data = emotion_result[0]['emotion']
            dominant_emotion = max(emotion_data, key=emotion_data.get)
            confidence = emotion_data[dominant_emotion]
            
            # Emotion stability check
            if dominant_emotion == previous_emotion:
                stable_count += 1
            else:
                stable_count = 0
                previous_emotion = dominant_emotion
            
            # Display emotion if stable
            if stable_count >= STABLE_THRESHOLD:
                emotion_color = (0, 255, 0) if confidence > 50 else (0, 255, 255)
                cv2.putText(frame, f"Emotion: {dominant_emotion.upper()}", 
                           (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Face mesh for key points
        mesh_results = face_mesh.process(rgb_frame)
        
        if mesh_results.multi_face_landmarks:
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Draw key points
                coordinates = draw_key_points(frame, face_landmarks, h, w)
                
                # Draw connections between points
                draw_face_connections(frame, coordinates)
                
                # Display coordinate info (optional)
                cv2.putText(frame, f"Key Points: {len(coordinates)}", 
                           (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No face detected", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    except Exception as e:
        print(f"Error: {e}")
        cv2.putText(frame, "Processing error", (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display frame
    cv2.imshow("Emotion Detection + Key Points", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Emotion detection stopped")