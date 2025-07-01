import cv2
from deepface import DeepFace
import os

img_folder = "images"  # yeh folder hai jisme images hain

image_files = [f for f in os.listdir(img_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for file in image_files:
    image_path = os.path.join(img_folder, file)
    img = cv2.imread(image_path)

    if img is None:
        print(f"{file} load nahi hui.")
        continue

    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            emotion_scores = first_result['emotion']
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)

            print(f"\nImage: {file}")
            print(f"Dominant Emotion: {dominant_emotion}")
            for emotion, score in emotion_scores.items():
                print(f"   {emotion}: {score:.2f}%")

            # yeh bs isleye kara hai ki image ko resize karke dikha sake
            # taaki sab images ek hi size mein aaye aur achhe se dikhe
            resized_img = cv2.resize(img, (500, 600))

            # Dominant emotion 
            cv2.putText(resized_img, f"Emotion: {dominant_emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # scores of emotions 
            for i, (emotion, score) in enumerate(emotion_scores.items()):
                text = f"{emotion}: {score:.2f}%"
                y = 80 + i * 30  #for gap
                cv2.putText(resized_img, text, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show image
            cv2.imshow(file, resized_img)
            cv2.waitKey(0)
            cv2.destroyWindow(file)

        else:
            print(f"{file} mein face detect nahi hua.")

    except Exception as e:
        print(f"Error in {file}: {e}")
