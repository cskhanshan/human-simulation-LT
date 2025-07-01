import cv2
from deepface import DeepFace

# webcam sy video lega yaha sy
cap = cv2.VideoCapture(0)

# agr koi emotion pehle se hai to usko track karne ke liye
# aur stable hone tak wait karne ke liye
previous_emotion = ""
stable_count = 0
STABLE_THRESHOLD = 5  # yeh bata rahaega ki emotion kitne baar same hona chahiye tabhi dikhaaye

while True:
    ret, frame = cap.read()

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            emotion_scores = first_result['emotion']

            # yaha wo emotion ko le raha hai jo sabse zyada score kiya hai
            # aur usko dominant emotion kehte hain
            # yeh emotion_scores dictionary se sabse zyada value wala emotion nikaal raha hai
            # jisse humein pata chalega ki kaunsa emotion sabse zyada hai
            # aur usko dominant_emotion kehte hain 

            dominant_emotion = max(emotion_scores, key=emotion_scores.get)

            # bs wahi emotions ko dikha raha hai jo imp hain
            # jese happy, neutral, surprise, sad
            
            filtered_emotions = {key: emotion_scores[key] for key in ['happy', 'neutral', 'surprise', 'sad'] if key in emotion_scores}

            # Stability logic: emotion bar bar same aa raha hai ya nahi
            if dominant_emotion == previous_emotion:
                stable_count += 1
            else:
                stable_count = 0
                previous_emotion = dominant_emotion

            # Jab same emotion stable ho jaaye tabhi dikhao
            if stable_count >= STABLE_THRESHOLD:
                cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Emotions list side mein show karo
            for i, (emotion, score) in enumerate(filtered_emotions.items()):
                text = f"{emotion}: {score:.2f}%"
                cv2.putText(frame, text, (50, 100 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        else:
            print("Chehra detect nahi hua.")

    except Exception as e:
        print(f"DeepFace analysis mein error aayi: {e}")

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
