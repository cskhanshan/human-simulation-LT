import cv2
import os
from ultralytics import YOLO

# Load YOLOv8x model
model = YOLO("yolov8x.pt")  # Make sure file is in same folder or give full path

# Image folder path
folder_path = "yolo-images"
valid_exts = ['.jpg', '.jpeg', '.png']

# Class names from COCO
COCO_NAMES = model.model.names

# Log file
log_file = open("yolo-detection-log.txt", "w", encoding="utf-8")

# Loop over images
for filename in sorted(os.listdir(folder_path)):
    if not any(filename.lower().endswith(ext) for ext in valid_exts):
        continue

    img_path = os.path.join(folder_path, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not read {filename}")
        continue

    print(f"\nProcessing: {filename}")
    results = model(image)[0]

    person_count = 0

    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        class_id = int(class_id)
        label = COCO_NAMES[class_id]

        if label != "person":
            continue

        person_count += 1
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"#{person_count} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Log
        coords = f"Person {person_count} - x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}"
        print(coords)
        log_file.write(f"{filename} - {coords}\n")

    # Total label
    cv2.putText(image, f"Total People: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show image
    cv2.imshow("YOLO Detection", image)

    # Wait for 'q' to move to next image
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyWindow("YOLO Detection")

log_file.close()
cv2.destroyAllWindows()
