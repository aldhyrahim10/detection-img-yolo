from ultralytics import YOLO
import cv2
import json

model = YOLO("runs/detect/train/weights/best.pt")


image_path = "/Users/aldirahim/Documents/Project/Python/motorcycle.v3i.yolov8/test/images/frame_0654_png.rf.48f3f1a0eca45ab3cc382b91cf8434d8.jpg"
img = cv2.imread(image_path)

results = model(img, conf=0.4)

detections = []

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        detections.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        })

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{class_name} {confidence:.2f}"
        cv2.putText(img, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)


cv2.imwrite("output.jpg", img)


print(json.dumps(detections, indent=2))
