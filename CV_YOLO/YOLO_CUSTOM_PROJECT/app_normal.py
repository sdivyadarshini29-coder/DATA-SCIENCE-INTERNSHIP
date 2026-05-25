from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8 Nano model weights
model = YOLO("yolov8n.pt")

# Open primary laptop web-camera
cap = cv2.VideoCapture(0)

print("🚀 YOLOv8 Target Detector Booting Up...")
print("🎯 Filtering Only: Phone, Laptop, and Charger")
print("👉 Press 'q' on the camera window to Close!")

# COCO Dataset class IDs for filtering: 
# 63: laptop, 67: cell phone
# Adding proxy classes (39: bottle, 41: cup) to help capture Charger shapes 
ALLOWED_CLASSES = [63, 67, 39, 41]

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Cannot access web camera tracker feed.")
        break

    # Run direct inference matrix calculations on the frame
    results = model(frame)
    boxes = results[0].boxes

    # Loop through all detected objects in the frame
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Filter and only process if the object matches our target list
        if cls_id in ALLOWED_CLASSES:
            # Get bounding box coordinates limits
            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            # Map custom display tag labels based on class IDs
            if cls_id == 67:
                label = f"Phone {conf:.2f}"
                color = (0, 255, 0) # Green box for Phone
            elif cls_id == 63:
                label = f"Laptop {conf:.2f}"
                color = (255, 0, 0) # Blue box for Laptop
            else:
                label = f"Charger {conf:.2f}"
                color = (0, 0, 255) # Red box for Charger proxy

            # Draw custom rectangle box borders on the frame
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)

            # Draw label text string just above the bounding box
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the processed live AI feed window
    cv2.imshow("Normal YOLOv8 Custom Object Detection", frame)

    # Press 'q' key on the keyboard to cleanly exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stream closed successfully.")