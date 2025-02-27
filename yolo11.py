import cv2
from ultralytics import YOLO
import numpy as np
# Load the YOLO model (replace "yolo11n.pt" with your model file)
model = YOLO("yolo11n.pt")  # Load an official or custom model

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Perform detection on the frame
    results = model(frame)


    # Annotate the frame with the detection results
    annotated_frame = results[0].plot()

    # Display the frameQ
    cv2.imshow('Live Object Detection', annotated_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
# metrics = model.val(data="coco128.yaml")
# print(f"mAP@50-95: {metrics.box.map:.4f}")
# print(f"mAP@50: {metrics.box.map50:.4f}")

