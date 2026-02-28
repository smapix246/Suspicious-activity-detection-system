from ultralytics import YOLO
import cv2

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")  # Automatically downloads on first use

# Use a video file or webcam (0 = default webcam)
cap = cv2.VideoCapture("videos/sample.mp4")  # Use 0 for webcam or "videos/sample.mp4" for video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection on frame
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press ESC key to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
