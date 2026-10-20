import cv2
import time
from ultralytics import YOLO

def main():
    # 1. Load the YOLOv8 nano model
    print("Initializing YOLOv8 AI Model...")
    model = YOLO("yolov8n.pt") 
    
    # 2. Define Target Classes for Rider Safety
    # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light, 67: cell phone (for indoor testing)
    TARGET_CLASSES = [0, 2, 3, 5, 7, 9, 67]

    # 3. Initialize the camera (Added AVFOUNDATION for macOS stability)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    # Set standard resolution for consistent performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not access the camera. Please check macOS permissions.")
        return

    # Warm up the camera to prevent initial frame drops on Mac
    time.sleep(1.0)
    print("AR Safety System v2.0 Active. Press 'q' in the video window to exit.")

    while True:
        ret, frame = cap.read()
        
        # Stability check: skip briefly if a frame drops instead of crashing immediately
        if not ret or frame is None:
            print("Warning: Frame dropped, retrying...")
            time.sleep(0.1)
            continue

        frame_h, frame_w, _ = frame.shape

        # 4. Run YOLOv8 inference (stream=False is more stable for single webcam loops)
        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                
                # --- FEATURE 1: CATEGORY FILTERING ---
                if class_id in TARGET_CLASSES:
                    conf = float(box.conf[0])
                    if conf < 0.4:  # Ignore low-confidence detections
                        continue

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    
                    # --- FEATURE 2: DISTANCE & RISK ESTIMATION ---
                    # Calculate the screen ratio occupied by the object
                    area_ratio = (w * h) / (frame_w * frame_h)
                    
                    # Determine hazard level based on proximity (bounding box size)
                    if area_ratio > 0.20:       # Occupies > 20% of view -> Critical Danger
                        color = (0, 0, 255)     # Red (BGR format)
                        status = "DANGER: TOO CLOSE!"
                    elif area_ratio > 0.08:     # Occupies > 8% of view -> Caution
                        color = (0, 255, 255)   # Yellow
                        status = "CAUTION"
                    else:                       # Safe distance
                        color = (0, 255, 0)     # Green
                        status = "SAFE"

                    # --- FEATURE 3: AR HUD RENDERING ---
                    class_name = model.names[class_id].upper()
                    label = f"{class_name} ({conf:.2f}) | {status}"
                    
                    # Draw bounding box and status label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw static HUD overlay (System Status & Center Crosshair)
        cv2.putText(frame, "AR HELMET HUD: ONLINE", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.line(frame, (frame_w // 2 - 15, frame_h // 2), (frame_w // 2 + 15, frame_h // 2), (255, 255, 255), 1)
        cv2.line(frame, (frame_w // 2, frame_h // 2 - 15), (frame_w // 2, frame_h // 2 + 15), (255, 255, 255), 1)

        # 5. Display the AR output
        cv2.imshow('AR Helmet - Collision Warning System v2.0', frame)

        # Press 'q' to cleanly exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Shutting down AR System...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()