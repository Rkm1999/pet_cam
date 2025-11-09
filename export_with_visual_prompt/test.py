import cv2
# from picamera2 import Picamera2  <-- This line is removed
from ultralytics import YOLO

# Set up the camera with OpenCV
# 0 is typically the default integrated webcam
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set desired resolution (optional, webcam will use the closest available)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# Load YOLOE model (e.g., the default or your custom .onnx file)
model = YOLO("custom_model.onnx", task='segment') 
# Or, use your custom model:
# model = YOLO("your-custom-model-seg.onnx")

print("Running YOLOE on laptop webcam...")
print("Press 'q' to quit")

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Run YOLOE model on the captured frame
    results = model.predict(frame)
    
    # Output the visual detection data
    annotated_frame = results[0].plot(boxes=True, masks=False)
    
    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'
    
    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top
    
    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow("Laptop Webcam - YOLOE", annotated_frame)
    
    # Exit the program if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()