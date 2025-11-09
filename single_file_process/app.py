import cv2
import numpy as np
import time
import os
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ========== CONFIGURATION ==========
REFERENCE_IMAGE_FILENAME = "reference_image.jpg"
OUTPUT_MODEL_FILENAME = "custom_model.onnx"
# Use "yoloe-l-seg.pt" or your "yoloe-11l-seg.pt"
BASE_MODEL_NAME = "yoloe-11l-seg.pt" 
# ===================================


class BoundingBoxTool:
    """
    This class is taken from your Image-Prompt Draw Box.py script.
    It's used to draw annotations on the captured reference image.
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"❌ Could not load image: {image_path}")
        
        self.original_image = self.image.copy()
        self.boxes = []
        self.current_box = []
        self.drawing = False
        self.class_id = 0
        self.mouse_x = 0
        self.mouse_y = 0
        
        cv2.namedWindow("Bounding Box Tool", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Bounding Box Tool", self.mouse_callback)
        
    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_x = x
        self.mouse_y = y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y]
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                temp_image = self.image.copy()
                cv2.rectangle(temp_image, (self.current_box[0], self.current_box[1]), 
                            (x, y), (0, 255, 0), 2)
                self.draw_crosshairs(temp_image, x, y)
                cv2.imshow("Bounding Box Tool", temp_image)
            else:
                temp_image = self.image.copy()
                self.draw_crosshairs(temp_image, x, y)
                cv2.imshow("Bounding Box Tool", temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current_box.extend([x, y])
            
            box = [
                min(self.current_box[0], self.current_box[2]),
                min(self.current_box[1], self.current_box[3]),
                max(self.current_box[0], self.current_box[2]),
                max(self.current_box[1], self.current_box[3])
            ]
            
            self.boxes.append({
                'bbox': box,
                'class_id': self.class_id
            })
            
            print(f"📦 Box {len(self.boxes)}: {box} (Class ID: {self.class_id})")
            self.class_id += 1
            self.update_display()
    
    def draw_crosshairs(self, image, x, y):
        height, width = image.shape[:2]
        cv2.line(image, (x, 0), (x, height), (255, 255, 0), 1)
        cv2.line(image, (0, y), (width, y), (255, 255, 0), 1)
        cv2.putText(image, f"({x}, {y})", (x + 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    def update_display(self):
        self.image = self.original_image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        for i, box_info in enumerate(self.boxes):
            bbox = box_info['bbox']
            class_id = box_info['class_id']
            color = colors[class_id % len(colors)]
            
            cv2.rectangle(self.image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(self.image, f"ID:{class_id}", (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.putText(self.image, "Draw boxes around objects", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.image, "R: Reset | Q: Quit & Save", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.image, f"Next Class ID: {self.class_id}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if hasattr(self, 'mouse_x') and hasattr(self, 'mouse_y'):
            self.draw_crosshairs(self.image, self.mouse_x, self.mouse_y)
        
        cv2.imshow("Bounding Box Tool", self.image)
        
    def run(self):
        print(f"🎯 Bounding Box Tool - Loaded: {self.image_path}")
        print("📋 Instructions:")
        print("   • Click and drag to draw boxes around objects")
        print("   • Each box gets a unique Class ID (0, 1, 2...)")
        print("   • Press 'R' to reset all boxes")
        print("   • Press 'Q' to quit and start training")
        print()
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.boxes = []
                self.class_id = 0
                self.update_display()
                print("🔄 Reset all boxes")
        
        cv2.destroyAllWindows()
        # Return the collected boxes for the next step
        return self.boxes

def step_1_capture_image():
    """
    Logic from Image-Prompt Capture.py
    """
    print("--- STEP 1: CAPTURE REFERENCE PHOTO ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print("Position your object(s) in the frame.")
    print("Press SPACE to capture, ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        display_frame = frame.copy()
        cv2.putText(display_frame, "SPACE: Take Photo | ESC: Exit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Capture Reference Photo", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            cv2.imwrite(REFERENCE_IMAGE_FILENAME, frame)
            print(f"✅ Photo saved: {REFERENCE_IMAGE_FILENAME}")
            
            confirm_frame = frame.copy()
            cv2.putText(confirm_frame, f"Saved: {REFERENCE_IMAGE_FILENAME}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Capture Reference Photo", confirm_frame)
            cv2.waitKey(1000)
            cap.release()
            cv2.destroyAllWindows()
            return True
        
        elif key == 27:
            print("Capture cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return False

def step_2_annotate_image():
    """
    Logic from Image-Prompt Draw Box.py
    """
    print("\n--- STEP 2: ANNOTATE REFERENCE PHOTO ---")
    if not os.path.exists(REFERENCE_IMAGE_FILENAME):
        print(f"❌ Image not found: {REFERENCE_IMAGE_FILENAME}")
        return None

    tool = BoundingBoxTool(REFERENCE_IMAGE_FILENAME)
    boxes = tool.run()
    
    if not boxes:
        print("⚠️ No boxes were drawn.")
        return None
    
    print(f"✅ Annotation complete. {len(boxes)} boxes captured.")
    return boxes

def step_3_train_model(boxes):
    """
    Logic from Image-Prompt ONNX Conversion.py
    """
    print("\n--- STEP 3: TRAINING & EXPORTING MODEL ---")
    
    all_bboxes = [box_info['bbox'] for box_info in boxes]
    all_class_ids = [box_info['class_id'] for box_info in boxes]

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = YOLOE(BASE_MODEL_NAME)

    visual_prompts = {
        'bboxes': np.array(all_bboxes),
        'cls': np.array(all_class_ids)
    }

    print("Prompting model with annotations...")
    # Run predict once to "train" the model with the prompts
    model.predict(
        REFERENCE_IMAGE_FILENAME,
        refer_image=REFERENCE_IMAGE_FILENAME, 
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=0.1
    )

    # --- START: MODIFIED EXPORT ---
    print(f"Exporting model to ONNX...")
    
    # Call export without the 'file' argument.
    # It returns the path to the file it created (e.g., 'yoloe-11l-seg.onnx')
    try:
        exported_file_path = model.export(format="onnx", imgsz=320)
    except Exception as e:
        print(f"❌ Error during model.export(): {e}")
        return False

    print(f"Model exported to default path: {exported_file_path}")

    # Rename the exported file to our desired output name
    try:
        # Remove the target file if it already exists (prevents error on Windows)
        if os.path.exists(OUTPUT_MODEL_FILENAME):
            os.remove(OUTPUT_MODEL_FILENAME)
            print(f"Removed existing model: {OUTPUT_MODEL_FILENAME}")
            
        os.rename(exported_file_path, OUTPUT_MODEL_FILENAME)
        print(f"Successfully renamed model to: {OUTPUT_MODEL_FILENAME}")
    except Exception as e:
        print(f"❌ Error renaming model: {e}")
        print(f"   Please manually rename '{exported_file_path}' to '{OUTPUT_MODEL_FILENAME}'")
        return False
    # --- END: MODIFIED EXPORT ---

    print("✅ Model export complete!")
    print("Object mapping:")
    for box_info in boxes:
        print(f"  ID {box_info['class_id']} -> Box {box_info['bbox']}")
    
    return True

def step_4_run_detector():
    """
    Logic from test.py
    """
    print("\n--- STEP 4: RUNNING LIVE DETECTOR ---")
    
    if not os.path.exists(OUTPUT_MODEL_FILENAME):
        print(f"❌ Error: Model file not found: {OUTPUT_MODEL_FILENAME}")
        return

    print(f"Loading custom ONNX model: {OUTPUT_MODEL_FILENAME}")
    # --- START: MODIFIED LINE ---
    # Explicitly tell YOLO this is a segmentation model
    model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
    # --- END: MODIFIED LINE ---
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    
    print("Running YOLOE on laptop webcam...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        results = model.predict(frame, verbose=False) # Added verbose=False to clean up logs
        annotated_frame = results[0].plot(boxes=True, masks=False)
        
        try:
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time
            text = f'FPS: {fps:.1f}'
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = annotated_frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            
            cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            # Squelch errors if speed isn't in results
            pass
        
        cv2.imshow(f"Laptop Webcam - {OUTPUT_MODEL_FILENAME}", annotated_frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detector stopped.")

def main():
    if step_1_capture_image():
        boxes = step_2_annotate_image()
        if boxes:
            if step_3_train_model(boxes):
                step_4_run_detector()

    print("\nPipeline finished.")

if __name__ == "__main__":
    main()
