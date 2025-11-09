import cv2
import numpy as np
import time
import os
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# ========== CONFIGURATION ==========
# We'll save images as reference_0.jpg, reference_1.jpg, etc.
REFERENCE_IMAGE_PREFIX = "reference"
OUTPUT_MODEL_FILENAME = "custom_model.onnx"
# Use "yoloe-l-seg.pt" or your "yoloe-11l-seg.pt"
BASE_MODEL_NAME = "yoloe-11l-seg.pt" 
# ===================================


class BoundingBoxTool:
    """
    Modified to support an ongoing class_id count and to
    quit without saving (ESC) or save and continue (Q).
    """
    def __init__(self, image_path, starting_class_id=0):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"❌ Could not load image: {image_path}")
        
        self.original_image = self.image.copy()
        self.boxes = []
        self.current_box = []
        self.drawing = False
        # Store the starting class ID
        self.starting_class_id = starting_class_id
        # The class ID now starts from where the last image left off
        self.class_id = starting_class_id
        self.next_class_id = starting_class_id
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
                # Only draw crosshairs if not drawing a box
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
            
            # Ask user for class ID
            try:
                # Get class ID from a popup window
                class_id_str = cv2.selectROI("Enter Class ID (then press ENTER)", 
                                             np.zeros((100, 300), dtype=np.uint8), 
                                             showCrosshair=False)
                cv2.destroyWindow("Enter Class ID (then press ENTER)")

                # Hacky way to get text input with OpenCV
                # Create a blank image and wait for key presses
                input_img = np.zeros((100, 400), dtype=np.uint8)
                cv2.putText(input_img, "Enter Class ID, then press ENTER:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                class_id_input = ""
                while True:
                    cv2.putText(input_img, class_id_input, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow("Class ID Input", input_img)
                    key = cv2.waitKey(0)

                    if key == 13:  # ENTER key
                        break
                    elif key == 8:  # Backspace
                        class_id_input = class_id_input[:-1]
                    elif 48 <= key <= 57:  # 0-9
                        class_id_input += chr(key)
                    
                    # Redraw input text
                    input_img = np.zeros((100, 400), dtype=np.uint8)
                    cv2.putText(input_img, "Enter Class ID, then press ENTER:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.destroyWindow("Class ID Input")
                
                if not class_id_input:
                    print("⚠️ No class ID entered. Box cancelled.")
                    self.update_display()
                    return
                    
                self.class_id = int(class_id_input)

            except Exception as e:
                print(f"Error getting class ID: {e}. Box cancelled.")
                self.update_display()
                return

            self.boxes.append({
                'bbox': box,
                'class_id': self.class_id
            })
            
            print(f"📦 Box {len(self.boxes)}: {box} (Class ID: {self.class_id})")
            
            # Update the next class_id to be the max of current + 1
            all_ids = [b['class_id'] for b in self.boxes] + [self.starting_class_id]
            self.next_class_id = max(all_ids) + 1
            
            self.update_display()
    
    def draw_crosshairs(self, image, x, y):
        height, width = image.shape[:2]
        cv2.line(image, (x, 0), (x, height), (255, 255, 0), 1)
        cv2.line(image, (0, y), (width, y), (255, 255, 0), 1)
        cv2.putText(image, f"({x}, {y})", (x + 10, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    def update_display(self):
        self.image = self.original_image.copy()
        # Generate more colors
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), 
                  (255,0,255), (128,0,0), (0,128,0), (0,0,128), (128,128,0), 
                  (0,128,128), (128,0,128)]
                  
        for i, box_info in enumerate(self.boxes):
            bbox = box_info['bbox']
            class_id = box_info['class_id']
            color = colors[class_id % len(colors)]
            
            cv2.rectangle(self.image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(self.image, f"ID:{class_id}", (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(self.image, f"Annotating: {os.path.basename(self.image_path)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.image, "Click/drag. Enter Class ID. R: Reset | Q: Save & Next | ESC: Cancel", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show all unique class IDs added so far
        unique_ids = sorted(list(set([b['class_id'] for b in self.boxes])))
        cv2.putText(self.image, f"Current IDs: {unique_ids}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if hasattr(self, 'mouse_x') and hasattr(self, 'mouse_y'):
            self.draw_crosshairs(self.image, self.mouse_x, self.mouse_y)
        
        cv2.imshow("Bounding Box Tool", self.image)
        
    def run(self):
        print(f"🎯 Bounding Box Tool - Loaded: {self.image_path}")
        print("📋 Instructions:")
        print("   • Click and drag to draw a box.")
        print("   • Enter the Class ID for that object (e.g., 0, 1, 2...).")
        print("   • You can use the SAME Class ID for the same object across different images.")
        print("   • Press 'R' to reset boxes for THIS image.")
        print("   • Press 'Q' to SAVE boxes for this image and move to the next.")
        print("   • Press 'ESC' to CANCEL boxes for this image and move to the next.")
        print()
        
        self.update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): # Quit and SAVE
                cv2.destroyAllWindows()
                print(f"✅ Saved {len(self.boxes)} boxes for this image.")
                return self.boxes, self.next_class_id
                
            elif key == ord('r'): # Reset
                self.boxes = []
                self.class_id = self.starting_class_id
                self.next_class_id = self.starting_class_id
                self.update_display()
                print("🔄 Reset all boxes for this image")
                
            elif key == 27: # ESC
                cv2.destroyAllWindows()
                print("🚫 Cancelled annotations for this image.")
                return [], self.starting_class_id # Return no boxes and the original class_id
        

def step_1_capture_images():
    """
    Modified to capture multiple images in a loop.
    """
    print("--- STEP 1: CAPTURE REFERENCE PHOTOS ---")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    image_filenames = []
    image_count = 0

    print("Position your object(s) in the frame.")
    print("Press SPACE to capture, Q to finish capturing, ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        display_frame = frame.copy()
        cv2.putText(display_frame, "SPACE: Take Photo | Q: Finish | ESC: Exit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Captured: {image_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Capture Reference Photo", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            filename = f"{REFERENCE_IMAGE_PREFIX}_{image_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Photo saved: {filename}")
            image_filenames.append(filename)
            image_count += 1
            
            # Show confirmation
            confirm_frame = frame.copy()
            cv2.putText(confirm_frame, f"Saved: {filename}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Capture Reference Photo", confirm_frame)
            cv2.waitKey(500) # Show for 0.5 sec
        
        elif key == ord('q'):
            print(f"Finished capturing. Total images: {len(image_filenames)}")
            break

        elif key == 27:
            print("Capture cancelled.")
            image_filenames = [] # Discard all images
            break

    cap.release()
    cv2.destroyAllWindows()
    return image_filenames

def step_2_annotate_images(image_filenames):
    """
    Modified to loop through each captured image for annotation.
    """
    print("\n--- STEP 2: ANNOTATE REFERENCE PHOTOS ---")
    if not image_filenames:
        print("❌ No images to annotate.")
        return None

    all_annotations = {} # Will map image_path -> list of boxes
    next_class_id = 0 # Start class ID at 0 for the first image

    for image_path in image_filenames:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}. Skipping.")
            continue

        tool = BoundingBoxTool(image_path, starting_class_id=next_class_id)
        boxes, next_class_id = tool.run()
        
        if boxes:
            all_annotations[image_path] = boxes
    
    if not all_annotations:
        print("⚠️ No boxes were drawn on any image.")
        return None
    
    print(f"\n✅ Annotation complete. {len(all_annotations)} images annotated.")
    return all_annotations

def step_3_train_model(all_annotations):
    """
    Modified to build the nested visual_prompts structure
    and pass the list of images to the model.
    """
    print("\n--- STEP 3: TRAINING & EXPORTING MODEL ---")
    
    all_refer_images = list(all_annotations.keys())
    nested_bboxes = []
    nested_classes = []
    
    object_map = {} # To store a simple map of ID -> count

    print("Building visual prompts...")
    for image_path in all_refer_images:
        annotations = all_annotations[image_path]
        
        image_bboxes = []
        image_classes = []
        
        for box_info in annotations:
            image_bboxes.append(box_info['bbox'])
            image_classes.append(box_info['class_id'])
            
            # Update object map for final summary
            class_id = box_info['class_id']
            if class_id not in object_map:
                object_map[class_id] = 0
            object_map[class_id] += 1

        nested_bboxes.append(np.array(image_bboxes))
        nested_classes.append(np.array(image_classes))

    visual_prompts = {
        'bboxes': nested_bboxes,
        'cls': nested_classes
    }

    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = YOLOE(BASE_MODEL_NAME)

    print("Prompting model with annotations...")
    # Run predict once to "train" the model with the prompts
    # We pass the *list* of images to refer_image
    # and the *nested list* structure to visual_prompts.
    # The first argument (source) can be any of the images, or even a different one.
    # We'll just use the first reference image.
    model.predict(
        all_refer_images[0],
        refer_image=all_refer_images, 
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=0.1
    )

    print(f"Exporting model to ONNX...")
    try:
        exported_file_path = model.export(format="onnx", imgsz=320)
    except Exception as e:
        print(f"❌ Error during model.export(): {e}")
        return False

    print(f"Model exported to default path: {exported_file_path}")

    try:
        if os.path.exists(OUTPUT_MODEL_FILENAME):
            os.remove(OUTPUT_MODEL_FILENAME)
            print(f"Removed existing model: {OUTPUT_MODEL_FILENAME}")
            
        os.rename(exported_file_path, OUTPUT_MODEL_FILENAME)
        print(f"Successfully renamed model to: {OUTPUT_MODEL_FILENAME}")
    except Exception as e:
        print(f"❌ Error renaming model: {e}")
        print(f"   Please manually rename '{exported_file_path}' to '{OUTPUT_MODEL_FILENAME}'")
        return False

    print("✅ Model export complete!")
    print("Object mapping summary:")
    for class_id, count in sorted(object_map.items()):
        print(f"  ID {class_id} -> {count} examples provided")
    
    return True

def step_4_run_detector():
    """
    This step remains unchanged. It just loads the final ONNX model
    and runs it on the webcam feed.
    """
    print("\n--- STEP 4: RUNNING LIVE DETECTOR ---")
    
    if not os.path.exists(OUTPUT_MODEL_FILENAME):
        print(f"❌ Error: Model file not found: {OUTPUT_MODEL_FILENAME}")
        return

    print(f"Loading custom ONNX model: {OUTPUT_MODEL_FILENAME}")
    model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    
    print("Running YOLOE on laptop webcam...")
    print("Press 'q' to quit")
    
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=True, masks=False)
        
        try:
            # Calculate FPS manually for more stability
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            text = f'FPS: {fps:.1f}'
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = annotated_frame.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            
            cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            pass
        
        cv2.imshow(f"Laptop Webcam - {OUTPUT_MODEL_FILENAME}", annotated_frame)
        
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Detector stopped.")

def main():
    # Clear old reference images
    for f in os.listdir("."):
        if f.startswith(REFERENCE_IMAGE_PREFIX) and f.endswith(".jpg"):
            os.remove(f)
            print(f"Removed old file: {f}")

    image_files = step_1_capture_images()
    if image_files:
        annotations = step_2_annotate_images(image_files)
        if annotations:
            if step_3_train_model(annotations):
                step_4_run_detector()

    print("\nPipeline finished.")

if __name__ == "__main__":
    main()