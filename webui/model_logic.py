import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import anyio # For running blocking code in async
import json
import glob # --- ADD THIS ---

# ========== CONFIGURATION ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR_NAME = "generated"
GENERATED_FILES_DIR = os.path.join(SCRIPT_DIR, GENERATED_FILES_DIR_NAME)

# --- NEW: Use a prefix for multiple images ---
REFERENCE_IMAGE_PREFIX = "reference"
OUTPUT_MODEL_NAME = "custom_model.onnx"
OUTPUT_NAMES_NAME = "custom_model_names.json"
BASE_MODEL_NAME = "yoloe-11l-seg.pt"

os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

# --- We no longer have one single filename ---
OUTPUT_MODEL_FILENAME = os.path.join(GENERATED_FILES_DIR, OUTPUT_MODEL_NAME)
OUTPUT_NAMES_FILENAME = os.path.join(GENERATED_FILES_DIR, OUTPUT_NAMES_NAME)
# ===================================

# --- Global state ---
detection_model = None
custom_names = {} 
model_load_lock = threading.Lock()

# --- MODIFIED ---
def get_reference_image_paths():
    """Returns a glob path to find all reference images."""
    return os.path.join(GENERATED_FILES_DIR, f"{REFERENCE_IMAGE_PREFIX}_*.jpg")

def get_output_model_path():
    """Returns the full path to the output model."""
    return OUTPUT_MODEL_FILENAME

def model_is_ready():
    """Checks if the ONNX model file exists."""
    return os.path.exists(OUTPUT_MODEL_FILENAME)

# --- MODIFIED ---
def clear_generated_files():
    """Deletes the old model and ALL reference images to start over."""
    print("🔄 Clearing generated files...")
    
    # Use glob to find all reference images and delete them
    image_files = glob.glob(get_reference_image_paths())
    for f in image_files:
        if os.path.exists(f):
            os.remove(f)
            
    if os.path.exists(OUTPUT_MODEL_FILENAME):
        os.remove(OUTPUT_MODEL_FILENAME)
    if os.path.exists(OUTPUT_NAMES_FILENAME):
        os.remove(OUTPUT_NAMES_FILENAME)

    global detection_model, custom_names
    with model_load_lock:
        detection_model = None
        custom_names = {} 
    print("✅ Files cleared.")


# --- NEW: This is the multi-image training logic from your app.py ---
def step_3_train_model(all_annotations: dict, class_names: list):
    """
    Trains the model using multiple reference images and their annotations.
    
    Args:
        all_annotations (dict): 
            A dict mapping image filenames to their annotations.
            e.g., {"reference_0.jpg": [{'bbox': [x,y,x,y], 'class_id': 0}, ...],
                   "reference_1.jpg": [{'bbox': [x,y,x,y], 'class_id': 1}, ...]}
        class_names (list): 
            List of strings, e.g., ["key", "phone"]
    """
    print("\n--- STEP 3: TRAINING & EXPORTING MODEL (MULTI-IMAGE) ---")
    
    if not all_annotations:
        raise Exception("No annotations provided for training.")

    all_refer_images_names = list(all_annotations.keys())
    # --- IMPORTANT: We need the FULL paths for the model ---
    all_refer_images_paths = [os.path.join(GENERATED_FILES_DIR, name) for name in all_refer_images_names]
    
    nested_bboxes = []
    nested_classes = []
    
    names_dict = {i: name for i, name in enumerate(class_names)}
    print(f"Set model class names: {names_dict}")

    print("Building visual prompts...")
    for image_name in all_refer_images_names:
        annotations = all_annotations[image_name]
        
        image_bboxes = []
        image_classes = []
        
        for box_info in annotations:
            image_bboxes.append(box_info['bbox'])
            image_classes.append(box_info['class_id'])

        nested_bboxes.append(np.array(image_bboxes))
        nested_classes.append(np.array(image_classes))

    visual_prompts = {
        'bboxes': nested_bboxes,
        'cls': nested_classes
    }

    print(f"Loading base model: {BASE_MODEL_NAME}")
    try:
        model = YOLOE(BASE_MODEL_NAME)
    except Exception as e:
        raise Exception(f"Failed to load base model: {e}")

    print("Prompting model with annotations...")
    # We pass the *list* of images to refer_image
    # and the *nested list* structure to visual_prompts.
    model.predict(
        all_refer_images_paths[0], # Source can be any of the images
        refer_image=all_refer_images_paths, 
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=0.1
    )

    print(f"Exporting model to ONNX...")
    try:
        exported_file_path = model.export(format="onnx", imgsz=640)
    except Exception as e:
        raise Exception(f"Error during model export: {e}")

    print(f"Model exported to default path: {exported_file_path}")

    try:
        if os.path.exists(OUTPUT_MODEL_FILENAME):
            os.remove(OUTPUT_MODEL_FILENAME)
        
        os.rename(exported_file_path, OUTPUT_MODEL_FILENAME)
        print(f"Successfully renamed model to: {OUTPUT_MODEL_FILENAME}")
    except Exception as e:
        raise Exception(f"Error renaming model: {e}")

    try:
        with open(OUTPUT_NAMES_FILENAME, 'w') as f:
            json.dump(names_dict, f)
        print(f"✅ Names saved to {OUTPUT_NAMES_FILENAME}")
    except Exception as e:
        raise Exception(f"Error saving names file: {e}")

    print("✅ Model export complete!")
    
    # --- Load the new model for detection ---
    global detection_model, custom_names
    with model_load_lock:
        print(f"Loading new ONNX model for detection: {OUTPUT_MODEL_FILENAME}")
        detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
        custom_names = names_dict
        print(f"✅ Detection model is loaded and ready.")
        
    return True

# --- run_detection_on_frame remains the same as your WebUI version ---
def run_detection_on_frame(frame: np.ndarray) -> np.ndarray:
    global detection_model, custom_names
    
    with model_load_lock:
        if detection_model is None:
            if model_is_ready():
                print(f"Model file found. Loading {OUTPUT_MODEL_FILENAME} into memory...")
                try:
                    detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
                    print(f"✅ Detection model is loaded. Will now apply custom names.")

                    try:
                        with open(OUTPUT_NAMES_FILENAME, 'r') as f:
                            names_from_file = json.load(f)
                            custom_names = {int(k): v for k, v in names_from_file.items()}
                        print(f"✅ Auto-loaded model. Names manually set: {custom_names}")
                    except Exception as e:
                        print(f"❌ Failed to auto-load/apply names: {e}")
                        
                except Exception as e:
                    print(f"❌ Failed to auto-load model: {e}")
                    cv2.putText(frame, f"Error loading {OUTPUT_MODEL_NAME}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return frame
            else:
                cv2.putText(frame, "Model not trained", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame

    # Run prediction
    try:
        results = detection_model.predict(frame, verbose=False, imgsz=640)
        
        if custom_names and results:
            results[0].names = custom_names

        annotated_frame = results[0].plot(boxes=True, masks=True) 
        
        try:
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time
            text = f'FPS: {fps:.1f}'
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass 

        return annotated_frame

    except Exception as e:
        print(f"Error during detection: {e}")
        cv2.putText(frame, f"Detection Error: {e}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

# --- CameraManager remains identical ---
class CameraManager:
    def __init__(self, camera_id=0):
        print("📷 Initializing Camera Manager...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"✅ Camera initialized with resolution: {self.width}x{self.height}")

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            time.sleep(0.01) 

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    # --- MODIFIED to accept a filename ---
    def capture_reference_image(self, filename: str):
        frame = self.get_frame()
        if frame is not None:
            full_path = os.path.join(GENERATED_FILES_DIR, filename)
            cv2.imwrite(full_path, frame)
            print(f"📸 Reference image captured and saved to {full_path}")
            return True, full_path
        return False, None

    def release(self):
        self.cap.release()

try:
    camera = CameraManager()
except Exception as e:
    print(f"🚨 FATAL: Could not initialize camera. {e}")
    print("🚨 Please ensure a webcam is connected and permissions are set.")
    camera = None