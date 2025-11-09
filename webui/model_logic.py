import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import anyio # For running blocking code in async
import json
import glob

# ========== CONFIGURATION ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR_NAME = "generated"
GENERATED_FILES_DIR = os.path.join(SCRIPT_DIR, GENERATED_FILES_DIR_NAME)

REFERENCE_IMAGE_PREFIX = "reference"
OUTPUT_MODEL_NAME = "custom_model.onnx"
# --- NEW: This is our persistent "database" file ---
PROMPT_LIBRARY_NAME = "prompt_library.json" 
BASE_MODEL_NAME = "yoloe-11l-seg.pt"

os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

OUTPUT_MODEL_FILENAME = os.path.join(GENERATED_FILES_DIR, OUTPUT_MODEL_NAME)
PROMPT_LIBRARY_FILENAME = os.path.join(GENERATED_FILES_DIR, PROMPT_LIBRARY_NAME)
# ===================================

# --- Global state ---
detection_model = None
custom_names = {} 
model_load_lock = threading.Lock()

# --- NEW: Prompt Library "Database" Functions ---

def load_prompt_library():
    """Loads the prompt library JSON from disk."""
    if os.path.exists(PROMPT_LIBRARY_FILENAME):
        try:
            with open(PROMPT_LIBRARY_FILENAME, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading prompt library: {e}")
    # Return a default empty library if file doesn't exist or is corrupt
    return {"class_names": {}, "annotations": {}}

def save_prompt_library(data: dict):
    """Saves the prompt library JSON to disk."""
    try:
        with open(PROMPT_LIBRARY_FILENAME, 'w') as f:
            json.dump(data, f, indent=2)
        print("✅ Prompt library saved.")
    except Exception as e:
        print(f"❌ Error saving prompt library: {e}")

def get_reference_image_paths():
    """Returns a glob path to find all reference images."""
    return os.path.join(GENERATED_FILES_DIR, f"{REFERENCE_IMAGE_PREFIX}_*.jpg")

def get_output_model_path():
    """Returns the full path to the output model."""
    return OUTPUT_MODEL_FILENAME

def model_is_ready():
    """Checks if the ONNX model file exists."""
    return os.path.exists(OUTPUT_MODEL_FILENAME)

def clear_generated_files():
    """Deletes the model, ALL reference images, and the prompt library."""
    print("🔄 Clearing all generated files...")
    
    # Use glob to find all reference images and delete them
    image_files = glob.glob(get_reference_image_paths())
    for f in image_files:
        if os.path.exists(f):
            os.remove(f)
            
    if os.path.exists(OUTPUT_MODEL_FILENAME):
        os.remove(OUTPUT_MODEL_FILENAME)
    if os.path.exists(PROMPT_LIBRARY_FILENAME):
        os.remove(PROMPT_LIBRARY_FILENAME)

    global detection_model, custom_names
    with model_load_lock:
        detection_model = None
        custom_names = {} 
    print("✅ All files cleared.")


# --- MODIFIED: This function is now called by the API ---
def step_3_train_model(all_annotations: dict, class_names_dict: dict):
    """
    Trains the model using annotations and names from the prompt library.
    
    Args:
        all_annotations (dict): 
            e.g., {"reference_0.jpg": [{'bbox': [x,y,x,y], 'class_id': 0}, ...]}
        class_names_dict (dict): 
            e.g., {"0": "key", "1": "phone"}
    """
    print("\n--- STEP 3: TRAINING & EXPORTING MODEL ---")
    
    if not all_annotations:
        print("⚠️ No annotations provided for training. Aborting.")
        return False

    all_refer_images_names = list(all_annotations.keys())
    all_refer_images_paths = [os.path.join(GENERATED_FILES_DIR, name) for name in all_refer_images_names]
    
    # Check if images exist
    existing_image_paths = []
    existing_image_names = []
    for i, path in enumerate(all_refer_images_paths):
        if os.path.exists(path):
            existing_image_paths.append(path)
            existing_image_names.append(all_refer_images_names[i])
        else:
            print(f"⚠️ Missing image {path}, it will be skipped.")

    if not existing_image_paths:
        print("❌ No valid image paths found for training. Aborting.")
        return False
        
    nested_bboxes = []
    nested_classes = []
    
    print("Building visual prompts...")
    for image_name in existing_image_names:
        annotations = all_annotations.get(image_name, [])
        
        image_bboxes = [box_info['bbox'] for box_info in annotations]
        image_classes = [box_info['class_id'] for box_info in annotations]

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
    model.predict(
        existing_image_paths[0],
        refer_image=existing_image_paths, 
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

    print("✅ Model export complete!")
    
    # --- Load the new model for detection ---
    global detection_model, custom_names
    with model_load_lock:
        print(f"Loading new ONNX model for detection: {OUTPUT_MODEL_FILENAME}")
        detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
        # Convert string keys from JSON back to int keys for model
        custom_names = {int(k): v for k, v in class_names_dict.items()}
        print(f"✅ Detection model is loaded and ready.")
        
    return True

# --- NEW: Load model and names on startup ---
def load_detection_model_on_startup():
    """Tries to load an existing model and prompt library on server start."""
    global detection_model, custom_names
    with model_load_lock:
        if not model_is_ready():
            print("ℹ️ No pre-trained model found on startup.")
            return

        print(f"Found model {OUTPUT_MODEL_FILENAME}. Loading into memory...")
        try:
            detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
            print("✅ Detection model loaded.")
            
            # Load corresponding names from our library
            library = load_prompt_library()
            if library and library.get('class_names'):
                custom_names = {int(k): v for k, v in library['class_names'].items()}
                print(f"✅ Names loaded from library: {custom_names}")
            else:
                print("⚠️ Model loaded, but no prompt library found. Names may be incorrect.")
                
        except Exception as e:
            print(f"❌ Failed to auto-load model on startup: {e}")
            detection_model = None
            custom_names = {}


# --- MODIFIED: Simplified to only use global state ---
def run_detection_on_frame(frame: np.ndarray) -> np.ndarray:
    """Runs detection on a frame using the globally loaded model and names."""
    
    with model_load_lock:
        if detection_model is None:
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

    def capture_reference_image(self, filename: str):
        frame = self.get_frame()
        if frame is not None:
            full_path = os.path.join(GENERATED_FILES_DIR, filename)
            cv2.imwrite(full_path, frame)
            print(f"📸 Reference image captured and saved to {full_path}")
            return True, full_path, self.width, self.height
        return False, None, 0, 0

    def release(self):
        self.cap.release()

try:
    camera = CameraManager()
except Exception as e:
    print(f"🚨 FATAL: Could not initialize camera. {e}")
    camera = None