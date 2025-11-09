import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import anyio # For running blocking code in async
import json # --- ADD THIS IMPORT ---

# ========== CONFIGURATION ==========
# --- Use absolute paths based on this file's location ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR_NAME = "generated"
GENERATED_FILES_DIR = os.path.join(SCRIPT_DIR, GENERATED_FILES_DIR_NAME)
# --- End of new path logic ---

REFERENCE_IMAGE_NAME = "reference_image.jpg"
OUTPUT_MODEL_NAME = "custom_model.onnx"
# --- ADD THIS NEW FILE NAME ---
OUTPUT_NAMES_NAME = "custom_model_names.json"
BASE_MODEL_NAME = "yoloe-11l-seg.pt"

# Ensure the directory exists
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

REFERENCE_IMAGE_FILENAME = os.path.join(GENERATED_FILES_DIR, REFERENCE_IMAGE_NAME)
OUTPUT_MODEL_FILENAME = os.path.join(GENERATED_FILES_DIR, OUTPUT_MODEL_NAME)
# --- ADD THIS NEW FILE PATH ---
OUTPUT_NAMES_FILENAME = os.path.join(GENERATED_FILES_DIR, OUTPUT_NAMES_NAME)
# ===================================

# --- Global state for the model ---
# We need to load the model into memory for detection
detection_model = None
# --- ADD THIS ---
custom_names = {} # Global cache for our custom names
model_load_lock = threading.Lock()

def get_reference_image_path():
    """Returns the full path to the reference image."""
    return REFERENCE_IMAGE_FILENAME

def get_output_model_path():
    """Returns the full path to the output model."""
    return OUTPUT_MODEL_FILENAME

def model_is_ready():
    """Checks if the ONNX model file exists."""
    return os.path.exists(OUTPUT_MODEL_FILENAME)

def clear_generated_files():
    """Deletes the old model and reference image to start over."""
    print("🔄 Clearing generated files...")
    if os.path.exists(REFERENCE_IMAGE_FILENAME):
        os.remove(REFERENCE_IMAGE_FILENAME)
    if os.path.exists(OUTPUT_MODEL_FILENAME):
        os.remove(OUTPUT_MODEL_FILENAME)
    # --- ADD THIS ---
    if os.path.exists(OUTPUT_NAMES_FILENAME):
        os.remove(OUTPUT_NAMES_FILENAME)
    # --- END ADD ---
    global detection_model, custom_names
    with model_load_lock:
        detection_model = None
        custom_names = {} # Clear cached names
    print("✅ Files cleared.")


def step_3_train_model(boxes_data: list, class_names: list):
    """
    Logic from your Image-Prompt ONNX Conversion.py
    This is a BLOCKING, CPU/GPU-intensive function.
    It should be run in a separate thread.
    
    Args:
        boxes_data (list): List of dicts, e.g., [{'bbox': [x1,y1,x2,y2], 'class_id': 0}, ...]
        class_names (list): List of strings, e.g., ["key", "phone"]
    """
    print("\n--- STEP 3: TRAINING & EXPORTING MODEL ---")
    
    # The data from the web UI is a list of dicts.
    # We need to parse it back into the format your script expects.
    all_bboxes = [box_info['bbox'] for box_info in boxes_data]
    all_class_ids = [box_info['class_id'] for box_info in boxes_data]

    if not all_bboxes:
        print("❌ No bounding boxes provided. Aborting training.")
        return False

    print(f"Loading base model: {BASE_MODEL_NAME}")
    # This will download if not present
    try:
        model = YOLOE(BASE_MODEL_NAME)
    except Exception as e:
        print(f"❌ Failed to load base model {BASE_MODEL_NAME}. Ensure you are online.")
        print(f"Error: {e}")
        raise Exception(f"Failed to load base model: {e}") # Raise error

    # --- NEW: Assign class names to the model ---
    # This creates the mapping: {0: "key", 1: "phone", ...}
    # We create the dict here to pass it to the export() function
    names_dict = {i: name for i, name in enumerate(class_names)}
    print(f"Set model class names: {names_dict}")
    # --- End of new logic ---

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

    print(f"Exporting model to ONNX...")
    try:
        # Call export. The 'names_dict' will be saved in the ONNX metadata.
        # --- REMOVE 'names' ARGUMENT FROM HERE ---
        exported_file_path = model.export(format="onnx", imgsz=640) # Pass names here
    except Exception as e:
        print(f"❌ Error during model.export(): {e}")
        raise Exception(f"Error during model export: {e}") # Raise error

    print(f"Model exported to default path: {exported_file_path}")

    # Rename the exported file to our desired output name
    try:
        if os.path.exists(OUTPUT_MODEL_FILENAME):
            os.remove(OUTPUT_MODEL_FILENAME)
        
        os.rename(exported_file_path, OUTPUT_MODEL_FILENAME)
        print(f"Successfully renamed model to: {OUTPUT_MODEL_FILENAME}")
    except Exception as e:
        print(f"❌ Error renaming model: {e}")
        raise Exception(f"Error renaming model: {e}") # Raise error

    # --- ADD THIS NEW BLOCK ---
    # Save the names_dict to a JSON file
    try:
        with open(OUTPUT_NAMES_FILENAME, 'w') as f:
            json.dump(names_dict, f)
        print(f"✅ Names saved to {OUTPUT_NAMES_FILENAME}")
    except Exception as e:
        print(f"❌ Error saving names file: {e}")
        raise Exception(f"Error saving names file: {e}") # Raise error
    # --- END NEW BLOCK ---

    print("✅ Model export complete!")
    print("Object mapping:")
    for i, name in names_dict.items():
        print(f"  ID {i} -> {name}")
    
    # --- Load the new model for detection ---
    global detection_model, custom_names
    with model_load_lock:
        print(f"Loading new ONNX model for detection: {OUTPUT_MODEL_FILENAME}")
        detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
        # --- REMOVE FAILING LINE ---
        # detection_model.names = names_dict
        
        # --- ADD THIS ---
        # Cache the names globally instead
        custom_names = names_dict
        # --- END ADD ---
        
        print(f"✅ Detection model is loaded and ready. Names will be applied from cache.")
        
    return True

def run_detection_on_frame(frame: np.ndarray) -> np.ndarray:
    global detection_model, custom_names
    
    with model_load_lock:
        if detection_model is None:
            # Model is not in memory. Check if the file exists on disk.
            if model_is_ready():
                print(f"Model file found. Loading {OUTPUT_MODEL_FILENAME} into memory...")
                try:
                    # YOLO() will automatically read the class names from the ONNX metadata
                    detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
                    print(f"✅ Detection model is loaded. Will now apply custom names.")

                    # --- ADD THIS BLOCK TO LOAD NAMES ON-THE-FLY ---
                    try:
                        with open(OUTPUT_NAMES_FILENAME, 'r') as f:
                            # json keys are strings, need to convert back to int
                            names_from_file = json.load(f)
                            # --- MODIFY THIS ---
                            # Load into our global cache
                            custom_names = {int(k): v for k, v in names_from_file.items()}
                        # --- REMOVE FAILING LINE ---
                        # detection_model.names = names_dict
                        print(f"✅ Auto-loaded model. Names manually set: {custom_names}")
                    except Exception as e:
                        print(f"❌ Failed to auto-load/apply names: {e}")
                        # Model will run with wrong (COCO) names, but it's better than crashing
                    # --- END ADD ---
                        
                except Exception as e:
                    print(f"❌ Failed to auto-load model: {e}")
                    # Put a permanent error on the frame
                    cv2.putText(frame, f"Error loading {OUTPUT_MODEL_NAME}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return frame
            else:
                # Model file doesn't exist, and it's not in memory.
                cv2.putText(frame, "Model not trained", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return frame

    # Run prediction
    try:
        results = detection_model.predict(frame, verbose=False, imgsz=640)
        
        # --- THIS IS THE KEY FIX ---
        # Overwrite the names on the *results* object, not the model object
        if custom_names and results:
            results[0].names = custom_names
        # --- END FIX ---

        # results[0].plot() will automatically use detection_model.names for labels
        annotated_frame = results[0].plot(boxes=True, masks=True) 
        
        # Add FPS text
        try:
            inference_time = results[0].speed['inference']
            fps = 1000 / inference_time
            text = f'FPS: {fps:.1f}'
            cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass # Squelch speed errors

        return annotated_frame

    except Exception as e:
        print(f"Error during detection: {e}")
        cv2.putText(frame, f"Detection Error: {e}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

# This class will manage the camera in a separate thread
# This is CRITICAL to allow multiple web clients to see the same stream
# and to allow API calls without interrupting the stream.
class CameraManager:
    def __init__(self, camera_id=0):
        print("📷 Initializing Camera Manager...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Set higher resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual resolution
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
            time.sleep(0.01) # ~100 FPS cap, adjust as needed

    def get_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def capture_reference_image(self):
        frame = self.get_frame()
        if frame is not None:
            cv2.imwrite(REFERENCE_IMAGE_FILENAME, frame)
            print(f"📸 Reference image captured and saved to {REFERENCE_IMAGE_FILENAME}")
            return True
        return False

    def release(self):
        self.cap.release()

# --- Initialize the Camera Manager globally ---
try:
    camera = CameraManager()
except Exception as e:
    print(f"🚨 FATAL: Could not initialize camera. {e}")
    print("🚨 Please ensure a webcam is connected and permissions are set.")
    camera = None