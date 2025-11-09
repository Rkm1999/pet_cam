import cv2
import numpy as np
import os
import time
import threading
from ultralytics import YOLO, YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
import anyio # For running blocking code in async

# ========== CONFIGURATION ==========
# --- Use absolute paths based on this file's location ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_FILES_DIR_NAME = "generated"
GENERATED_FILES_DIR = os.path.join(SCRIPT_DIR, GENERATED_FILES_DIR_NAME)
# --- End of new path logic ---

REFERENCE_IMAGE_NAME = "reference_image.jpg"
OUTPUT_MODEL_NAME = "custom_model.onnx"
BASE_MODEL_NAME = "yoloe-11l-seg.pt"

# Ensure the directory exists
os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

REFERENCE_IMAGE_FILENAME = os.path.join(GENERATED_FILES_DIR, REFERENCE_IMAGE_NAME)
OUTPUT_MODEL_FILENAME = os.path.join(GENERATED_FILES_DIR, OUTPUT_MODEL_NAME)
# ===================================

# --- Global state for the model ---
# We need to load the model into memory for detection
detection_model = None
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
    global detection_model
    with model_load_lock:
        detection_model = None
    print("✅ Files cleared.")


def step_3_train_model(boxes_data: list):
    """
    Logic from your Image-Prompt ONNX Conversion.py
    This is a BLOCKING, CPU/GPU-intensive function.
    It should be run in a separate thread.
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
        return False

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
        # Call export without the 'file' argument.
        exported_file_path = model.export(format="onnx", imgsz=640) # Using 640 for better accuracy
    except Exception as e:
        print(f"❌ Error during model.export(): {e}")
        return False

    print(f"Model exported to default path: {exported_file_path}")

    # Rename the exported file to our desired output name
    try:
        if os.path.exists(OUTPUT_MODEL_FILENAME):
            os.remove(OUTPUT_MODEL_FILENAME)
        
        os.rename(exported_file_path, OUTPUT_MODEL_FILENAME)
        print(f"Successfully renamed model to: {OUTPUT_MODEL_FILENAME}")
    except Exception as e:
        print(f"❌ Error renaming model: {e}")
        return False

    print("✅ Model export complete!")
    print("Object mapping:")
    for box_info in boxes_data:
        print(f"  ID {box_info['class_id']} -> Box {box_info['bbox']}")
    
    # --- Load the new model for detection ---
    global detection_model
    with model_load_lock:
        print(f"Loading new ONNX model for detection: {OUTPUT_MODEL_FILENAME}")
        detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
        print("✅ Detection model is loaded and ready.")
        
    return True

def run_detection_on_frame(frame: np.ndarray) -> np.ndarray:
    """
    Runs the loaded ONNX model on a single frame and returns the annotated frame.
    Will auto-load the model from disk if it exists but isn't in memory.
    """
    global detection_model
    
    with model_load_lock:
        if detection_model is None:
            # Model is not in memory. Check if the file exists on disk.
            if model_is_ready():
                print(f"Model file found. Loading {OUTPUT_MODEL_FILENAME} into memory...")
                try:
                    detection_model = YOLO(OUTPUT_MODEL_FILENAME, task='segment')
                    print("✅ Detection model is loaded and ready.")
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
        annotated_frame = results[0].plot(boxes=True, masks=True) # Show masks too!
        
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