import uvicorn
from fastapi import FastAPI, Response, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import anyio 
import os # --- ADD THIS ---

import model_logic

app = FastAPI()

# --- App State ---
# MODIFIED: State now holds a list of captured images
app_state = {"status": "CAPTURE", "images": []}

# --- Static File Mounting ---
app.mount(f"/{model_logic.GENERATED_FILES_DIR_NAME}", StaticFiles(directory=model_logic.GENERATED_FILES_DIR), name="generated")

# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --- API Endpoints ---

@app.get("/api/status")
async def get_status():
    if app_state['status'] == 'TRAINING':
        return {"status": "TRAINING"}
    
    if model_logic.model_is_ready():
        app_state['status'] = 'DETECT'
    
    # Send image list along with status
    return JSONResponse(content=app_state)

@app.post("/api/reset")
async def reset_app():
    """Resets the application back to the CAPTURE state."""
    model_logic.clear_generated_files()
    # MODIFIED: Clear image list
    app_state['status'] = 'CAPTURE'
    app_state['images'] = []
    print("🔄 Application Reset")
    return JSONResponse(content={"message": "Application reset", "status": "CAPTURE", "images": []})

# --- MODIFIED: Capture now adds to a list ---
@app.post("/api/capture")
async def capture_image():
    """Captures a frame and adds it to the list of reference images."""
    if not model_logic.camera:
        return JSONResponse(content={"error": "Camera not initialized"}, status_code=500)

    # Generate a new filename based on count
    image_count = len(app_state['images'])
    filename = f"{model_logic.REFERENCE_IMAGE_PREFIX}_{image_count}.jpg"
    
    success, full_path = model_logic.camera.capture_reference_image(filename)

    if success:
        # Add the *base filename* to the state
        app_state['images'].append(filename)
        # We set status to ANNOTATE, but the UI will decide what to do
        app_state['status'] = 'ANNOTATE'
        
        return JSONResponse(content={
            "message": f"Image {filename} captured",
            "image_name": filename,
            "image_url": f"/{model_logic.GENERATED_FILES_DIR_NAME}/{filename}",
            "all_images": app_state['images'], # Send the full list back
            "status": "ANNOTATE"
        })
    else:
        return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

# --- MODIFIED: Train now expects a different data structure ---
@app.post("/api/train")
async def start_training(data: dict = Body(...)):
    """
    Receives all annotations for all images and starts training.
    """
    # NEW: Expects a dictionary of annotations
    annotations = data.get('annotations')
    class_names = data.get('class_names') 
    
    if not annotations or not class_names:
        return JSONResponse(content={"error": "No annotations or class_names data provided"}, status_code=400)
    
    app_state['status'] = 'TRAINING'
    print(f"Received annotations for {len(annotations)} images. Starting training in background...")

    try:
        # Pass the new data structures to the (new) training function
        await anyio.to_thread.run_sync(model_logic.step_3_train_model, annotations, class_names)
        app_state['status'] = 'DETECT'
        return JSONResponse(content={"message": "Training complete", "status": "DETECT"})
    except Exception as e:
        app_state['status'] = 'ANNOTATE' # Revert state on failure
        print(f"❌ Training failed: {e}")
        return JSONResponse(content={"error": f"Training failed: {e}"}, status_code=500)

# --- Video Streaming Endpoints (Unchanged) ---

def encode_frame_for_stream(frame):
    """Encodes a single frame to JPEG for streaming."""
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return None
    return buffer.tobytes()

async def capture_feed_generator():
    """Generator for the 'CAPTURE' video stream (raw camera)."""
    if not model_logic.camera:
        print("Camera not available for capture feed.")
        return

    print("Starting capture feed stream...")
    # MODIFIED: Stay in CAPTURE or ANNOTATE (while not training)
    while app_state.get('status') in ['CAPTURE', 'ANNOTATE']:
        frame = model_logic.camera.get_frame()
        if frame is None:
            await anyio.sleep(0.01)
            continue
        
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
        
        jpg_bytes = encode_frame_for_stream(frame)
        if jpg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        await anyio.sleep(0.033) 
    print("Stopping capture feed stream.")

async def detection_feed_generator():
    if not model_logic.camera:
        print("Camera not available for detection feed.")
        return

    print("Starting detection feed stream...")
    while app_state.get('status') == 'DETECT':
        frame = model_logic.camera.get_frame()
        if frame is None:
            await anyio.sleep(0.01)
            continue
        
        annotated_frame = model_logic.run_detection_on_frame(frame)
        
        jpg_bytes = encode_frame_for_stream(annotated_frame)
        if jpg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        await anyio.sleep(0.033) 
    print("Stopping detection feed stream.")

@app.get("/video/capture_feed")
async def video_capture_feed():
    """Streams the raw camera feed."""
    return StreamingResponse(capture_feed_generator(), 
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video/detection_feed")
async def video_detection_feed():
    """Streams the camera feed with object detection overlays."""
    return StreamingResponse(detection_feed_generator(), 
                             media_type='multipart/x-mixed-replace; boundary=frame')

# --- Main entry point ---
if __name__ == "__main__":
    if model_logic.camera is None:
        print("🚨 Cannot start server, camera initialization failed.")
    else:
        print("🚀 Starting FastAPI server at http://127.0.0.1:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)