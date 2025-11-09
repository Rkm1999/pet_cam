import uvicorn
from fastapi import FastAPI, Response, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import anyio # Used by FastAPI to run blocking code

# Import our custom logic
import model_logic

app = FastAPI()

# --- App State ---
# This dictionary will hold the simple state of our application
# 'CAPTURE', 'ANNOTATE', 'TRAINING', 'DETECT'
app_state = {"status": "CAPTURE"}

# --- Static File Mounting ---
# This serves files from the 'generated' directory using the absolute path
app.mount(f"/{model_logic.GENERATED_FILES_DIR_NAME}", StaticFiles(directory=model_logic.GENERATED_FILES_DIR), name="generated")

# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    """Serves the main index.html file."""
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --- API Endpoints ---

@app.get("/api/status")
async def get_status():
    """Returns the current state of the application."""
    if app_state['status'] == 'TRAINING':
        return {"status": "TRAINING"}
    
    if model_logic.model_is_ready():
        app_state['status'] = 'DETECT'
    
    return JSONResponse(content=app_state)

@app.post("/api/reset")
async def reset_app():
    """Resets the application back to the CAPTURE state."""
    model_logic.clear_generated_files()
    app_state['status'] = 'CAPTURE'
    print("🔄 Application Reset")
    return JSONResponse(content={"message": "Application reset", "status": "CAPTURE"})

@app.post("/api/capture")
async def capture_image():
    """Captures the current camera frame and saves it as the reference."""
    if not model_logic.camera:
        return JSONResponse(content={"error": "Camera not initialized"}, status_code=500)

    if model_logic.camera.capture_reference_image():
        app_state['status'] = 'ANNOTATE'
        return JSONResponse(content={
            "message": "Image captured",
            # Send the correct relative URL for the browser
            "image_url": f"/{model_logic.GENERATED_FILES_DIR_NAME}/{model_logic.REFERENCE_IMAGE_NAME}",
            "image_width": model_logic.camera.width,
            "image_height": model_logic.camera.height,
            "status": "ANNOTATE"
        })
    else:
        return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

@app.post("/api/train")
async def start_training(data: dict = Body(...)):
    """
    Receives bounding box data and starts the training process.
    This runs the blocking 'step_3_train_model' in a background thread.
    """
    boxes = data.get('boxes')
    if not boxes:
        return JSONResponse(content={"error": "No boxes data provided"}, status_code=400)
    
    app_state['status'] = 'TRAINING'
    print(f"Received {len(boxes)} boxes. Starting training in background...")

    # Run the blocking function in a thread pool
    try:
        await anyio.to_thread.run_sync(model_logic.step_3_train_model, boxes)
        app_state['status'] = 'DETECT'
        return JSONResponse(content={"message": "Training complete", "status": "DETECT"})
    except Exception as e:
        app_state['status'] = 'ANNOTATE' # Revert state on failure
        print(f"❌ Training failed: {e}")
        return JSONResponse(content={"error": f"Training failed: {e}"}, status_code=500)

# --- Video Streaming Endpoints ---

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
    while app_state.get('status') == 'CAPTURE':
        frame = model_logic.camera.get_frame()
        if frame is None:
            await anyio.sleep(0.01)
            continue
        
        # Add 'aiming' crosshairs
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 1)
        cv2.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 1)
        
        jpg_bytes = encode_frame_for_stream(frame)
        if jpg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        await anyio.sleep(0.033) # ~30 FPS
    print("Stopping capture feed stream.")

async def detection_feed_generator():
    """Generator for the 'DETECT' video stream (with model inference)."""
    if not model_logic.camera:
        print("Camera not available for detection feed.")
        return

    print("Starting detection feed stream...")
    while app_state.get('status') == 'DETECT':
        frame = model_logic.camera.get_frame()
        if frame is None:
            await anyio.sleep(0.01)
            continue
        
        # Run detection
        annotated_frame = model_logic.run_detection_on_frame(frame)
        
        jpg_bytes = encode_frame_for_stream(annotated_frame)
        if jpg_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        await anyio.sleep(0.033) # ~30 FPS (or slower if model is slow)
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
        print("    Open this URL in your browser to use the Web UI.")
        uvicorn.run(app, host="127.0.0.1", port=8000)