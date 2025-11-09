import uvicorn
from fastapi import FastAPI, Response, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import anyio 
import os
import time

import model_logic

app = FastAPI()

# --- NEW: Load model on server startup ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Server starting up...")
    model_logic.load_detection_model_on_startup()

# --- Static File Mounting ---
app.mount(f"/{model_logic.GENERATED_FILES_DIR_NAME}", StaticFiles(directory=model_logic.GENERATED_FILES_DIR), name="generated")

# --- HTML Endpoint ---
@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --- API Endpoints ---

@app.get("/api/prompt_library")
async def get_prompt_library():
    """Returns the current prompt library from disk."""
    library = model_logic.load_prompt_library()
    return JSONResponse(content=library)

@app.post("/api/reset")
async def reset_app():
    """Deletes all images, models, and the prompt library."""
    model_logic.clear_generated_files()
    print("🔄 Application Reset")
    # Return a new empty library
    return JSONResponse(content=model_logic.load_prompt_library())

@app.post("/api/capture")
async def capture_image():
    """Captures a new image, saves it, and returns its info."""
    if not model_logic.camera:
        return JSONResponse(content={"error": "Camera not initialized"}, status_code=500)

    # Generate a unique filename
    filename = f"{model_logic.REFERENCE_IMAGE_PREFIX}_{int(time.time())}.jpg"
    
    success, full_path, width, height = model_logic.camera.capture_reference_image(filename)

    if success:
        return JSONResponse(content={
            "message": f"Image {filename} captured",
            "image_name": filename,
            "image_url": f"/{model_logic.GENERATED_FILES_DIR_NAME}/{filename}",
            "image_width": width,
            "image_height": height,
        })
    else:
        return JSONResponse(content={"error": "Failed to capture image"}, status_code=500)

@app.post("/api/train")
async def start_training(data: dict = Body(...)):
    """
    Receives the *entire* prompt library, saves it, and starts training.
    """
    class_names = data.get('class_names') 
    annotations = data.get('annotations')
    
    if class_names is None or annotations is None:
        return JSONResponse(content={"error": "Invalid library data provided"}, status_code=400)
    
    # 1. Save the new library as the "single source of truth"
    try:
        model_logic.save_prompt_library(data)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to save library: {e}"}, status_code=500)

    # 2. Start the training process in a background thread
    print("Starting training in background...")
    try:
        # Pass the data directly to the training function
        await anyio.to_thread.run_sync(model_logic.step_3_train_model, annotations, class_names)
        return JSONResponse(content={"message": "Training complete"})
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return JSONResponse(content={"error": f"Training failed: {e}"}, status_code=500)

# --- Video Streaming Endpoints ---

def encode_frame_for_stream(frame):
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
    while True: # Stream will be started/stopped by client
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
    """Generator for the 'DETECT' video stream (with model inference)."""
    if not model_logic.camera:
        print("Camera not available for detection feed.")
        return

    print("Starting detection feed stream...")
    while True: # Stream will be started/stopped by client
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
    return StreamingResponse(capture_feed_generator(), 
                             media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/video/detection_feed")
async def video_detection_feed():
    return StreamingResponse(detection_feed_generator(), 
                             media_type='multipart/x-mixed-replace; boundary=frame')

# --- Main entry point ---
if __name__ == "__main__":
    if model_logic.camera is None:
        print("🚨 Cannot start server, camera initialization failed.")
    else:
        print("🚀 Starting FastAPI server at http://127.0.0.1:8000")
        uvicorn.run(app, host="127.0.0.1", port=8000)