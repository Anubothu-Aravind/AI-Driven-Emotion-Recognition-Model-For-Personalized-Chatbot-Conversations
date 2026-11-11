"""
FastAPI Backend — Emotion-Aware Chatbot Server
===============================================
Endpoints:
  POST /emotion         → receive per-frame emotion from webcam detector
  POST /chat            → receive user message, return emotion-adapted reply
  GET  /emotion/status  → debug: show live emotion counters
  GET  /emotion/history → debug: show aggregation history
  POST /chat/reset      → reset conversation history
  GET  /health          → health check

Also serves the frontend static files and provides a video feed
endpoint that runs the emotion detection in-process.
"""

import os
import sys
import time
import threading
import traceback
import asyncio
from contextlib import asynccontextmanager

import os
import sys

# ── Suppress ALL TensorFlow noise and warnings (CRITICAL for client submission) ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Also suppress some Keras 3 warnings
os.environ["KERAS_BACKEND"] = "tensorflow"

import time
import threading
import traceback
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cv2
import numpy as np
from statistics import mode

# ── Add src/ to path so we can import the existing ML utilities ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from utils.datasets import get_labels
from utils.inference import detect_faces, apply_offsets, load_detection_model
from utils.inference import draw_text, draw_bounding_box
from utils.preprocessor import preprocess_input

# ── Load .env if present ──
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Local imports ──
from backend.emotion_manager import emotion_manager
from backend.chatbot import get_chat_response, reset_conversation
from backend.text_emotion import get_combined_emotion

# ══════════════════════════════════════════════
#  Load ML Models (once at startup)
# ══════════════════════════════════════════════
from keras.models import load_model as keras_load_model

DETECTION_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "trained_models", "detection_models", "haarcascade_frontalface_default.xml"
)
EMOTION_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "trained_models", "emotion_models", "fer2013_mini_XCEPTION.102-0.66.hdf5"
)

# ── Load everything with clean error handling ──
try:
    emotion_labels = get_labels("fer2013")
    face_detection = load_detection_model(DETECTION_MODEL_PATH)
    emotion_classifier = keras_load_model(EMOTION_MODEL_PATH, compile=False)
    emotion_target_size = emotion_classifier.input_shape[1:3]
except Exception as e:
    print(f"\n[CRITICAL ERROR] Failed to load ML models: {e}")
    sys.exit(1)

# ── Webcam + emotion window ──
video_capture = cv2.VideoCapture(0)
_camera_lock = threading.Lock()

if not video_capture.isOpened():
    print("\n  - Camera status: No camera found. Will retry on start.\n")
else:
    print("\n  - Camera status: Connected! ✓\n")

FPS_LIMIT = 15  # max frames per second
_FRAME_DELAY = 1.0 / FPS_LIMIT

frame_window = 10
emotion_offsets = (0, 0)
emotion_window: list[str] = []
_camera_lock = threading.Lock()

# The latest detected emotion text for the UI overlay
_latest_frame_emotion = "neutral"

# Shutdown flag to gracefully stop video generation
_shutdown_flag = threading.Event()

# ══════════════════════════════════════════════
#  Lifespan (Startup/Shutdown)
# ══════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic can go here if needed
    yield
    # Shutdown logic - set the shutdown flag first
    _shutdown_flag.set()
    # Give the video generator a moment to finish cleanly
    await asyncio.sleep(0.2)
    # Then release the camera
    try:
        with _camera_lock:
            if video_capture.isOpened():
                video_capture.release()
    except Exception:
        pass

# ══════════════════════════════════════════════
#  FastAPI App
# ══════════════════════════════════════════════
app = FastAPI(title="Emotion-Aware Chatbot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ══════════════════════════════════════════════
#  Pydantic Models
# ══════════════════════════════════════════════
class EmotionPayload(BaseModel):
    emotion: str

class ChatPayload(BaseModel):
    message: str

class EmojiPayload(BaseModel):
    emotion: str | None = None   # None or "" to clear


# ══════════════════════════════════════════════
#  Video Frame Generator (with emotion detection)
# ══════════════════════════════════════════════
def generate_frames():
    """Yield MJPEG frames with emotion overlay + feed emotions to manager."""
    global emotion_window, _latest_frame_emotion, video_capture

    _consecutive_failures = 0

    try:
        while not _shutdown_flag.is_set():
            frame_start = time.time()

            with _camera_lock:
                if not video_capture.isOpened():
                    video_capture = cv2.VideoCapture(0)
                    time.sleep(1)
                    continue
                success, frame = video_capture.read()

            if not success:
                _consecutive_failures += 1
                if _consecutive_failures > 30:
                    # Try to re-open the camera
                    with _camera_lock:
                        video_capture.release()
                        video_capture = cv2.VideoCapture(0)
                    _consecutive_failures = 0
                time.sleep(0.1)
                continue

            _consecutive_failures = 0
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(face_detection, gray_image)

            for face_coordinates in faces:
                x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
                gray_face = gray_image[y1:y2, x1:x2]
                try:
                    gray_face = cv2.resize(gray_face, emotion_target_size)
                except Exception:
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)
                emotion_probability = np.max(emotion_prediction)
                emotion_label_arg = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[emotion_label_arg]

                # ── Feed into 1-minute aggregator ──
                emotion_manager.record_emotion(emotion_text, float(emotion_probability))
                _latest_frame_emotion = emotion_text

                emotion_window.append(emotion_text)
                if len(emotion_window) > frame_window:
                    emotion_window.pop(0)
                try:
                    emotion_mode = mode(emotion_window)
                except Exception:
                    continue

                # Color mapping
                if emotion_text == "angry":
                    color = emotion_probability * np.asarray((255, 0, 0))
                elif emotion_text == "sad":
                    color = emotion_probability * np.asarray((0, 0, 255))
                elif emotion_text == "happy":
                    color = emotion_probability * np.asarray((255, 255, 0))
                elif emotion_text == "surprise":
                    color = emotion_probability * np.asarray((0, 255, 255))
                else:
                    color = emotion_probability * np.asarray((0, 255, 0))

                color = color.astype(int).tolist()

                draw_bounding_box(face_coordinates, frame, color)
                draw_text(face_coordinates, frame, emotion_mode, color, 0, -45, 1, 1)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            # ── Frame rate limiter ──
            elapsed = time.time() - frame_start
            if elapsed < _FRAME_DELAY:
                time.sleep(_FRAME_DELAY - elapsed)
    except (asyncio.CancelledError, GeneratorExit, StopAsyncIteration):
        # Normal shutdown, don't print anything
        pass
    except Exception as e:
        if not _shutdown_flag.is_set():
            print(f"Error in video generator: {e}")
    finally:
        # Ensure camera is released if this is the last generator
        pass


# ══════════════════════════════════════════════
#  Routes
# ══════════════════════════════════════════════

# ── Frontend ──
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── Video feed ──
@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Emotion endpoints ──
@app.post("/emotion")
async def receive_emotion(payload: EmotionPayload):
    """Receive per-frame emotion from an external detector (optional)."""
    emotion_manager.record_emotion(payload.emotion)
    return {"status": "ok", "recorded": payload.emotion}


@app.get("/emotion/status")
async def emotion_status():
    """Debug endpoint: show current emotion counters."""
    return emotion_manager.get_live_counts()


@app.get("/emotion/history")
async def emotion_history():
    """Debug endpoint: show aggregation history."""
    return {"history": emotion_manager.get_history()}


@app.get("/emotion/current")
async def emotion_current():
    """Get the current dominant emotion and effective emotion."""
    return {
        "dominant_emotion": emotion_manager.get_current_emotion(),
        "frame_emotion": _latest_frame_emotion,
        "latest_confidence": round(emotion_manager.get_latest_confidence(), 4),
        "emoji_override": emotion_manager.get_emoji_emotion(),
        "effective_emotion": emotion_manager.get_effective_emotion(),
    }


# ── Emoji override endpoints ──
@app.post("/emoji")
async def set_emoji(payload: EmojiPayload):
    """Set or clear the emoji emotion override."""
    emotion = payload.emotion
    if not emotion or emotion.strip() == "":
        emotion_manager.set_emoji_emotion(None)
        return {"status": "emoji cleared", "emoji_emotion": None}
    emotion_manager.set_emoji_emotion(emotion)
    return {"status": "emoji set", "emoji_emotion": emotion_manager.get_emoji_emotion()}


@app.get("/emoji")
async def get_emoji():
    """Get the current emoji override state."""
    return {"emoji_emotion": emotion_manager.get_emoji_emotion()}


# ── Chat endpoint ──
@app.post("/chat")
async def chat(payload: ChatPayload):
    """
    Receive user message → resolve effective emotion via priority logic → generate adapted reply.
    Priority: emoji override > face emotion (confidence >= 0.6) > neutral
    """
    # Get the effective emotion using priority logic
    effective_emotion = emotion_manager.get_effective_emotion()

    # Text emotion is a secondary signal ONLY when no emoji override is active
    # Emoji override is the ABSOLUTE top priority and must not be overridden
    if emotion_manager.get_emoji_emotion() is None:
        effective_emotion = get_combined_emotion(effective_emotion, payload.message)

    response_text = get_chat_response(payload.message, effective_emotion)

    return {
        "reply": response_text,
        "emotion_used": effective_emotion,  # for debug; frontend can ignore this
        "emoji_override": emotion_manager.get_emoji_emotion(),
        "face_confidence": round(emotion_manager.get_latest_confidence(), 4),
    }


@app.post("/chat/reset")
async def chat_reset():
    """Reset conversation history."""
    reset_conversation()
    return {"status": "conversation reset"}


@app.get("/health")
async def health():
    return {"status": "running", "timestamp": time.time()}


# ── Run directly ──
if __name__ == "__main__":
    import uvicorn
    import signal
    import logging

    logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)

    print("=" * 60)
    print("  Emotion-Aware Chatbot Server")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 60)

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        pass
