# AI-Driven Emotion Recognition Model for Personalized Chatbot Conversations

An end-to-end **emotion-aware chatbot system** that uses real-time facial emotion detection via webcam to silently adapt chatbot conversation tone. The system detects emotions frame-by-frame using a CNN (mini-XCEPTION), aggregates them over a 1-minute window, and uses the dominant emotion to modulate the LLM chatbot's response style — without ever mentioning emotions to the user.

---

## Features

- **Real-time Facial Emotion Detection** — 7 emotions (angry, disgust, fear, happy, sad, surprise, neutral) via webcam using pre-trained mini-XCEPTION CNN.
- **1-Minute Emotion Aggregation** — Frame-by-frame emotions are counted; every 60 seconds the dominant emotion is computed and used.
- **Emotion-Aware Chatbot** — Powered by Groq LLaMA 3 (70B). Tone silently adapts:
  - `SAD` → warm, empathetic, gently encouraging
  - `ANGRY` → calm, patient, de-escalating
  - `HAPPY` → upbeat, positive, energetic
  - `FEAR` → reassuring, supportive
  - `NEUTRAL` → professional, balanced
- **Silent Adaptation** — The chatbot NEVER mentions emotion detection to the user.
- **Text Emotion Analysis** — Optional secondary signal from chat text keywords.
- **Web UI** — Split-screen: live camera feed + chat interface in one page.
- **Gender Classification** — Detects gender of individuals (legacy feature).
- **Multiple Demo Modes** — Image, video, webcam, GradCAM visualization.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BROWSER (Frontend)                      │
│  ┌──────────────────┐    ┌────────────────────────────────┐ │
│  │  Camera Preview   │    │     Chat Interface             │ │
│  │  (MJPEG Stream)   │    │  User ↔ Bot messages           │ │
│  │  /video_feed      │    │  POST /chat                    │ │
│  └──────────────────┘    └────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────┐
│                  FastAPI Backend (app.py)                    │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Video Feed   │  │ Emotion Manager │  │   Chatbot      │ │
│  │ Generator    │──▶│ (1-min counter) │──▶│  (Groq LLM)   │ │
│  │ + ML Model   │  │ get_current()   │  │  tone adapt    │ │
│  └──────────────┘  └─────────────────┘  └────────────────┘ │
│         │                                       │           │
│    Uses existing                         System prompt      │
│    trained models                        changes per        │
│    (no retraining)                       emotion            │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
face_classification-master/
├── backend/                          # NEW — FastAPI backend
│   ├── __init__.py
│   ├── app.py                        # FastAPI server (video + chat + API)
│   ├── emotion_manager.py            # 1-minute emotion aggregator
│   ├── chatbot.py                    # Groq LLM chatbot with tone adaptation
│   └── text_emotion.py              # Text-based emotion analysis (secondary)
├── frontend/                         # NEW — Web UI
│   └── index.html                    # Camera + Chat split-screen UI
├── src/                              # Existing ML code
│   ├── models/cnn.py                 # CNN architectures (mini-XCEPTION etc.)
│   ├── utils/                        # Inference, preprocessing, datasets
│   ├── templates/index.html          # Legacy Flask template
│   ├── video_emotion_color_demo.py   # Modified — now feeds EmotionManager
│   ├── video_emotion_gender_demo.py
│   ├── image_emotion_gender_demo.py
│   ├── web_video_server.py           # Legacy Flask video server
│   └── ...
├── trained_models/                   # Pre-trained weights
│   ├── detection_models/             # Haar cascade face detector
│   ├── emotion_models/               # fer2013 mini-XCEPTION checkpoints
│   └── gender_models/                # Gender CNN checkpoints
├── datasets/                         # Dataset scripts
├── images/                           # Sample test images
├── .env                              # Groq API key (DO NOT commit)
├── run.py                            # Single-command launcher
├── REQUIREMENTS.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/LasyWorks/AI-Driven-Emotion-Recognition-Model-For-Personalized-Chatbot-Conversations.git
cd AI-Driven-Emotion-Recognition-Model-For-Personalized-Chatbot-Conversations
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r REQUIREMENTS.txt
```

### 4. Set Up Groq API Key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key from [https://console.groq.com](https://console.groq.com).

---

## Usage

### Run the Full Chatbot System (Recommended)
```bash
python run.py
```
Then open **http://localhost:8000** in your browser. You'll see:
- **Left panel**: Live webcam feed with emotion detection overlays
- **Right panel**: Chat interface — type messages and get emotion-adapted replies

### Run Standalone Demos (Legacy)

**Webcam emotion detection only (no chatbot):**
```bash
cd src
python video_emotion_color_demo.py
```

**Image emotion/gender recognition:**
```bash
cd src
python image_emotion_gender_demo.py ../images/test_image.jpg
```

**Flask web server (legacy):**
```bash
cd src
python web_video_server.py
```

---

## How the 1-Minute Emotion Logic Works

```
  Frame 1: happy ──┐
  Frame 2: happy   │
  Frame 3: sad     │  ← All emotions counted
  Frame 4: happy   │     in a 60-second window
  Frame 5: neutral │
  ...              │
  Frame N: happy ──┘
                   │
       After 60 seconds:
       ┌──────────────────────────┐
       │ happy: 42 frames        │
       │ sad: 8 frames           │ → Dominant = HAPPY
       │ neutral: 10 frames      │
       │ angry: 0 frames         │
       └──────────────────────────┘
                   │
       Counters reset, new window begins.
       Chatbot system prompt switches to "HAPPY" tone.
```

---

## Emotion Flow: Camera → Chatbot

```
Webcam Frame
    │
    ▼
Face Detection (Haar Cascade)
    │
    ▼
Emotion Classification (mini-XCEPTION CNN)
    │
    ▼
EmotionManager.record_emotion("happy")   ← per-frame
    │
    ▼
60-second aggregation window
    │
    ▼
EmotionManager.get_current_emotion()  → "happy"
    │
    ▼
Chatbot receives user message
    │
    ▼
System prompt selected based on dominant emotion
    │
    ▼
Groq LLaMA 3 generates response with adapted tone
    │
    ▼
Response sent to user (emotion NEVER mentioned)
```

---

## API Endpoints

| Method | Endpoint           | Description                               |
|--------|--------------------|-------------------------------------------|
| GET    | `/`                | Serve the frontend UI                     |
| GET    | `/video_feed`      | MJPEG stream of webcam + emotion overlay  |
| POST   | `/emotion`         | Receive per-frame emotion (external use)  |
| GET    | `/emotion/status`  | Live emotion counters (debug)             |
| GET    | `/emotion/current` | Current dominant emotion                  |
| GET    | `/emotion/history` | Aggregation history log                   |
| POST   | `/chat`            | Send message → get emotion-adapted reply  |
| POST   | `/chat/reset`      | Reset conversation history                |
| GET    | `/health`          | Health check                              |

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `GROQ_API_KEY not set` | Add your key to the `.env` file |
| `Camera not found` | Ensure webcam is connected and not used by another app |
| `Module not found: keras` | Run `pip install -r REQUIREMENTS.txt` |
| `Port 8000 already in use` | Kill the existing process or use `--port 8001` |
| TensorFlow warnings | These are suppressed automatically; safe to ignore |

---

## Training Custom Models

To retrain emotion or gender classifiers:
```bash
cd src
python train_emotion_classifier.py
python train_gender_classifier.py
```
Requires the fer2013 and IMDB datasets in the `datasets/` folder.

---

## Technologies

- **ML/DL**: Keras, TensorFlow, OpenCV, mini-XCEPTION CNN
- **Backend**: FastAPI, Uvicorn, Python
- **LLM**: Groq API (LLaMA 3 70B Versatile)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Emotion Aggregation**: Custom 60-second sliding window counter
