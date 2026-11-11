# Running Instructions

## Quick Start

Simply run the following command from the root directory:

```bash
python run.py
```

The launcher script will automatically:
- Detect and use the virtual environment (`venv/`)
- Check for missing dependencies and install them
- Start the FastAPI server on http://localhost:8000

## What the Script Does

1. **Auto-detects the virtual environment** - No need to manually activate it
2. **Installs missing packages** - Automatically checks and installs any missing dependencies
3. **Prevents package conflicts** - Sets `PYTHONNOUSERSITE=1` to avoid conflicts between global and venv packages
4. **Clean, professional output** - No verbose logs unless there's an error

## Accessing the Application

Once the server starts, you'll see:
```
==============================================================
  AI-Driven Emotion-Aware Chatbot System
==============================================================

  Components:
  [1] Webcam Emotion Detection (ML)  ✓
  [2] 1-Minute Emotion Aggregator     ✓
  [3] Groq LLaMA 3 Chatbot            ✓
  [4] Web Frontend (Camera + Chat)     ✓

  ✓  Groq API Key loaded (gsk_NzwN...)

  Starting server...
  Open http://localhost:8000 in your browser
  Press Ctrl+C to stop
==============================================================

  - Camera status: Connected! ✓

INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Then open your browser and navigate to: **http://localhost:8000**

## Troubleshooting

### Port Already in Use
If you see `ERROR: [Errno 10048] error while attempting to bind on address`, it means port 8000 is already occupied.

Solution:
```powershell
# Kill all Python processes
Get-Process -Name python* | Stop-Process -Force
# Then run again
python run.py
```

### Camera Not Found
If you see "Camera status: No camera found", the application will still run but the video feed won't work until a camera is connected.

### Missing Dependencies
If the auto-installer fails, manually install dependencies:
```bash
venv\Scripts\activate
pip install -r face_classification-master\REQUIREMENTS.txt
```

## Client Submission Notes

The application is production-ready with:
- ✅ Clean, professional startup logs
- ✅ No verbose TensorFlow/Keras warnings
- ✅ Automatic dependency management
- ✅ Error handling for missing camera/API keys
- ✅ All Keras 3 compatibility issues resolved
- ✅ Proper package isolation to prevent conflicts
