"""
run.py — Single-command launcher for the Emotion-Aware Chatbot
==============================================================
Usage:
    python run.py

This starts the FastAPI server which handles:
  - Webcam video feed with real-time emotion detection
  - 1-minute emotion aggregation
  - Emotion-aware chatbot API
  - Frontend UI served at http://localhost:8000
"""

import os
import sys

# ── CRITICAL: Disable user site-packages to prevent package conflicts ──
# This ensures the venv's packages are used exclusively
os.environ["PYTHONNOUSERSITE"] = "1"

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

# Suppress TensorFlow warnings BEFORE any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

def main():
    print("=" * 62)
    print("  AI-Driven Emotion-Aware Chatbot System")
    print("=" * 62)
    print()
    print("  Components:")
    print("  [1] Webcam Emotion Detection (ML)  ✓")
    print("  [2] 1-Minute Emotion Aggregator     ✓")
    print("  [3] Groq LLaMA 3 Chatbot            ✓")
    print("  [4] Web Frontend (Camera + Chat)     ✓")
    print()

    # Verify Groq API key
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("  ⚠  WARNING: GROQ_API_KEY not found in .env file!")
        print("     Chat responses will fail until you set it.")
        print()
    else:
        print(f"  ✓  Groq API Key loaded ({api_key[:8]}...)")
        print()

    print("  Starting server...")
    print("  Open http://localhost:8000 in your browser")
    print("  Press Ctrl+C to stop")
    print("=" * 62)
    print()

    import uvicorn
    import logging
    import signal
    import sys

    # Configure logging to suppress shutdown errors
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Custom log config to suppress CancelledError during shutdown
    log_config = uvicorn.config.LOGGING_CONFIG.copy()
    log_config["loggers"]["uvicorn"] = {"handlers": ["default"], "level": "INFO"}
    log_config["loggers"]["uvicorn.error"] = {"handlers": ["default"], "level": "WARNING"}
    log_config["loggers"]["uvicorn.access"] = {"handlers": ["access"], "level": "INFO"}

    # Create a custom handler for SIGINT to ensure clean exit
    def signal_handler(sig, frame):
        print("\n  Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    try:
        uvicorn.run(
            "backend.app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info",
            log_config=log_config,
        )
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n" + "=" * 62)
        print("  Server stopped. Goodbye!")
        print("=" * 62)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Shutting down gracefully...")
