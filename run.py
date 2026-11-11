import os
import sys
import subprocess
import importlib.util

# ── CRITICAL: Disable user site-packages FIRST to prevent package conflicts ──
os.environ["PYTHONNOUSERSITE"] = "1"

# This script allows running the project from the root directory with automatic dependency checking.
PROJECT_NAME = "face_classification-master"
PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), PROJECT_NAME)

def check_and_install_dependencies():
    """Checks if required packages are installed and installs missing ones."""
    # Mapping of import names to pip package names
    requirements = {
        "tensorflow": "tensorflow",
        "keras": "keras",
        "pandas": "pandas",
        "numpy": "numpy",
        "h5py": "h5py",
        "cv2": "opencv-python",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "dotenv": "python-dotenv",
        "groq": "groq",
        "pydantic": "pydantic",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "PIL": "pillow"
    }
    
    missing = []
    # Temporarily suppress warnings from imports
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
    for import_name, package_name in requirements.items():
        try:
            # First try a lightweight find_spec
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                # If find_spec fails (often happens with cv2), try actual import
                __import__(import_name)
        except ImportError:
            missing.append(package_name)
        except Exception:
            # If some other weird error happens during import, count as missing just in case
            missing.append(package_name)
    
    if missing:
        # Hide the detailed installation logs to keep it clean for the client
        print(f"--- Setting up environment: {', '.join(missing)} ---")
        try:
            # Use subprocess.DEVNULL to keep the output clean
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Successfully initialized environment.\n")
        except subprocess.CalledProcessError:
            # If it fails, fallback to showing output so user can see why
            print(f"Error initializing dependencies. Trying with verbose logs...")
            subprocess.call([sys.executable, "-m", "pip", "install", *missing])

if __name__ == "__main__":
    # ── Critical: Disable user site-packages to prevent package conflicts ──
    os.environ["PYTHONNOUSERSITE"] = "1"
    
    if os.path.exists(PROJECT_DIR):
        # ── Detect and use venv if available ──
        venv_python = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
        if os.path.exists(venv_python):
            python_executable = venv_python
        else:
            python_executable = sys.executable
        
        # 1. Dependency Check (using detected Python)
        print(f"Using Python: {python_executable}")
        original_executable =  sys.executable
        sys.executable = python_executable
        check_and_install_dependencies()
        sys.executable = original_executable
        
        # 2. Run the actual project
        print(f"--- Entering project directory: {PROJECT_NAME} ---")
        os.chdir(PROJECT_DIR)
        
        try:
            # Run using the venv python executable with environment variable
            subprocess.run([python_executable, "run.py"], env=os.environ.copy())
        except KeyboardInterrupt:
            # The inner script also handles this, but we catch it here to be silent
            pass
    else:
        print(f"CRITICAL ERROR: Could not find project directory: {PROJECT_DIR}")
        print("Please ensure you are running this from the correct folder.")
        sys.exit(1)
