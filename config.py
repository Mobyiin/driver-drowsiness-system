import os

# Paths for model and assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

WARNING_SOUND_PATH = os.path.join(BASE_DIR, "assets", "warning.mp3")
CRITICAL_SOUND_PATH = os.path.join(BASE_DIR, "assets", "critical.mp3")

# YOLOv8 Model Configuration
CONFIDENCE_THRESHOLD = 0.7

# Time Thresholds 
DROWSY_TIME_LIMIT = 5.0
CRITICAL_SOUND_DURATION_SEC = 15.0

# Fatigue Detection Configuration
FATIGUE_WINDOW_SEC = 120   
FATIGUE_ALERT_LIMIT = 4

# Camera Configuration
CAMERA_INDEX = 0            
IDLE_CAMERA_FPS = 5.0
WINDOW_NAME = "Driver Drowsiness System"
