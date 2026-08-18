# Real-Time Driver Drowsiness & Fatigue Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0.0.93-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pygame](https://img.shields.io/badge/Pygame--CE-2.5.0-red.svg)](https://pyga.me)

An end-to-end, real-time computer vision system engineered to monitor driver attentiveness, track micro-sleep events, and evaluate overall physical fatigue. Powered by a custom-trained **YOLOv8** object detection model, an adaptive **Sliding Window Fatigue Queue**, and a **Dynamic State Machine**, this application provides non-intrusive driver monitoring with instant, multi-channel audio warnings and a live web analytics dashboard.

---

## 📸 Screenshots & Demo

### Live Detection & UI Overlay

| Awake State | Drowsy Warning (5s) | Danger State | Critical Alert |
| :---: | :---: | :---: | :---: |
| ![Awake Status](assets/images/awake_status.png) | ![Drowsy Warning](assets/images/drowsy_warning.png) | ![Drowsy Danger](assets/images/drowsy_danger.png) | ![Critical Alert](assets/images/critical_alert.png) |
| *Normal driving condition* | *Short-term audio loop active* | *High risk drowsiness detected* | *Persistent fatigue alert* |

## Key Features

- **Live Web Dashboard (Streamlit):** A decoupled UI architecture providing a real-time monitoring dashboard, live video rendering, and dynamic 60-second charts tracking drowsiness trends and system FPS.
- **Smart Target Locking (Center-Gravity Selection):** Prevents the system from jumping to passengers. It dynamically locks onto the primary driver using a weighted scoring formula based on bounding box area and proximity to the frame's center.
- **Active/Idle Power Management (Adaptive Throttling):** Intelligently reduces CPU/GPU load. When the driver leaves the frame, the system enters an `IDLE` state, throttling camera hardware limits to 5 FPS and reducing inference frequency.
- **Robust Occlusion Handling (3-State Logic):** Prevents false resets of the fatigue timer when the driver briefly drops their head or leaves the frame, triggering an immediate "Driver Missing" critical alert if the target is lost for $\ge 3.0$ seconds.
- **Dual-Stage Adaptive Alert Framework:**
  - **Stage 1 (Micro-sleep Alert):** Triggers a continuous warning audio loop if eye closure persists for $\ge 5.0$ seconds.
  - **Stage 2 (Chronic Fatigue Break Warning):** Evaluates cumulative fatigue over a 2-minute rolling window, triggering an emergency "Take a Break" alert upon repeated offenses.
- **Non-Blocking Multi-Threaded Video & Audio:** A dedicated `CameraStream` thread eliminates video lag, while an asynchronous `pygame.mixer` engine handles concurrent audio channels without freezing frames.

---

## Project Architecture

```text
driver-drowsiness-system/
│
├── assets/
│   ├── images/                      
│   ├── warning.mp3
│   └── critical.mp3
|
├── models/                  # Custom trained YOLOv8 model weights
│   └── best.pt
│
├── trains/                  # Colab training workflow
│   └── yolo_drowsiness_training.ipynb
│
├── src/                     # Core system modules
│   ├── __init__.py
│   ├── camera_stream.py     # Multi-threaded async video capture
│   ├── detector.py          # YOLOv8 inference & Target Locking pipeline
│   ├── activeidle_sm.py     # State Machine for adaptive resource scaling
│   ├── fatigue_analyzer.py  # Sliding-window fatigue algorithm
│   └── audio_player.py      # Non-blocking multi-channel sound engine
│
├── utils/                   # Helper functions
│   ├── __init__.py
│   └── helpers.py           # OpenCV UI rendering engine
│
├── app.py                   # Streamlit Web Dashboard entrypoint
├── main.py                  # Standard Desktop/OpenCV entrypoint
├── config.py                # Centralized project configuration
└── requirements.txt         # Environment dependencies

```

## Dataset & Training Pipeline

### 1. Dataset Curation & Data Split Ratio
The dataset was curated and structured using **Roboflow Universe**, specifically annotated to distinguish between driver states (`awake` and `drowsy`).

To ensure robust generalization and prevent overfitting, the dataset was split into three independent subsets:
- **Train Set (68%):** Used for weight optimization during training.
- **Validation Set (19%):** Used for monitoring loss and hyperparameter tuning during 50 epochs.
- **Test Set (13%):** Reserved for unbiased performance evaluation.

- **Source:** Roboflow Universe
- **Format:** YOLOv8 PyTorch format
- 🔗 **Roboflow Dataset Link:** [Driver Drowsiness System Dataset](https://universe.roboflow.com/mobin-yaghooti/driver-drowsiness-system)

### 2. Training Configuration (Google Colab T4 GPU)
The model was fine-tuned on Google Colab using a Tesla T4 GPU instance:
- **Base Model:** `yolov8n.pt` (Nano variant for high FPS edge inference)
- **Epochs:** 50
- **Image Resolution:** $640 \times 640$ pixels
- **Notebook Reference:** `trains/train_drowsy_detection_model.ipynb`

## Technical Deep-Dive & System Methodology

### 1. Weighted Target Locking (`src/detector.py`)
To prevent the model from falsely switching focus to passengers in the background, detections are evaluated using a Center-Gravity scoring system. The system locks onto the detection with the highest score:

$$\text{Score} = \text{Area} \times (1 - \hat{d})$$

Where normalized distance $\hat{d}$ is defined over the maximum frame diagonal distance $d_{\max}$:

$$\hat{d} = \frac{d_{\text{center}}}{d_{\max}} = \frac{\sqrt{(x_{\text{center}} - x_{\text{frame}})^2 + (y_{\text{center}} - y_{\text{frame}})^2}}{\sqrt{\left(\frac{W}{2}\right)^2 + \left(\frac{H}{2}\right)^2}}$$

This guarantees the driver (largest area, closest to the camera center) is always the primary target.

### 2. Active/Idle State Machine (`src/activeidle_sm.py`)
To optimize deployment on edge devices, the system modulates inference rates and hardware capture speeds based on target presence:
- **ACTIVE State:** Driver is present $\rightarrow$ Hardware limits lifted, inference runs at target $10\text{ FPS}$.
- **IDLE State:** Driver missing for $\ge 10\text{s}$ $\rightarrow$ Camera hardware throttled to $5\text{ FPS}$, inference dropped to $0.5\text{ FPS}$, drastically cooling the CPU.

### 3. Smart Occlusion Handling (3-State Logic)
Standard fatigue monitors reset their timers if the face is temporarily lost (e.g., driver drops their head). This system implements a 3-state memory loop:

- **Awake:** Resets all timers.
- **Drowsy:** Increases drowsiness duration:
$$T_{\text{drowsy}} = t_{\text{now}} - t_{\text{start}}$$
- **No Target:** Pauses drowsiness duration without resetting, triggering a critical alarm if missing time exceeds threshold:
$$T_{\text{missing}} = t_{\text{now}} - t_{\text{lost}} \ge 3.0\text{ s}$$

### 4. Sliding Window Fatigue Algorithm (`src/fatigue_analyzer.py`)
Driver fatigue is cumulative. The system maintains a time-stamped history queue of all triggered alerts and evaluates fatigue over a moving window:

$$\mathcal{Q}_{\text{active}} = \{ t \in \mathcal{T}_{\text{alerts}} \mid (t_{\text{current}} - t) \le W_{\text{fatigue}} \}$$

- **Window Size ($W_{\text{fatigue}}$):** $120\text{ seconds}$ (2 minutes).
- **Trigger Condition:** When $|\mathcal{Q}_{\text{active}}| \ge 4$, a Critical Fatigue State is declared, activating the prolonged emergency warning.

## Key Technical Challenges & Engineering Solutions

- **High CPU Utilization & Thermal Throttling:**
  - *Solution:* Implemented `torch.set_num_threads(2)` and an Adaptive Throttling architecture (`activeidle_sm.py`), dropping CPU usage from ~40% to ~5% during idle periods.
- **Preventing Video Stuttering During Sound Playback:** 
  - *Solution:* Built `AudioPlayer` using `pygame.mixer` with dedicated asynchronous channels, allowing sound loops to execute without blocking the main OpenCV thread.
- **Web App Memory Leaks:**
  - *Solution:* Utilized Python's `collections.deque` with a `maxlen=60` in the Streamlit dashboard to maintain $O(1)$ performance for live trend charts without overloading RAM.

---

## Getting Started

### Prerequisites
- **Python:** 3.10+ installed.
- **Hardware:** Integrated or USB Webcam (IP Cameras supported via config string).


### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/mobyiin/driver-drowsiness-system.git
cd driver-drowsiness-system
```

2. **Create and activate a Virtual Environment:**

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

3. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

### Running the Application

This project supports a decoupled architecture with two distinct interfaces. Choose the one that fits your use case:

**Option A: Web Dashboard Analytics (Recommended)**
Launch the modern Streamlit dashboard featuring live telemetry and behavioral charts.

```bash
streamlit run app.py
```

**Option B: Standard Desktop Application**
Run the lightweight, headless OpenCV window.

```bash
python main.py
```

To safely terminate either application, press `q` on the video feed or use your OS window controls.

### Author & Acknowledgments

- **Developer:** Mobin Yaghooti
- **Special Thanks:** [Ultralytics](https://github.com/ultralytics) for the YOLOv8 framework, [Streamlit](https://streamlit.io/) for the web UI framework, and [Roboflow](https://roboflow.com) for dataset management tools.

