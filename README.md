
# 🖐️ AirMouse Pro — Five-Finger Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange" alt="MediaPipe">
  <img src="https://img.shields.io/badge/Platform-Windows-lightgrey?logo=windows&logoColor=white" alt="Windows">
  <img src="https://img.shields.io/badge/Status-Development-yellow" alt="Development">
</p>

**AirMouse Pro** is a real-time computer vision project that detects and labels five fingers using a webcam. Built with Python, OpenCV, and the MediaPipe Hand Landmarker Tasks API, it displays hand landmarks, finger states, tracking status, and FPS in a custom visual dashboard.

The project demonstrates how computer vision can be used to build interactive, touch-free applications.

---

## ✨ Features

- 🖐️ **Five-Finger Recognition** — Detects Thumb, Index, Middle, Ring, and Pinky.
- 🎯 **Hand Landmark Tracking** — Tracks 21 hand landmarks using MediaPipe.
- 🏷️ **Finger Name Labels** — Displays finger names near their fingertips.
- 🟢 **Finger State Detection** — Shows whether each finger is estimated to be open or closed.
- 🎨 **Creative HUD** — Custom dashboard with a dark interface, neon-style landmarks, and colored indicators.
- 📊 **Live FPS Counter** — Displays approximate processing frames per second.
- 📷 **Real-Time Webcam Processing** — Processes camera frames continuously.
- 🔄 **Mirrored Camera View** — Provides a natural selfie-style experience.
- 🧹 **Resource Cleanup** — Releases the camera and tracking resources when the application exits.
- 👨‍💻 **Developer Information** — Includes developer credits in the application and source code.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| OpenCV | Webcam access, image processing, and visualization |
| MediaPipe Tasks API | Hand landmark detection |
| NumPy | Used internally by image-processing dependencies |
| Git & GitHub | Version control and project hosting |

## 📋 Requirements

- Python 3.10 or newer, subject to dependency compatibility.
- A working webcam.
- An internet connection for the first model download.
- Windows, Linux, or macOS with a supported graphical desktop.
- Compatible versions of OpenCV and MediaPipe.

**Recommendation:** Python 3.12 is a good starting point if a compatible MediaPipe package is available for your platform.

## 📂 Project Structure

```text
AirMouse-OpenCV/
│
├── main.py                  # Main application
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .gitignore               # Git exclusions
│
└── models/
    └── hand_landmarker.task # Downloaded model (generated)
```

The `models/` directory and model file are created automatically when the application downloads the model.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ATIQULTIU/AirMouse-OpenCV.git
cd AirMouse-OpenCV
```

Replace the repository URL if you publish the project under a different GitHub repository name.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python executable directly instead.

### 3. Create `requirements.txt`

Add the following dependencies:

```text
opencv-python
mediapipe
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Alternatively, without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. Run the application

```powershell
python main.py
```

Or:

```powershell
.\.venv\Scripts\python.exe main.py
```

On the first run, the application downloads the MediaPipe hand landmark model automatically if it is missing.

## 🎮 How to Use

1. Start the application.
2. Allow access to your webcam if requested.
3. Position your hand in front of the camera.
4. Keep your hand within the camera frame and ensure sufficient lighting.
5. Observe the hand skeleton and fingertip labels.
6. Check the dashboard to see each finger's estimated state.
7. Press `Q` or `ESC` to close the application.

## 🖐️ Finger Recognition

| Finger | Landmark Index | Display |
|---|---:|---|
| Thumb | 4 | Thumb |
| Index | 8 | Index |
| Middle | 12 | Middle |
| Ring | 16 | Ring |
| Pinky | 20 | Pinky |

MediaPipe provides 21 hand landmarks. The application uses fingertip locations and finger-joint geometry to estimate whether each finger is extended.

**Note:** Finger-state detection is a geometric approximation, not a perfect classifier. Hand rotation, occlusion, camera angle, and lighting can affect accuracy. Thumb detection may be less reliable than detection of the other four fingers.

## 🖥️ Dashboard

The live interface displays:

- Application title
- Webcam tracking status
- FPS counter
- Hand skeleton and landmark connections
- Finger names beside detected fingertips
- Green indicators for estimated extended fingers
- Red indicators for estimated curled fingers
- Developer name and email

## ⚙️ Troubleshooting

### Camera not opening

- Check whether your webcam is connected.
- Close other applications that might be using the camera.
- Allow camera access in your operating system's privacy settings.
- If necessary, change the camera index in `main.py`.

For example:

```python
camera = cv2.VideoCapture(1)
```

### MediaPipe installation fails

Check your Python version:

```powershell
py -0p
```

Try creating a virtual environment with a supported Python version and reinstalling the dependencies.

### Hand is not detected

- Improve the lighting.
- Keep the entire hand visible.
- Move closer to the webcam if the hand is too small.
- Avoid rapid movements.
- Verify that the model downloaded successfully.

### Application cannot download the model

Check your internet connection and try downloading the official model manually:

[MediaPipe Hand Landmarker Model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

Save it as:

```text
models/hand_landmarker.task
```

### Dashboard text overlaps

Use a higher-resolution camera frame or adjust the dashboard text positions in `draw_dashboard()`.

## 🔒 Privacy

- Camera frames are processed locally by the application.
- The hand landmark model is downloaded from Google's MediaPipe model hosting when needed.
- The application does not intentionally upload webcam frames to a server.
- Camera access is required for live hand tracking.

## 🗺️ Future Improvements

- [ ] Mouse movement using the index fingertip.
- [ ] Left-click and right-click gestures.
- [ ] Double-click and drag controls.
- [ ] Improved finger-state classification.
- [ ] Support for tracking multiple hands.
- [ ] Gesture-based volume and brightness controls.
- [ ] Configurable sensitivity and smoothing.
- [ ] Settings panel and customizable themes.
- [ ] Performance optimization and automated tests.

## 👨‍💻 Developer

**MD. Atiqul Islam (Atik)**

- **Email:** [atik.cmttiu1001@gmail.com](mailto:atik.cmttiu1001@gmail.com)
- **GitHub:** [ATIQULTIU](https://github.com/ATIQULTIU)
- **Portfolio:** [View Portfolio](https://md-atiqul-islam.my.canva.site/)

## 📄 License

This project is intended for learning, experimentation, and portfolio development. Add a `LICENSE` file to the repository if you want to publish it under a specific open-source license.

---

<p align="center">
  <b>Built with Python, OpenCV, and MediaPipe</b><br>
  Developed by MD. Atiqul Islam (Atik)
</p>
