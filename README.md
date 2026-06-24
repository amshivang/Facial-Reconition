# Offline Facial Recognition Video Analyzer

A high-performance, secure, 100% on-device facial recognition application designed to scan security camera footage and identify timestamps when a target person is present. 

Developed with a privacy-first architecture, this tool does not connect to the cloud or make external API requests, making it compliant with strict chain-of-custody and confidentiality protocols.

---

## 🚀 Windows Desktop App (Direct Download)

For the easiest experience on Windows, you can download the pre-compiled standalone executable. This packages the GUI, Python runtime, OpenCV processing library, and the AI models into a single file that runs fully offline without any configuration.

### How to Download & Run:
1. Go to the **[Releases](https://github.com/amshivang/Facial-Reconition/releases)** section of this repository.
2. Download the latest `video_ai.exe`.
3. Double-click the file to launch the application.
   *(Note: Since it is an unsigned executable, Windows SmartScreen may show a warning. Click **More Info** -> **Run anyway** to start it).*

---

## Key Features

- **Interactive Scanning (Pause-on-Match):** The scanner automatically pauses as soon as a target face is detected, showing a side-by-side view of the face crop next to the full surveillance camera frame. You can choose to **Continue Scan** or **Terminate**.
- **80% Accuracy Optimization:** The system is pre-configured with a similarity threshold of `0.31` using OpenCV SFace, optimized to flag suspects even in poor lighting, at angles, or with motion blur.
- **Motion Gating Filter:** Skips sections of static, unchanging video footage (e.g., empty hallways) automatically to accelerate analysis.
- **Dynamic Sampling Rate (FPS):** Allows you to choose how many frames to analyze per second of video to trade off speed vs. thoroughness.
- **Hardware Acceleration:** Native GPU acceleration using ONNX Runtime with NVIDIA CUDA/TensorRT, automatically falling back to optimized multi-threaded CPU execution.
- **Self-Contained Executable:** The models are embedded in the desktop application so you can run it completely offline out-of-the-box.

---

## Tech Stack

- **Desktop UI:** CustomTkinter (Fluent-like native Windows GUI).
- **Web UI:** Streamlit (local browser dashboard).
- **Core Processing:** Python 3.10+ & OpenCV.
- **Face Detection:** YuNet (via OpenCV FaceDetectorYN).
- **Face Recognition:** SFace (via OpenCV FaceRecognizerSF).
- **GPU Inference Engine:** ONNX Runtime GPU (CUDA/TensorRT).

---

## Developer Installation & Setup

If you prefer to run the application from source code or modify it, follow these setup steps:

### 1. Prerequisites
- **Python 3.10 to 3.14** installed.
- (Optional) **NVIDIA GPU** with CUDA Toolkit (v11.8 or v12.x/v13.x) & cuDNN installed for hardware acceleration.

### 2. Clone and Setup
Clone this repository to your local directory:
```bash
git clone https://github.com/amshivang/Facial-Reconition.git
cd Facial-Reconition
```

Create a virtual environment and activate it:
*   **Windows (PowerShell):**
    ```powershell
    python -m venv .venv
    .venv\Scripts\activate
    ```
*   **Linux / macOS:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

Install the dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Application from Source

You have two options when running from source:

### Option A: Launch the Desktop GUI
Run the native Windows desktop app:
```bash
python gui.py
```

### Option B: Launch the Web Dashboard
Launch the local web dashboard:
```bash
streamlit run app.py
```
The interface will automatically open in your default browser at `http://localhost:8501`.

### How to use it:
1. **Choose a Target Photo:** Click "Choose Target Photo" and select an image of the person you want to find. It will show a cropped preview of the detected target face.
2. **Choose a Video File:** Click "Choose Video File" and select your surveillance video.
3. **Start Analysis:** Click **Start Video Analysis**. The live feed will display the scanning in progress. When the target face matches, the scan will pause for verification. Click **Continue Scan** to resume searching.
