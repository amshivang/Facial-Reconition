# Offline Facial Recognition Video Analyzer

A high-performance, secure, 100% on-device facial recognition application designed to scan security camera footage and identify timestamps when a target person is present. 

Developed with a privacy-first architecture, this tool does not connect to the cloud or make external API requests, making it compliant with strict chain-of-custody and confidentiality protocols.

---

## 🚀 Windows Desktop App (Direct Download)

For the easiest experience on Windows, you can download the pre-compiled standalone executable. This packages the GUI, Python runtime, OpenCV processing library, and the AI models into a single file that runs fully offline without any configuration.

### How to Download & Run:
1. Go to the **[Releases](https://github.com/amshivang/Facial-Recognition/releases)** section of this repository.
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

## Developer Installation & Setup

If you prefer to run the application from source code or modify it, follow these setup steps:

### 1. Prerequisites
- **Python 3.10 to 3.14** installed.
- (Optional) **NVIDIA GPU** with CUDA Toolkit (v11.8 or v12.x/v13.x) & cuDNN installed for hardware acceleration.

### 2. Run from Source (Step-by-Step Commands)

Choose the block of commands corresponding to your terminal and copy-paste them to set up and run the application:

#### 💻 Windows (Command Prompt - cmd)
```cmd
git clone https://github.com/amshivang/Facial-Recognition.git
cd Facial-Recognition
python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

#### 🐧 Linux / macOS (Terminal)
```bash
git clone https://github.com/amshivang/Facial-Recognition.git
cd Facial-Recognition
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## How to Use the Web Dashboard

The interface will automatically open in your default browser at `http://localhost:8501`.

1. **Choose a Target Photo:** Upload a photo of the target individual, or input their local absolute file path.
2. **Choose a Video File:** Provide the absolute path to your security recording on your SSD (highly recommended for multi-gigabyte surveillance recordings to prevent browser memory overflows).
3. **Configure Settings:** Keep the default settings (Threshold: `0.31`, Sampling: `2.0 FPS`, Motion Filter: `ON`) for the pre-configured 80% accuracy scan.
4. **Click "Start Video Analysis":** The scanner will run, showing a live frame feed. When the target person is found, it will pause and display the matches. Click **Continue Scan** to keep searching.
