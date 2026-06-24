# Offline Facial Recognition Video Analyzer

A high-performance, secure, 100% on-device facial recognition application designed to scan security camera footage and identify timestamps when a target person is present. 

Developed with a privacy-first architecture, this tool does not connect to the cloud or make external API requests, making it compliant with strict chain-of-custody and confidentiality protocols.

---

## Key Features

- **Interactive Scanning (Pause-on-Match):** The scanner automatically pauses as soon as a target face is detected, showing a side-by-side view of the face crop next to the full surveillance camera frame. You can choose to **Continue Scan** or **Terminate**.
- **80% Accuracy Optimization:** The system is pre-configured with a similarity threshold of `0.31` using OpenCV SFace, optimized to flag suspects even in poor lighting, at angles, or with motion blur.
- **Motion Gating Filter:** Skips sections of static, unchanging video footage (e.g., empty hallways) automatically to accelerate analysis.
- **Dynamic Sampling Rate (FPS):** Allows you to choose how many frames to analyze per second of video to trade off speed vs. thoroughness.
- **Hardware Acceleration:** Native GPU acceleration using ONNX Runtime with NVIDIA CUDA/TensorRT, automatically falling back to optimized multi-threaded CPU execution.
- **No Large Binaries in Git:** Pre-trained AI models (`YuNet` and `SFace`) are automatically downloaded from the official OpenCV Model Zoo on the first launch.

---

## Tech Stack

- **UI Dashboard:** Streamlit (customized with CSS to run cleanly as a local desktop app).
- **Core Processing:** Python 3.10+ & OpenCV.
- **Face Detection:** YuNet (via OpenCV FaceDetectorYN).
- **Face Recognition:** SFace (via OpenCV FaceRecognizerSF).
- **GPU Inference Engine:** ONNX Runtime GPU (CUDA/TensorRT).

---

## Installation & Setup

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

## Running the Application

Launch the local web dashboard:
```bash
streamlit run app.py
```

The interface will automatically open in your default browser at `http://localhost:8501`.

### How to use it:
1. **Choose a Target Photo:** Upload a photo of the target individual, or input their local absolute file path.
2. **Choose a Video File:** Provide the absolute path to your security recording on your SSD (highly recommended for multi-gigabyte surveillance recordings to prevent browser memory overflows).
3. **Configure Settings:** Keep the default settings (Threshold: `0.31`, Sampling: `2.0 FPS`, Motion Filter: `ON`) for the pre-configured 80% accuracy scan.
4. **Click "Start Video Analysis":** The scanner will run, showing a live frame feed. When the target person is found, it will pause and display the matches. Click **Continue Scan** to keep searching.
