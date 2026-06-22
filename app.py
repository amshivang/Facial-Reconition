import streamlit as st
import cv2
import numpy as np
import os
import json
import glob
import time
from analyzer import VideoFaceAnalyzer

# Page configuration
st.set_page_config(
    page_title="Local Face Recognition Video Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Law Enforcement theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    h1, h2, h3, h4 {
        color: #4da6ff !important;
        font-family: 'Outfit', sans-serif;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 0 10px rgba(0, 102, 204, 0.5);
    }
    .report-card {
        background-color: #1a1c23;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d303d;
        margin-bottom: 20px;
    }
    .match-banner {
        background-color: #3b181a;
        border: 1px solid #ff4d4d;
        padding: 15px;
        border-radius: 8px;
        color: #ffcccc;
        font-weight: bold;
        margin-bottom: 15px;
    }
    /* Hide Deploy button and default Header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    /* Hide default Streamlit footer */
    footer {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 Offline Facial Recognition Video Analyzer")
st.markdown("##### Designed for secure, on-device law enforcement and surveillance video scanning.")

# Initialize the analyzer
@st.cache_resource
def load_analyzer():
    os.makedirs("models", exist_ok=True)
    return VideoFaceAnalyzer()

try:
    analyzer = load_analyzer()
    st.sidebar.success("✅ Face Recognition Engine Loaded (CPU & CUDA Ready)")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.error("Model files are missing. Please verify that `models/face_detection_yunet.onnx` and `models/face_recognition_sface.onnx` exist in the project folder.")
    st.stop()

# Initialize session state variables
if "scan_in_progress" not in st.session_state:
    st.session_state.scan_in_progress = False
if "paused_on_match" not in st.session_state:
    st.session_state.paused_on_match = False
if "current_frame_idx" not in st.session_state:
    st.session_state.current_frame_idx = 0
if "matches" not in st.session_state:
    st.session_state.matches = []
if "latest_match" not in st.session_state:
    st.session_state.latest_match = None
if "target_embedding" not in st.session_state:
    st.session_state.target_embedding = None
if "prev_gray_frame" not in st.session_state:
    st.session_state.prev_gray_frame = None

# Fixed 80% accuracy settings (applied silently in background)
similarity_threshold = 0.31
sample_rate_fps = 2.0
enable_motion_filter = True
motion_threshold = 15

# Main Interface Grid
col_inputs, col_target = st.columns([2, 1])

target_img_path = None
video_path = None

with col_inputs:
    st.markdown("### 1. Configure Inputs")
    
    # Target Image Choice
    target_option = st.radio("Target Image Source", ["Upload a Photo", "Enter Local Photo Path"], horizontal=True)
    
    if target_option == "Upload a Photo":
        target_img_uploaded = st.file_uploader("Upload Target Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
        if target_img_uploaded:
            os.makedirs("temp", exist_ok=True)
            target_img_path = os.path.join("temp", "uploaded_target.jpg")
            with open(target_img_path, "wb") as f:
                f.write(target_img_uploaded.getbuffer())
    else:
        target_path_input = st.text_input("Absolute Path to Target Image File", placeholder="C:\\Users\\...\\photo.jpg")
        if target_path_input and os.path.exists(target_path_input):
            target_img_path = target_path_input
        elif target_path_input:
            st.error("Target image path does not exist.")

    # Video Source Choice
    video_option = st.radio("Video Source", ["Enter Local Video Path (Recommended for large files)", "Upload Video File"], horizontal=True)
    
    if video_option == "Upload Video File":
        video_uploaded = st.file_uploader("Upload Video (MP4, AVI, MKV)", type=["mp4", "avi", "mkv"])
        if video_uploaded:
            os.makedirs("temp", exist_ok=True)
            video_path = os.path.join("temp", "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(video_uploaded.getbuffer())
    else:
        video_path_input = st.text_input("Absolute Path to Video File on Local SSD", placeholder="D:\\Surveillance\\Camera_1_Hour_Recording.mp4")
        if video_path_input and os.path.exists(video_path_input):
            video_path = video_path_input
        elif video_path_input:
            st.error("Video file path does not exist.")

with col_target:
    st.markdown("### 2. Target Profile")
    if target_img_path:
        img = cv2.imread(target_img_path)
        if img is not None:
            # Try to detect face and display crop
            detector = cv2.FaceDetectorYN.create(
                model="models/face_detection_yunet.onnx",
                config="",
                input_size=(img.shape[1], img.shape[0])
            )
            retval, faces = detector.detect(img)
            
            if faces is not None and len(faces) > 0:
                bbox = faces[0][0:4].astype(int)
                x, y, w, h = bbox
                y_min, y_max = max(0, y), min(img.shape[0], y+h)
                x_min, x_max = max(0, x), min(img.shape[1], x+w)
                crop = img[y_min:y_max, x_min:x_max]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
                st.image(crop_rgb, caption="Detected Target Face", width='stretch')
                st.success("Target face detected successfully!")
            else:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Target Image", width='stretch')
                st.warning("No face detected in the photo. The scan may fail or be inaccurate.")
    else:
        st.info("Upload a target photo or specify a path to see the preview.")

# Controls & Live Status area
st.markdown("---")

# Handle Interactive Control Buttons
if st.session_state.paused_on_match:
    st.markdown(f"""
    <div class="match-banner">
        🚨 SCAN PAUSED: Target Face Matched at {st.session_state.latest_match['timestamp']} 
        (Confidence Similarity: {st.session_state.latest_match['similarity']:.2f})
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("➡️ Continue Scan", use_container_width=True, on_click=lambda: setattr(st.session_state, "paused_on_match", False))
    with col2:
        def terminate_scan():
            st.session_state.scan_in_progress = False
            st.session_state.paused_on_match = False
            st.session_state.current_frame_idx = 0
        st.button("⏹️ Terminate Scan", use_container_width=True, on_click=terminate_scan)
        
    # Match Preview Display
    col_crop, col_context = st.columns([1, 3])
    with col_crop:
        st.subheader("Face Match")
        crop_img = cv2.imread(st.session_state.latest_match['crop_path'])
        if crop_img is not None:
            st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), width='stretch')
    with col_context:
        st.subheader("Full Security Camera Frame")
        context_img = cv2.imread(st.session_state.latest_match['context_path'])
        if context_img is not None:
            st.image(cv2.cvtColor(context_img, cv2.COLOR_BGR2RGB), width='stretch')

elif not st.session_state.scan_in_progress:
    # Reset/Start Scan button
    if st.button("🚀 Start Video Analysis", use_container_width=True):
        if not target_img_path or not video_path:
            st.error("Please configure both a target image and a video file path before starting.")
        else:
            # 1. Load target image and get its embedding
            target_img = cv2.imread(target_img_path)
            if target_img is None:
                st.error("Could not load target image.")
            else:
                target_embedding, target_face = analyzer.get_face_embedding(target_img)
                if target_embedding is None:
                    st.error("No face detected in the target image. Please use a clearer photo.")
                else:
                    # Save target face crop reference
                    target_bbox = target_face[0:4].astype(int)
                    tx, ty, tw, th = target_bbox
                    target_crop = target_img[max(0, ty):min(target_img.shape[0], ty+th), max(0, tx):min(target_img.shape[1], tx+tw)]
                    os.makedirs("output", exist_ok=True)
                    cv2.imwrite("output/target_face_crop.jpg", target_crop)

                    # Initialize scanning session state
                    st.session_state.scan_in_progress = True
                    st.session_state.paused_on_match = False
                    st.session_state.current_frame_idx = 0
                    st.session_state.matches = []
                    st.session_state.latest_match = None
                    st.session_state.target_embedding = target_embedding
                    st.session_state.prev_gray_frame = None
                    st.rerun()

# Run the scan loop if in progress and not paused
if st.session_state.scan_in_progress and not st.session_state.paused_on_match:
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    live_feed_placeholder = st.empty()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Could not open video file: {video_path}")
        st.session_state.scan_in_progress = False
        st.stop()
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(video_fps / sample_rate_fps)) if video_fps > 0 else 1
    
    # Seek to last saved index
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame_idx)
    frame_idx = st.session_state.current_frame_idx
    
    target_embedding = st.session_state.target_embedding
    prev_gray_frame = st.session_state.prev_gray_frame
    
    found_match_this_step = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # End of video reached
            st.session_state.scan_in_progress = False
            st.session_state.current_frame_idx = 0
            cap.release()
            st.success("🎉 Video scan complete! Checked the entire recording.")
            st.rerun()
            break
            
        if frame_idx % frame_step == 0:
            current_time_sec = frame_idx / video_fps
            timestamp_str = analyzer.format_timestamp(current_time_sec)
            
            # Update progress UI
            progress_val = float(frame_idx / total_frames) if total_frames > 0 else 0.0
            progress_bar.progress(progress_val)
            status_text.markdown(f"**Scanning Video... {progress_val*100:.1f}%** | Timestamp: `{timestamp_str}` | Matches Found So Far: `{len(st.session_state.matches)}`")
            
            # Live frame preview in UI (downscaled for speed)
            frame_preview = cv2.resize(frame, (320, 240))
            live_feed_placeholder.image(cv2.cvtColor(frame_preview, cv2.COLOR_BGR2RGB), caption="Live Scanner Feed", width=320)
            
            # Motion Filter
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            
            if enable_motion_filter and prev_gray_frame is not None:
                frame_delta = cv2.absdiff(prev_gray_frame, gray_frame)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                motion_score = (np.sum(thresh == 255) / thresh.size) * 100
                if motion_score < (motion_threshold / 100):
                    frame_idx += 1
                    continue
                    
            prev_gray_frame = gray_frame
            st.session_state.prev_gray_frame = prev_gray_frame
            
            # Face Detection & Recognition
            h, w, _ = frame.shape
            analyzer.detector.setInputSize((w, h))
            retval, faces = analyzer.detector.detect(frame)
            
            if faces is not None:
                for i, face in enumerate(faces):
                    try:
                        aligned_face = analyzer.recognizer.alignCrop(frame, face)
                        face_embedding = analyzer.recognizer.feature(aligned_face)
                    except Exception as e:
                        continue
                        
                    similarity = analyzer.compute_similarity(target_embedding, face_embedding)
                    
                    if similarity >= similarity_threshold:
                        # Found a match! Save crops
                        os.makedirs("output", exist_ok=True)
                        bbox = face[0:4].astype(int)
                        x, y, bw, bh = bbox
                        y_min, y_max = max(0, y), min(h, y+bh)
                        x_min, x_max = max(0, x), min(w, x+bw)
                        face_crop = frame[y_min:y_max, x_min:x_max]
                        
                        match_filename = f"match_{timestamp_str.replace(':', '-').replace('.', '_')}_{i}.jpg"
                        match_filepath = os.path.join("output", match_filename)
                        cv2.imwrite(match_filepath, face_crop)
                        
                        context_frame = frame.copy()
                        cv2.rectangle(context_frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                        cv2.putText(context_frame, f"Match: {similarity:.2f}", (x, y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        context_filename = f"context_{timestamp_str.replace(':', '-').replace('.', '_')}_{i}.jpg"
                        context_filepath = os.path.join("output", context_filename)
                        cv2.imwrite(context_filepath, context_frame)
                        
                        # Store in state
                        match_data = {
                            "timestamp_seconds": current_time_sec,
                            "timestamp": timestamp_str,
                            "similarity": float(similarity),
                            "crop_path": match_filepath,
                            "context_path": context_filepath
                        }
                        
                        st.session_state.matches.append(match_data)
                        st.session_state.latest_match = match_data
                        
                        # Advance current frame index for continuation
                        st.session_state.current_frame_idx = frame_idx + 1
                        st.session_state.paused_on_match = True
                        found_match_this_step = True
                        break
                        
            if found_match_this_step:
                break
                
        frame_idx += 1
        
    cap.release()
    if found_match_this_step:
        st.rerun()

# Display historical matches list
if len(st.session_state.matches) > 0:
    st.markdown("---")
    st.markdown("### 📊 Historic Matches Saved")
    
    # Render matches in a grid
    match_cols = st.columns(min(4, len(st.session_state.matches)))
    for idx, match in enumerate(st.session_state.matches):
        col_idx = idx % len(match_cols)
        with match_cols[col_idx]:
            st.markdown(f"**Match at {match['timestamp']}**")
            crop_img = cv2.imread(match['crop_path'])
            if crop_img is not None:
                st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), caption=f"Score: {match['similarity']:.2f}", width='stretch')
