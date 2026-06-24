import os
import sys
import cv2
import numpy as np
import json
import time
import threading
import queue
import customtkinter as ctk
from tkinter import filedialog, messagebox
import PIL.Image
from analyzer import VideoFaceAnalyzer

# Configure CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VideoAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Window settings
        self.title("Offline Facial Recognition Video Analyzer")
        self.geometry("1100x700")
        self.minsize(950, 650)
        self.configure(fg_color="#0e1117")
        
        # State variables
        self.target_image_path = None
        self.video_path = None
        self.analyzer = None
        self.scan_thread = None
        self.scan_queue = queue.Queue()
        self.scan_active = False
        self.scan_paused = False
        self.resume_event = threading.Event()
        self.stop_event = threading.Event()
        self.matches = []
        self.current_frame_idx = 0
        self.prev_gray_frame = None
        self.target_embedding = None
        
        # Resolve Model Paths (PyInstaller bundle vs normal run)
        self.resolve_model_paths()
        
        # Scanning parameters (matching web app's silent defaults)
        self.similarity_threshold = 0.31
        self.sample_rate_fps = 2.0
        self.enable_motion_filter = True
        self.motion_threshold = 15
        
        # UI components layout
        self.setup_ui()
        
        # Handle close window event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start queue polling
        self.after(50, self.poll_queue)
        
        # Async load the AI Engine
        self.load_engine_async()

    def resolve_model_paths(self):
        """Resolves model paths depending on whether the app is run from source or packaged .exe"""
        if hasattr(sys, '_MEIPASS'):
            # Packaged in single-file exe
            self.yunet_path = os.path.join(sys._MEIPASS, "models", "face_detection_yunet.onnx")
            self.sface_path = os.path.join(sys._MEIPASS, "models", "face_recognition_sface.onnx")
            # Fallback if bundle didn't extract correctly
            if not os.path.exists(self.yunet_path) or not os.path.exists(self.sface_path):
                self.yunet_path = "models/face_detection_yunet.onnx"
                self.sface_path = "models/face_recognition_sface.onnx"
        else:
            # Normal python execution
            self.yunet_path = "models/face_detection_yunet.onnx"
            self.sface_path = "models/face_recognition_sface.onnx"

    def setup_ui(self):
        # Configure grid structure (2 columns, left for controls/previews, right for alerts/history)
        self.grid_columnconfigure(0, weight=4, minsize=480)
        self.grid_columnconfigure(1, weight=5, minsize=450)
        self.grid_rowconfigure(0, weight=1)
        
        # Left Panel (Controls and Live Feed)
        self.left_panel = ctk.CTkFrame(self, fg_color="#0e1117")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(2, weight=1) # Live feed grows
        
        # Header Title
        self.header_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        self.title_lbl = ctk.CTkLabel(
            self.header_frame, 
            text="🔍 Offline Face Recognition Video Analyzer", 
            font=("Segoe UI", 18, "bold"), 
            text_color="#4da6ff",
            anchor="w"
        )
        self.title_lbl.pack(fill="x", anchor="w")
        
        self.subtitle_lbl = ctk.CTkLabel(
            self.header_frame, 
            text="Designed for secure, on-device surveillance video scanning.", 
            font=("Segoe UI", 11), 
            text_color="#8b949e",
            anchor="w"
        )
        self.subtitle_lbl.pack(fill="x", anchor="w")
        
        # Input Configuration Card
        self.input_card = ctk.CTkFrame(self.left_panel, fg_color="#1a1c23", border_color="#2d303d", border_width=1)
        self.input_card.grid(row=1, column=0, sticky="ew", pady=10, padx=2)
        self.input_card.grid_columnconfigure(0, weight=1)
        
        self.input_lbl = ctk.CTkLabel(self.input_card, text="1. Configure Inputs", font=("Segoe UI", 13, "bold"), text_color="#4da6ff")
        self.input_lbl.grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))
        
        # Target Image Selection Row
        self.target_frame = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.target_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)
        self.target_frame.grid_columnconfigure(0, weight=1)
        
        self.target_btn = ctk.CTkButton(
            self.target_frame, 
            text="Choose Target Photo", 
            command=self.browse_target_image,
            fg_color="#30363d", 
            hover_color="#484f58",
            height=32
        )
        self.target_btn.grid(row=0, column=0, sticky="w")
        self.target_file_lbl = ctk.CTkLabel(self.target_frame, text="No target photo selected", font=("Segoe UI", 11), text_color="#8b949e", anchor="w")
        self.target_file_lbl.grid(row=0, column=1, sticky="ew", padx=15)
        
        # Target Face Preview Container
        self.target_preview_lbl = ctk.CTkLabel(self.input_card, text="", text_color="#8b949e")
        self.target_preview_lbl.grid(row=2, column=0, sticky="w", padx=15, pady=(2, 5))
        
        # Video File Selection Row
        self.video_frame = ctk.CTkFrame(self.input_card, fg_color="transparent")
        self.video_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(5, 12))
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_btn = ctk.CTkButton(
            self.video_frame, 
            text="Choose Video File", 
            command=self.browse_video,
            fg_color="#30363d", 
            hover_color="#484f58",
            height=32
        )
        self.video_btn.grid(row=0, column=0, sticky="w")
        self.video_file_lbl = ctk.CTkLabel(self.video_frame, text="No video file selected", font=("Segoe UI", 11), text_color="#8b949e", anchor="w")
        self.video_file_lbl.grid(row=0, column=1, sticky="ew", padx=15)
        
        # Scanner Feed and Progress Card
        self.feed_card = ctk.CTkFrame(self.left_panel, fg_color="#1a1c23", border_color="#2d303d", border_width=1)
        self.feed_card.grid(row=2, column=0, sticky="nsew", pady=10, padx=2)
        self.feed_card.grid_columnconfigure(0, weight=1)
        self.feed_card.grid_rowconfigure(1, weight=1) # Live feed expands
        
        self.feed_title_lbl = ctk.CTkLabel(self.feed_card, text="2. Live Scanner Feed & Status", font=("Segoe UI", 13, "bold"), text_color="#4da6ff")
        self.feed_title_lbl.grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))
        
        # Scanner Canvas Frame
        self.feed_canvas_frame = ctk.CTkFrame(self.feed_card, fg_color="#0e1117", border_color="#2d303d", border_width=1)
        self.feed_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=5)
        self.feed_canvas_frame.grid_columnconfigure(0, weight=1)
        self.feed_canvas_frame.grid_rowconfigure(0, weight=1)
        
        # Label to host Live Feed frame
        self.live_feed_lbl = ctk.CTkLabel(self.feed_canvas_frame, text="Feed Offline", font=("Segoe UI", 14), text_color="#484f58")
        self.live_feed_lbl.grid(row=0, column=0, sticky="nsew")
        
        # Progress Bar & Status Text
        self.status_bar_lbl = ctk.CTkLabel(self.feed_card, text="System Idle", font=("Segoe UI", 11, "bold"), text_color="#8b949e")
        self.status_bar_lbl.grid(row=2, column=0, sticky="ew", padx=15, pady=(5, 2))
        
        self.progress_bar = ctk.CTkProgressBar(self.feed_card, fg_color="#30363d", progress_color="#0066cc")
        self.progress_bar.set(0.0)
        self.progress_bar.grid(row=3, column=0, sticky="ew", padx=15, pady=(2, 12))
        
        # Main Start / Stop Scan Button
        self.action_btn = ctk.CTkButton(
            self.left_panel, 
            text="🚀 Start Video Analysis", 
            command=self.toggle_scanning,
            font=("Segoe UI", 14, "bold"),
            fg_color="#0066cc",
            hover_color="#0052a3",
            height=40,
            state="disabled" # Disabled until AI loads
        )
        self.action_btn.grid(row=3, column=0, sticky="ew", pady=(10, 0), padx=2)
        
        # Right Panel (Alerts and Match History)
        self.right_panel = ctk.CTkFrame(self, fg_color="#0e1117")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(1, weight=1) # History frame grows
        
        # Match Alert Card (Hidden by default, shown when paused on match)
        self.match_alert_card = ctk.CTkFrame(self.right_panel, fg_color="#1a1c23", border_color="#ff4d4d", border_width=1)
        # Initially not packed/gridded
        self.match_alert_card.grid_columnconfigure(0, weight=1)
        
        # Banner Header
        self.banner_frame = ctk.CTkFrame(self.match_alert_card, fg_color="#3b181a", height=35)
        self.banner_frame.grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        self.banner_frame.pack_propagate(False)
        self.banner_lbl = ctk.CTkLabel(
            self.banner_frame, 
            text="🚨 SCAN PAUSED: Target Face Matched!", 
            font=("Segoe UI", 12, "bold"), 
            text_color="#ffcccc"
        )
        self.banner_lbl.pack(fill="both", expand=True, padx=10)
        
        # Match Previews Container
        self.previews_frame = ctk.CTkFrame(self.match_alert_card, fg_color="transparent")
        self.previews_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=10)
        self.previews_frame.grid_columnconfigure(0, weight=1)
        self.previews_frame.grid_columnconfigure(1, weight=3)
        
        # Crop Preview
        self.crop_container = ctk.CTkFrame(self.previews_frame, fg_color="#0e1117", border_color="#2d303d", border_width=1)
        self.crop_container.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.crop_container.grid_columnconfigure(0, weight=1)
        self.crop_container.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(self.crop_container, text="Face Match", font=("Segoe UI", 10, "bold"), text_color="#4da6ff").grid(row=0, column=0, pady=2)
        self.match_crop_lbl = ctk.CTkLabel(self.crop_container, text="No Image", text_color="#484f58")
        self.match_crop_lbl.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        
        # Context Preview
        self.context_container = ctk.CTkFrame(self.previews_frame, fg_color="#0e1117", border_color="#2d303d", border_width=1)
        self.context_container.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.context_container.grid_columnconfigure(0, weight=1)
        self.context_container.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(self.context_container, text="Full Camera Frame", font=("Segoe UI", 10, "bold"), text_color="#4da6ff").grid(row=0, column=0, pady=2)
        self.match_context_lbl = ctk.CTkLabel(self.context_container, text="No Image", text_color="#484f58")
        self.match_context_lbl.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        
        # Alert Interactive Controls
        self.alert_controls = ctk.CTkFrame(self.match_alert_card, fg_color="transparent")
        self.alert_controls.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.alert_controls.grid_columnconfigure(0, weight=1)
        self.alert_controls.grid_columnconfigure(1, weight=1)
        
        self.continue_btn = ctk.CTkButton(
            self.alert_controls, 
            text="➡️ Continue Scan", 
            command=self.continue_scan,
            font=("Segoe UI", 12, "bold"),
            fg_color="#0066cc",
            hover_color="#0052a3",
            height=32
        )
        self.continue_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        self.terminate_btn = ctk.CTkButton(
            self.alert_controls, 
            text="⏹️ Terminate Scan", 
            command=self.terminate_scan,
            font=("Segoe UI", 12, "bold"),
            fg_color="#8f2022",
            hover_color="#70181a",
            height=32
        )
        self.terminate_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        # History Card
        self.history_card = ctk.CTkFrame(self.right_panel, fg_color="#1a1c23", border_color="#2d303d", border_width=1)
        self.history_card.grid(row=1, column=0, sticky="nsew", pady=(10, 0), padx=2)
        self.history_card.grid_columnconfigure(0, weight=1)
        self.history_card.grid_rowconfigure(1, weight=1) # Scrollable area expands
        
        self.history_lbl = ctk.CTkLabel(self.history_card, text="📊 Historic Matches Saved", font=("Segoe UI", 13, "bold"), text_color="#4da6ff")
        self.history_lbl.grid(row=0, column=0, sticky="w", padx=15, pady=(10, 5))
        
        # Scrollable Matches Frame
        self.history_scroll = ctk.CTkScrollableFrame(self.history_card, fg_color="transparent")
        self.history_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        self.history_scroll.grid_columnconfigure(0, weight=1)
        
        self.no_matches_lbl = ctk.CTkLabel(self.history_scroll, text="No matches recorded yet.", font=("Segoe UI", 12), text_color="#8b949e")
        self.no_matches_lbl.grid(row=0, column=0, pady=40, sticky="ew")
        
        # Engine status bar
        self.engine_status_lbl = ctk.CTkLabel(self.left_panel, text="🤖 Loading Face Recognition Engine...", font=("Segoe UI", 10), text_color="#8b949e", anchor="w")
        self.engine_status_lbl.grid(row=4, column=0, sticky="ew", pady=(5, 0))

    # Async Engine Loading
    def load_engine_async(self):
        def worker():
            try:
                # Load analyzer using resolved model paths
                analyzer = VideoFaceAnalyzer(yunet_model_path=self.yunet_path, sface_model_path=self.sface_path)
                self.scan_queue.put(('engine_ready', analyzer))
            except Exception as e:
                self.scan_queue.put(('engine_failed', str(e)))
        
        threading.Thread(target=worker, daemon=True).start()

    # File browsing functions
    def browse_target_image(self):
        path = filedialog.askopenfilename(
            title="Select Target Face Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if path:
            self.target_image_path = path
            self.target_file_lbl.configure(text=os.path.basename(path), text_color="#ffffff")
            
            # Detect target face and display crop
            self.load_target_face_crop()

    def load_target_face_crop(self):
        if not self.analyzer:
            self.target_preview_lbl.configure(text="AI Engine loading, preview deferred...")
            return
            
        img = cv2.imread(self.target_image_path)
        if img is not None:
            try:
                # Detect face using resolved YuNet model path
                detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_path,
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
                    
                    # Convert to PIL Image for display
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = PIL.Image.fromarray(crop_rgb)
                    
                    # Size of preview
                    preview_w, preview_h = self.get_scaled_size(pil_img, max_w=100, max_h=100)
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(preview_w, preview_h))
                    
                    self.target_preview_lbl.configure(image=ctk_img, text="")
                    self.target_preview_lbl.image = ctk_img # Keep reference
                else:
                    # Fallback to whole image
                    pil_img = PIL.Image.open(self.target_image_path)
                    preview_w, preview_h = self.get_scaled_size(pil_img, max_w=100, max_h=100)
                    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(preview_w, preview_h))
                    self.target_preview_lbl.configure(image=ctk_img, text="⚠️ No face detected in photo")
                    self.target_preview_lbl.image = ctk_img
            except Exception as e:
                self.target_preview_lbl.configure(image=None, text="⚠️ Face detection error")
                print(f"Error drawing target crop: {e}")

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Security Camera Recording",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if path:
            self.video_path = path
            self.video_file_lbl.configure(text=os.path.basename(path), text_color="#ffffff")

    def toggle_scanning(self):
        if self.scan_active:
            self.stop_scan()
        else:
            self.start_scan()

    def start_scan(self):
        if not self.target_image_path or not self.video_path:
            messagebox.showerror("Configuration Error", "Please configure both a target image and a video file path before starting.")
            return
            
        if not self.analyzer:
            messagebox.showerror("System Error", "AI Engine is still loading. Please wait.")
            return

        # 1. Load target embedding
        target_img = cv2.imread(self.target_image_path)
        if target_img is None:
            messagebox.showerror("Error", "Could not load target image.")
            return
            
        target_embedding, target_face = self.analyzer.get_face_embedding(target_img)
        if target_embedding is None:
            messagebox.showerror("Error", "No face detected in the target image. Please use a clearer photo.")
            return
            
        # Crop target face reference
        bbox = target_face[0:4].astype(int)
        tx, ty, tw, th = bbox
        target_crop = target_img[max(0, ty):min(target_img.shape[0], ty+th), max(0, tx):min(target_img.shape[1], tx+tw)]
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/target_face_crop.jpg", target_crop)
        
        self.target_embedding = target_embedding
        self.prev_gray_frame = None
        self.current_frame_idx = 0
        self.matches = []
        
        # Reset History list
        for child in self.history_scroll.winfo_children():
            child.destroy()
        self.no_matches_lbl = ctk.CTkLabel(self.history_scroll, text="No matches recorded yet.", font=("Segoe UI", 12), text_color="#8b949e")
        self.no_matches_lbl.grid(row=0, column=0, pady=40, sticky="ew")
        
        # Reset variables and UI
        self.scan_active = True
        self.scan_paused = False
        self.resume_event.clear()
        self.stop_event.clear()
        
        self.action_btn.configure(text="⏹️ Terminate Scan", fg_color="#8f2022", hover_color="#70181a")
        self.target_btn.configure(state="disabled")
        self.video_btn.configure(state="disabled")
        self.progress_bar.set(0.0)
        self.status_bar_lbl.configure(text="Scan Initiating...", text_color="#4da6ff")
        
        # Hide match alert banner
        self.match_alert_card.grid_forget()
        
        # Launch Scanner Thread
        self.scan_thread = threading.Thread(target=self.scan_loop_worker, daemon=True)
        self.scan_thread.start()

    def stop_scan(self):
        if not self.scan_active:
            return
        
        # Signal stop
        self.stop_event.set()
        self.resume_event.set() # Unblock if paused on match
        self.status_bar_lbl.configure(text="Terminating Scan...", text_color="#ff4d4d")

    def continue_scan(self):
        if self.scan_active and self.scan_paused:
            self.scan_paused = False
            self.match_alert_card.grid_forget()
            self.status_bar_lbl.configure(text="Resuming Scan...", text_color="#4da6ff")
            self.resume_event.set()
            self.resume_event.clear()

    def terminate_scan(self):
        self.stop_scan()
        self.match_alert_card.grid_forget()

    # Scanning loop thread function
    def scan_loop_worker(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.scan_queue.put(('error', f"Could not open video file: {self.video_path}"))
            return
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(video_fps / self.sample_rate_fps)) if video_fps > 0 else 1
        
        frame_idx = 0
        prev_gray_frame = None
        
        while cap.isOpened() and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_step == 0:
                current_time_sec = frame_idx / video_fps
                timestamp_str = self.analyzer.format_timestamp(current_time_sec)
                
                # Check stop
                if self.stop_event.is_set():
                    break
                    
                # Downscaled feed preview for queue
                frame_preview = cv2.resize(frame, (320, 240))
                self.scan_queue.put(('live_frame', frame_preview))
                
                # Send progress update
                progress_val = float(frame_idx / total_frames) if total_frames > 0 else 0.0
                self.scan_queue.put(('progress', (progress_val, timestamp_str)))
                
                # Motion Filter
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
                
                if self.enable_motion_filter and prev_gray_frame is not None:
                    frame_delta = cv2.absdiff(prev_gray_frame, gray_frame)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    motion_score = (np.sum(thresh == 255) / thresh.size) * 100
                    if motion_score < (self.motion_threshold / 100):
                        frame_idx += 1
                        continue
                        
                prev_gray_frame = gray_frame
                
                # Face Detection & Recognition
                h, w, _ = frame.shape
                self.analyzer.detector.setInputSize((w, h))
                retval, faces = self.analyzer.detector.detect(frame)
                
                if faces is not None:
                    found_match_in_frame = False
                    for i, face in enumerate(faces):
                        try:
                            aligned_face = self.analyzer.recognizer.alignCrop(frame, face)
                            face_embedding = self.analyzer.recognizer.feature(aligned_face)
                        except Exception:
                            continue
                            
                        similarity = self.analyzer.compute_similarity(self.target_embedding, face_embedding)
                        
                        if similarity >= self.similarity_threshold:
                            # Match Found!
                            bbox = face[0:4].astype(int)
                            x, y, bw, bh = bbox
                            y_min, y_max = max(0, y), min(h, y+bh)
                            x_min, x_max = max(0, x), min(w, x+bw)
                            face_crop = frame[y_min:y_max, x_min:x_max]
                            
                            # Save files
                            timestamp_clean = timestamp_str.replace(':', '-').replace('.', '_')
                            match_filename = f"match_{timestamp_clean}_{i}.jpg"
                            match_filepath = os.path.join("output", match_filename)
                            cv2.imwrite(match_filepath, face_crop)
                            
                            context_frame = frame.copy()
                            cv2.rectangle(context_frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                            cv2.putText(context_frame, f"Match: {similarity:.2f}", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            context_filename = f"context_{timestamp_clean}_{i}.jpg"
                            context_filepath = os.path.join("output", context_filename)
                            cv2.imwrite(context_filepath, context_frame)
                            
                            match_data = {
                                "timestamp_seconds": current_time_sec,
                                "timestamp": timestamp_str,
                                "similarity": float(similarity),
                                "crop_path": match_filepath,
                                "context_path": context_filepath
                            }
                            
                            # Update GUI with match details and trigger pause
                            self.scan_queue.put(('match_found', match_data))
                            found_match_in_frame = True
                            
                            # Wait for GUI resume signal
                            self.resume_event.wait()
                            break
                    
                    if found_match_in_frame:
                        # Skip processing further faces in this frame
                        pass
                        
            frame_idx += 1
            
        cap.release()
        
        # Complete/Stop signal
        if self.stop_event.is_set():
            self.scan_queue.put(('scan_stopped', None))
        else:
            self.scan_queue.put(('scan_complete', None))

    # Thread-safe queue poller
    def poll_queue(self):
        try:
            while True:
                msg_type, data = self.scan_queue.get_nowait()
                
                if msg_type == 'engine_ready':
                    self.analyzer = data
                    self.engine_status_lbl.configure(
                        text="✅ Face Recognition Engine Loaded (CPU & CUDA Ready)", 
                        text_color="#4da6ff"
                    )
                    self.action_btn.configure(state="normal")
                    # If target image was already chosen, update its crop preview
                    if self.target_image_path:
                        self.load_target_face_crop()
                        
                elif msg_type == 'engine_failed':
                    self.engine_status_lbl.configure(
                        text=f"❌ Model load error: {data}", 
                        text_color="#ff4d4d"
                    )
                    messagebox.showerror("Model Load Failure", f"Error initializing AI engine models: {data}\nPlease verify models exist in models/ directory.")
                    
                elif msg_type == 'live_frame':
                    self.update_live_feed(data)
                    
                elif msg_type == 'progress':
                    progress_val, timestamp_str = data
                    self.progress_bar.set(progress_val)
                    self.status_bar_lbl.configure(
                        text=f"Scanning Video... {progress_val*100:.1f}% | Timestamp: {timestamp_str} | Matches: {len(self.matches)}",
                        text_color="#4da6ff"
                    )
                    
                elif msg_type == 'match_found':
                    self.handle_match_found(data)
                    
                elif msg_type == 'scan_complete':
                    self.handle_scan_finished(success=True)
                    
                elif msg_type == 'scan_stopped':
                    self.handle_scan_finished(success=False)
                    
                elif msg_type == 'error':
                    messagebox.showerror("Error during scan", data)
                    self.handle_scan_finished(success=False)
                    
        except queue.Empty:
            pass
            
        # Continue polling
        self.after(50, self.poll_queue)

    # UI updates for scanning events
    def update_live_feed(self, frame_bgr):
        # Convert BGR (OpenCV) to RGB (Pillow)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)
        
        # Create CTkImage
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(320, 240))
        self.live_feed_lbl.configure(image=ctk_img, text="")
        self.live_feed_lbl.image = ctk_img

    def handle_match_found(self, match_data):
        self.scan_paused = True
        self.matches.append(match_data)
        
        # 1. Update Alert Banner and show Alert Card
        self.banner_lbl.configure(
            text=f"🚨 SCAN PAUSED: Target Face Matched at {match_data['timestamp']} (Confidence: {match_data['similarity']:.2f})"
        )
        self.match_alert_card.grid(row=0, column=0, sticky="ew", padx=2, pady=(0, 10))
        
        # 2. Update Alert Previews
        # Load Crop Image
        crop_pil = PIL.Image.open(match_data['crop_path'])
        crop_w, crop_h = self.get_scaled_size(crop_pil, max_w=100, max_h=100)
        crop_ctk = ctk.CTkImage(light_image=crop_pil, dark_image=crop_pil, size=(crop_w, crop_h))
        self.match_crop_lbl.configure(image=crop_ctk, text="")
        self.match_crop_lbl.image = crop_ctk
        
        # Load Context Image
        ctx_pil = PIL.Image.open(match_data['context_path'])
        ctx_w, ctx_h = self.get_scaled_size(ctx_pil, max_w=280, max_h=170)
        ctx_ctk = ctk.CTkImage(light_image=ctx_pil, dark_image=ctx_pil, size=(ctx_w, ctx_h))
        self.match_context_lbl.configure(image=ctx_ctk, text="")
        self.match_context_lbl.image = ctx_ctk
        
        # 3. Add to scrollable history cards
        self.add_history_card(match_data)
        
        # 4. Sound notification or visual flash
        self.bell() # Standard system beep

    def add_history_card(self, match_data):
        # Remove "no matches" placeholder if this is first match
        if len(self.matches) == 1:
            self.no_matches_lbl.grid_forget()
            
        # Card container
        card = ctk.CTkFrame(self.history_scroll, fg_color="#1a1c23", border_color="#30363d", border_width=1)
        card.pack(fill="x", pady=4, padx=5)
        
        # Grid structure for card
        card.grid_columnconfigure(1, weight=1)
        
        # Crop preview thumbnail (50x50)
        crop_pil = PIL.Image.open(match_data['crop_path'])
        crop_w, crop_h = self.get_scaled_size(crop_pil, max_w=50, max_h=50)
        crop_ctk = ctk.CTkImage(light_image=crop_pil, dark_image=crop_pil, size=(crop_w, crop_h))
        
        thumbnail_lbl = ctk.CTkLabel(card, image=crop_ctk, text="")
        thumbnail_lbl.image = crop_ctk
        thumbnail_lbl.grid(row=0, column=0, padx=10, pady=8, sticky="w")
        
        # Details text
        details_txt = f"Match at {match_data['timestamp']}\nSimilarity: {match_data['similarity']:.2f}"
        details_lbl = ctk.CTkLabel(card, text=details_txt, font=("Segoe UI", 11, "bold"), text_color="#ffffff", justify="left", anchor="w")
        details_lbl.grid(row=0, column=1, sticky="ew", padx=5)
        
        # View full image button
        view_btn = ctk.CTkButton(
            card, 
            text="👁️ View Details", 
            width=85,
            height=26,
            fg_color="#30363d",
            hover_color="#484f58",
            command=lambda: self.show_match_details_popup(match_data)
        )
        view_btn.grid(row=0, column=2, padx=15, pady=8, sticky="e")

    def show_match_details_popup(self, match_data):
        # Create a new top-level window for detail view
        popup = ctk.CTkToplevel(self)
        popup.title(f"Match Details - {match_data['timestamp']}")
        popup.geometry("750x550")
        popup.configure(fg_color="#0e1117")
        popup.transient(self) # Keep on top of main window
        
        # Configure layout
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(1, weight=1)
        
        # Top banner details
        header = ctk.CTkFrame(popup, fg_color="#1a1c23", border_color="#2d303d", border_width=1, height=60)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        header.pack_propagate(False)
        
        header_text = f"📍 Occurrence Timestamp: {match_data['timestamp']}    |    🎯 Confidence Score: {match_data['similarity']:.2f}"
        header_lbl = ctk.CTkLabel(header, text=header_text, font=("Segoe UI", 13, "bold"), text_color="#4da6ff")
        header_lbl.pack(fill="both", expand=True, padx=15)
        
        # Context Image Preview (large size)
        content_frame = ctk.CTkFrame(popup, fg_color="#1a1c23", border_color="#2d303d", border_width=1)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)
        
        img_lbl = ctk.CTkLabel(content_frame, text="Loading context frame...")
        img_lbl.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Load and resize image to fit the window safely
        ctx_pil = PIL.Image.open(match_data['context_path'])
        scaled_w, scaled_h = self.get_scaled_size(ctx_pil, max_w=700, max_h=400)
        ctx_ctk = ctk.CTkImage(light_image=ctx_pil, dark_image=ctx_pil, size=(scaled_w, scaled_h))
        img_lbl.configure(image=ctx_ctk, text="")
        img_lbl.image = ctx_ctk
        
        # Bottom Close Button
        close_btn = ctk.CTkButton(
            popup, 
            text="Close Window", 
            command=popup.destroy,
            width=120,
            height=32,
            fg_color="#30363d",
            hover_color="#484f58"
        )
        close_btn.grid(row=2, column=0, pady=(10, 15))

    def handle_scan_finished(self, success=True):
        self.scan_active = False
        self.scan_paused = False
        
        # UI reset
        self.action_btn.configure(text="🚀 Start Video Analysis", fg_color="#0066cc", hover_color="#0052a3")
        self.target_btn.configure(state="normal")
        self.video_btn.configure(state="normal")
        self.live_feed_lbl.configure(image=None, text="Feed Offline", font=("Segoe UI", 14), text_color="#484f58")
        
        if success:
            self.progress_bar.set(1.0)
            self.status_bar_lbl.configure(text=f"🎉 Video scan complete! Checked entire video. Matches: {len(self.matches)}", text_color="#2ea043")
            messagebox.showinfo("Scan Complete", f"🎉 Video scan complete!\n\nProcessed the entire recording and found {len(self.matches)} matches.\nAll crop files are saved in the 'output' directory.")
        else:
            self.status_bar_lbl.configure(text="Scan Terminated", text_color="#ff4d4d")
            messagebox.showwarning("Scan Terminated", "The video scanning process was stopped by the user.")

    # Utility layout sizing helper
    def get_scaled_size(self, pil_image, max_w, max_h):
        w, h = pil_image.size
        aspect_ratio = w / h
        
        # Scale to max width first
        scaled_w = max_w
        scaled_h = int(scaled_w / aspect_ratio)
        
        # If height exceeds max height, scale by height instead
        if scaled_h > max_h:
            scaled_h = max_h
            scaled_w = int(scaled_h * aspect_ratio)
            
        return max(1, scaled_w), max(1, scaled_h)

    # Window closing cleanly
    def on_closing(self):
        if self.scan_active:
            if messagebox.askokcancel("Quit", "Scanning is currently running. Do you want to terminate scan and quit?"):
                self.stop_scan()
                # Wait briefly for thread to unlock
                if self.scan_thread and self.scan_thread.is_alive():
                    self.scan_thread.join(timeout=1.0)
                self.destroy()
        else:
            self.destroy()

if __name__ == "__main__":
    app = VideoAIApp()
    app.mainloop()
