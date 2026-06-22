import cv2
import numpy as np
import os
import json
import time
import urllib.request

class VideoFaceAnalyzer:
    def __init__(self, yunet_model_path="models/face_detection_yunet.onnx", sface_model_path="models/face_recognition_sface.onnx"):
        self.yunet_path = yunet_model_path
        self.sface_path = sface_model_path
        self.detector = None
        self.recognizer = None
        
        # Auto-download models if they are missing locally
        self._ensure_models_exist()
        self._init_models()

    def _ensure_models_exist(self):
        os.makedirs(os.path.dirname(self.yunet_path), exist_ok=True)
        
        yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        
        # Set headers to look like browser requests
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        
        if not os.path.exists(self.yunet_path):
            print(f"Downloading face detection model (YuNet)...")
            req = urllib.request.Request(yunet_url, headers=headers)
            with urllib.request.urlopen(req) as response, open(self.yunet_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
            
        if not os.path.exists(self.sface_path):
            print(f"Downloading face recognition model (SFace)...")
            req = urllib.request.Request(sface_url, headers=headers)
            with urllib.request.urlopen(req) as response, open(self.sface_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")

    def _init_models(self):
        # We start with a default input size for YuNet, but we will update it dynamically for each image/frame size.
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.8, # Confidence threshold to detect faces
            nms_threshold=0.3,
            top_k=5000
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config=""
        )

    def get_face_embedding(self, img, align=True):
        """
        Detects the largest face in an image and extracts its SFace embedding.
        If align=True, it aligns the crop first.
        """
        h, w, _ = img.shape
        self.detector.setInputSize((w, h))
        retval, faces = self.detector.detect(img)
        
        if faces is None or len(faces) == 0:
            return None, None
            
        # If multiple faces are detected, select the largest one (by area)
        largest_face = None
        max_area = 0
        for face in faces:
            # face format: [x, y, w, h, x_re, y_re, ...]
            bbox = face[0:4].astype(int)
            area = bbox[2] * bbox[3]
            if area > max_area:
                max_area = area
                largest_face = face
                
        if align:
            aligned_face = self.recognizer.alignCrop(img, largest_face)
            embedding = self.recognizer.feature(aligned_face)
            return embedding, largest_face
        else:
            # Fallback to direct crop if alignment is skipped
            bbox = largest_face[0:4].astype(int)
            x, y, w, h = bbox
            x, y = max(0, x), max(0, y)
            crop = img[y:y+h, x:x+w]
            if crop.size == 0:
                return None, None
            crop_resized = cv2.resize(crop, (112, 112))
            embedding = self.recognizer.feature(crop_resized)
            return embedding, largest_face

    def compute_similarity(self, feat1, feat2):
        """
        Computes cosine similarity between two feature embeddings.
        OpenCV SFace match output: Cosine Similarity >= 0.36 is considered a match.
        """
        return self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)

    def format_timestamp(self, seconds):
        """Converts seconds into HH:MM:SS format."""
        milliseconds = int((seconds - int(seconds)) * 1000)
        seconds = int(seconds)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}.{milliseconds:03d}"

    def analyze_video(self, video_path, target_img_path, output_dir="output", 
                      sample_rate_fps=2.0, similarity_threshold=0.36, 
                      enable_motion_filter=True, motion_threshold=15, 
                      progress_callback=None):
        """
        Scans a video for the target person.
        - sample_rate_fps: How many frames to process per second of video.
        - similarity_threshold: Match threshold (OpenCV SFace standard is 0.36).
        - enable_motion_filter: Skip frames with no motion.
        - motion_threshold: Threshold value for background subtraction differences.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load target image and get its embedding
        target_img = cv2.imread(target_img_path)
        if target_img is None:
            raise ValueError(f"Could not load target image from {target_img_path}")
            
        target_embedding, target_face = self.get_face_embedding(target_img)
        if target_embedding is None:
            raise ValueError("No face detected in the target image. Please use a clearer photo.")
            
        # Save a crop of the target face for validation reference
        target_bbox = target_face[0:4].astype(int)
        tx, ty, tw, th = target_bbox
        # Ensure coordinates are within image boundaries
        ty_min, ty_max = max(0, ty), min(target_img.shape[0], ty+th)
        tx_min, tx_max = max(0, tx), min(target_img.shape[1], tx+tw)
        target_crop = target_img[ty_min:ty_max, tx_min:tx_max]
        cv2.imwrite(os.path.join(output_dir, "target_face_crop.jpg"), target_crop)

        # 2. Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / video_fps if video_fps > 0 else 0
        
        # Calculate step size based on requested sampling rate
        # e.g., if video is 30fps and we want 2fps, we step by 15 frames.
        frame_step = max(1, int(video_fps / sample_rate_fps)) if video_fps > 0 else 1

        matches = []
        prev_gray_frame = None
        
        print(f"Starting video scan...")
        print(f"Duration: {self.format_timestamp(duration_sec)}")
        print(f"Total Frames: {total_frames} | Frame Step: {frame_step}")

        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Only process sampled frames
            if frame_idx % frame_step == 0:
                current_time_sec = frame_idx / video_fps
                
                # Check cancellation or update progress
                if progress_callback:
                    progress_callback(frame_idx / total_frames, current_time_sec)

                # Convert to grayscale for motion detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

                # Motion detection filter
                if enable_motion_filter and prev_gray_frame is not None:
                    frame_delta = cv2.absdiff(prev_gray_frame, gray_frame)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    # Calculate percentage of changed pixels
                    motion_score = (np.sum(thresh == 255) / thresh.size) * 100
                    
                    if motion_score < (motion_threshold / 100):
                        # Skip frame: no significant motion detected
                        frame_idx += 1
                        continue

                prev_gray_frame = gray_frame

                # Detect faces in current frame
                h, w, _ = frame.shape
                self.detector.setInputSize((w, h))
                retval, faces = self.detector.detect(frame)

                if faces is not None:
                    for i, face in enumerate(faces):
                        # Extract embedding for this face
                        try:
                            aligned_face = self.recognizer.alignCrop(frame, face)
                            face_embedding = self.recognizer.feature(aligned_face)
                        except Exception as e:
                            # Skip if alignment fails
                            continue

                        # Compare with target
                        similarity = self.compute_similarity(target_embedding, face_embedding)

                        if similarity >= similarity_threshold:
                            timestamp_str = self.format_timestamp(current_time_sec)
                            
                            # Save crop of the matched face
                            bbox = face[0:4].astype(int)
                            x, y, bw, bh = bbox
                            y_min, y_max = max(0, y), min(h, y+bh)
                            x_min, x_max = max(0, x), min(w, x+bw)
                            face_crop = frame[y_min:y_max, x_min:x_max]
                            
                            match_filename = f"match_{timestamp_str.replace(':', '-').replace('.', '_')}_{i}.jpg"
                            match_filepath = os.path.join(output_dir, match_filename)
                            cv2.imwrite(match_filepath, face_crop)

                            # Save context frame (with bounding box)
                            context_frame = frame.copy()
                            cv2.rectangle(context_frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                            cv2.putText(context_frame, f"Match: {similarity:.2f}", (x, y-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            context_filename = f"context_{timestamp_str.replace(':', '-').replace('.', '_')}_{i}.jpg"
                            context_filepath = os.path.join(output_dir, context_filename)
                            cv2.imwrite(context_filepath, context_frame)

                            matches.append({
                                "timestamp_seconds": current_time_sec,
                                "timestamp": timestamp_str,
                                "similarity": float(similarity),
                                "crop_path": match_filepath,
                                "context_path": context_filepath
                            })

            frame_idx += 1

        cap.release()
        scan_duration = time.time() - start_time
        
        # Save results log
        results = {
            "video_path": video_path,
            "target_image": target_img_path,
            "scan_time_seconds": scan_duration,
            "total_matches": len(matches),
            "matches": matches
        }
        
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"Scan complete in {scan_duration:.2f} seconds. Found {len(matches)} matches.")
        return results

if __name__ == "__main__":
    # Test initialization
    try:
        analyzer = VideoFaceAnalyzer()
        print("Success: Analyzer successfully initialized.")
    except Exception as e:
        print(f"Error during initialization: {e}")
