import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import base64
import json
import requests
from datetime import datetime
import customtkinter as ctk
import mediapipe as mp
import zlib
import base91
from io import BytesIO

# =========================
# FIREBASE CONFIG
# =========================
FIREBASE_CONFIG = {
    "YOUR_FIREBASE_CONFIGURATION"
}

FIREBASE_RTDB_URL = "FIREBASE_JSON_CONFIG_FILE"

# =========================
# CONFIG
# =========================
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
MODEL_PATH = r"C:\Users\sys\PycharmProjects\Major_Project\curriculum_model.pth"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
LOW_LIGHT_THRESHOLD = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# MODEL
# =========================
class SimpleFERCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# =========================
# PREPROCESSING FUNCTIONS
# =========================
def detect_and_enhance_low_light(gray):
    mean_brightness = gray.mean()
    if mean_brightness >= LOW_LIGHT_THRESHOLD:
        return gray, False, mean_brightness

    target = 120.0
    gamma = np.log(target + 1e-6) / np.log(mean_brightness + 1e-6)
    gamma = np.clip(gamma, 0.6, 2.0)

    norm = gray.astype(np.float32) / 255.0
    gamma_corrected = np.power(norm, 1.0 / gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gamma_corrected)
    denoised = cv2.bilateralFilter(clahe_enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised, True, mean_brightness


def align_face_with_landmarks(rgb_img, gray_img):
    rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    results = face_mesh.process(rgb_bgr)
    if not results.multi_face_landmarks:
        return gray_img, [0.0, 0.0, 0.0], False

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = gray_img.shape

    def get_point(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    left_eye = get_point(33)
    right_eye = get_point(263)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_gray = cv2.warpAffine(gray_img, rot_mat, (w, h), flags=cv2.INTER_CUBIC)

    return aligned_gray, [0.0, 0.0, 0.0], True


def detect_face(gray, face_cascade):
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    return max(faces, key=lambda b: b[2] * b[3]) if len(faces) > 0 else None


transform_input = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# =========================
# ULTIMATE COMPRESSION FUNCTIONS
# =========================
# =========================
# ULTIMATE COMPRESSION FUNCTIONS (FIXED)
# =========================
def ultimate_string_compression(frame_rgb, face_box):
    """
    BEST POSSIBLE STRING ENCODING FOR FIREBASE RTDB
    Pipeline: Face-only → WebP → Zlib → Base91
    Expected: 23KB JPEG → 5-8KB string (70-80% reduction)
    """
    try:
        x, y, w, h = face_box

        # Step 1: Extract face only (reduces pixel count by 70-85%)
        face_rgb = frame_rgb[y:y + h, x:x + w]

        # Step 2: Convert to WebP (more compatible than AVIF, still 30-40% smaller than JPEG)
        img = Image.fromarray(face_rgb)
        buffer = BytesIO()
        img.save(buffer, format='WEBP', quality=75, method=6)
        webp_bytes = buffer.getvalue()

        # Step 3: Zlib compress (squeeze out remaining redundancy)
        compressed = zlib.compress(webp_bytes, level=9)

        # Step 4: Base91 encoding (most efficient text encoding)
        base91_string = base91.encode(compressed)

        # ========== SIZE REPORT ==========
        # Calculate original full-frame JPEG size for comparison
        _, orig_buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
        original_size = len(orig_buffer.tobytes())

        print("\n========== COMPRESSION REPORT ==========")
        print(f"Original full frame JPEG:     {original_size:,} bytes")
        print(
            f"Face-only WebP:               {len(webp_bytes):,} bytes ({100 - (len(webp_bytes) / original_size) * 100:.1f}% reduction)")
        print(
            f"Zlib compressed:              {len(compressed):,} bytes ({100 - (len(compressed) / original_size) * 100:.1f}% reduction)")
        print(
            f"Final Base91 string:          {len(base91_string):,} chars ({100 - (len(base91_string) / original_size) * 100:.1f}% reduction)")
        print(f"Total size reduction:         {100 - (len(base91_string) / original_size) * 100:.1f}%")
        print("================================================\n")

        return {
            'compressed_data': base91_string,
            'face_coords': [int(x), int(y), int(w), int(h)],  # FIX: Convert to Python int
            'original_shape': [int(frame_rgb.shape[0]), int(frame_rgb.shape[1])],  # FIX: Convert to Python int
            'encoding': 'webp_zlib_base91'
        }

    except Exception as e:
        print(f"Compression error: {e}")
        # Fallback to simpler JPEG compression
        return fallback_compression(frame_rgb, face_box)


def fallback_compression(frame_rgb, face_box):
    """
    Fallback if WebP not available
    Pipeline: Face-only JPEG → Zlib → Base91
    """
    try:
        x, y, w, h = face_box
        face_rgb = frame_rgb[y:y + h, x:x + w]

        # Encode face as JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR),
                                 [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = buffer.tobytes()

        # Compress + Base91 encode
        compressed = zlib.compress(jpeg_bytes, level=9)
        base91_string = base91.encode(compressed)

        # Calculate original size
        _, orig_buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
        original_size = len(orig_buffer.tobytes())

        print("\n========== FALLBACK COMPRESSION REPORT ==========")
        print(f"Original full frame JPEG:     {original_size:,} bytes")
        print(f"Face-only JPEG:               {len(jpeg_bytes):,} bytes")
        print(f"Zlib compressed:              {len(compressed):,} bytes")
        print(f"Final Base91 string:          {len(base91_string):,} chars")
        print(f"Total size reduction:         {100 - (len(base91_string) / original_size) * 100:.1f}%")
        print("================================================\n")

        return {
            'compressed_data': base91_string,
            'face_coords': [int(x), int(y), int(w), int(h)],  # FIX: Convert to Python int
            'original_shape': [int(frame_rgb.shape[0]), int(frame_rgb.shape[1])],  # FIX: Convert to Python int
            'encoding': 'jpeg_zlib_base91'
        }
    except Exception as e:
        print(f"Fallback compression error: {e}")
        return None


def decompress_from_firebase(data):
    """
    Retrieve and decompress image from Firebase
    """
    try:
        # Decode Base91 → Zlib decompress
        compressed = base91.decode(data['compressed_data'])
        decompressed_bytes = zlib.decompress(compressed)

        # Load image (WebP or JPEG depending on encoding)
        img = Image.open(BytesIO(decompressed_bytes))
        face_array = np.array(img)

        return face_array, data['face_coords']
    except Exception as e:
        print(f"Decompression error: {e}")
        return None, None


# =========================
# FIREBASE SAVE FUNCTION (FIXED)
# =========================
def save_to_firebase_compressed(self, frame_rgb, face_box, emotion_text, conf_text, brightness_numeric, pose_numeric):
    """
    Save compressed face image to Firebase RTDB
    """
    try:
        # 1. Compress image
        compressed_result = ultimate_string_compression(frame_rgb, face_box)

        if compressed_result is None:
            self.status_label.configure(text="Compression failed!")
            return False

        # 2. Confidence cleanup
        conf_value = float(conf_text.replace('%', '')) if conf_text not in ['-', ''] else 0.0

        # 3. Build RTDB entry (FIX: Convert all NumPy types to Python types)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "timestamp": timestamp,
            "emotion": str(emotion_text),
            "confidence_percent": float(conf_value),
            "brightness": float(brightness_numeric),
            "pose_yaw_pitch_roll": [float(p) for p in pose_numeric],  # FIX: Convert list elements
            **compressed_result  # Unpacks compressed_data, face_coords, etc.
        }

        # 4. Upload to Firebase
        response = requests.post(FIREBASE_RTDB_URL, json=payload)

        if response.status_code in [200, 201]:
            self.status_label.configure(text="✓ Saved to Firebase Cloud!")
            print(f"✓ Saved: {emotion_text} ({conf_value:.1f}%) at {timestamp}")
            return True

        self.status_label.configure(text=f"Firebase HTTP {response.status_code}")
        print(f"Firebase response: {response.text}")
        return False

    except Exception as e:
        self.status_label.configure(text=f"Cloud save error: {str(e)}")
        print(f"Cloud save error: {e}")
        import traceback
        traceback.print_exc()
        return False


def fallback_compression(frame_rgb, face_box):
    """
    Fallback if AVIF not available
    Pipeline: Face-only JPEG → Zlib → Base91
    """
    try:
        x, y, w, h = face_box
        face_rgb = frame_rgb[y:y + h, x:x + w]

        # Encode face as JPEG
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR),
                                 [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpeg_bytes = buffer.tobytes()

        # Compress + Base91 encode
        compressed = zlib.compress(jpeg_bytes, level=9)
        base91_string = base91.encode(compressed)

        print(f"Using fallback JPEG compression: {len(base91_string):,} chars")

        return {
            'compressed_data': base91_string,
            'face_coords': [x, y, w, h],
            'original_shape': list(frame_rgb.shape[:2]),
            'encoding': 'jpeg_zlib_base91'
        }
    except Exception as e:
        print(f"Fallback compression error: {e}")
        return None


def decompress_from_firebase(data):
    """
    Retrieve and decompress image from Firebase
    """
    try:
        # Decode Base91 → Zlib decompress
        compressed = base91.decode(data['compressed_data'])
        decompressed_bytes = zlib.decompress(compressed)

        # Load image (AVIF or JPEG depending on encoding)
        img = Image.open(BytesIO(decompressed_bytes))
        face_array = np.array(img)

        return face_array, data['face_coords']
    except Exception as e:
        print(f"Decompression error: {e}")
        return None, None


# =========================
# FIREBASE SAVE FUNCTION
# =========================
def save_to_firebase_compressed(self, frame_rgb, face_box, emotion_text, conf_text, brightness_numeric, pose_numeric):
    """
    Save compressed face image to Firebase RTDB
    """
    try:
        # 1. Compress image
        compressed_result = ultimate_string_compression(frame_rgb, face_box)

        if compressed_result is None:
            self.status_label.configure(text="Compression failed!")
            return False

        # 2. Confidence cleanup
        conf_value = float(conf_text.replace('%', '')) if conf_text not in ['-', ''] else 0.0

        # 3. Build RTDB entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload = {
            "timestamp": timestamp,
            "emotion": emotion_text,
            "confidence_percent": conf_value,
            "brightness": float(brightness_numeric),
            "pose_yaw_pitch_roll": pose_numeric,
            **compressed_result  # Unpacks compressed_data, face_coords, etc.
        }

        # 4. Upload to Firebase
        response = requests.post(FIREBASE_RTDB_URL, json=payload)

        if response.status_code in [200, 201]:
            self.status_label.configure(text="✓ Saved to Firebase Cloud!")
            print(f"✓ Saved: {emotion_text} ({conf_value:.1f}%) at {timestamp}")
            return True

        self.status_label.configure(text=f"Firebase HTTP {response.status_code}")
        return False

    except Exception as e:
        self.status_label.configure(text=f"Cloud save error: {str(e)}")
        print(f"Cloud save error: {e}")
        return False


# =========================
# CUSTOMTKINTER UI + CAMERA LOOP
# =========================
class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Emotion Detection")
        self.geometry("1100x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.video_label = ctk.CTkLabel(self, text="", width=640, height=480)
        self.video_label.place(x=20, y=20)

        self.align_var = ctk.BooleanVar(value=True)
        self.light_var = ctk.BooleanVar(value=True)

        ctk.CTkLabel(self, text="Preprocessing", font=ctk.CTkFont(size=18, weight="bold")).place(x=690, y=20)
        ctk.CTkSwitch(self, text="Face alignment", variable=self.align_var).place(x=690, y=60)
        ctk.CTkSwitch(self, text="Low-light enhancement", variable=self.light_var).place(x=690, y=100)

        ctk.CTkLabel(self, text="Results", font=ctk.CTkFont(size=18, weight="bold")).place(x=690, y=150)
        self.emotion_label = ctk.CTkLabel(self, text="Emotion: -", font=ctk.CTkFont(size=20))
        self.emotion_label.place(x=690, y=190)
        self.conf_label = ctk.CTkLabel(self, text="Confidence: -", font=ctk.CTkFont(size=18))
        self.conf_label.place(x=690, y=230)

        ctk.CTkLabel(self, text="Environment", font=ctk.CTkFont(size=18, weight="bold")).place(x=690, y=280)
        self.brightness_label = ctk.CTkLabel(self, text="Brightness: -", font=ctk.CTkFont(size=14))
        self.brightness_label.place(x=690, y=320)
        self.pose_label = ctk.CTkLabel(self, text="Pose: -", font=ctk.CTkFont(size=14))
        self.pose_label.place(x=690, y=350)

        self.cloud_button = ctk.CTkButton(
            self,
            text="Save to Firebase (Ultra Compressed)",
            command=self.save_current_frame_to_cloud,
            width=280,
            height=50,
        )
        self.cloud_button.place(x=690, y=400)

        self.status_label = ctk.CTkLabel(self, text="Status: Ready", font=ctk.CTkFont(size=14))
        self.status_label.place(x=690, y=470)

        self.quit_button = ctk.CTkButton(self, text="Quit", command=self.on_quit, width=200, height=40)
        self.quit_button.place(x=690, y=520)

        self.model = SimpleFERCNN(num_classes=len(EMOTION_CLASSES)).to(DEVICE)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.face_cascade = cv2.CascadeClassifier(HAAR_PATH)

        self.current_frame_rgb = None
        self.current_face_box = None
        self.current_emotion = "-"
        self.current_conf = "-"
        self.current_brightness_numeric = 0.0
        self.current_pose_numeric = [0.0, 0.0, 0.0]

        self.update_frame()

    def save_current_frame_to_cloud(self):
        if self.current_frame_rgb is None or self.current_emotion == "-":
            self.status_label.configure(text="No face/emotion detected!")
            return

        if self.current_face_box is None:
            self.status_label.configure(text="No face box available!")
            return

        save_to_firebase_compressed(
            self,
            self.current_frame_rgb,
            self.current_face_box,
            self.current_emotion,
            self.current_conf,
            self.current_brightness_numeric,
            self.current_pose_numeric
        )

    def on_quit(self):
        if self.cap.isOpened():
            self.cap.release()
        face_mesh.close()
        self.destroy()

    def update_frame(self):
        if not self.cap.isOpened():
            self.after(50, self.update_frame)
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(50, self.update_frame)
            return

        frame_bgr = frame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        self.current_frame_rgb = frame_rgb.copy()

        original_brightness = gray.mean()
        brightness_val = original_brightness
        if self.light_var.get():
            gray, enhanced, brightness = detect_and_enhance_low_light(gray)
            brightness_val = brightness
            brightness_text = f"Brightness: {original_brightness:.1f} → {brightness:.1f}"
            if enhanced:
                brightness_text += " (ENHANCED)"
        else:
            brightness_text = f"Brightness: {original_brightness:.1f}"

        self.current_brightness_numeric = brightness_val
        self.brightness_label.configure(text=brightness_text)

        face_box = detect_face(gray, self.face_cascade)
        emotion_text = "-"
        conf_text = "-"
        pose_text = "Pose: -"

        if face_box is not None:
            self.current_face_box = face_box  # Store face box
            x, y, w, h = face_box
            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_gray = gray[y:y + h, x:x + w]
            face_rgb_crop = frame_rgb[y:y + h, x:x + w]

            if self.align_var.get():
                face_gray, pose_numeric, success = align_face_with_landmarks(face_rgb_crop, face_gray)
                if success:
                    self.current_pose_numeric = pose_numeric
                    pose_text = f"Pose: {pose_numeric[0]:.0f}/{pose_numeric[1]:.0f}/{pose_numeric[2]:.0f}"

            self.pose_label.configure(text=pose_text)

            pil_face = Image.fromarray(face_gray)
            inp = transform_input(pil_face).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                emotion_text = EMOTION_CLASSES[pred_idx]
                conf_text = f"{conf * 100:.1f}%"

            self.current_emotion = emotion_text
            self.current_conf = conf_text

            cv2.putText(frame_rgb, f"{emotion_text} ({conf_text})",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2, cv2.LINE_AA)
        else:
            self.current_face_box = None

        self.emotion_label.configure(text=f"Emotion: {emotion_text}")
        self.conf_label.configure(text=f"Confidence: {conf_text}")

        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.after(30, self.update_frame)


# =========================
# MAIN APP LAUNCH
# =========================
if __name__ == "__main__":
    print("Starting Emotion Detection with Ultimate Compression")
    print("Compression: Face-only → AVIF → Zlib → Base91")
    print("=" * 50)
    app = EmotionApp()
    app.mainloop()
