FIREBASE_RTDB_URL = "Use_Your_Firebase_DB"
import requests
import base64
import zlib
import io
from PIL import Image, ImageTk
import customtkinter as ctk
import matplotlib.pyplot as plt
import base91
import numpy as np
from datetime import datetime
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use('Agg')

# ============================
# CONFIGURATION
# ============================
FIREBASE_RTDB_URL = "USE_YOUR_FIREBASE_REAL_TIME_DB"
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ============================
# MODERN FETCH + DECODE (WITH LOADING)
# ============================
class ImageDataFetcher:
    """Modern fetcher with progress tracking and error handling"""

    @staticmethod
    def fetch_and_decode(callback=None):
        """Fetch and decode images with progress callback"""
        try:
            response = requests.get(FIREBASE_RTDB_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            images = []
            total_full_frame_size = 0
            total_compressed = 0
            skipped = 0

            total_items = len(data)

            for idx, (key, entry) in enumerate(data.items()):
                if callback:
                    callback(idx, total_items, f"Processing {key[:20]}...")

                try:
                    result = ImageDataFetcher._process_entry(key, entry)
                    if result:
                        images.append(result["image_data"])
                        total_full_frame_size += result["estimated_full"]
                        total_compressed += result["compressed_size"]
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"Error processing {key}: {e}")
                    skipped += 1

            print(f"\n{'=' * 60}")
            print(f"Successfully loaded {len(images)} images")
            print(f"Skipped {skipped} invalid entries")
            print(f"Would-be Full Frame: {total_full_frame_size / 1024:.2f} KB")
            print(f"Actual Compressed: {total_compressed / 1024:.2f} KB")
            print(f"Space Saved: {(total_full_frame_size - total_compressed) / 1024:.2f} KB")
            print(f"⚡ Compression Ratio: {100 * (1 - total_compressed / total_full_frame_size):.1f}%")
            print(f"{'=' * 60}\n")

            return images, total_full_frame_size, total_compressed

        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            return [], 0, 0

    @staticmethod
    def _process_entry(key, entry):
        """Process individual database entry"""
        emotion = entry.get("emotion", "Unknown")
        timestamp = entry.get("timestamp", datetime.now().isoformat())
        confidence = entry.get("confidence_percent", 0)

        # New format: compressed_data (Base91)
        if "compressed_data" in entry:
            return ImageDataFetcher._process_new_format(key, entry, emotion, timestamp, confidence)

        # Old format: compressed_image (Hex)
        elif "compressed_image" in entry:
            return ImageDataFetcher._process_old_format(key, entry, emotion, timestamp, confidence)

        return None

    @staticmethod
    def _process_new_format(key, entry, emotion, timestamp, confidence):
        """Process new compression format"""
        try:
            compressed_base91 = entry.get("compressed_data")
            encoding_type = entry.get("encoding", "WebP")
            face_coords = entry.get("face_coords", [0, 0, 100, 100])
            original_shape = entry.get("original_shape", [480, 640])

            # Decode Base91 → Zlib decompress
            compressed_bytes = base91.decode(compressed_base91)
            image_bytes = zlib.decompress(compressed_bytes)

            # Load image
            img = Image.open(io.BytesIO(image_bytes))

            # Calculate compression metrics
            face_w, face_h = img.size
            frame_h, frame_w = original_shape
            face_area = face_w * face_h
            frame_area = frame_h * frame_w
            area_ratio = face_area / frame_area if frame_area > 0 else 0.2
            estimated_full_frame = int(len(image_bytes) / area_ratio) if area_ratio > 0 else len(image_bytes) * 5

            comp_size = len(compressed_base91)

            print(f"{emotion:12} |  {face_w}x{face_h} | "
                  f"{comp_size / 1024:.1f}KB | "
                  f"{100 * (1 - comp_size / estimated_full_frame):.0f}% saved")

            return {
                "image_data": {
                    "image": img,
                    "emotion": emotion,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "face_bytes": len(image_bytes),
                    "estimated_full": estimated_full_frame,
                    "comp": comp_size,
                    "encoding": encoding_type,
                    "face_coords": face_coords,
                    "key": key
                },
                "estimated_full": estimated_full_frame,
                "compressed_size": comp_size
            }

        except Exception as e:
            print(f"Skipped new format {key}: {e}")
            return None

    @staticmethod
    def _process_old_format(key, entry, emotion, timestamp, confidence):
        """Process old compression format"""
        try:
            compressed_hex = entry.get("compressed_image")

            if not isinstance(compressed_hex, str):
                return None

            compressed_bytes = bytes.fromhex(compressed_hex)
            base64_text = zlib.decompress(compressed_bytes).decode("utf-8")
            img_bytes = base64.b64decode(base64_text)
            img = Image.open(io.BytesIO(img_bytes))

            full_frame_size = len(img_bytes)
            comp_size = len(compressed_hex)

            print(f"{emotion:12} | 📏 Full Frame | "
                  f"{comp_size / 1024:.1f}KB | "
                  f"{100 * (1 - comp_size / full_frame_size):.0f}% saved")

            return {
                "image_data": {
                    "image": img,
                    "emotion": emotion,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "face_bytes": full_frame_size,
                    "estimated_full": full_frame_size,
                    "comp": comp_size,
                    "encoding": "Legacy Format",
                    "face_coords": None,
                    "key": key
                },
                "estimated_full": full_frame_size,
                "compressed_size": comp_size
            }

        except Exception as e:
            print(f"Skipped old format {key}: {e}")
            return None


# ============================
# MODERN UI COMPONENTS
# ============================
class ModernCard(ctk.CTkFrame):
    """Modern card component for image display"""

    def __init__(self, parent, image_data, **kwargs):
        super().__init__(parent, **kwargs)
        self.image_data = image_data
        self.configure(corner_radius=15, fg_color=("gray85", "gray25"))
        self._create_widgets()

    def _create_widgets(self):
        # Header with emotion and confidence
        emotion_color = self._get_emotion_color(self.image_data["emotion"])

        header_frame = ctk.CTkFrame(self, fg_color="transparent", height=40)
        header_frame.pack(fill="x", padx=10, pady=(10, 0))

        emotion_label = ctk.CTkLabel(
            header_frame,
            text=self.image_data["emotion"].upper(),
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=emotion_color
        )
        emotion_label.pack(side="left")

        confidence_label = ctk.CTkLabel(
            header_frame,
            text=f"{self.image_data['confidence']:.1f}%",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        confidence_label.pack(side="right")

        # Image display
        img = self.image_data["image"].resize((220, 220), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)

        image_frame = ctk.CTkFrame(self, fg_color="transparent")
        image_frame.pack(pady=10)

        image_label = ctk.CTkLabel(image_frame, image=self.photo, text="")
        image_label.pack()

        # Compression info
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=(0, 10))

        # File sizes
        estimated_kb = self.image_data["estimated_full"] / 1024
        comp_kb = self.image_data["comp"] / 1024
        saved_pct = 100 * (1 - self.image_data["comp"] / self.image_data["estimated_full"])

        size_text = f"{comp_kb:.1f}KB / ⏳ {estimated_kb:.1f}KB"
        saved_text = f"{saved_pct:.1f}% saved"

        ctk.CTkLabel(
            info_frame,
            text=size_text,
            font=ctk.CTkFont(size=11),
            text_color="gray70"
        ).pack()

        ctk.CTkLabel(
            info_frame,
            text=saved_text,
            font=ctk.CTkFont(size=11),
            text_color="#51cf66"
        ).pack()

        # Encoding badge
        encoding_frame = ctk.CTkFrame(
            info_frame,
            fg_color=self._get_encoding_color(self.image_data["encoding"]),
            corner_radius=10,
            height=24
        )
        encoding_frame.pack(pady=5)

        ctk.CTkLabel(
            encoding_frame,
            text=self.image_data["encoding"],
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="white"
        ).pack(padx=10, pady=2)

    def _get_emotion_color(self, emotion):
        """Return color based on emotion"""
        colors = {
            "happy": "#FFD166",
            "sad": "#118AB2",
            "angry": "#EF476F",
            "surprise": "#06D6A0",
            "neutral": "#8A8A8A",
            "fear": "#7209B7"
        }
        return colors.get(emotion.lower(), "#118AB2")

    def _get_encoding_color(self, encoding):
        """Return color based on encoding type"""
        colors = {
            "WebP": "#4285F4",
            "JPEG": "#EA4335",
            "Legacy Format": "#FBBC05",
            "unknown": "#8A8A8A"
        }
        return colors.get(encoding, "#34A853")


# ============================
# MAIN APPLICATION
# ============================
class ModernGalleryApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Emotion Dashboard")
        self.geometry("1600x950")
        self.minsize(1200, 700)

        # App state
        self.images = []
        self.total_orig = 0
        self.total_comp = 0

        # UI setup
        self._setup_ui()

        # Initial data load
        self._load_data_async()

    def _setup_ui(self):
        """Setup the modern UI layout"""

        # Create main container with padding
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        self._create_header(main_container)

        # Tab view for switching between views
        self.tabview = ctk.CTkTabview(main_container)
        self.tabview.pack(fill="both", expand=True, pady=(20, 0))

        # Create tabs
        self.gallery_tab = self.tabview.add("Gallery")
        self.stats_tab = self.tabview.add("Analytics")

        # Setup tab contents
        self._setup_gallery_tab()
        self._setup_stats_tab()

        # Status bar
        self._create_status_bar(main_container)

    def _create_header(self, parent):
        """Create modern header with logo and controls"""
        header = ctk.CTkFrame(parent, height=80, fg_color="transparent")
        header.pack(fill="x")

        # Left side: Logo and title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", fill="y")

        ctk.CTkLabel(
            title_frame,
            text="Emotion Gallery",
            font=ctk.CTkFont(size=28, weight="bold")
        ).pack(side="left", padx=(0, 20))

        ctk.CTkLabel(
            title_frame,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray70"
        ).pack(side="left")

        # Right side: Controls
        controls_frame = ctk.CTkFrame(header, fg_color="transparent")
        controls_frame.pack(side="right", fill="y")

        # Refresh button with icon
        self.refresh_btn = ctk.CTkButton(
            controls_frame,
            text="Refresh",
            command=self._refresh_data,
            width=120,
            height=40,
            fg_color="#4285F4",
            hover_color="#3367D6"
        )
        self.refresh_btn.pack(side="right", padx=(10, 0))

        # Filter dropdown
        self.filter_var = ctk.StringVar(value="All Emotions")
        filter_menu = ctk.CTkOptionMenu(
            controls_frame,
            values=["All Emotions", "Happy", "Sad", "Angry", "Surprise", "Neutral"],
            variable=self.filter_var,
            width=140,
            height=40,
            command=self._filter_gallery
        )
        filter_menu.pack(side="right", padx=(10, 0))

        # Search bar
        search_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        search_frame.pack(side="right")

        ctk.CTkLabel(search_frame, text="", font=ctk.CTkFont(size=18)).pack(side="left", padx=(0, 5))
        self.search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search emotions...",
            width=200,
            height=40
        )
        self.search_entry.pack(side="left")
        self.search_entry.bind("<KeyRelease>", self._search_gallery)

    def _setup_gallery_tab(self):
        """Setup gallery with grid layout"""
        self.gallery_container = ctk.CTkScrollableFrame(
            self.gallery_tab,
            fg_color="transparent"
        )
        self.gallery_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Loading placeholder
        self.loading_label = ctk.CTkLabel(
            self.gallery_container,
            text="Loading images...",
            font=ctk.CTkFont(size=16)
        )
        self.loading_label.pack(pady=50)

    def _setup_stats_tab(self):
        """Setup statistics dashboard"""
        # Main stats container
        self.stats_container = ctk.CTkScrollableFrame(
            self.stats_tab,
            fg_color="transparent"
        )
        self.stats_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Placeholder for stats
        ctk.CTkLabel(
            self.stats_container,
            text="Loading statistics...",
            font=ctk.CTkFont(size=16)
        ).pack(pady=50)




    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_bar = ctk.CTkFrame(parent, height=40, fg_color="transparent")
        status_bar.pack(fill="x", pady=(10, 0))

        self.status_label = ctk.CTkLabel(
            status_bar,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        self.status_label.pack(side="left")

        self.stats_label = ctk.CTkLabel(
            status_bar,
            text="Images: 0 | Size: 0KB",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        self.stats_label.pack(side="right")

    def _load_data_async(self):
        """Load data in background thread"""
        self.status_label.configure(text="Fetching data from Firebase...")
        self.refresh_btn.configure(state="disabled")

        threading.Thread(
            target=self._fetch_data_thread,
            daemon=True
        ).start()

    def _fetch_data_thread(self):
        """Thread for fetching data"""

        def progress_callback(current, total, message):
            self.after(0, lambda: self._update_progress(current, total, message))

        self.images, self.total_orig, self.total_comp = ImageDataFetcher.fetch_and_decode(
            callback=progress_callback
        )

        self.after(0, self._on_data_loaded)

    def _update_progress(self, current, total, message):
        """Update progress in UI"""
        if total > 0:
            percent = (current / total) * 100
            self.status_label.configure(
                text=f" {message} ({percent:.0f}%)"
            )

    def _on_data_loaded(self):
        """Called when data loading is complete"""
        self.status_label.configure(text="Data loaded successfully")
        self.refresh_btn.configure(state="normal")

        # Update stats label
        self.stats_label.configure(
            text=f"Images: {len(self.images)} | "
                 f"Saved: {(self.total_orig - self.total_comp) / 1024:.1f}KB"
        )

        # Rebuild views
        self._rebuild_gallery()
        self._rebuild_stats()

    def _rebuild_gallery(self):
        """Rebuild gallery view"""
        # Clear loading placeholder
        for widget in self.gallery_container.winfo_children():
            widget.destroy()

        if not self.images:
            ctk.CTkLabel(
                self.gallery_container,
                text="No images found",
                font=ctk.CTkFont(size=16)
            ).pack(pady=50)
            return

        # Create grid layout
        cols = 4
        for i, item in enumerate(self.images):
            row, col = divmod(i, cols)

            card = ModernCard(self.gallery_container, item)
            card.grid(
                row=row,
                column=col,
                padx=10,
                pady=10,
                sticky="nsew"
            )

        # Configure grid weights
        for i in range(cols):
            self.gallery_container.grid_columnconfigure(i, weight=1)

    def _rebuild_stats(self):
        """Rebuild statistics view"""
        for widget in self.stats_container.winfo_children():
            widget.destroy()

        if not self.images:
            return

        # Create stats dashboard
        self._create_stats_dashboard()
        self._create_visualizations()

    def _create_stats_dashboard(self):
        """Create statistics dashboard"""
        # KPI cards
        kpi_frame = ctk.CTkFrame(self.stats_container, fg_color="transparent")
        kpi_frame.pack(fill="x", pady=(0, 20))

        kpis = [
            {
                "title": "Total Images",
                "value": len(self.images),
                "color": "#4285F4",
                "icon": ""
            },
            {
                "title": "Space Saved",
                "value": f"{(self.total_orig - self.total_comp) / 1024:.1f} KB",
                "color": "#34A853",
                "icon": ""
            },
            {
                "title": "Compression Ratio",
                "value": f"{100 * (1 - self.total_comp / self.total_orig):.1f}%",
                "color": "#FBBC05",
                "icon": ""
            },
            {
                "title": "Avg. per Image",
                "value": f"{(self.total_comp / len(self.images)) / 1024:.1f} KB",
                "color": "#EA4335",
                "icon": ""
            }
        ]

        for i, kpi in enumerate(kpis):
            card = ctk.CTkFrame(
                kpi_frame,
                width=200,
                height=120,
                corner_radius=15,
                fg_color=kpi["color"]
            )
            card.pack(side="left", expand=True, fill="both", padx=(0 if i == 3 else 0, 10))

            # Card content
            ctk.CTkLabel(
                card,
                text=kpi["icon"],
                font=ctk.CTkFont(size=32)
            ).pack(pady=(15, 5))

            ctk.CTkLabel(
                card,
                text=str(kpi["value"]),
                font=ctk.CTkFont(size=24, weight="bold"),
                text_color="white"
            ).pack()

            ctk.CTkLabel(
                card,
                text=kpi["title"],
                font=ctk.CTkFont(size=12),
                text_color="white"
            ).pack(pady=(5, 15))

    def _create_visualizations(self):
        """Create charts and visualizations"""
        # Emotion distribution
        emotions_frame = ctk.CTkFrame(self.stats_container)
        emotions_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(
            emotions_frame,
            text="Emotion Distribution",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        # Calculate emotion counts
        emotion_counts = {}
        for item in self.images:
            emotion = item["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Create pie chart
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')

        if emotion_counts:
            colors = ['#FFD166', '#118AB2', '#EF476F', '#06D6A0', '#7209B7']
            wedges, texts, autotexts = ax.pie(
                emotion_counts.values(),
                labels=emotion_counts.keys(),
                autopct='%1.1f%%',
                colors=colors[:len(emotion_counts)],
                startangle=90
            )

            # Style text
            for text in texts:
                text.set_color('white')
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)

        ax.set_title("", color='white', pad=20)

        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, emotions_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)

    def _filter_gallery(self, choice):
        """Filter gallery by emotion"""
        # Implementation for filtering
        pass

    def _search_gallery(self, event):
        """Search gallery by text"""
        # Implementation for searching
        pass

    def _refresh_data(self):
        """Refresh data from Firebase"""
        self._load_data_async()


# ============================
# APPLICATION ENTRY POINT
# ============================
if __name__ == "__main__":
    print("Starting...")
    print("=" * 60)

    app = ModernGalleryApp()
    app.mainloop()
