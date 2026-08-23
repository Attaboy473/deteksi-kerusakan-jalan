import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import av
import cv2
import pandas as pd
import streamlit as st
import torch
from PIL import ExifTags, Image
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

st.set_page_config(
    page_title="Deteksi Kerusakan Jalan YOLOv11",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# KONFIGURASI MODEL (versi lokal: model dicari di folder yang sama dengan app.py)
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "best_yolo11n_jalanrusak_4kelas.pt")

if not os.path.exists(MODEL_PATH):
    candidates = [
        p for p in BASE_DIR.rglob("best*.pt")
        if "yolo11n.pt" not in str(p) and "yolo26n.pt" not in str(p)
    ]

    if candidates:
        MODEL_PATH = str(max(candidates, key=lambda p: p.stat().st_mtime))

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model .pt tidak ditemukan. Pastikan file "
        "best_yolo11n_jalanrusak_4kelas.pt sudah ada."
    )
    st.stop()


@st.cache_resource
def load_model(model_path):
    loaded_model = YOLO(model_path)
    # Fuse Conv + BatchNorm untuk mengurangi overhead inferensi.
    try:
        loaded_model.fuse()
    except Exception:
        pass
    return loaded_model


model = load_model(MODEL_PATH)
INFERENCE_DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = bool(torch.cuda.is_available())

# YOLO/PyTorch tidak selalu aman jika satu objek model dipanggil oleh
# beberapa thread WebRTC secara bersamaan.
model_lock = threading.Lock()


# =========================
# CUSTOM STYLING
# =========================
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ---------- Hero header ---------- */
        .hero-wrap { padding: 0.5rem 0 1.25rem 0; }
        .hero-title {
            font-size: 2.6rem; font-weight: 800; line-height: 1.15;
            background: linear-gradient(92deg, #FFC65C 0%, #FFB020 45%, #FF7A1A 100%);
            -webkit-background-clip: text; background-clip: text;
            -webkit-text-fill-color: transparent; color: transparent;
            margin-bottom: 0.4rem;
        }
        .hero-sub { color: #9AA4B2; font-size: 1.02rem; margin-bottom: 1rem; }
        .chip {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.4rem 0.95rem; margin: 0 0.45rem 0.45rem 0;
            border-radius: 999px; font-weight: 600; font-size: 0.9rem;
            background: rgba(255, 176, 32, 0.10);
            border: 1px solid rgba(255, 176, 32, 0.35);
            color: #FFC65C;
        }

        /* ---------- Section cards ---------- */
        .section-card {
            background: #151A23;
            border: 1px solid #232B38;
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            margin: 0.9rem 0 1.4rem 0;
        }
        .section-heading {
            font-size: 1.25rem; font-weight: 700; color: #F5F7FA;
            margin-bottom: 0.9rem;
        }

        /* ---------- Metric cards ---------- */
        div[data-testid="stMetric"] {
            background: #151A23;
            border: 1px solid #232B38;
            border-radius: 14px;
            padding: 0.85rem 1rem;
        }
        div[data-testid="stMetricLabel"] { color: #9AA4B2; }
        div[data-testid="stMetricValue"] { color: #F5F7FA; font-size: 1.7rem; }

        /* ---------- Sidebar spec list ---------- */
        .sidebar-brand {
            font-size: 1.15rem; font-weight: 800; color: #FFB020;
            margin-bottom: 0.2rem;
        }
        .status-pill {
            display: inline-block; padding: 0.25rem 0.8rem; border-radius: 999px;
            background: rgba(46, 204, 113, 0.12);
            border: 1px solid rgba(46, 204, 113, 0.4);
            color: #6EE7A0; font-size: 0.85rem; font-weight: 600;
            margin-bottom: 1.1rem;
        }
        .status-dot {
            display: inline-block; width: 7px; height: 7px; border-radius: 50%;
            background: #6EE7A0; margin-right: 0.45rem; vertical-align: middle;
        }
        .spec-row {
            display: flex; justify-content: space-between; gap: 0.75rem;
            padding: 0.55rem 0; border-bottom: 1px solid #232B38;
        }
        .spec-row:last-child { border-bottom: none; }
        .spec-label { color: #9AA4B2; font-size: 0.88rem; }
        .spec-value { color: #F5F7FA; font-size: 0.88rem; font-weight: 600; text-align: right; }

        /* ---------- Bersihkan UI default Streamlit ---------- */
        #MainMenu, footer, header {visibility: hidden;}
        div[data-testid="stSidebar"] { border-right: 1px solid #232B38; }
        div[data-testid="stFileUploader"] > div { border-radius: 14px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()

def render_hero():
    class_names = [model.names[k] for k in sorted(model.names)]
    chips = "".join(
        f'<span class="chip">{name}</span>'
        for name in class_names
    )
    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-title">Deteksi Kerusakan Jalan<br>menggunakan YOLOv11</div>
            <div class="hero-sub">
                Sistem mendeteksi empat jenis kerusakan permukaan jalan secara otomatis —
                melalui upload gambar, kamera, maupun video real-time (WebRTC),
                lengkap dengan pembacaan lokasi foto (EXIF/GPS).
            </div>
            {chips}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(text):
    st.markdown(f'<div class="section-heading">{text}</div>', unsafe_allow_html=True)


# =========================
# FUNGSI METADATA EXIF
# =========================
def convert_to_degrees(value):
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None


def extract_exif_metadata(image_bytes):
    metadata = {
        "has_exif": False,
        "datetime": None,
        "latitude": None,
        "longitude": None
    }

    try:
        image = Image.open(BytesIO(image_bytes))
        exif_data = image.getexif()

        if not exif_data:
            return metadata

        metadata["has_exif"] = True

        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                metadata["datetime"] = str(value)

        try:
            gps_info = exif_data.get_ifd(ExifTags.IFD.GPSInfo)
        except Exception:
            gps_info = None

        if gps_info:
            gps_data = {
                ExifTags.GPSTAGS.get(key, key): value
                for key, value in gps_info.items()
            }

            gps_latitude = gps_data.get("GPSLatitude")
            gps_latitude_ref = gps_data.get("GPSLatitudeRef")
            gps_longitude = gps_data.get("GPSLongitude")
            gps_longitude_ref = gps_data.get("GPSLongitudeRef")

            if (
                gps_latitude
                and gps_latitude_ref
                and gps_longitude
                and gps_longitude_ref
            ):
                lat = convert_to_degrees(gps_latitude)
                lon = convert_to_degrees(gps_longitude)

                if lat is not None and lon is not None:
                    if gps_latitude_ref != "N":
                        lat = -lat
                    if gps_longitude_ref != "E":
                        lon = -lon

                    metadata["latitude"] = lat
                    metadata["longitude"] = lon

        return metadata

    except Exception:
        return metadata


# =========================
# DETEKSI GAMBAR STATIS
# =========================
def process_image(image_bytes, source_label, conf, imgsz):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    metadata = extract_exif_metadata(image_bytes)

    with st.spinner("Mendeteksi kerusakan jalan..."):
        with model_lock:
            results = model.predict(
                source=image,
                conf=conf,
                imgsz=imgsz,
                verbose=False
            )

    # Ultralytics menghasilkan array BGR dari plot().
    result_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    section_heading("Hasil Deteksi")

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            image,
            caption=f"Gambar Input ({source_label})",
            width="stretch"
        )
    with col2:
        st.image(
            result_img,
            caption="Hasil Deteksi",
            width="stretch"
        )

    st.markdown("")
    section_heading("Detail Deteksi")

    boxes = results[0].boxes
    num_detections = 0 if boxes is None or len(boxes) == 0 else len(boxes)

    if num_detections == 0:
        st.warning("Tidak ada kerusakan jalan yang terdeteksi.")
    else:
        rows = []
        class_counter = {}

        for i, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            class_counter[class_name] = class_counter.get(class_name, 0) + 1
            rows.append(
                {
                    "No": i,
                    "Kelas": class_name,
                    "Confidence": f"{confidence:.2f}",
                    "x1": round(xyxy[0], 1),
                    "y1": round(xyxy[1], 1),
                    "x2": round(xyxy[2], 1),
                    "y2": round(xyxy[3], 1),
                }
            )

        metric_cols = st.columns(5)
        metric_cols[0].metric("Total Deteksi", num_detections)
        class_names = [model.names[k] for k in sorted(model.names)]
        for col, name in zip(metric_cols[1:], class_names):
            col.metric(name, class_counter.get(name, 0))

        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
        )

        btn_col, info_col = st.columns([1, 2], vertical_alignment="center")
        btn_col.download_button(
            label="Unduh Gambar Hasil",
            data=cv2.imencode(".png", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name="hasil_deteksi.png",
            mime="image/png",
            width="stretch",
        )

        inference_ms = float(results[0].speed["inference"])
        info_col.caption(
            f"Inferensi {inference_ms:.1f} ms &nbsp;•&nbsp; "
            f"Image size {imgsz} px &nbsp;•&nbsp; "
            f"Confidence threshold {conf:.3f}"
        )

    st.divider()
    section_heading("Metadata & Lokasi Foto")

    if metadata["has_exif"]:
        if metadata["datetime"]:
            st.write(f"**Waktu Pengambilan Foto:** {metadata['datetime']}")
        else:
            st.info("Metadata waktu pengambilan foto tidak ditemukan.")

        if (
            metadata["latitude"] is not None
            and metadata["longitude"] is not None
        ):
            lat = metadata["latitude"]
            lon = metadata["longitude"]

            loc_col1, loc_col2 = st.columns(2)
            loc_col1.metric("Latitude", f"{lat:.6f}")
            loc_col2.metric("Longitude", f"{lon:.6f}")

            map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
            st.map(map_data, zoom=16)

            google_maps_url = f"https://www.google.com/maps?q={lat},{lon}"
            st.markdown(f"[Buka lokasi di Google Maps]({google_maps_url})")
        else:
            st.warning("Koordinat GPS tidak ditemukan pada metadata foto.")
    else:
        st.warning(
            "Metadata EXIF tidak ditemukan. Jika gambar diambil dari fitur "
            "kamera browser, metadata GPS biasanya tidak ikut tersimpan. "
            "Untuk menampilkan lokasi, gunakan upload JPG original dari kamera HP."
        )


# =========================
# DETEKSI REAL-TIME WEBRTC (512 px, NON-BLOCKING)
# =========================
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]}
        ]
    }
)


def create_video_frame_callback(conf, target_inference_fps):
    """
    Callback WebRTC berlatensi rendah.

    - Video dikembalikan pada setiap frame tanpa menunggu YOLO.
    - Inferensi selalu memakai imgsz=512.
    - Inferensi dibatasi 1-3 kali/detik.
    - Hanya frame terbaru yang diproses; tidak ada antrean frame lama.
    - Bounding box terakhir digambar ulang pada frame kamera terbaru.
    """
    inference_interval = 1.0 / float(target_inference_fps)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-webrtc")

    state = {
        "last_submit_at": 0.0,
        "inference_running": False,
        "detections": [],
        "inference_ms": 0.0,
        "last_error": None,
    }
    state_lock = threading.Lock()

    def run_inference(frame_bgr):
        started = time.perf_counter()
        try:
            with model_lock:
                results = model.predict(
                    source=frame_bgr,
                    conf=conf,
                    imgsz=512,
                    device=INFERENCE_DEVICE,
                    half=USE_HALF,
                    max_det=50,
                    verbose=False,
                )

            detections = []
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    detections.append((x1, y1, x2, y2, cls_id, confidence))

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with state_lock:
                state["detections"] = detections
                state["inference_ms"] = elapsed_ms
                state["last_error"] = None
        except Exception as exc:
            with state_lock:
                state["last_error"] = str(exc)
        finally:
            with state_lock:
                state["inference_running"] = False

    def video_frame_callback(frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        now = time.perf_counter()

        # Submit maksimal sesuai target FPS dan hanya jika worker sedang kosong.
        # frame.copy() diperlukan karena worker berjalan setelah callback selesai.
        with state_lock:
            can_submit = (
                not state["inference_running"]
                and now - state["last_submit_at"] >= inference_interval
            )
            if can_submit:
                state["inference_running"] = True
                state["last_submit_at"] = now

        if can_submit:
            executor.submit(run_inference, image_bgr.copy())

        # Callback tidak menunggu inferensi: frame kamera terbaru langsung dipakai.
        output = image_bgr.copy()
        with state_lock:
            detections = list(state["detections"])
            inference_ms = state["inference_ms"]
            inference_running = state["inference_running"]
            last_error = state["last_error"]

        for x1, y1, x2, y2, cls_id, confidence in detections:
            label = f"{model.names[cls_id]} {confidence:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y = max(y1, text_h + 8)
            cv2.rectangle(
                output,
                (x1, label_y - text_h - 8),
                (x1 + text_w + 8, label_y),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                output,
                label,
                (x1 + 4, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        status = "processing" if inference_running else "ready"
        info_text = (
            f"YOLO 512 | Target: {target_inference_fps} inferensi/detik | "
            f"{inference_ms:.0f} ms | {status}"
        )
        cv2.putText(
            output,
            info_text,
            (12, output.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if last_error:
            cv2.putText(
                output,
                "Inferensi gagal - video tetap berjalan",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(output, format="bgr24")

    return video_frame_callback


def show_realtime_detection(conf, target_inference_fps):
    section_heading("Deteksi Real-Time melalui WebRTC")
    st.caption(
        "Video dikirim pada setiap frame, sementara YOLO berjalan secara "
        f"non-blocking ±{target_inference_fps}×/detik pada resolusi inferensi "
        "512 piksel. Bounding box hasil deteksi digambar ulang di atas frame terbaru."
    )

    video_callback = create_video_frame_callback(
        conf=conf,
        target_inference_fps=target_inference_fps,
    )

    webrtc_ctx = webrtc_streamer(
        key="jalan-realtime-512",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 24, "max": 30},
                "facingMode": "environment",
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                    ]
                }
            ]
        },
        async_processing=False,
    )

    if webrtc_ctx.state.playing:
        st.success("Kamera aktif dan deteksi real-time sedang berjalan.")
    else:
        st.info("Kamera belum aktif. Tekan **START** lalu izinkan akses kamera.")


# =========================
# UI STREAMLIT
# =========================
render_hero()

device_label = (
    f"GPU: {torch.cuda.get_device_name(0)}"
    if torch.cuda.is_available()
    else "CPU"
)

# Parameter dikunci pada nilai optimal hasil evaluasi model.
# conf = 0.423 adalah titik F1-score terbaik pada kurva F1-Confidence validasi.
conf = 0.423
imgsz = 512
target_inference_fps = 2

# ---------- Sidebar: Tentang Model ----------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Tentang Model</div>', unsafe_allow_html=True)
    st.markdown('<span class="status-pill"><span class="status-dot"></span>Model aktif & siap</span>', unsafe_allow_html=True)

    try:
        n_params = sum(p.numel() for p in model.model.parameters())
        param_text = f"{n_params / 1e6:.2f} juta"
    except Exception:
        param_text = "±2.58 juta"

    spec_items = [
        ("Arsitektur", "YOLOv11n (nano)"),
        ("Parameter", param_text),
        ("Komputasi", "±6.3 GFLOPs"),
        ("Jumlah kelas", "4"),
        ("Ukuran input", "512 × 512 px"),
        ("Perangkat", device_label),
        ("Conf. threshold", f"{conf:.3f}"),
        ("Epoch training", "50"),
    ]
    spec_html = "".join(
        f'<div class="spec-row"><span class="spec-label">{label}</span>'
        f'<span class="spec-value">{value}</span></div>'
        for label, value in spec_items
    )
    st.markdown(f'<div>{spec_html}</div>', unsafe_allow_html=True)

    st.caption(
        "Kelas: Lubang • Retak Buaya • Memanjang • Melintang"
    )

# ---------- Pilih metode input ----------
section_heading("Metode Input")

input_method = st.pills(
    "Pilih metode input:",
    ["Upload Gambar", "Ambil Foto dari Kamera", "Deteksi Real-Time (WebRTC)"],
    selection_mode="single",
    default="Upload Gambar",
    label_visibility="collapsed",
)

st.markdown("")

if input_method == "Upload Gambar":
    uploaded_image = st.file_uploader(
        "Pilih gambar JPG/PNG",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_image is not None:
        process_image(
            uploaded_image.getvalue(),
            "Upload Gambar",
            conf,
            imgsz
        )
    else:
        st.info(
            "**Belum ada gambar.** Upload foto permukaan jalan untuk memulai "
            "deteksi — hasil berupa bounding box, tabel detail, dan lokasi foto."
        )

elif input_method == "Ambil Foto dari Kamera":
    camera_image = st.camera_input(
        "Ambil foto jalan menggunakan kamera"
    )

    if camera_image is not None:
        process_image(
            camera_image.getvalue(),
            "Kamera",
            conf,
            imgsz
        )
    else:
        st.info("Arahkan kamera ke permukaan jalan, lalu ambil foto.")

else:
    show_realtime_detection(
        conf=conf,
        target_inference_fps=target_inference_fps,
    )

st.divider()
st.caption(
    "Sistem Deteksi Kerusakan Jalan • YOLOv11 • "
    "Confidence threshold dikunci 0.423 (titik F1 terbaik hasil validasi)."
)
