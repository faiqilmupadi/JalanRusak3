import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from PIL import Image
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# =====================================
# Konfigurasi Halaman
# =====================================
st.set_page_config(
    page_title="Survey Jalan Akurat", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk tampilan mobile
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stApp { max-width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🕳️ Survey Lubang Jalan")

# =====================================
# Load Model (Cached)
# =====================================
@st.cache_resource
def load_model():
    try:
        model_path = "konfigurasi.pt"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file '{model_path}' tidak ditemukan!")
            st.stop()
        return YOLO(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model = load_model()
class_names = model.names

# =====================================
# Helper Function: Drawing
# =====================================
def draw_on_frame(frame, results, is_video=False):
    """Menggambar bounding box dan menghitung deteksi"""
    detected_in_frame = 0
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # ID Tracking (hanya untuk video upload)
        ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [None]*len(boxes)
        
        for obj_id, box, cls, conf in zip(ids, boxes, clss, confs):
            detected_in_frame += 1
            
            # Logika Counting untuk Video
            if is_video and obj_id is not None:
                if obj_id not in st.session_state['master_ids']:
                    st.session_state['master_ids'].add(obj_id)
                    st.session_state['total_count'] = len(st.session_state['master_ids'])
                label = f"ID:{obj_id} {class_names[cls]} {conf:.2f}"
            else:
                label = f"{class_names[cls]} {conf:.2f}"
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return frame, detected_in_frame

# =====================================
# WebRTC Video Processor (Untuk Webcam)
# =====================================
class PotholeTransformer(VideoTransformerBase):
    def __init__(self, conf_threshold):
        self.conf_threshold = conf_threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Predict tanpa tracking untuk performa webcam
        results = model.predict(img, conf=self.conf_threshold, verbose=False)
        processed_frame, _ = draw_on_frame(img, results, is_video=False)
        return processed_frame

# =====================================
# Sidebar & Session State
# =====================================
if 'master_ids' not in st.session_state: st.session_state['master_ids'] = set()
if 'total_count' not in st.session_state: st.session_state['total_count'] = 0

st.sidebar.header("🎛️ Kontrol & Input")

input_type = st.sidebar.selectbox(
    "Pilih Metode Input", 
    ["Webcam Real-time", "Foto", "Video Upload"]
)

conf_threshold = st.sidebar.slider(
    "Sensitivity (Confidence)", 0.1, 1.0, 0.3, 0.05
)

if st.sidebar.button("🔄 Reset Counter"):
    st.session_state['master_ids'] = set()
    st.session_state['total_count'] = 0
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.metric("Total Unik (Session)", st.session_state['total_count'])

# Placeholder Utama
col1, col2 = st.columns(2)
metric_total = col1.empty()
metric_status = col2.empty()
frame_window = st.empty()

# =====================================
# LOGIKA INPUT
# =====================================

# 1. WEBCAM REAL-TIME (WebRTC)
if input_type == "Webcam Real-time":
    st.markdown("### 🤳 Live Survey (Kamera HP)")
    st.info("Gunakan kamera belakang untuk hasil survey yang lebih baik.")
    
    webrtc_streamer(
        key="pothole-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_processor_factory=lambda: PotholeTransformer(conf_threshold),
        media_stream_constraints={
            "video": {"facingMode": "environment"}, # Trigger kamera belakang HP
            "audio": False
        },
        async_processing=True,
    )

# 2. FOTO
elif input_type == "Foto":
    st.markdown("### 📸 Upload Foto")
    uploaded_file = st.file_uploader("Pilih foto", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model.predict(frame_bgr, conf=conf_threshold, verbose=False)
        processed_frame, detected_count = draw_on_frame(frame_bgr, results)
        
        frame_window.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        metric_total.metric("🕳️ Terdeteksi", detected_count)

# 3. VIDEO UPLOAD
elif input_type == "Video Upload":
    st.markdown("### 🎥 Upload Video")
    uploaded_file = st.file_uploader("Pilih video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Tracking diaktifkan untuk Video Upload
            results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
            processed_frame, cur_count = draw_on_frame(frame, results, is_video=True)
            
            frame_window.image(processed_frame, channels="BGR", use_container_width=True)
            metric_total.metric("📊 Total Survey", st.session_state['total_count'])
            metric_status.metric("🎞️ Frame Ini", cur_count)
            
            progress_bar.progress(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / total_frames)
            
        cap.release()
        os.unlink(tfile.name)
        st.success("✅ Pemrosesan video selesai!")

# Footer
st.markdown("---")
st.caption("🚂 Running on Railway/Cloud | 🤖 YOLO Powered")