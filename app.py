import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from PIL import Image
import os

# =====================================
# Konfigurasi Halaman (Mobile Responsive)
# =====================================
st.set_page_config(
    page_title="Survey Jalan Akurat", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Tambahan agar tampilan lebih rapi di HP
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .main { padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🕳️ Survey Lubang Jalan")

# =====================================
# Load Model
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

with st.spinner("🔄 Loading model YOLO..."):
    model = load_model()
    class_names = model.names

st.success("✅ Model berhasil dimuat!")

# =====================================
# Session State
# =====================================
if 'master_ids' not in st.session_state:
    st.session_state['master_ids'] = set()
if 'total_count' not in st.session_state:
    st.session_state['total_count'] = 0

# =====================================
# Sidebar & Input Selection
# =====================================
st.sidebar.header("🎛️ Kontrol & Input")

# Info deployment
st.sidebar.info("🚂 Running on Railway")

input_type = st.sidebar.selectbox(
    "Pilih Metode Input", 
    ["Foto", "Video Upload"]
)

conf_threshold = st.sidebar.slider(
    "Sensitivity (Confidence)", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.3,
    step=0.05
)

if st.sidebar.button("🔄 Reset Counter / Survey Baru"):
    st.session_state['master_ids'] = set()
    st.session_state['total_count'] = 0
    st.rerun()

# Info Session
st.sidebar.markdown("---")
st.sidebar.metric("Total Terdeteksi (Session)", st.session_state['total_count'])

# =====================================
# Fungsi Gambar (Helper)
# =====================================
def draw_on_frame(frame, results, is_video=False):
    """Menggambar bounding box dan label pada frame"""
    current_ids = []
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].boxes.conf.cpu().numpy()
        
        # Ambil ID jika mode tracking aktif (Video)
        ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else [None]*len(boxes)
        
        for obj_id, box, cls, conf in zip(ids, boxes, clss, confs):
            if is_video and obj_id is not None:
                if obj_id not in st.session_state['master_ids']:
                    st.session_state['master_ids'].add(obj_id)
                    st.session_state['total_count'] = len(st.session_state['master_ids'])
                current_ids.append(obj_id)
                label = f"ID:{obj_id} {class_names[cls]} {conf:.2f}"
            else:
                label = f"{class_names[cls]} {conf:.2f}"
            
            x1, y1, x2, y2 = map(int, box)
            
            # Gambar rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Background untuk text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0], y1), (0, 255, 0), -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
    return frame, len(current_ids)

# =====================================
# Logika Utama Berdasarkan Input
# =====================================

# PLACEHOLDERS UNTUK METRIK (Mobile Friendly)
col1, col2 = st.columns(2)
metric_total = col1.empty()
metric_status = col2.empty()
frame_window = st.empty()

# 1. FOTO
if input_type == "Foto":
    st.markdown("### 📸 Upload Foto Jalan")
    uploaded_file = st.file_uploader(
        "Pilih foto (JPG, PNG)", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload foto jalan yang akan dianalisis"
    )
    
    if uploaded_file:
        with st.spinner("🔍 Menganalisis foto..."):
            # Baca image
            img = Image.open(uploaded_file)
            frame = np.array(img)
            
            # Convert RGB ke BGR untuk OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Predict
            results = model.predict(frame_bgr, conf=conf_threshold, verbose=False)
            
            # Draw boxes
            processed_frame, detected_count = draw_on_frame(frame_bgr, results, is_video=False)
            
            # Convert BGR back to RGB untuk display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display
            frame_window.image(processed_frame_rgb, use_container_width=True, caption="Hasil Deteksi")
            
            # Metrics
            metric_total.metric("🕳️ Lubang Terdeteksi", detected_count)
            
            if detected_count == 0:
                st.info("✅ Tidak ada lubang jalan terdeteksi dalam foto ini.")
            else:
                st.warning(f"⚠️ Ditemukan {detected_count} lubang jalan!")
                
                # Detail detections
                with st.expander("📊 Detail Deteksi"):
                    for i, box in enumerate(results[0].boxes):
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"**Objek {i+1}:** {class_names[cls]} (Confidence: {conf:.2%})")

# 2. VIDEO UPLOAD
elif input_type == "Video Upload":
    st.markdown("### 🎥 Upload Video Jalan")
    uploaded_file = st.file_uploader(
        "Pilih video (MP4, AVI, MOV)", 
        type=['mp4', 'avi', 'mov'],
        help="Upload video survey jalan"
    )
    
    if uploaded_file:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Open video
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.info(f"📹 Video: {total_frames} frames @ {fps} FPS")
        
        frame_count = 0
        process_every_n_frames = 2  # Process setiap 2 frame untuk performa lebih baik
        
        # Stop button
        stop_button = st.button("⏹️ Stop Processing")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process hanya setiap N frames
            if frame_count % process_every_n_frames == 0:
                # Track dengan YOLO
                results = model.track(
                    frame, 
                    conf=conf_threshold, 
                    persist=True, 
                    tracker="botsort.yaml", 
                    verbose=False
                )
                
                # Draw boxes
                processed_frame, cur_count = draw_on_frame(frame, results, is_video=True)
                
                # Display
                frame_window.image(processed_frame, channels="BGR", use_container_width=True)
                
                # Update metrics
                metric_total.metric("📊 Total Survey", st.session_state['total_count'])
                metric_status.metric("🎞️ Frame Ini", cur_count)
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        # Clean up temp file
        os.unlink(tfile.name)
        
        st.success(f"✅ Video selesai diproses! Total lubang terdeteksi: **{st.session_state['total_count']}**")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>🚂 Powered by Railway | 🤖 YOLO + Streamlit | 🇮🇩 Made in Indonesia</small>
    </div>
""", unsafe_allow_html=True)