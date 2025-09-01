import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

st.title("🎨 Deteksi Warna Dominan Real-time")

# Pilih mode
mode = st.radio("Pilih mode input:", ["Kamera Real-time", "Upload Foto"])

def get_dominant_color(image_array):
    """Mendapatkan warna dominan dari array gambar"""
    # Reshape gambar menjadi array 2D
    img = image_array.reshape((-1, 3))
    img = np.float32(img)
    
    # Kriteria untuk k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 1
    
    # K-means clustering untuk mendapatkan warna dominan
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = centers[0].astype(int)
    
    return tuple(dominant_color)

def draw_detection_box(image, x, y, width, height, color_bgr, color_name=""):
    """Menggambar bounding box pada gambar"""
    # Konversi ke BGR untuk OpenCV
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Gambar rectangle
    cv2.rectangle(image_bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Tambahkan teks informasi warna
    text = f"RGB: {color_bgr[2]},{color_bgr[1]},{color_bgr[0]}"
    if color_name:
        text = f"{color_name} - {text}"
    
    # Background untuk teks
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image_bgr, (x, y - text_height - 10), (x + text_width, y), (0, 0, 0), -1)
    
    # Teks
    cv2.putText(image_bgr, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Konversi kembali ke RGB
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def get_color_name(bgr_color):
    """Mendapatkan nama warna berdasarkan nilai BGR"""
    b, g, r = bgr_color
    
    # Definisi warna dasar
    if r > 200 and g < 100 and b < 100:
        return "Merah"
    elif g > 200 and r < 100 and b < 100:
        return "Hijau"
    elif b > 200 and r < 100 and g < 100:
        return "Biru"
    elif r > 200 and g > 200 and b < 100:
        return "Kuning"
    elif r > 200 and g < 100 and b > 200:
        return "Magenta"
    elif r < 100 and g > 200 and b > 200:
        return "Cyan"
    elif r > 200 and g > 200 and b > 200:
        return "Putih"
    elif r < 50 and g < 50 and b < 50:
        return "Hitam"
    elif 100 < r < 200 and 100 < g < 200 and 100 < b < 200:
        return "Abu-abu"
    elif r > 150 and g > 100 and b < 100:
        return "Oranye"
    elif r > 100 and g < 100 and b > 100:
        return "Ungu"
    elif r > 150 and g > 100 and b > 100:
        return "Pink"
    else:
        return "Campuran"

if mode == "Kamera Real-time":
    st.markdown("### 📹 Mode Kamera Real-time")
    st.info("Aplikasi akan mendeteksi warna dominan dari area yang dipilih secara real-time")
    
    # Settings untuk deteksi
    col1, col2 = st.columns(2)
    with col1:
        box_size = st.slider("Ukuran Box Deteksi", 50, 200, 100)
    with col2:
        update_interval = st.slider("Interval Update (detik)", 0.1, 2.0, 0.5)
    
    # Placeholder untuk kamera dan hasil
    camera_placeholder = st.empty()
    result_placeholder = st.empty()
    color_placeholder = st.empty()
    
    # Tombol kontrol
    col1, col2 = st.columns(2)
    with col1:
        start_detection = st.button("🎯 Mulai Deteksi")
    with col2:
        stop_detection = st.button("⏹️ Stop Deteksi")
    
    # Session state untuk kontrol
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    
    if start_detection:
        st.session_state.detection_active = True
    
    if stop_detection:
        st.session_state.detection_active = False
    
    if st.session_state.detection_active:
        try:
            # Ambil gambar dari kamera
            img_file = st.camera_input("📸 Kamera", key="realtime_camera")
            
            if img_file is not None:
                # Baca gambar
                img = Image.open(img_file)
                img_array = np.array(img)
                
                # Hitung posisi box di tengah gambar
                height, width = img_array.shape[:2]
                x = (width - box_size) // 2
                y = (height - box_size) // 2
                
                # Pastikan box tidak keluar dari batas gambar
                x = max(0, min(x, width - box_size))
                y = max(0, min(y, height - box_size))
                
                # Ekstrak area dalam box
                box_area = img_array[y:y+box_size, x:x+box_size]
                
                # Deteksi warna dominan dari area box
                dominant_color_bgr = get_dominant_color(box_area)
                color_name = get_color_name(dominant_color_bgr)
                
                # Gambar box deteksi pada gambar
                img_with_box = draw_detection_box(img, x, y, box_size, box_size, 
                                                dominant_color_bgr, color_name)
                
                # Tampilkan gambar dengan box
                camera_placeholder.image(img_with_box, caption="Live Detection", use_column_width=True)
                
                # Tampilkan hasil deteksi
                with result_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Warna Terdeteksi", color_name)
                    with col2:
                        st.metric("RGB", f"{dominant_color_bgr[2]}, {dominant_color_bgr[1]}, {dominant_color_bgr[0]}")
                    with col3:
                        st.metric("Hex", f"#{dominant_color_bgr[2]:02x}{dominant_color_bgr[1]:02x}{dominant_color_bgr[0]:02x}")
                
                # Preview warna
                hex_color = f"#{dominant_color_bgr[2]:02x}{dominant_color_bgr[1]:02x}{dominant_color_bgr[0]:02x}"
                color_placeholder.color_picker("🎨 Preview Warna", value=hex_color, key=f"color_{time.time()}")
                
                # Auto refresh
                time.sleep(update_interval)
                st.rerun()
                
        except Exception as e:
            st.error(f"Error dalam deteksi real-time: {str(e)}")
            st.warning("Pastikan kamera tersedia dan izin akses kamera telah diberikan.")

elif mode == "Upload Foto":
    st.markdown("### 📁 Mode Upload Foto")
    
    # Settings untuk deteksi area
    use_custom_area = st.checkbox("Gunakan Area Deteksi Kustom")
    
    uploaded_file = st.file_uploader("Upload foto", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        st.image(img, caption="Foto yang diupload", use_column_width=True)
        
        if use_custom_area:
            st.markdown("#### Pilih Area Deteksi")
            col1, col2 = st.columns(2)
            with col1:
                x_start = st.slider("X Start", 0, width-50, 0)
                y_start = st.slider("Y Start", 0, height-50, 0)
            with col2:
                box_width = st.slider("Lebar Box", 50, min(width-x_start, 300), 100)
                box_height = st.slider("Tinggi Box", 50, min(height-y_start, 300), 100)
            
            # Ekstrak area yang dipilih
            selected_area = img_array[y_start:y_start+box_height, x_start:x_start+box_width]
        else:
            # Gunakan seluruh gambar
            selected_area = img_array
            x_start, y_start = 0, 0
            box_width, box_height = width, height
        
        # Deteksi warna dominan
        if st.button("🔍 Analisis Warna"):
            with st.spinner("Menganalisis warna..."):
                dominant_color_bgr = get_dominant_color(selected_area)
                color_name = get_color_name(dominant_color_bgr)
                
                # Gambar box jika menggunakan area kustom
                if use_custom_area:
                    img_with_box = draw_detection_box(img, x_start, y_start, 
                                                    box_width, box_height, 
                                                    dominant_color_bgr, color_name)
                    st.image(img_with_box, caption="Area Deteksi", use_column_width=True)
                
                # Tampilkan hasil
                st.success("✅ Analisis selesai!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎨 Warna Dominan", color_name)
                with col2:
                    st.metric("🔢 Nilai RGB", f"{dominant_color_bgr[2]}, {dominant_color_bgr[1]}, {dominant_color_bgr[0]}")
                with col3:
                    hex_color = f"#{dominant_color_bgr[2]:02x}{dominant_color_bgr[1]:02x}{dominant_color_bgr[0]:02x}"
                    st.metric("📝 Kode Hex", hex_color)
                
                # Preview warna dengan ukuran lebih besar
                st.markdown("#### 🎨 Preview Warna")
                st.color_picker("Warna terdeteksi", value=hex_color, disabled=True)
                
                # Informasi tambahan
                with st.expander("📊 Informasi Detail"):
                    st.write(f"**Dimensi Gambar:** {width} x {height} pixels")
                    if use_custom_area:
                        st.write(f"**Area Deteksi:** {box_width} x {box_height} pixels")
                        st.write(f"**Posisi Box:** ({x_start}, {y_start})")
                    st.write(f"**Total Pixel Dianalisis:** {selected_area.shape[0] * selected_area.shape[1]:,}")

# Sidebar dengan informasi
with st.sidebar:
    st.markdown("### ℹ️ Informasi Aplikasi")
    st.write("""
    **Fitur:**
    - ✅ Deteksi real-time dari kamera
    - ✅ Upload dan analisis foto
    - ✅ Bounding box untuk area deteksi
    - ✅ Preview warna hasil
    - ✅ Nama warna dalam Bahasa Indonesia
    
    **Cara Penggunaan:**
    1. Pilih mode (Kamera/Upload)
    2. Untuk real-time: Klik 'Mulai Deteksi'
    3. Untuk upload: Pilih foto dan klik 'Analisis'
    
    **Tips:**
    - Pastikan pencahayaan yang baik
    - Area deteksi sebaiknya tidak terlalu kecil
    - Gunakan objek dengan warna yang kontras
    """)
    
    st.markdown("### 🎯 Pengaturan")
    if mode == "Kamera Real-time":
        st.info("Gunakan pengaturan di atas untuk mengatur ukuran box dan interval update.")
    else:
        st.info("Centang 'Area Deteksi Kustom' untuk memilih area spesifik.")
