# Import library yang diperlukan
import streamlit as st              # Framework utama untuk UI web
import requests                     # Untuk melakukan HTTP requests ke API
import time                         # Untuk menghitung waktu eksekusi
import base64                       # Untuk decode/encode data gambar
from PIL import Image               # Untuk manipulasi gambar
import io                           # Untuk operasi input/output binary

# Definisi URL API endpoint untuk deteksi objek
API_URL = "http://localhost:8080/predict"

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="YOLO11 Object Detection", layout="wide")
st.title("📦 YOLO11 Object Detection Inference")
st.write("Upload image(s) and adjust the parameters for object detection.")

# --- Sidebar untuk parameter model ---
st.sidebar.header("🔧 Parameters")
# Parameter ukuran gambar (imgsz) - Menentukan resolusi input untuk model
imgsz = st.sidebar.slider("Image Size", 320, 1280, 640, step=32)
# Parameter confidence threshold - Menentukan batas minimal kepercayaan deteksi
conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, step=0.01)
# Parameter IoU threshold - Menentukan batas IoU untuk Non-Maximum Suppression
iou = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, step=0.01)

# --- File Uploader untuk input gambar ---
uploaded_files = st.file_uploader(
    "Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# --- Button untuk memulai inferensi ---
if st.button("🚀 Run Inference") and uploaded_files:
    with st.spinner("Running inference..."):  # Menampilkan spinner selama proses
        # Menyiapkan file untuk dikirim ke API
        files = [("images", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        # Menyiapkan parameter untuk dikirim ke API
        data = {
            "imgsz": str(imgsz),
            "conf": str(conf),
            "iou": str(iou)
        }

        try:
            # Mencatat waktu mulai untuk menghitung durasi inferensi
            start_time = time.time()
            # Mengirim request POST ke FastAPI server
            response = requests.post(API_URL, files=files, data=data)
            # Menghitung waktu yang dibutuhkan untuk inferensi
            elapsed = time.time() - start_time
            # Mengkonversi response JSON ke dict Python
            result = response.json()

            # Menampilkan info sukses dan waktu inferensi
            st.success(f"Inference completed in {elapsed:.2f} seconds.")
            st.write(f"Total Images: {result['total_images']}")
            
            # Iterasi melalui setiap gambar dan hasil deteksinya
            for i, (img_file, detections, speed, annotated_base64) in enumerate(zip(
                uploaded_files, result["data"], result["speed"], result["images"]
            )):
                # Menampilkan header untuk setiap gambar
                st.subheader(f"📸 Image {i + 1}: {img_file.name}")

                # --- Decode base64 string ke format gambar ---
                img_bytes = base64.b64decode(annotated_base64)

                # --- Menampilkan gambar yang telah dianotasi ---
                annotated_image = Image.open(io.BytesIO(img_bytes))
                st.image(annotated_image, caption="📍 Annotated Detection", width=600)

                # --- Menampilkan data deteksi dalam format JSON ---
                st.write("🧠 Detections:")
                st.json(detections)

                # --- Menampilkan informasi kecepatan proses ---
                st.write("⏱️ Speed (ms):")
                st.json(speed)

        except Exception as e:
            # Menampilkan pesan error jika terjadi masalah
            st.error(f"Error: {str(e)}")

# Pesan info jika gambar sudah diupload tapi belum diproses
elif uploaded_files:
    st.info("Klik tombol 🚀 Run Inference untuk mulai.")