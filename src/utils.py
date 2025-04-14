# Import library yang diperlukan
import cv2                  # OpenCV untuk manipulasi dan anotasi gambar
import numpy as np          # NumPy untuk operasi array dan matriks
from PIL import Image       # Pillow untuk operasi gambar
from io import BytesIO      # BytesIO untuk operasi I/O dalam memori
import base64               # Base64 untuk encoding gambar


def draw_boxes_on_image(image: Image.Image, detections: list) -> bytes:
    """
    Fungsi untuk menggambar kotak (bounding box) hasil deteksi pada gambar.
    
    Args:
        image (Image.Image): Gambar PIL yang akan dianotasi
        detections (list): Daftar deteksi objek, masing-masing berisi koordinat, 
                          nama kelas, dan confidence
    
    Returns:
        bytes: String base64 dari gambar yang telah dianotasi
    """
    # Konversi PIL Image ke format OpenCV (numpy array)
    img_array = np.array(image)
    # OpenCV menggunakan format BGR, sedangkan PIL menggunakan RGB
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # Dapatkan dimensi gambar untuk perhitungan koordinat
    img_h, img_w = img_cv.shape[:2]

    # Iterasi melalui setiap deteksi objek
    for det in detections:
        # Ekstrak informasi dari deteksi
        xcenter = det["xcenter"]  # Koordinat x pusat objek
        ycenter = det["ycenter"]  # Koordinat y pusat objek
        width = det["width"]      # Lebar objek
        height = det["height"]    # Tinggi objek
        class_name = det["name"]  # Nama kelas objek terdeteksi
        confidence = det["confidence"]  # Nilai confidence deteksi (0-1)

        # Hitung koordinat sudut kotak berdasarkan pusat dan dimensi
        x1 = int(xcenter - width / 2)   # Koordinat x sudut kiri atas
        y1 = int(ycenter - height / 2)  # Koordinat y sudut kiri atas
        x2 = int(xcenter + width / 2)   # Koordinat x sudut kanan bawah
        y2 = int(ycenter + height / 2)  # Koordinat y sudut kanan bawah
            
        # Warna untuk bounding box - BGR format (Biru tua)
        box_color = (255, 100, 0)  
        
        # Ketebalan garis kotak bersifat dinamis berdasarkan ukuran kotak
        thickness = max(2, int((x2-x1) * 0.005))
        # Gambar kotak pada gambar
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), box_color, thickness)
        
        # Buat label dengan nama kelas dan nilai confidence
        label = f"{class_name}: {confidence*100:.1f}%"
        
        # Posisikan label DI ATAS kotak
        label_y = y1 - 10  # 10 piksel di atas kotak
        
        # Dapatkan ukuran teks untuk menentukan ukuran background
        font_scale = 3
        font_thickness = 3
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Pastikan label tetap terlihat jika berada di bagian atas gambar
        if label_y < label_height + 10:
            label_y = y1 + 30  # Letakkan label di dalam kotak bagian atas
        
        # Gambar background label (persegi panjang hitam)
        cv2.rectangle(
            img_cv, 
            (x1, label_y - label_height - 5),  # Koordinat sudut kiri atas background
            (x1 + label_width + 10, label_y + 5),  # Koordinat sudut kanan bawah
            (0, 0, 0),  # Warna hitam untuk kontras maksimum
            -1  # -1 berarti persegi panjang terisi (filled)
        )
        
        # Gambar border putih di sekitar background label untuk visibilitas ekstra
        cv2.rectangle(
            img_cv, 
            (x1, label_y - label_height - 5), 
            (x1 + label_width + 10, label_y + 5),
            (255, 255, 255),  # Warna putih untuk border
            3  # Ketebalan border
        )
        
        # Gambar teks label
        cv2.putText(
            img_cv, 
            label, 
            (x1 + 5, label_y),  # Posisikan teks di dalam background
            cv2.FONT_HERSHEY_SIMPLEX,  # Jenis font
            font_scale,  # Skala ukuran font
            (255, 255, 255),  # Warna putih untuk teks
            font_thickness  # Ketebalan font
        )
        
    # Konversi kembali ke format PIL Image
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB
    pil_img = Image.fromarray(img_rgb)  # Konversi array NumPy ke PIL Image

    # Encode gambar ke format base64 untuk dikirim ke frontend
    buffered = BytesIO()  # Buffer memory untuk menyimpan gambar
    pil_img.save(buffered, format="PNG")  # Simpan gambar sebagai PNG
    # Encode buffer ke base64 dan konversi ke string UTF-8
    return base64.b64encode(buffered.getvalue()).decode("utf-8")