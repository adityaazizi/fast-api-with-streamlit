from pydantic import BaseModel        # Library untuk validasi data dan serialisasi
from typing import List, Optional     # Type hints untuk tipe data kompleks


class Detection(BaseModel):
    """
    Model Pydantic untuk hasil deteksi objek tunggal.
    
    Mewakili satu objek terdeteksi dengan informasi tentang
    kelas, posisi, ukuran, dan nilai confidence.
    """
    class_id: int      # ID numerik kelas objek
    name: str          # Nama kelas objek (string)
    confidence: float  # Nilai confidence deteksi (0-1)
    xcenter: float     # Koordinat x pusat bounding box
    ycenter: float     # Koordinat y pusat bounding box
    width: float       # Lebar bounding box
    height: float      # Tinggi bounding box


class SpeedLog(BaseModel):
    """
    Model Pydantic untuk log kecepatan prediksi.
    
    Menyimpan waktu yang dibutuhkan untuk setiap tahap inferensi
    dalam milidetik.
    """
    preprocess: float   # Waktu preprocessing dalam milidetik
    inference: float    # Waktu inferensi model dalam milidetik
    postprocess: float  # Waktu postprocessing dalam milidetik


class InferenceResponse(BaseModel):
    """
    Model Pydantic untuk respons API endpoint prediksi.
    
    Menyediakan struktur respons yang konsisten termasuk status,
    hasil deteksi, statistik kecepatan, dan gambar teranotasi.
    """
    success: bool                # Status keberhasilan operasi
    message: str                 # Pesan status atau error
    total_images: int            # Jumlah gambar yang diproses
    data: List[List[Detection]]  # List deteksi untuk setiap gambar
    speed: List[SpeedLog]        # Statistik kecepatan untuk setiap gambar
    images: List[str]            # Gambar teranotasi dalam format base64