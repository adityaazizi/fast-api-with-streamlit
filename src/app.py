# Import library yang diperlukan
import os                          # Untuk akses variabel lingkungan
import uvicorn                     # ASGI server untuk menjalankan FastAPI
import pyrootutils                 # Untuk setup path root project

from fastapi import FastAPI, UploadFile, File, Form  # Framework utama dan komponen untuk API
from fastapi.responses import JSONResponse           # Respon dalam format JSON
from typing import List                              # Untuk type hint List
from schema.response import InferenceResponse        # Schema response model
from engine.object_detection import ObjectDetector   # Class untuk deteksi objek


# Setup root directory project untuk memastikan import relatif bekerja dengan baik
# Ini membantu menemukan file-file dalam project tanpa masalah path
root = pyrootutils.setup_root(
    search_from=__file__,          # Mulai pencarian dari file ini
    indicator=[".git", "pyproject.toml", "src"],  # Indikator root directory
    pythonpath=True,               # Tambahkan root ke PYTHONPATH
    dotenv=True,                   # Load variabel dari .env file jika ada
)

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Inisialisasi model detector dengan file weight
# yolo11n.pt adalah file model YOLO11 nano yang telah dilatih
detector = ObjectDetector("weight/yolo11n.pt")

# Endpoint untuk prediksi/deteksi objek
@app.post("/predict", response_model=InferenceResponse)
async def predict(
    images: List[UploadFile] = File(...),  # Menerima multiple gambar
    imgsz: int = Form(640),                # Ukuran gambar input, default 640px
    conf: float = Form(0.25),              # Confidence threshold, default 0.25
    iou: float = Form(0.45)                # IoU threshold untuk NMS, default 0.45
):
    """
    Endpoint untuk melakukan deteksi objek pada gambar yang diupload.
    
    Args:
        images: Daftar file gambar yang diupload
        imgsz: Ukuran gambar untuk inferensi
        conf: Threshold confidence untuk deteksi (0-1)
        iou: Threshold IoU untuk Non-Maximum Suppression (0-1)
        
    Returns:
        JSONResponse dengan hasil deteksi objek
    """
    try:
        # Memanggil method predict_images dari detector
        results = await detector.predict_images(images, imgsz, conf, iou)
        return JSONResponse(content=results)
    except Exception as e:
        # Handling error dan mengembalikan response error dengan format yang konsisten
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": str(e),
            "total_images": 0,
            "data": [],            # Data deteksi kosong
            "speed": []            # Informasi kecepatan kosong
        })


# Entry point untuk menjalankan aplikasi
if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",                        # Path ke instance FastAPI
        host=os.getenv("HOST", "0.0.0.0"),  # Host dari env var atau default
        port=int(os.getenv("PORT", 8080)),    # Port dari env var atau default
        reload=True                           # Auto-reload saat kode berubah
    )