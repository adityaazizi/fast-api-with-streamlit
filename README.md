# YOLO11n Object Detection API

Aplikasi deteksi objek dengan YOLOv8 yang terdiri dari backend FastAPI dan frontend Streamlit.

## 📋 Fitur

- 🔍 Deteksi objek menggunakan model YOLO11
- 🖼️ Upload dan proses multiple gambar
- ⚙️ Konfigurasi parameter deteksi (confidence, IoU threshold, image size)
- 📊 Visualisasi hasil deteksi dengan bounding box dan label
- 🚀 API endpoint untuk integrasi dengan aplikasi lain
- 🎛️ Interface Streamlit yang user-friendly

## 🏗️ Struktur Proyek

```
FASTAPI/
├── src/
│   ├── __pycache__/
│   ├── engine/
│   │   ├── __pycache__/
│   │   └── object_detection.py    # Kelas untuk deteksi objek dengan YOLOv8
│   ├── schema/
│   │   ├── __pycache__/
│   │   └── response.py            # Model Pydantic untuk respons API
│   ├── app.py                     # Aplikasi FastAPI utama
│   ├── object_detection.py        # Implementasi modul deteksi objek
│   ├── streamlit.py               # Aplikasi Streamlit untuk frontend
│   └── utils.py                   # Fungsi utilitas untuk anotasi gambar
├── venv/                          # Virtual environment Python
├── weight/                        # Direktori untuk model YOLOv8
├── .python-version                # Versi Python yang digunakan
├── docker-compose.yaml            # Konfigurasi Docker Compose
├── Dockerfile                     # Instruksi build Docker image
├── poetry.lock                    # Lock file dependensi Poetry
├── pyproject.toml                 # Konfigurasi proyek Python
└── requirements.txt               # Daftar dependensi Python
```

## 🛠️ Teknologi yang Digunakan

- [FastAPI](https://fastapi.tiangolo.com/): Framework API modern dan berkinerja tinggi
- [Streamlit](https://streamlit.io/): Framework untuk membuat aplikasi data dengan cepat
- [YOLOv8](https://github.com/ultralytics/ultralytics): Model deteksi objek state-of-the-art
- [OpenCV](https://opencv.org/): Library untuk pemrosesan gambar
- [Pydantic](https://pydantic-docs.helpmanual.io/): Library validasi data
- [Docker](https://www.docker.com/): Untuk kontainerisasi aplikasi

## 🚀 Cara Menjalankan

### Menggunakan Docker (Direkomendasikan)

1. Pastikan Docker dan Docker Compose terinstal
2. Clone repository ini
3. Jalankan aplikasi dengan Docker Compose:

```bash
docker-compose up --build
```

4. Akses aplikasi:
   - FastAPI: http://localhost:8080
   - Swagger UI: http://localhost:8080/docs
   - Streamlit UI: http://localhost:8501

### Pengembangan Lokal

1. Pastikan Python 3.10 terinstal
2. Clone repository ini
3. Buat dan aktifkan virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Untuk Unix/Linux
venv\Scripts\activate  # Untuk Windows
```

4. Instal dependensi:

```bash
pip install -r requirements.txt
```

5. Jalankan aplikasi:

```bash
# Terminal 1 - FastAPI
python src/app.py

# Terminal 2 - Streamlit
streamlit run src/streamlit.py
```

## 🔧 Parameter Konfigurasi

Aplikasi ini mendukung beberapa parameter konfigurasi untuk penyesuaian prediksi:

| Parameter | Deskripsi                     | Default | Range    |
| --------- | ----------------------------- | ------- | -------- |
| imgsz     | Ukuran gambar untuk inferensi | 640     | 320-1280 |
| conf      | Threshold confidence          | 0.25    | 0.0-1.0  |
| iou       | Threshold IoU untuk NMS       | 0.45    | 0.0-1.0  |

## 📦 API Endpoints

### POST /predict

Endpoint untuk mendeteksi objek dalam gambar.

**Request:**

- Content-Type: multipart/form-data
- Body:
  - images: List file gambar (jpg, jpeg, png)
  - imgsz: Ukuran gambar (integer)
  - conf: Threshold confidence (float)
  - iou: Threshold IoU (float)

**Response:**

```json
{
  "success": true,
  "message": "Inference complete.",
  "total_images": 1,
  "data": [
    [
      {
        "class_id": 0,
        "name": "person",
        "confidence": 0.95,
        "xcenter": 320.5,
        "ycenter": 250.2,
        "width": 50.0,
        "height": 120.0
      }
    ]
  ],
  "speed": [
    {
      "preprocess": 12.3,
      "inference": 45.6,
      "postprocess": 7.8
    }
  ],
  "images": ["base64_encoded_image_string"]
}
```

## 📝 Catatan

- Model YOLO11 diharapkan berada di direktori `weight/` dengan nama file `yolo11n.pt` (atau sesuaikan path di `app.py`)
- Pastikan port 8080 dan 8501 tersedia saat menjalankan aplikasi

## 🔄 Pengembangan Lebih Lanjut

Beberapa ide untuk pengembangan lebih lanjut:

1. Tambahkan dukungan untuk video streaming
2. Implementasikan caching untuk hasil deteksi
3. Tambahkan dukungan untuk lebih banyak model
4. Tingkatkan UI dengan fitur analitik lebih lanjut
5. Tambahkan autentikasi untuk API
6. Implementasikan logging dan monitoring
7. Tambahkan unit test dan integration test

## 📄 Lisensi

[MIT](LICENSE)
