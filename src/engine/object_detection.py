from ultralytics import YOLO         # Library utama untuk model YOLO11
from PIL import Image                # Untuk manipulasi gambar
from io import BytesIO               # Untuk operasi I/O dalam memori
from typing import List              # Type hint untuk list
from utils import draw_boxes_on_image  # Fungsi untuk menggambar kotak deteksi


class ObjectDetector:
    """
    Kelas untuk mendeteksi objek dalam gambar menggunakan model YOLO11.
    
    Kelas ini menyediakan antarmuka untuk memuat model YOLO,
    melakukan prediksi pada gambar, dan memformat hasil deteksi.
    """
    
    def __init__(self, model_path: str):
        """
        Inisialisasi detektor objek dengan model YOLO11.
        
        Args:
            model_path (str): Path ke file model YOLO (.pt)
        """
        # Memuat model YOLO dari file
        self.model = YOLO(model_path)
        # Menyimpan dictionary nama kelas (id: nama)
        self.names = self.model.names

    async def predict_images(self, image_files: List, imgsz: int, conf: float, iou: float):
        """
        Melakukan prediksi deteksi objek pada daftar gambar.
        
        Args:
            image_files (List): Daftar file gambar untuk diproses
            imgsz (int): Ukuran gambar untuk inferensi
            conf (float): Threshold confidence untuk deteksi (0-1)
            iou (float): Threshold IoU untuk Non-Maximum Suppression (0-1)
            
        Returns:
            dict: Dictionary hasil deteksi dengan format:
                {
                    "success": bool,
                    "message": str,
                    "total_images": int,
                    "data": List[List[dict]],  # Deteksi untuk setiap gambar
                    "speed": List[dict],       # Statistik kecepatan
                    "images": List[str]        # Gambar teranotasi dalam format base64
                }
        """
        all_detections = []      # Menyimpan deteksi untuk semua gambar
        annotated_images = []    # Menyimpan gambar yang sudah dianotasi
        speeds = []              # Menyimpan statistik kecepatan untuk setiap gambar

        # Iterasi melalui setiap file gambar yang diunggah
        for image_file in image_files:
            # Membaca konten file
            content = await image_file.read()
            # Konversi ke objek PIL Image dan pastikan dalam format RGB
            image = Image.open(BytesIO(content)).convert("RGB")

            # Jalankan prediksi model YOLO pada gambar
            results = self.model.predict(
                image,              # Gambar input
                imgsz=imgsz,        # Ukuran gambar untuk inferensi
                conf=conf,          # Threshold confidence
                iou=iou,            # Threshold IoU untuk NMS
                verbose=False       # Nonaktifkan output verbose
            )

            # Parse hasil prediksi ke format yang lebih mudah digunakan
            detections = self._parse_results(results)
            all_detections.append(detections)

            # Gambar kotak dan label pada gambar asli
            annotated_img = draw_boxes_on_image(image, detections)
            annotated_images.append(annotated_img)

            # Simpan statistik kecepatan dari hasil YOLO
            # results[0].speed berisi kecepatan dalam format:
            # {'preprocess': ms, 'inference': ms, 'postprocess': ms}
            speeds.append(results[0].speed)

        # Kembalikan hasil dalam format yang terstruktur
        return {
            "success": True,
            "message": "Inference complete.",
            "total_images": len(image_files),
            "data": all_detections,         # Daftar deteksi untuk setiap gambar
            "speed": speeds,                # Statistik kecepatan untuk setiap gambar
            "images": annotated_images      # Gambar teranotasi dalam format base64
        }

    def _parse_results(self, results):
        """
        Mengkonversi hasil prediksi YOLO ke format yang lebih sesuai untuk respons API.
        
        Args:
            results: Hasil dari model.predict()
            
        Returns:
            list: Daftar deteksi, masing-masing berisi informasi kelas, 
                 confidence, dan koordinat
        """
        detections = []

        # Iterasi melalui setiap hasil prediksi
        for r in results:
            # Iterasi melalui setiap bounding box
            for box in r.boxes:
                # Dapatkan ID kelas dan konversi ke integer
                class_id = int(box.cls[0])
                # Dapatkan nama kelas dari dictionary names
                name = self.names[class_id]
                
                # Tambahkan informasi deteksi ke dalam daftar
                detections.append({
                    "class_id": class_id,                # ID kelas numerik
                    "name": name,                        # Nama kelas string
                    "confidence": float(box.conf[0]),    # Nilai confidence (0-1)
                    # Koordinat x pusat kotak
                    "xcenter": float(box.xywh[0][0]),
                    # Koordinat y pusat kotak
                    "ycenter": float(box.xywh[0][1]),
                    # Lebar kotak
                    "width": float(box.xywh[0][2]),
                    # Tinggi kotak
                    "height": float(box.xywh[0][3])
                })

        return detections