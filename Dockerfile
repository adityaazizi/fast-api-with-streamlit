# Base image - Menggunakan Python 3.10 versi slim untuk mengurangi ukuran image
FROM python:3.10-slim

# Set working directory - Tempat semua operasi berikutnya akan dijalankan
WORKDIR /app

# Install system dependencies for OpenCV
# libgl1-mesa-glx: Library untuk OpenGL yang dibutuhkan oleh OpenCV
# libglib2.0-0: Library C yang dibutuhkan untuk aplikasi Gnome/GTK+
# rm -rf: Membersihkan cache apt untuk mengurangi ukuran image
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file terlebih dahulu untuk memanfaatkan Docker cache layer
# Jika requirements.txt tidak berubah, layer ini tidak perlu di-rebuild
COPY requirements.txt .

# Install Python dependencies dari requirements.txt
# --no-cache-dir: Mengurangi ukuran image dengan tidak menyimpan cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi ke container
# Langkah ini diletakkan setelah instalasi dependencies untuk memanfaatkan Docker cache
COPY . .

# Expose port yang digunakan aplikasi
# 8080: Port untuk FastAPI
# 8501: Port default untuk Streamlit
EXPOSE 8080 8501

# Jalankan FastAPI dan Streamlit secara bersamaan
# Menggunakan shell command dengan & untuk menjalankan proses secara paralel
CMD ["sh", "-c", "python src/app.py & streamlit run src/streamlit.py"]