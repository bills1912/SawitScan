# 🌴 SawitScan - Sistem Deteksi Kematangan Kelapa Sawit

Aplikasi web untuk mendeteksi tingkat kematangan buah kelapa sawit menggunakan model YOLO dan FastAPI.

## 📁 Struktur Proyek

```
palm-detection-app/
├── backend/
│   ├── main.py              # API FastAPI
│   ├── requirements.txt     # Dependencies Python
│   └── best.pt              # Model YOLO (perlu diunduh)
├── frontend/
│   └── index.html           # Web interface
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Unduh Model YOLO

Unduh model dari link berikut dan simpan ke folder `backend/`:

```bash
cd backend
curl -L -o best.pt "https://github.com/bills1912/computer-vision/raw/refs/heads/main/best.pt"
```

Atau unduh manual dari:
https://github.com/bills1912/computer-vision/raw/refs/heads/main/best.pt

### 2. Setup Backend

```bash
# Masuk ke folder backend
cd backend

# Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Jalankan server
python main.py
# atau
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan berjalan di: http://localhost:8000

### 3. Jalankan Frontend

Buka file `frontend/index.html` di browser, atau gunakan server sederhana:

```bash
# Dari folder frontend
cd frontend
python -m http.server 3000
```

Kemudian buka: http://localhost:3000

## 📚 API Endpoints

| Endpoint | Method | Deskripsi |
|----------|--------|-----------|
| `/` | GET | Informasi API |
| `/health` | GET | Status kesehatan API |
| `/classes` | GET | Daftar kelas deteksi |
| `/detect` | POST | Upload gambar untuk deteksi |
| `/docs` | GET | Swagger UI Documentation |

### Contoh Request Deteksi

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@gambar_sawit.jpg"
```

### Contoh Response

```json
{
  "success": true,
  "filename": "gambar_sawit.jpg",
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "total_detections": 3,
  "detections": [
    {
      "class_id": 2,
      "class_name": "Matang (Ripe)",
      "confidence": 95.5,
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 300.7,
        "y2": 450.2
      },
      "description": "Buah kelapa sawit yang sudah matang sempurna",
      "color": "#f97316",
      "recommendation": "Segera panen untuk kualitas optimal"
    }
  ],
  "class_summary": {
    "Matang (Ripe)": {
      "count": 2,
      "avg_confidence": 94.5,
      "color": "#f97316"
    }
  },
  "annotated_image": "data:image/png;base64,..."
}
```

## 🎨 Fitur Frontend

- ✅ Drag & Drop upload gambar
- ✅ Preview gambar sebelum deteksi
- ✅ Visualisasi hasil dengan bounding box
- ✅ Detail informasi per deteksi
- ✅ Ringkasan per kelas
- ✅ Rekomendasi penanganan
- ✅ Status koneksi API real-time
- ✅ Responsive design

## 🏷️ Kelas Deteksi

| ID | Kelas | Deskripsi | Rekomendasi |
|----|-------|-----------|-------------|
| 0 | Mentah (Unripe) | Buah masih mentah | Tunggu 2-4 minggu |
| 1 | Kurang Matang | Buah kurang matang | Tunggu 1-2 minggu |
| 2 | Matang (Ripe) | Buah matang sempurna | Segera panen |
| 3 | Terlalu Matang | Buah terlalu matang | Panen segera |
| 4 | Busuk (Rotten) | Buah sudah busuk | Tidak layak panen |
| 5 | Tandan Kosong | Tandan tanpa buah | - |

> **Catatan:** Mapping kelas di atas adalah contoh. Sesuaikan dengan kelas yang ada di model Anda.

## 🛠️ Troubleshooting

### Model tidak ditemukan
Pastikan file `best.pt` ada di folder `backend/` dan path sudah benar.

### CORS Error
Pastikan CORS middleware sudah dikonfigurasi di backend (sudah termasuk dalam kode).

### Port sudah digunakan
Ganti port dengan menambahkan parameter:
```bash
uvicorn main:app --port 8001
```

Jangan lupa update `API_URL` di `index.html`:
```javascript
const API_URL = 'http://localhost:8001';
```

## 📝 Lisensi

MIT License

## 👨‍💻 Kontributor

Dibuat dengan ❤️ menggunakan:
- FastAPI
- Ultralytics YOLO
- Vanilla JavaScript
