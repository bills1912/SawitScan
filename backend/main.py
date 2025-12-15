"""
Palm Oil Fruit Detection API
Backend menggunakan FastAPI dan YOLO untuk mendeteksi buah kelapa sawit
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from PIL import Image
import io
import base64
import numpy as np
import cv2
from typing import List, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv

# --- 1. KONFIGURASI ENV (FIXED) ---
# Mengambil path folder backend saat ini
BACKEND_DIR = Path(__file__).resolve().parent
# Naik satu level ke root, lalu masuk ke frontend/.env
ENV_PATH = BACKEND_DIR.parent / 'frontend' / '.env'
load_dotenv(ENV_PATH)

# Inisialisasi FastAPI
app = FastAPI(
    title="Palm Oil Fruit Detection API",
    description="API untuk mendeteksi tingkat kematangan buah kelapa sawit menggunakan YOLO",
    version="1.0.0"
)

# CORS middleware untuk mengizinkan akses dari frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path ke model YOLO
MODEL_PATH = "https://github.com/bills1912/SawitScan/raw/refs/heads/main/backend/yolov9.pt"

# Load model YOLO
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Mapping kelas deteksi kelapa sawit (sesuaikan dengan model Anda)
CLASS_INFO = {
    0: {
        "name": "Tidak Normal (Abnormal)",
        "description": "Buah kelapa sawit yang sudah busuk",
        "color": "#7c3aed",  # Purple
        "recommendation": "Tidak layak panen, buang"
    },
    1: {
        "name": "Tandan Kosong (Empty Bunch)",
        "description": "Tandan kelapa sawit kosong",
        "color": "#6b7280",  # Gray
        "recommendation": "Tidak ada buah untuk dipanen"
    },
    2: {
        "name": "Terlalu Matang (Over-ripe)",
        "description": "Buah kelapa sawit yang terlalu matang",
        "color": "#ef4444",  # Red
        "recommendation": "Panen segera, kualitas minyak menurun"
    },
    3: {
        "name": "Matang (Ripe)",
        "description": "Buah kelapa sawit yang sudah matang sempurna, siap panen",
        "color": "#f97316",  # Orange
        "recommendation": "Segera panen untuk kualitas optimal"
    },
    4: {
        "name": "Kurang Matang (Under-ripe)", 
        "description": "Buah kelapa sawit yang kurang matang",
        "color": "#eab308",  # Yellow
        "recommendation": "Tunggu 1-2 minggu sebelum panen"
    },
    5: {
        "name": "Mentah (Unripe)",
        "description": "Buah kelapa sawit yang masih mentah, belum siap panen",
        "color": "#22c55e",  # Green
        "recommendation": "Tunggu 2-4 minggu sebelum panen"
    },
}

def get_class_info(class_id: int) -> Dict[str, Any]:
    """Mendapatkan informasi kelas berdasarkan ID"""
    if class_id in CLASS_INFO:
        return CLASS_INFO[class_id]
    return {
        "name": f"Kelas {class_id}",
        "description": "Kelas tidak dikenal",
        "color": "#9ca3af",
        "recommendation": "Tidak ada rekomendasi"
    }

def process_detection(image: Image.Image) -> Dict[str, Any]:
    """
    Memproses gambar dan melakukan deteksi menggunakan YOLO
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model tidak tersedia")
    
    # Konversi PIL Image ke numpy array
    img_array = np.array(image)
    
    # Jika gambar RGBA, konversi ke RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    elif len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Jalankan deteksi
    results = model(img_array, conf=0.25)
    
    # Proses hasil deteksi
    detections = []
    class_summary = {}
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Ambil koordinat bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Dapatkan nama kelas dari model atau mapping
                if hasattr(result, 'names') and class_id in result.names:
                    class_name = result.names[class_id]
                else:
                    class_info = get_class_info(class_id)
                    class_name = class_info["name"]
                
                class_info = get_class_info(class_id)
                
                detection = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence * 100, 2),
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2)
                    },
                    "description": class_info["description"],
                    "color": class_info["color"],
                    "recommendation": class_info["recommendation"]
                }
                detections.append(detection)
                
                # Update summary
                if class_name not in class_summary:
                    class_summary[class_name] = {
                        "count": 0,
                        "avg_confidence": 0,
                        "color": class_info["color"],
                        "description": class_info["description"],
                        "recommendation": class_info["recommendation"]
                    }
                class_summary[class_name]["count"] += 1
                class_summary[class_name]["avg_confidence"] += confidence
    
    # Hitung rata-rata confidence untuk setiap kelas
    for class_name in class_summary:
        count = class_summary[class_name]["count"]
        class_summary[class_name]["avg_confidence"] = round(
            (class_summary[class_name]["avg_confidence"] / count) * 100, 2
        )
    
    # Gambar bounding box pada gambar
    annotated_img = results[0].plot()
    
    # Konversi gambar hasil ke base64
    annotated_pil = Image.fromarray(annotated_img)
    buffered = io.BytesIO()
    annotated_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "total_detections": len(detections),
        "detections": detections,
        "class_summary": class_summary,
        "annotated_image": f"data:image/png;base64,{img_base64}"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Palm Oil Fruit Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "/detect": "POST - Upload image for detection",
            "/health": "GET - Check API health",
            "/classes": "GET - Get available detection classes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.get("/classes")
async def get_classes():
    """Mendapatkan daftar kelas yang dapat dideteksi"""
    classes = []
    for class_id, info in CLASS_INFO.items():
        classes.append({
            "id": class_id,
            **info
        })
    return {"classes": classes}

@app.post("/detect")
async def detect_palm_fruit(file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi buah kelapa sawit dari gambar
    
    - **file**: File gambar (jpg, png, jpeg)
    
    Returns:
    - Hasil deteksi dengan bounding box dan informasi kelas
    """
    # Validasi tipe file
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Tipe file tidak didukung. Gunakan: {', '.join(allowed_types)}"
        )
    
    try:
        # Baca file gambar
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Konversi ke RGB jika perlu
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Proses deteksi
        result = process_detection(image)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            **result
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saat memproses gambar: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
