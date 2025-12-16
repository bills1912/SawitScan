"""
Palm Oil Fruit Detection API
Backend menggunakan FastAPI, YOLO, dan Roboflow Inference
Fitur: Multi-stage Detection (Janjang -> Kematangan)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import numpy as np
import cv2
from typing import List, Dict, Any
import os
from pathlib import Path
from dotenv import load_dotenv
import uuid

# --- 1. KONFIGURASI ENV & APP ---
# Mengambil path folder backend saat ini
BACKEND_DIR = Path(__file__).resolve().parent
# Naik satu level ke root, lalu masuk ke frontend/.env
ENV_PATH = BACKEND_DIR.parent / 'frontend' / '.env'
load_dotenv(ENV_PATH)

# Inisialisasi FastAPI
app = FastAPI(
    title="Palm Oil Fruit Detection API",
    description="API Deteksi Bertingkat: Janjang Sawit -> Tingkat Kematangan",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. INISIALISASI MODEL ---

# A. Model Kematangan (Lokal YOLO)
MODEL_PATH = "yolov9.pt"
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Model Kematangan loaded: {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: File {MODEL_PATH} tidak ditemukan.")
        model = None
except Exception as e:
    print(f"❌ Error loading model YOLO: {e}")
    model = None

# B. Model Janjang/Bunch (Roboflow Cloud)
ROBOFLOW_API_KEY = "CaKJLDG1Q8mLEWjXAZ4B" # API Key Anda
ROBOFLOW_MODEL_ID = "palm-fruit-ahuai/3"

try:
    CLIENT = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=ROBOFLOW_API_KEY
    )
    print(f"✅ Roboflow Client initialized for: {ROBOFLOW_MODEL_ID}")
except Exception as e:
    print(f"❌ Error initializing Roboflow client: {e}")
    CLIENT = None

# --- 3. DATA KELAS (DIPERTAHANKAN) ---
# Mapping kelas deteksi kelapa sawit (Sesuai kode lama Anda)
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

# --- 4. HELPER FUNCTIONS ---

def draw_roboflow_boxes(image: Image.Image, predictions: List[Dict], filename: str):
    """
    Menggambar kotak manual untuk hasil Roboflow (Janjang)
    jika model Kematangan tidak menemukan hasil.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    annotated_detections = []
    
    for i, pred in enumerate(predictions):
        # Roboflow: x, y (center), width, height
        x_center, y_center = pred['x'], pred['y']
        w, h = pred['width'], pred['height']
        
        # Convert ke x1, y1, x2, y2
        x1 = x_center - (w / 2)
        y1 = y_center - (h / 2)
        x2 = x_center + (w / 2)
        y2 = y_center + (h / 2)
        
        confidence = round(pred['confidence'] * 100, 2)
        class_name = pred['class'] # Biasanya "Fresh Fruit Bunch"
        
        # ID UNIQUE (Sama formatnya dengan YOLO)
        unique_id = f"{filename}_{i + 1}"
        
        # Gambar Kotak (Biru untuk Janjang)
        color = "#3b82f6" 
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Label Text
        text = f"#{i+1} {class_name} {confidence}%"
        # Background text simple
        draw.rectangle([x1, y1 - 15, x1 + (len(text)*6), y1], fill=color)
        draw.text((x1 + 2, y1 - 15), text, fill="white", font=font)

        # Format Data Output
        detection = {
            "id": unique_id,
            "index": i + 1,
            "class_id": 99, # ID Khusus Janjang
            "class_name": class_name,
            "confidence": confidence,
            "bbox": {"x1": round(x1,2), "y1": round(y1,2), "x2": round(x2,2), "y2": round(y2,2)},
            "description": "Deteksi Janjang Utuh (Model Tahap 1)",
            "color": color,
            "recommendation": "Janjang terdeteksi, namun tingkat kematangan spesifik tidak terbaca."
        }
        annotated_detections.append(detection)
        
    return image, annotated_detections

def process_detection(image: Image.Image, filename: str) -> Dict[str, Any]:
    """
    Logika Utama:
    1. Cek Janjang (Roboflow).
    2. Jika Ada Janjang -> Cek Kematangan (YOLO).
    3. Jika Kematangan Ada -> Tampilkan Kematangan.
    4. Jika Kematangan Kosong -> Tampilkan Janjang saja.
    """
    
    # 1. Persiapan File Temp untuk Roboflow
    temp_filename = f"/tmp/{uuid.uuid4()}.jpg"
    image.save(temp_filename)
    
    try:
        # --- TAHAP 1: DETEKSI JANJANG ---
        print("🕵️ Memulai Deteksi Janjang (Roboflow)...")
        bunch_result = CLIENT.infer(temp_filename, model_id=ROBOFLOW_MODEL_ID)
        
        # Hapus file temp
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        bunch_predictions = bunch_result.get('predictions', [])
        
        # LOGIKA: Jika TIDAK ADA Janjang sama sekali -> Stop
        if not bunch_predictions:
            return {
                "total_detections": 0,
                "detections": [],
                "class_summary": {},
                "annotated_image": None
            }

        print(f"✅ Janjang ditemukan: {len(bunch_predictions)} buah.")

        # --- TAHAP 2: DETEKSI KEMATANGAN (YOLO) ---
        print("🕵️ Memulai Deteksi Kematangan (YOLOv9)...")
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
        ripeness_results = model(img_array, conf=0.25)
        
        # Cek apakah YOLO menemukan sesuatu
        ripeness_found = False
        for result in ripeness_results:
            if result.boxes is not None and len(result.boxes) > 0:
                ripeness_found = True
                
        # --- LOGIKA OUTPUT FINAL ---
        
        if ripeness_found:
            # === SKENARIO A: Kematangan Ditemukan (PRIORITAS) ===
            print("✅ Kematangan terdeteksi! Menggunakan hasil YOLO.")
            
            detections = []
            class_summary = {}
            annotated_img = ripeness_results[0].plot() # Pakai plotter bawaan
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            for result in ripeness_results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Ambil Info Kelas (Warna, Deskripsi, dll)
                        class_info = get_class_info(class_id)
                        class_name = class_info["name"]
                        
                        # ID UNIK: filename_index
                        unique_id = f"{filename}_{i + 1}"
                        
                        detection = {
                            "id": unique_id,
                            "index": i + 1,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(confidence * 100, 2),
                            "bbox": {
                                "x1": round(x1, 2), "y1": round(y1, 2),
                                "x2": round(x2, 2), "y2": round(y2, 2)
                            },
                            "description": class_info["description"],
                            "color": class_info["color"],
                            "recommendation": class_info["recommendation"]
                        }
                        detections.append(detection)
                        
                        # Update Summary
                        if class_name not in class_summary:
                            class_summary[class_name] = {
                                "count": 0, "avg_confidence": 0,
                                "color": class_info["color"],
                                "description": class_info["description"],
                                "recommendation": class_info["recommendation"]
                            }
                        class_summary[class_name]["count"] += 1
                        class_summary[class_name]["avg_confidence"] += confidence

            # Hitung Rata-rata Confidence Summary
            for class_name in class_summary:
                count = class_summary[class_name]["count"]
                class_summary[class_name]["avg_confidence"] = round(
                    (class_summary[class_name]["avg_confidence"] / count) * 100, 2
                )
            
            final_detections = detections
            final_image_pil = Image.fromarray(annotated_img[..., ::-1]) # RGB correct
            
        else:
            # === SKENARIO B: Janjang Ada, Tapi Kematangan Nihil (FALLBACK) ===
            print("⚠️ Kematangan nihil. Menampilkan kotak Janjang (Roboflow).")
            
            # Gambar kotak manual berdasarkan data Roboflow
            image_copy = image.copy()
            final_image_pil, final_detections = draw_roboflow_boxes(image_copy, bunch_predictions, filename)
            
            class_summary = {
                "Janjang Sawit": {
                    "count": len(final_detections),
                    "avg_confidence": 0, 
                    "color": "#3b82f6",
                    "description": "Deteksi Objek Janjang",
                    "recommendation": "Cek visual manual"
                }
            }

        # Konversi Hasil Akhir ke Base64
        buffered = io.BytesIO()
        final_image_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "total_detections": len(final_detections),
            "detections": final_detections,
            "class_summary": class_summary,
            "annotated_image": f"data:image/png;base64,{img_base64}"
        }

    except Exception as e:
        # Cleanup jika error
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print(f"❌ ERROR Processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. ENDPOINTS ---

@app.get("/")
async def root():
    """Root endpoint (Tetap dipertahankan)"""
    return {
        "message": "Palm Oil Fruit Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "mode": "Multi-Stage (Roboflow -> YOLO)",
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
        "model_yolo": model is not None,
        "model_roboflow": CLIENT is not None,
        "model_path": MODEL_PATH
    }

@app.get("/classes")
async def get_classes():
    """Mendapatkan daftar kelas (Tetap dipertahankan)"""
    classes = []
    for class_id, info in CLASS_INFO.items():
        classes.append({
            "id": class_id,
            **info
        })
    return {"classes": classes}

@app.post("/detect")
async def detect_palm_fruit(file: UploadFile = File(...)):
    """Endpoint utama untuk deteksi"""
    # Validasi tipe file
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Tipe file tidak didukung.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Panggil logic utama dengan nama file untuk ID Unik
        result = process_detection(image, file.filename)
        
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
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)