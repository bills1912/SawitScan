#!/bin/bash

# SawitScan - Script untuk menjalankan aplikasi

echo "🌴 SawitScan - Palm Oil Fruit Detection"
echo "========================================"
echo ""

# Check if model exists
if [ ! -f "backend/best.pt" ]; then
    echo "⚠️  Model YOLO tidak ditemukan!"
    echo ""
    echo "Silakan unduh model terlebih dahulu:"
    echo "curl -L -o backend/best.pt 'https://github.com/bills1912/computer-vision/raw/refs/heads/main/best.pt'"
    echo ""
    read -p "Apakah Anda ingin melanjutkan tanpa model? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 tidak ditemukan. Silakan install Python 3 terlebih dahulu."
    exit 1
fi

# Install dependencies
echo "📦 Menginstall dependencies..."
cd backend
pip install -r requirements.txt --quiet

# Start backend
echo ""
echo "🚀 Menjalankan Backend API di http://localhost:8000"
echo "📚 API Docs tersedia di http://localhost:8000/docs"
echo ""
echo "Tekan Ctrl+C untuk menghentikan server"
echo ""

python main.py
