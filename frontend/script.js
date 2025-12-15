// Configuration
const API_URL = "https://billsar1912-sawit-scan.hf.space";

// DOM Elements
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const previewFilename = document.getElementById("previewFilename");
const previewSize = document.getElementById("previewSize");
const detectBtn = document.getElementById("detectBtn");
const clearBtn = document.getElementById("clearBtn");
const loadingOverlay = document.getElementById("loadingOverlay");
const errorMessage = document.getElementById("errorMessage");
const resultsPlaceholder = document.getElementById("resultsPlaceholder");
const resultsContent = document.getElementById("resultsContent");
const resultImage = document.getElementById("resultImage");
const resultBadgeText = document.getElementById("resultBadgeText");
const totalDetections = document.getElementById("totalDetections");
const totalClasses = document.getElementById("totalClasses");
const detectionList = document.getElementById("detectionList");
const summaryGrid = document.getElementById("summaryGrid");
const statusBadge = document.getElementById("statusBadge");

let selectedFile = null;

// Check API Health
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        if (data.status === "healthy") {
            statusBadge.innerHTML =
                '<div class="status-dot"></div><span>API Connected</span>';
            statusBadge.style.borderColor = "rgba(34, 197, 94, 0.3)";
        } else {
            throw new Error("API not healthy");
        }
    } catch (error) {
        statusBadge.innerHTML =
            '<div class="status-dot" style="background: var(--danger)"></div><span>API Disconnected</span>';
        statusBadge.style.borderColor = "rgba(239, 68, 68, 0.3)";
    }
}

// Initialize
checkApiHealth();
setInterval(checkApiHealth, 30000);

// File Upload Handlers
uploadZone.addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
});

uploadZone.addEventListener("dragleave", () => {
    uploadZone.classList.remove("dragover");
});

uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

function handleFile(file) {
    // Validate file type
    const validTypes = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    ];
    if (!validTypes.includes(file.type)) {
        showError("Format file tidak didukung. Gunakan JPG, PNG, atau WEBP.");
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError("Ukuran file terlalu besar. Maksimum 10MB.");
        return;
    }

    selectedFile = file;
    hideError();

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewFilename.textContent = file.name;
        previewSize.textContent = formatFileSize(file.size);
        previewContainer.classList.add("active");
        uploadZone.style.display = "none";
        detectBtn.disabled = false;
        clearBtn.style.display = "block";
    };
    reader.readAsDataURL(file);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

// Clear Button
clearBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    previewContainer.classList.remove("active");
    uploadZone.style.display = "block";
    detectBtn.disabled = true;
    clearBtn.style.display = "none";
    hideError();
});

// Detect Button
detectBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    showLoading();
    hideError();

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(`${API_URL}/detect`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Gagal memproses gambar");
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        showError(
            error.message || "Terjadi kesalahan saat menghubungi server"
        );
    } finally {
        hideLoading();
    }
});

// Display Results
function displayResults(data) {
    resultsPlaceholder.style.display = "none";
    resultsContent.style.display = "block";
    resultsContent.classList.add("fade-in");

    // Set result image
    resultImage.src = data.annotated_image;

    // Update stats
    totalDetections.textContent = data.total_detections;
    totalClasses.textContent = Object.keys(data.class_summary).length;
    resultBadgeText.textContent = `${data.total_detections} Terdeteksi`;

    // Clear previous results
    detectionList.innerHTML = "";
    summaryGrid.innerHTML = "";

    // Render detection cards
    data.detections.forEach((detection, index) => {
        const card = document.createElement("div");
        card.className = "detection-card slide-in";
        card.style.animationDelay = `${index * 0.1}s`;
        card.innerHTML = `
                    <div class="detection-header">
                        <div class="detection-color" style="background: ${detection.color}"></div>
                        <span class="detection-class">${detection.class_name}</span>
                        <span class="detection-confidence">${detection.confidence}%</span>
                    </div>
                    <div class="detection-description">${detection.description}</div>
                    <div class="detection-recommendation">
                        <span class="detection-recommendation-icon">💡</span>
                        <span>${detection.recommendation}</span>
                    </div>
                `;
        detectionList.appendChild(card);
    });

    // Render summary cards
    Object.entries(data.class_summary).forEach(([className, info]) => {
        const card = document.createElement("div");
        card.className = "summary-card";
        card.style.borderColor = info.color;
        card.innerHTML = `
                    <div class="summary-count" style="color: ${info.color}">${info.count}</div>
                    <div class="summary-label">${className}</div>
                    <div class="summary-confidence">Avg: ${info.avg_confidence}%</div>
                `;
        summaryGrid.appendChild(card);
    });

    // Handle empty results
    if (data.total_detections === 0) {
        detectionList.innerHTML = `
                    <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
                        <p>Tidak ada buah kelapa sawit yang terdeteksi pada gambar ini.</p>
                        <p style="margin-top: 0.5rem; font-size: 0.85rem;">Pastikan gambar menampilkan buah kelapa sawit dengan jelas.</p>
                    </div>
                `;
    }
    const inputMetadata = {
        method: "POST",
        endpoint: "/detect",
        payload: {
            file_name: selectedFile.name,
            file_type: selectedFile.type,
            file_size_bytes: selectedFile.size,
            file_size_readable: formatFileSize(selectedFile.size),
            content_type: "multipart/form-data",
        },
        timestamp: new Date().toISOString(),
    };

    // Tampilkan Input
    document.getElementById("jsonInputContent").textContent =
        JSON.stringify(inputMetadata, null, 2);

    // 2. Siapkan data Output
    // Kita buat copy dari data agar tidak merusak data asli
    const displayData = JSON.parse(JSON.stringify(data));

    // TRUNCATE BASE64 IMAGE:
    // String base64 gambar sangat panjang (ratusan kb), jadi kita pendekkan untuk display
    if (displayData.annotated_image) {
        displayData.annotated_image =
            "data:image/png;base64, [TRUNCATED_FOR_DISPLAY] ... (" +
            data.annotated_image.length +
            " chars)";
    }

    // Tampilkan Output
    document.getElementById("jsonOutputContent").textContent =
        JSON.stringify(displayData, null, 2);
}

// Utility Functions
function showLoading() {
    loadingOverlay.classList.add("active");
}

function hideLoading() {
    loadingOverlay.classList.remove("active");
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add("active");
}

function hideError() {
    errorMessage.classList.remove("active");
}

// Fungsi Helper untuk Copy JSON
function copyToClipboard(elementId) {
    const text = document.getElementById(elementId).textContent;
    navigator.clipboard.writeText(text).then(() => {
        // Visual feedback simpel (opsional)
        const btn = document.querySelector(
            `button[onclick="copyToClipboard('${elementId}')"]`
        );
        const originalText = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => {
            btn.textContent = originalText;
        }, 2000);
    });
}