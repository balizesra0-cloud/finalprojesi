document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const dropZonePrompt = document.getElementById("drop-zone-prompt");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const removeImgBtn = document.getElementById("remove-img-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const btnSpinner = document.getElementById("btn-spinner");
    
    const resultsEmpty = document.getElementById("results-empty");
    const resultsLoading = document.getElementById("results-loading");
    const resultsContent = document.getElementById("results-content");
    
    const diagnosisCard = document.getElementById("diagnosis-card");
    const diagnosisIcon = document.getElementById("diagnosis-icon");
    const diagnosisValue = document.getElementById("diagnosis-value");
    const analysisTimestamp = document.getElementById("analysis-timestamp");
    
    const gaugeProgressCircle = document.getElementById("gauge-progress-circle");
    const gaugePercentValue = document.getElementById("gauge-percent-value");
    
    const probMelanomaText = document.getElementById("prob-melanoma");
    const probBenignText = document.getElementById("prob-benign");
    const barMelanoma = document.getElementById("bar-melanoma");
    const barBenign = document.getElementById("bar-benign");

    let selectedFile = null;

    // Trigger file input click when clicking on drop zone
    dropZone.addEventListener("click", (e) => {
        // Prevent click trigger when clicking the remove button
        if (e.target.closest("#remove-img-btn")) return;
        fileInput.click();
    });

    // Drag & Drop event listeners
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle File processing
    function handleFile(file) {
        // Check if it's an image
        if (!file.type.startsWith("image/")) {
            alert("Lütfen yalnızca geçerli bir görsel dosyası (PNG, JPG, JPEG) yükleyin.");
            return;
        }

        selectedFile = file;
        
        // Read file for preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZonePrompt.style.display = "none";
            previewContainer.style.display = "flex";
            analyzeBtn.removeAttribute("disabled");
        };
        reader.readAsDataURL(file);

        // Reset results state if we load a new image
        resetResultsState();
    }

    // Remove Image button
    removeImgBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // Avoid triggering dropZone click
        resetUploadState();
    });

    function resetUploadState() {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.style.display = "none";
        dropZonePrompt.style.display = "flex";
        analyzeBtn.setAttribute("disabled", "true");
        resetResultsState();
    }

    function resetResultsState() {
        resultsEmpty.style.display = "flex";
        resultsLoading.style.display = "none";
        resultsContent.style.display = "none";
        
        // Reset gauge
        setGaugeValue(0);
        // Reset bars
        barMelanoma.style.width = "0%";
        barBenign.style.width = "0%";
    }

    // Set Gauge Progress & Animation
    function setGaugeValue(percent) {
        const radius = 50;
        const circumference = 2 * Math.PI * radius; // ~314.159
        
        // Calculate stroke offset
        const offset = circumference - (percent / 100) * circumference;
        gaugeProgressCircle.style.strokeDashoffset = offset;
        
        // Animate counter
        animateCounter(gaugePercentValue, percent);
    }

    // Animate Text Counter
    function animateCounter(element, targetValue) {
        let currentValue = 0;
        const duration = 1000; // 1s
        const steps = 60;
        const stepTime = duration / steps;
        const increment = targetValue / steps;
        
        const timer = setInterval(() => {
            currentValue += increment;
            if (currentValue >= targetValue) {
                currentValue = targetValue;
                clearInterval(timer);
            }
            element.textContent = `${currentValue.toFixed(1)}%`;
        }, stepTime);
    }

    // Analyze Button click
    analyzeBtn.addEventListener("click", () => {
        if (!selectedFile) return;

        // UI updates to Loading state
        analyzeBtn.setAttribute("disabled", "true");
        btnSpinner.style.display = "inline-block";
        
        resultsEmpty.style.display = "none";
        resultsLoading.style.display = "flex";
        resultsContent.style.display = "none";

        // Prepare FormData
        const formData = new FormData();
        formData.append("file", selectedFile);

        // Fetch prediction from backend API
        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error("Sunucu hatası veya geçersiz analiz isteği.");
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                renderResults(data);
            } else {
                throw new Error(data.message || "Bilinmeyen bir hata oluştu.");
            }
        })
        .catch(error => {
            console.error(error);
            alert("Analiz başarısız oldu: " + error.message);
            resetUploadState();
        })
        .finally(() => {
            // Restore button state
            analyzeBtn.removeAttribute("disabled");
            btnSpinner.style.display = "none";
        });
    });

    // Render results on the UI
    function renderResults(data) {
        // Show results panel, hide loader
        resultsLoading.style.display = "none";
        resultsContent.style.display = "flex";
        
        const isMelanoma = data.result === "MELANOM";
        
        // Update Timestamp
        const now = new Date();
        const timeString = now.toLocaleTimeString("tr-TR", { hour: '2-digit', minute: '2-digit' });
        analysisTimestamp.textContent = `Bugün saat ${timeString} itibarıyla analiz edildi`;

        // Update Diagnosis Card
        diagnosisCard.className = "diagnosis-card " + (isMelanoma ? "is-melanoma" : "is-benign");
        diagnosisValue.textContent = isMelanoma ? "MELANOM (Kötü Huylu)" : "BENİGN (İyi Huylu)";

        // Set Diagnosis Icon (SVG)
        if (isMelanoma) {
            diagnosisIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
            `;
        } else {
            diagnosisIcon.innerHTML = `
                <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
            `;
        }

        // Animate Visual Gauge (shows the prediction confidence)
        setTimeout(() => {
            setGaugeValue(data.confidence);
        }, 150);

        // Update & Animate Breakdown bars
        probMelanomaText.textContent = `${data.details.melanoma.toFixed(2)}%`;
        probBenignText.textContent = `${data.details.benign.toFixed(2)}%`;

        setTimeout(() => {
            barMelanoma.style.width = `${data.details.melanoma}%`;
            barBenign.style.width = `${data.details.benign}%`;
        }, 300);
    }
});
