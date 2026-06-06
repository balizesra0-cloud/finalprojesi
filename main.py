import io
import os
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# TensorFlow warnings to a minimum
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI(title="Melanoma Detection API")

# Mount static files directory
# We check if static directory exists; if not, we create it
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model globally
MODEL_PATH = "final_model.h5"
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file {MODEL_PATH} not found in root directory!")
        raise RuntimeError(f"Model file {MODEL_PATH} not found in root directory!")
    try:
        logging.info("Loading TensorFlow Keras model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logging.info("Model loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        raise e

@app.get("/")
def read_root():
    # Serve index.html directly from static directory
    html_path = os.path.join("static", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "DermoscopyAI Backend Active. Please place index.html in the static directory."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model henüz yüklenmedi, lütfen daha sonra tekrar deneyin.")

    # 1. Dosya Uzantısı Kontrolü
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    _, ext = os.path.splitext(file.filename.lower())
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı. Lütfen PNG veya JPG kullanın.")

    # 2. Dosya Boyutu Sınırlandırması (Maks. 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Dosya boyutu çok büyük (Maks. 10MB).")

    # 3. İstemci Content-Type Kontrolü
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Yüklenen dosya geçerli bir görsel değil.")

    try:
        file_bytes = await file.read()
        
        # 4. Pillow ile Görsel Yapısı Doğrulaması (Magic bytes/corrupt file kontrolü)
        try:
            img_check = Image.open(io.BytesIO(file_bytes))
            img_check.verify()
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Görsel dosyası bozuk veya geçersiz.")
        
        # Resize to 224x224 as required by the model
        img_resized = img.resize((224, 224))
        
        # Convert image to numpy array and normalize
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run prediction
        prediction = model.predict(img_array)
        
        benign_prob = float(prediction[0][0])
        melanoma_prob = float(prediction[0][1])
        
        # Format percentages
        benign_percentage = benign_prob * 100
        melanoma_percentage = melanoma_prob * 100
        
        # Determine result
        if melanoma_percentage > benign_percentage:
            result = "MELANOM"
            confidence = melanoma_percentage
        else:
            result = "BENIGN"
            confidence = benign_percentage
            
        return {
            "success": True,
            "result": result,
            "confidence": round(confidence, 2),
            "details": {
                "melanoma": round(melanoma_percentage, 2),
                "benign": round(benign_percentage, 2)
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        # Hata detaylarını sadece sunucu loglarında tut, istemciye sızdırma
        logging.error(f"Inference hatası: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analiz işlemi gerçekleştirilemedi.")

if __name__ == "__main__":
    import uvicorn
    # Run uvicorn server locally on port 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
