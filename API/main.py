from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions


app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
# model = load_model("soil_classifier.h5")
model = load_model("soil_classifier.h5", compile=False)

# Directory for storing uploaded images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

@app.get("/")
async def root():
    return {"message": "Welcome to the Soil Image Classification API"}

# Upload image endpoint
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"url": f"http://127.0.0.1:8000/images/{file.filename}"}

# Predict endpoint
class ImageRequest(BaseModel):
    image_url: str

@app.post("/predict")
async def predict_image(req: ImageRequest):
    try:
        img = Image.open(req.image_url).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img), axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        decoded = decode_predictions(preds, top=3)[0]
        result = [{"label": label, "description": desc, "probability": float(prob)} for label, desc, prob in decoded]
        return {"predictions": result}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
