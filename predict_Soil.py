import os
import warnings
import json
import sys

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = INFO+WARNING, 3 = ERROR only

# Suppress Python warnings (h5py, requests, etc.)
warnings.filterwarnings("ignore")

# Imports after suppression
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore
import requests
from io import BytesIO
from PIL import Image

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "soil_classifier.h5")
model = load_model(model_path, compile=False)

def predict_soil(image_url, model):
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        pred = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(pred)
        confidence = np.max(pred) * 100

        class_names = ['Alluvial_soil', 'Black_soil', 'Clay_soil', 'Lateritic_soil']
        return class_names[predicted_class], confidence

    except Exception as e:
        return f"Error: {e}", 0

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"error":"Expected one arguments: image URL"}))
        sys.exit(1)
    try:
        image_url = sys.argv[1]
        pred_class, confidence = predict_soil(image_url, model)
        results = {
            "predicted_class": pred_class,
            "confidence": confidence
        }
        print(json.dumps(results))
    except Exception as e:
        print(json.dumps({"error":str(e)}))
        sys.exit(1)
