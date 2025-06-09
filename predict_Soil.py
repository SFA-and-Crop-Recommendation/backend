from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore
import requests
from io import BytesIO
from PIL import Image
import os

model_path = os.path.join(os.path.dirname(__file__), "soil_classifier")
model = load_model(model_path)


def predict_soil(image_url, model):
    try:
        # Step 1: Download image from URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise error for bad response

        # Step 2: Open image using PIL
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))

        # Step 3: Convert image to array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Step 4: Predict
        pred = model.predict(img_array)
        predicted_class = np.argmax(pred)
        confidence = np.max(pred) * 100

        # Step 5: Class names
        class_names = ['Alluvial_soil', 'Black_soil', 'Clay_soil', 'Lateritic_soil']

        return class_names[predicted_class], confidence

    except Exception as e:
        return f"Error: {e}", 0

# Example usage
image_url = "http://localhost:3000/uploads/img_1749408475784.jpg"
pred_class, confidence = predict_soil(image_url, model)
print(f"Predicted Soil Type: {pred_class}")
print(f"Confidence: {confidence:.2f}%")
