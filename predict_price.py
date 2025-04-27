import sys
import json
import pandas as pd
import numpy as np
import pickle
import traceback
from train_model import fetch_and_save_commodity_data, train_and_predict_from_csv

def recommend_and_predict_prices(test_params, params):
    """Improved version with better error handling and frontend-friendly output"""
    try:
        # Load Crop Recommendation Model
        with open('crop_recommendation_model.pkl', 'rb') as f:
            model_pipeline = pickle.load(f)

        model = model_pipeline['model']
        scaler = model_pipeline['scaler']
        label_encoder = model_pipeline['label_encoder']
        feature_columns = model_pipeline['feature_columns']

        # Prepare input sample
        test_sample_df = pd.DataFrame([test_params], columns=feature_columns)
        test_sample_scaled = scaler.transform(test_sample_df)

        # Predict probabilities
        pred_proba = model.predict_proba(test_sample_scaled)
        top_5_indices = np.argsort(pred_proba[0])[::-1][:5]
        recommended_crops = label_encoder.inverse_transform(top_5_indices)

        # Predict future prices
        predictions = []
        debug_info = []

        for crop in recommended_crops:
            try:
                fetch_and_save_commodity_data(params, commodity_name=crop, filename="commodity_data.csv")
                _, _, predicted_price = train_and_predict_from_csv("commodity_data.csv")
                predictions.append({
                    "crop": crop,
                    "price": float(predicted_price)  # Ensure numeric value
                })
            except Exception as e:
                predictions.append({
                    "crop": crop,
                    "price": 0.0
                })
                debug_info.append(f"Could not predict for {crop}: {str(e)}")

        return {
            "success": True,
            "predictions": predictions,  # Now an array of objects
            "debug": debug_info,
            "input_params": {
                "test_params": test_params,
                "filters": params
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def main():
    try:
        # Parse command line arguments
        test_params = json.loads(sys.argv[1])
        params = json.loads(sys.argv[2])
        
        # Get predictions
        result = recommend_and_predict_prices(test_params, params)
        
        # Output final JSON
        print(json.dumps(result, ensure_ascii=False), flush=True)
        
    except Exception as e:
        error_output = {
            "success": False,
            "error": f"Main function error: {str(e)}",
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_output), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()