import pandas as pd
import numpy as np
import pickle
from train_model import fetch_and_save_commodity_data, train_and_predict_from_csv

def recommend_and_predict_prices(test_params,params):
    """
    Takes test parameters, recommends crops, predicts future prices, and returns a dictionary.
    """
    # Load Crop Recommendation Model
    with open(r'C:\\Users\\Nejarul\\Desktop\\Project Dataset\\crop_recommendation_model.pkl', 'rb') as f:
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

    # Recommend Top 5 Crops
    top_5_indices = np.argsort(pred_proba[0])[::-1][:5]
    recommended_crops = label_encoder.inverse_transform(top_5_indices)

    # Fixed Parameters for price prediction
    

    # Predict future prices for each recommended crop
    predicted_prices = {}

    for crop in recommended_crops:
        try:
            fetch_and_save_commodity_data(params, commodity_name=crop, filename="commodity_data.csv")
            price_model, price_scaler, predicted_price = train_and_predict_from_csv("commodity_data.csv")
            predicted_prices[crop] = predicted_price
        except ValueError as e:
            print(f"⚠️ Could not predict for {crop}: {e}")
            predicted_prices[crop] = 0  # or None if you prefer

    return predicted_prices
