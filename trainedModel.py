import warnings
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def crop_profit_recommendation_with_risk(df, crops, market):
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    yield_per_acre = {
        "Wheat": 20, "Rice": 22, "Maize": 18, "Sugarcane": 400,
        "Potato": 250, "Tomato": 180, "Onion": 120, "Mustard": 10,
        "Cotton": 8, "Apple": 7, "Banana": 35
    }

    cost_per_acre = {
        "Wheat": 25000, "Rice": 30000, "Maize": 20000, "Sugarcane": 50000,
        "Potato": 60000, "Tomato": 40000, "Onion": 35000, "Mustard": 18000,
        "Cotton": 45000, "Apple": 90000, "Banana": 50000
    }

    risk_factors = {
        "Wheat": "🟢 Low", "Rice": "🟢 Low", "Maize": "🟢 Low", "Sugarcane": "🟡 Medium",
        "Potato": "🔴 High", "Tomato": "🔴 High", "Onion": "🔴 High", "Mustard": "🟢 Low",
        "Cotton": "🟡 Medium", "Apple": "🔴 High", "Banana": "🔴 High"
    }

    # If only one crop, return just that crop name
    if len(crops) == 1:
        return crops[0]

    processed_crops = []
    available_markets = df['Market'].unique()

    for crop in crops:
        try:
            # Search for available model for the crop in any market
            model_found = False
            for mkt in available_markets:
                model_path = f"lstm_models/lstm_{crop}_{mkt}.h5"
                scaler_path = f"scalers/scaler_{crop}_{mkt}.pkl"
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    model = load_model(model_path, compile=False)
                    scaler = joblib.load(scaler_path)

                    # Prepare data
                    data = df[(df['Commodity'] == crop) & (df['Market'] == mkt)][['Arrival_Date', 'Modal_Price']].copy()
                    data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], dayfirst=True, errors='coerce')
                    data.dropna(inplace=True)
                    data.set_index('Arrival_Date', inplace=True)
                    data = data.resample("ME").mean().dropna()

                    if len(data) < 12:
                        break

                    last_12 = data[-12:].values
                    last_scaled = scaler.transform(last_12).reshape(1, 12, 1)
                    forecast_scaled = model.predict(last_scaled, verbose=0)[0]
                    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

                    avg_price = forecast.mean()
                    yield_qtl = yield_per_acre.get(crop)
                    cost = cost_per_acre.get(crop)

                    if yield_qtl is None or cost is None:
                        break

                    # Add to final result list
                    processed_crops.append(crop)
                    model_found = True
                    break

            if not model_found:
                processed_crops.append(crop)

        except Exception as e:
            print(f"❌ Error for {crop}: {e}")
            processed_crops.append(crop)

    # Final return: list of crop names
    return processed_crops if processed_crops else crops
