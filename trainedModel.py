# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib
# import os
# import warnings

# def predict_next_6_months(commodity, market, df):
#     """
#     Predict next 6 months of Modal Price using pretrained LSTM model.

#     Args:
#         commodity (str): Commodity name (e.g., 'Apple')
#         market (str): Market name (e.g., 'Mechua')
#         df (pd.DataFrame): Full historical dataset

#     Returns:
#         dict: {date (str): forecasted price (float)}
#     """
#     try:
#         model_path = f"lstm_models/lstm_{commodity}_{market}.h5"
#         scaler_path = f"scalers/scaler_{commodity}_{market}.pkl"

#         if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#             raise FileNotFoundError(f"Model or scaler for {commodity}-{market} not found.")

#         # Load model and scaler
#         model = load_model(model_path, compile=False)
#         scaler = joblib.load(scaler_path)

#         # Filter and prepare data
#         data = df[(df['Commodity'] == commodity) & (df['Market'] == market)][['Arrival_Date', 'Modal_Price']].copy()
#         data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], dayfirst=True, errors='coerce')
#         data.dropna(subset=['Arrival_Date', 'Modal_Price'], inplace=True)
#         data.set_index('Arrival_Date', inplace=True)
#         data.index = pd.DatetimeIndex(data.index)
#         monthly = data.resample("ME").mean().dropna()

#         if len(monthly) < 12:
#             raise ValueError(f"Not enough historical data for {commodity} - {market}.")

#         # Use the most recent 12 months from the latest available data
#         last_12 = monthly[-12:].values
#         last_12_scaled = scaler.transform(last_12).reshape(1, 12, 1)

#         # Predict next 6 months
#         forecast_scaled = model.predict(last_12_scaled, verbose=0)[0]
#         forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

#         # Use current date as base for future dates
#         from pandas.tseries.offsets import MonthEnd
#         today = pd.Timestamp.today().replace(day=1) + MonthEnd(0)  # End of this month
#         future_dates = pd.date_range(start=today + pd.DateOffset(months=1), periods=6, freq="ME")

#         forecast_dict = {
#             date.strftime("%Y-%m-%d"): round(price, 2)
#             for date, price in zip(future_dates, forecast)
#         }

#         return forecast_dict

#     except Exception as e:
#         print(f"❌ Error in prediction for {commodity}-{market}: {e}")
#         return None


import os
import sys
import json
import pandas as pd
import joblib
import warnings
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Yield and cost data
YIELD_PER_ACRE = {
    "Wheat": 20, "Rice": 22, "Maize": 18, "Sugarcane": 400,
    "Potato": 250, "Tomato": 180, "Onion": 120, "Mustard": 10,
    "Cotton": 8, "Apple": 7, "Banana": 35
}

COST_PER_ACRE = {
    "Wheat": 25000, "Rice": 30000, "Maize": 20000, "Sugarcane": 50000,
    "Potato": 60000, "Tomato": 40000, "Onion": 35000, "Mustard": 18000,
    "Cotton": 45000, "Apple": 90000, "Banana": 50000
}

RISK_FACTORS = {
    "Wheat": "🟢 Low", "Rice": "🟢 Low", "Maize": "🟢 Low",
    "Sugarcane": "🟡 Medium", "Potato": "🔴 High", "Tomato": "🔴 High",
    "Onion": "🔴 High", "Mustard": "🟢 Low", "Cotton": "🟡 Medium",
    "Apple": "🔴 High", "Banana": "🔴 High"
}

def crop_profit_recommendation_with_risk(df, crops, market):
    results = []

    for crop in crops:
        try:
            model_path = f"lstm_models/lstm_{crop}_{market}.h5"
            scaler_path = f"scalers/scaler_{crop}_{market}.pkl"

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                continue

            # Load model and scaler
            model = load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)

            # Prepare data
            data = df[(df['Commodity'] == crop) & (df['Market'] == market)][['Arrival_Date', 'Modal_Price']].copy()
            data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], dayfirst=True, errors='coerce')
            data.dropna(inplace=True)
            data.set_index('Arrival_Date', inplace=True)
            data = data.resample("ME").mean().dropna()

            if len(data) < 12:
                continue

            last_12 = data[-12:].values
            last_scaled = scaler.transform(last_12).reshape(1, 12, 1)
            forecast_scaled = model.predict(last_scaled, verbose=0)[0]
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

            avg_price = forecast.mean()
            yield_qtl = YIELD_PER_ACRE.get(crop)
            cost = COST_PER_ACRE.get(crop)
            risk = RISK_FACTORS.get(crop, "🟡 Medium")

            if yield_qtl is None or cost is None:
                continue

            revenue = avg_price * yield_qtl
            profit = revenue - cost

            # Future dates for predictions
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq="ME")
            monthly_prices = {
                date.strftime("%Y-%m-%d"): round(p, 2)
                for date, p in zip(future_dates, forecast)
            }

            results.append({
                "Crop": crop,
                "Avg_Predicted_Price (₹/qtl)": round(avg_price, 2),
                "Net_Profit_per_Acre (₹)": round(profit, 2),
                "Risk_Factor": risk,
                "Monthly_Prices": monthly_prices
            })

        except Exception as e:
            print(f"❌ Error for {crop}: {e}")
            continue

    results = sorted(results, key=lambda x: x["Net_Profit_per_Acre (₹)"], reverse=True)
    return results if results else [{"message": "No valid crop predictions available."}]


if __name__ == "__main__":
    import json

    if len(sys.argv) != 3:
        print(json.dumps({"error": "Expected two arguments: crops list (JSON) and market name"}))
        sys.exit(1)

    try:
        crops = json.loads(sys.argv[1])  # Example: '["Wheat", "Rice", "Potato"]'
        market = sys.argv[2]
        df = pd.read_csv("crop_prices.csv")

        result = crop_profit_recommendation_with_risk(df, crops, market)
        print(json.dumps(result, ensure_ascii=False))  # include emoji support
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
