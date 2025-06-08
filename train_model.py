import json
import pandas as pd
import sys
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import joblib
import os
import warnings

def predict_next_6_months(commodity, market, df):
    """
    Predict next 6 months of Modal Price using pretrained LSTM model.

    Args:
        commodity (str): Commodity name (e.g., 'Apple')
        market (str): Market name (e.g., 'Mechua')
        df (pd.DataFrame): Full historical dataset

    Returns:
        dict: {date (str): forecasted price (float)}
    """
    try:
        model_path = f"lstm_models/lstm_{commodity}_{market}.h5"
        scaler_path = f"scalers/scaler_{commodity}_{market}.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler for {commodity}-{market} not found.")

        # Load model and scaler
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        # Filter and prepare data
        data = df[(df['Commodity'] == commodity) & (df['Market'] == market)][['Arrival_Date', 'Modal_Price']].copy()
        data['Arrival_Date'] = pd.to_datetime(data['Arrival_Date'], dayfirst=True, errors='coerce')
        data.dropna(subset=['Arrival_Date', 'Modal_Price'], inplace=True)
        data.set_index('Arrival_Date', inplace=True)
        data.index = pd.DatetimeIndex(data.index)
        monthly = data.resample("ME").mean().dropna()

        if len(monthly) < 12:
            raise ValueError(f"Not enough historical data for {commodity} - {market}.")

        # Use the most recent 12 months from the latest available data
        last_12 = monthly[-12:].values
        last_12_scaled = scaler.transform(last_12).reshape(1, 12, 1)

        # Predict next 6 months
        forecast_scaled = model.predict(last_12_scaled, verbose=0)[0]
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

        # Use current date as base for future dates
        from pandas.tseries.offsets import MonthEnd
        today = pd.Timestamp.today().replace(day=1) + MonthEnd(0)  # End of this month
        future_dates = pd.date_range(start=today + pd.DateOffset(months=1), periods=6, freq="ME")

        forecast_dict = {
            date.strftime("%Y-%m-%d"): float(round(price, 2))
            for date, price in zip(future_dates, forecast)
        }

        return forecast_dict

    except Exception as e:
        # print(json.dumps({"error":"No forecast available for the given crop and market."}))
        return {"Success": False, "error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error":"Expected two arguments: params JSON and commodity name"}))
        sys.exit(1)

    try:
        df = pd.read_csv("crop_prices.csv")
        market = json.loads(sys.argv[1])
        commodity = json.loads(sys.argv[2])
        result = predict_next_6_months(commodity, market, df)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error":str(e)}))
        sys.exit(1)