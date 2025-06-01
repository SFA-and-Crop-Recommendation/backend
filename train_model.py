# import json
# import os
# from datetime import datetime
# import sys
# import traceback
# import pandas as pd
# import requests
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Input
# from sklearn.preprocessing import MinMaxScaler

# def fetch_and_save_commodity_data(params, commodity_name, filename="commodity_data.csv"):
#     """
#     Fetch commodity data from API, filter last 24 months, and save to CSV.
#     Returns True if successful, False otherwise.
#     """
#     api_key = "579b464db66ec23bdd00000107d05e8dc0f44b2264e86594199d6d7f"
#     url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    
#     query_params = params.copy()
#     query_params.update({
#         "api-key": api_key,
#         "format": "json",
#         "filters[Commodity]": commodity_name,
#         "limit": 100000,
#         "offset": 0
#     })

#     try:
#         response = requests.get(url, params=query_params)
#         response.raise_for_status()
#         data = response.json()

#         if 'records' not in data or len(data['records']) == 0:
#             raise ValueError(f"No records found for {commodity_name} with parameters {params}.")

#         records = data['records']
#         df = pd.DataFrame(records)

#         # Preprocessing
#         df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True, errors='coerce')
#         df = df.dropna(subset=['Arrival_Date'])  # Drop invalid dates
#         df = df.sort_values('Arrival_Date')

#         # Filter last 24 months
#         latest_date = df['Arrival_Date'].max()
#         cutoff_date = latest_date - pd.DateOffset(months=24)
#         df = df[df['Arrival_Date'] >= cutoff_date]

#         df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')
#         df = df.dropna(subset=['Modal_Price'])

#         # Save to CSV
#         df.to_csv(filename, index=False)
#         return True
        
#     except Exception as e:
#         print(f"Error in fetch_and_save_commodity_data: {str(e)}")
#         return False

# def train_and_predict_from_csv(filename="commodity_data.csv"):
#     """
#     Train LSTM model from CSV data and predict prices.
#     Returns model, scaler, and prediction data (dates, actual_prices, predicted_prices)
#     """
#     try:
#         # Load data
#         df = pd.read_csv(filename)
#         df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True, errors='coerce')
#         df = df.sort_values('Arrival_Date')

#         if df.shape[0] < 50:
#             raise ValueError(f"Not enough data to train model. Only {df.shape[0]} records found.")

#         prices = df['Modal_Price'].values.reshape(-1, 1)
#         dates = df['Arrival_Date'].values

#         # Scaling
#         scaler = MinMaxScaler()
#         prices_scaled = scaler.fit_transform(prices)

#         # Create sequences for LSTM
#         sequence_length = 6  # 6 months
#         X, y = [], []
#         for i in range(len(prices_scaled) - sequence_length):
#             X.append(prices_scaled[i:i+sequence_length])
#             y.append(prices_scaled[i+sequence_length])

#         X, y = np.array(X), np.array(y)

#         # Build model
#         model = Sequential([
#             LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
#             Dense(1)
#         ])
#         model.compile(optimizer='adam', loss='mean_squared_error')

#         # Train model
#         model.fit(X, y, epochs=50, batch_size=8, verbose=0)

#         # Predict prices
#         predicted_scaled = model.predict(X, verbose=0)
#         predicted_prices = scaler.inverse_transform(predicted_scaled).flatten()
#         actual_prices = scaler.inverse_transform(y).flatten()

#         return model, scaler, (dates[sequence_length:], actual_prices, predicted_prices)

#     except Exception as e:
#         raise Exception(f"Error in train_and_predict_from_csv: {str(e)}")

# def future_prices(params, commodity_name, filename="commodity_data.csv"):
#     try:
#         # Fetch and save data
#         success = fetch_and_save_commodity_data(params, commodity_name, filename)
#         if not success:
#             raise Exception("Failed to fetch commodity data")
        
#         # Train model and get predictions
#         model, scaler, (dates, actual_prices, predicted_prices) = train_and_predict_from_csv(filename)
        
#         # Prepare response
#         predictions = []
#         for date, actual, predicted in zip(dates, actual_prices, predicted_prices):
#             try:
#                 date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
#                 predictions.append({
#                     "date": date_str,
#                     "actual_price": float(actual),
#                     "predicted_price": float(predicted)
#                 })
#             except Exception as e:
#                 # print(f"Error processing entry: {e}")
#                 continue
        
#         response = {
#             "success": True,
#             "current_data": predictions[-1],
#             "predictions": predictions,
#             "metadata": {
#                 "commodity": commodity_name,
#                 "params": params,
#                 "generated_at": datetime.now().isoformat()
#             }
#         }
        
#         return response
        
#     except Exception as e:
#         error_response = {
#             "success": False,
#             "error": str(e),
#             "traceback": traceback.format_exc()
#         }
#         # print(f"Error in future_prices: {error_response}")
#         return error_response

import json
import pandas as pd
import sys
import numpy as np
from tensorflow.keras.models import load_model
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
            date.strftime("%Y-%m-%d"): round(price, 2)
            for date, price in zip(future_dates, forecast)
        }

        return forecast_dict

    except Exception as e:
        print(f"Error in prediction for {commodity}-{market}: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Expected two arguments: params JSON and commodity name"}))
        sys.exit(1)

    try:
        df = pd.read_csv("crop_prices.csv")
        market = json.loads(sys.argv[1])
        commodity = sys.argv[2]
        result = predict_next_6_months(commodity, market, df)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)