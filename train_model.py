import pandas as pd
import requests

def fetch_and_save_commodity_data(params, commodity_name, filename="commodity_data.csv"):
    """
    Fetch commodity data from API, filter last 24 months, and save to CSV.
    """
    api_key = "579b464db66ec23bdd00000107d05e8dc0f44b2264e86594199d6d7f"
    url = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
    
    query_params = params.copy()
    query_params.update({
        "api-key": api_key,
        "format": "json",
        "filters[Commodity]": commodity_name,
        "limit": 100000,
        "offset": 0
    })

    response = requests.get(url, params=query_params)
    data = response.json()

    if 'records' not in data or len(data['records']) == 0:
        raise ValueError(f"No records found for {commodity_name} with parameters {params}.")

    records = data['records']
    df = pd.DataFrame(records)

    # Preprocessing
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Arrival_Date'])  # Drop invalid dates
    df = df.sort_values('Arrival_Date')

    # Filter last 24 months
    latest_date = df['Arrival_Date'].max()
    cutoff_date = latest_date - pd.DateOffset(months=24)
    df = df[df['Arrival_Date'] >= cutoff_date]

    df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')
    df = df.dropna(subset=['Modal_Price'])

    # Save to CSV
    df.to_csv(filename, index=False)
    # print(f"Saved {len(df)} records to {filename}")
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

def train_and_predict_from_csv(filename="commodity_data.csv"):
    """
    Train LSTM model from CSV data and predict price 6 months later.
    """
    df = pd.read_csv(filename)
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Arrival_Date')

    if df.shape[0] < 50:
        raise ValueError(f"Not enough data to train model. Only {df.shape[0]} records found.")

    prices = df['Modal_Price'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    n_steps = 30

    if len(prices_scaled) <= n_steps:
        raise ValueError(f"Not enough sequences for LSTM. Need more than {n_steps} records.")

    X, y = [], []
    for i in range(n_steps, len(prices_scaled)):
        X.append(prices_scaled[i-n_steps:i, 0])
        y.append(prices_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build Model
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))  # <-- using Input layer properly
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    # Predict next 180 days (6 months)
    future_steps = 180
    last_sequence = prices_scaled[-n_steps:]

    predictions = []

    for _ in range(future_steps):
        pred_input = last_sequence.reshape((1, n_steps, 1))
        pred_price = model.predict(pred_input, verbose=0)
        predictions.append(pred_price[0, 0])

        last_sequence = np.append(last_sequence[1:], pred_price[0, 0])

    # Get final price after 180 days
    future_price_scaled = predictions[-1]
    future_price = scaler.inverse_transform(np.array(future_price_scaled).reshape(-1, 1))[0, 0]

    return model, scaler, future_price

def train_and_predict_from_csv_map(filename):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    # Load data
    df = pd.read_csv(filename)
    
    # Assume 'Price' and 'Date' columns exist
    prices = df['Modal_Price'].values.reshape(-1, 1)
    dates = pd.to_datetime(df['Arrival_Date'])

    # Scaling
    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    # Create sequences for LSTM
    X, y = [], []
    sequence_length = 6  # 6 months
    for i in range(len(prices_scaled) - sequence_length):
        X.append(prices_scaled[i:i+sequence_length])
        y.append(prices_scaled[i+sequence_length])

    X, y = np.array(X), np.array(y)

    # Build model
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    # Predict prices
    predicted_scaled = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_scaled).flatten()
    actual_prices = scaler.inverse_transform(y).flatten()

    return model, scaler, (dates[sequence_length:], actual_prices, predicted_prices)
