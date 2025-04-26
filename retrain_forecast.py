import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from datetime import datetime, timedelta

# === CONFIGURATION ===
THINGSPEAK_CHANNEL_ID = "2501725"
THINGSPEAK_API_KEY    = "JMHIQT5X1F9HCM95"
INLET_FIELD           = "field1"  # Inlet flow rate field
OUTLET_FIELD          = "field2"  # Outlet flow rate field
NUM_DAYS_FORECAST     = 7
FORECAST_JSON_FILE    = "forecast.json"

# === 1. Fetch latest data from ThingSpeak ===
def fetch_thingspeak_data(channel_id, api_key, inlet_field, outlet_field):
    url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results=10000"
    response = requests.get(url)
    data = response.json()
    
    feeds = data['feeds']
    records = []
    for entry in feeds:
        timestamp = entry['created_at']
        inlet = entry.get(inlet_field)
        outlet = entry.get(outlet_field)
        if inlet is not None and outlet is not None:
            try:
                inlet = float(inlet)
                outlet = float(outlet)
                records.append({"timestamp": timestamp, "inlet_flow": inlet, "outlet_flow": outlet})
            except:
                continue
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    print(f"Fetched {len(df)} records from ThingSpeak.")  # Debug: Print number of records fetched
    return df

# === 2a. Aggregate to daily usage ===
def aggregate_daily(df):
    df['net_flow'] = df['inlet_flow'] - df['outlet_flow']
    daily_usage = df['net_flow'].resample('D').sum() * (1/60)  # to liters/hour
    print(f"Aggregated {len(daily_usage)} daily records.")  # Debug: Print number of daily records
    return daily_usage.dropna()

# === 2b. Aggregate to hourly usage ===
def aggregate_hourly(df):
    df['net_flow'] = df['inlet_flow'] - df['outlet_flow']
    hourly_usage = df['net_flow'].resample('H').sum() * (1/60)  # to liters/hour
    print(f"Aggregated {len(hourly_usage)} hourly records.")  # Debug: Print number of hourly records
    return hourly_usage.dropna()

# === 3. Prepare data for LSTM/GRU ===
def prepare_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y

# === 4a. Build LSTM Model for daily prediction ===
def build_lstm_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 4b. Build GRU Model for hourly prediction ===
def build_gru_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 5a. Train and forecast daily usage ===
def train_and_forecast_daily(daily_series):
    WINDOW_SIZE = 7
    X, y = prepare_data(daily_series.values, window_size=WINDOW_SIZE)

    split_index = int(0.8 * len(X))
    X_train, y_train = X[:split_index], y[:split_index]
    
    model = build_lstm_model(WINDOW_SIZE)
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)  # Debug: Verbose for training info

    forecast_input = daily_series.values[-WINDOW_SIZE:]
    forecasted = []
    for _ in range(7):
        input_seq = np.expand_dims(forecast_input[-WINDOW_SIZE:], axis=(0,2))
        next_val = model.predict(input_seq)[0,0]
        forecasted.append(next_val)
        forecast_input = np.append(forecast_input, next_val)

    print("Daily forecast predictions:", forecasted)  # Debug: Print the forecasted values
    return forecasted

# === 5b. Train and forecast hourly usage ===
def train_and_forecast_hourly(hourly_series):
    WINDOW_SIZE = 24
    X, y = prepare_data(hourly_series.values, window_size=WINDOW_SIZE)

    split_index = int(0.8 * len(X))
    X_train, y_train = X[:split_index], y[:split_index]
    
    model = build_gru_model(WINDOW_SIZE)
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)  # Debug: Verbose for training info

    forecast_input = hourly_series.values[-WINDOW_SIZE:]
    forecasted = []
    for _ in range(24):
        input_seq = np.expand_dims(forecast_input[-WINDOW_SIZE:], axis=(0,2))
        next_val = model.predict(input_seq)[0,0]
        forecasted.append(next_val)
        forecast_input = np.append(forecast_input, next_val)

    print("Hourly forecast predictions:", forecasted)  # Debug: Print the forecasted values
    return forecasted

# === 6. Save forecast into JSON ===
def save_forecast_json(daily_forecast, hourly_forecast):
    today = datetime.utcnow().date()
    daily_forecast_dict = {}
    hourly_forecast_dict = {}

    for i, prediction in enumerate(daily_forecast):
        forecast_date = today + timedelta(days=i+1)
        daily_forecast_dict[str(forecast_date)] = round(float(prediction), 2)

    for i, prediction in enumerate(hourly_forecast):
        hour_label = f"{i:02d}:00"
        hourly_forecast_dict[hour_label] = round(float(prediction), 2)

    result = {
        "daily_forecast": daily_forecast_dict,
        "hourly_forecast": hourly_forecast_dict
    }

    with open(FORECAST_JSON_FILE, "w") as f:
        json.dump(result, f, indent=4)
    
    print(f"Saved forecast to {FORECAST_JSON_FILE}")
    # Check if file is being saved properly
    with open(FORECAST_JSON_FILE, "r") as f:
        print("Updated forecast.json:", json.load(f))  # Print the file contents to check if it's updated

# === MAIN FUNCTION ===
def main():
    print("Fetching data from ThingSpeak...")
    df = fetch_thingspeak_data(THINGSPEAK_CHANNEL_ID, THINGSPEAK_API_KEY, INLET_FIELD, OUTLET_FIELD)

    print("Aggregating daily and hourly usage...")
    daily_series = aggregate_daily(df)
    hourly_series = aggregate_hourly(df)

    if len(daily_series) < 30 or len(hourly_series) < 48:
        print("Not enough data points yet. Need at least 30 days and 48 hours.")
        return

    print("Training LSTM for daily forecast...")
    daily_forecast = train_and_forecast_daily(daily_series)

    print("Training GRU for hourly forecast...")
    hourly_forecast = train_and_forecast_hourly(hourly_series)

    print("Saving forecast...")
    save_forecast_json(daily_forecast, hourly_forecast)

if __name__ == "__main__":
    main()
