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
NUM_HOURS_FORECAST    = 7
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

# === 2a. Aggregate to hourly usage ===
def aggregate_hourly(df):
    df['net_flow'] = df['inlet_flow'] - df['outlet_flow']
    hourly_usage = df['net_flow'].resample('H').sum() * (1/60)  # to liters/minute
    print(f"Aggregated {len(hourly_usage)} hourly records.")  # Debug: Print number of hourly records
    return hourly_usage.dropna()

# === 2b. Aggregate to minutely usage ===
def aggregate_minutely(df):
    df['net_flow'] = df['inlet_flow'] - df['outlet_flow']
    minutely_usage = df['net_flow'].resample('T').sum() * (1/60)  # to liters/second
    print(f"Aggregated {len(minutely_usage)} minutely records.")  # Debug: Print number of minutely records
    return minutely_usage.dropna()

# === 3. Prepare data for LSTM/GRU ===
def prepare_data(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y

# === 4a. Build LSTM Model for hourly prediction ===
def build_lstm_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 4b. Build GRU Model for minutely prediction ===
def build_gru_model(window_size):
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(64, activation='relu', input_shape=(window_size, 1)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 5a. Train and forecast hourly usage ===
def train_and_forecast_hourly(hourly_series):
    WINDOW_SIZE = 7
    X, y = prepare_data(hourly_series.values, window_size=WINDOW_SIZE)

    split_index = int(0.8 * len(X))
    X_train, y_train = X[:split_index], y[:split_index]
    
    model = build_lstm_model(WINDOW_SIZE)
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)  # Debug: Verbose for training info

    forecast_input = hourly_series.values[-WINDOW_SIZE:]
    forecasted = []
    for _ in range(7):
        input_seq = np.expand_dims(forecast_input[-WINDOW_SIZE:], axis=(0,2))
        next_val = model.predict(input_seq)[0,0]
        forecasted.append(next_val)
        forecast_input = np.append(forecast_input, next_val)

    print("Hourly forecast predictions:", forecasted)  # Debug: Print the forecasted values
    return forecasted

# === 5b. Train and forecast minutely usage ===
def train_and_forecast_minutely(minutely_series):
    WINDOW_SIZE = 60
    X, y = prepare_data(minutely_series.values, window_size=WINDOW_SIZE)

    split_index = int(0.8 * len(X))
    X_train, y_train = X[:split_index], y[:split_index]
    
    model = build_gru_model(WINDOW_SIZE)
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)  # Debug: Verbose for training info

    forecast_input = minutely_series.values[-WINDOW_SIZE:]
    forecasted = []
    for _ in range(60):
        input_seq = np.expand_dims(forecast_input[-WINDOW_SIZE:], axis=(0,2))
        next_val = model.predict(input_seq)[0,0]
        forecasted.append(next_val)
        forecast_input = np.append(forecast_input, next_val)

    print("Minutely forecast predictions:", forecasted)  # Debug: Print the forecasted values
    return forecasted

# === 6. Save forecast into JSON ===
def save_forecast_json(hourly_forecast, minutely_forecast):
    now = datetime.utcnow()
    hourly_forecast_dict = {}
    minutely_forecast_dict = {}

    for i, prediction in enumerate(hourly_forecast):
        forecast_time = now + timedelta(hours=i+1)
        hour_label = forecast_time.strftime("%Y-%m-%d %H:00")
        hourly_forecast_dict[hour_label] = round(float(prediction), 2)

    for i, prediction in enumerate(minutely_forecast):
        forecast_time = now + timedelta(minutes=i+1)
        minute_label = forecast_time.strftime("%Y-%m-%d %H:%M")
        minutely_forecast_dict[minute_label] = round(float(prediction), 2)

    result = {
        "hourly_forecast": hourly_forecast_dict,
        "minutely_forecast": minutely_forecast_dict
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

    print("Aggregating hourly and minutely usage...")
    hourly_series = aggregate_hourly(df)
    minutely_series = aggregate_minutely(df)

    if len(hourly_series) < 1 or len(minutely_series) < 1:
        print("Not enough data points yet. Need at least 30 hours and 60 minutes.")
        return

    print("Training LSTM for hourly forecast...")
    hourly_forecast = train_and_forecast_hourly(hourly_series)

    print("Training GRU for minutely forecast...")
    minutely_forecast = train_and_forecast_minutely(minutely_series)

    print("Saving forecast...")
    save_forecast_json(hourly_forecast, minutely_forecast)

if __name__ == "__main__":
    main()
