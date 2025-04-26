import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# === CONFIGURATION ===
THINGSPEAK_CHANNEL_ID = "2501725"   # ✅ your channel ID
THINGSPEAK_API_KEY    = "JMHIQT5X1F9HCM95"  # ✅ your read API key

# ThingSpeak Fields
INLET_FIELD = "field1"
OUTLET_FIELD = "field2"

def fetch_data():
    print("Fetching data from ThingSpeak...")
    url = f'https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}&results=8000'
    response = requests.get(url)
    if response.status_code == 200:
        feeds = response.json()['feeds']
        print(f"Fetched {len(feeds)} records from ThingSpeak.")
        return feeds
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return []

def preprocess_data(feeds):
    print("Preprocessing data...")
    df = pd.DataFrame(feeds)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)

    df[INLET_FIELD] = pd.to_numeric(df[INLET_FIELD], errors='coerce')
    df[OUTLET_FIELD] = pd.to_numeric(df[OUTLET_FIELD], errors='coerce')

    df['net_flow'] = df[INLET_FIELD] - df[OUTLET_FIELD]

    hourly_usage = df['net_flow'].resample('h').sum() * (1/60)   # liters/min
    minutely_usage = df['net_flow'].resample('min').sum() * (1/60)  # liters/sec

    print(f"Aggregated {len(hourly_usage)} hourly records.")
    print(f"Aggregated {len(minutely_usage)} minutely records.")

    return hourly_usage.dropna(), minutely_usage.dropna()

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_and_forecast(series, seq_length):
    data = series.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=50, batch_size=8, verbose=0)

    last_seq = scaled_data[-seq_length:]
    last_seq = last_seq.reshape((1, seq_length, 1))

    prediction_scaled = model.predict(last_seq)
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction.flatten()[0]

def save_forecasts(hourly_forecast, minutely_forecast):
    print("Saving forecasts...")
    forecast = {
        "hourly_forecast": hourly_forecast,
        "minutely_forecast": minutely_forecast,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open('forecast.json', 'w') as f:
        json.dump(forecast, f, indent=4)
    print("Forecasts saved to forecast.json")

def main():
    feeds = fetch_data()
    if feeds:
        hourly_series, minutely_series = preprocess_data(feeds)

        if len(hourly_series) >= 30 and len(minutely_series) >= 100:
            print("Training models...")
            hourly_forecast = train_and_forecast(hourly_series, seq_length=24)
            minutely_forecast = train_and_forecast(minutely_series, seq_length=60)
            save_forecasts(hourly_forecast, minutely_forecast)
        else:
            print("Not enough data yet. Skipping training.")
            save_forecasts(None, None)
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()
