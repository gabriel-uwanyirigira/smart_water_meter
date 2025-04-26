import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# === CONFIGURATION ===
THINGSPEAK_CHANNEL_ID = "2501725"  # ✅ updated
THINGSPEAK_API_KEY    = "JMHIQT5X1F9HCM95"  # ✅ updated

# ThingSpeak Fields
INLET_FIELD = "field1"  # ✅ updated
OUTLET_FIELD = "field2"  # ✅ updated

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
    print("Preparing second-based flow data...")
    df = pd.DataFrame(feeds)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)
    
    df[INLET_FIELD] = pd.to_numeric(df[INLET_FIELD], errors='coerce')
    df[OUTLET_FIELD] = pd.to_numeric(df[OUTLET_FIELD], errors='coerce')
    
    df['net_flow'] = df[INLET_FIELD] - df[OUTLET_FIELD]
    
    second_usage = df['net_flow'].resample('S').sum()  # 'S' for second
    print(f"Aggregated {len(second_usage)} second-based records.")
    
    return second_usage

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def train_and_forecast_seconds(second_series):
    print("Training LSTM for second-based forecast...")
    data = second_series.values
    data = data.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    seq_length = 60  # use past 60 seconds for prediction

    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    last_seq = scaled_data[-seq_length:]
    last_seq = last_seq.reshape((1, seq_length, 1))

    prediction_scaled = model.predict(last_seq)
    prediction = scaler.inverse_transform(prediction_scaled)

    return prediction.flatten()[0]

def save_forecast(second_forecast):
    print("Saving forecast to forecast.json...")
    forecast = {
        "second_forecast": second_forecast,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open('forecast.json', 'w') as f:
        json.dump(forecast, f, indent=4)
    print("Forecast saved to forecast.json.")

def main():
    feeds = fetch_data()
    if feeds:
        second_series = preprocess_data(feeds)
        second_series = second_series.dropna()

        if len(second_series) >= 100:
            second_forecast = train_and_forecast_seconds(second_series)
            save_forecast(second_forecast)
        else:
            print("Not enough data to train the model. Skipping training...")
            save_forecast(None)
    else:
        print("No data to process.")

# === INFINITE RETRAINING LOOP EVERY 5 MINUTES ===
if __name__ == "__main__":
    while True:
        start_time = time.time()
        print(f"\n--- New cycle at {datetime.utcnow().isoformat()} ---")
        try:
            main()
        except Exception as e:
            print(f"Error occurred: {e}")

        elapsed_time = time.time() - start_time
        sleep_time = max(0, 300 - elapsed_time)  # 5 minutes minus time taken
        print(f"\nSleeping for {sleep_time:.2f} seconds...\n")
        time.sleep(sleep_time)
