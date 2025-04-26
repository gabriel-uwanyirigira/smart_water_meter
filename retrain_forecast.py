# retrain_forecast.py

import requests, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# === Config ===
CHANNEL_ID = '2501725'  
READ_API_KEY = 'JMHIQT5X1F9HCM95'
INLET_FIELD = 'field1'
OUTLET_FIELD = 'field2'
FORECAST_PATH = 'forecast.json'
NUM_RESULTS = 2000
SEQUENCE_LENGTH = 60  # 60 seconds history

# === Fetch sensor data ===
def fetch_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={NUM_RESULTS}"
    r = requests.get(url)
    feeds = r.json()['feeds']
    return pd.DataFrame(feeds)

# === Preprocess Data ===
def preprocess_data(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)
    df[INLET_FIELD] = pd.to_numeric(df[INLET_FIELD], errors='coerce')
    df[OUTLET_FIELD] = pd.to_numeric(df[OUTLET_FIELD], errors='coerce')
    df['net_flow'] = df[INLET_FIELD] - df[OUTLET_FIELD]
    second_series = df['net_flow'].resample('S').sum()
    return second_series.dropna()

# === Prepare input/output for training ===
def prepare_data(data, sequence_length=SEQUENCE_LENGTH):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length])
        y.append(scaled[i + sequence_length])

    return np.array(X), np.array(y), scaler

# === Train LSTM model ===
def train_model(X, y):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
    return model

# === Forecast next time step ===
def forecast(model, scaler, last_sequence):
    scaled_sequence = scaler.transform(last_sequence.reshape(-1, 1))
    input_seq = np.expand_dims(scaled_sequence, axis=0)  # Shape (1, 60, 1)
    prediction_scaled = model.predict(input_seq)[0]
    return scaler.inverse_transform([prediction_scaled])[0][0]

# === Save forecast.json ===
def save_forecast(second_forecast):
    with open(FORECAST_PATH, 'w') as f:
        json.dump({
            "second_forecast": round(second_forecast, 4),
            "timestamp": datetime.utcnow().isoformat()
        }, f, indent=2)

# === Main ===
def main():
    print("📡 Fetching data...")
    df = fetch_data()
    if df.empty:
        print("⚠️ No data fetched!")
        save_forecast(None)
        return

    print("🧹 Preprocessing data...")
    second_series = preprocess_data(df)

    if len(second_series) < SEQUENCE_LENGTH:
        print("⚠️ Not enough data to train!")
        save_forecast(None)
        return

    print("🧠 Preparing training data...")
    X, y, scaler = prepare_data(second_series.values)

    print("📈 Training model...")
    model = train_model(X, y)

    print("🔮 Forecasting...")
    prediction = forecast(model, scaler, second_series.values[-SEQUENCE_LENGTH:])

    print("💾 Saving forecast...")
    save_forecast(prediction)

    print("✅ Done!")

if __name__ == "__main__":
    main()
