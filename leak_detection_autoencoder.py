import requests, json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

# === Config ===
CHANNEL_ID = '2501725'
READ_API_KEY = 'JMHIQT5X1F9HCM95'
OUTLET_FIELD = 'field2'
FORECAST_PATH = 'leak_forecast.json'
NUM_RESULTS = 2000
SEQUENCE_LENGTH = 60  # seconds history
ANOMALY_THRESHOLD = 0.03  # can be tuned

# === Fetch sensor data ===
def fetch_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={NUM_RESULTS}"
    r = requests.get(url)
    feeds = r.json().get('feeds', [])
    return pd.DataFrame(feeds)

# === Preprocess Data ===
def preprocess_data(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.set_index('created_at', inplace=True)
    df[OUTLET_FIELD] = pd.to_numeric(df[OUTLET_FIELD], errors='coerce')
    outlet_series = df[OUTLET_FIELD].resample('S').mean().interpolate()
    return outlet_series.dropna()

# === Prepare sequences ===
def prepare_data(data, sequence_length=SEQUENCE_LENGTH):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    X = []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length])

    return np.array(X), scaler

# === Autoencoder model ===
def build_autoencoder(sequence_length):
    inputs = Input(shape=(sequence_length, 1))
    encoded = LSTM(64, activation='relu')(inputs)
    repeated = RepeatVector(sequence_length)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(repeated)
    output = TimeDistributed(Dense(1))(decoded)
    model = Model(inputs, output)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

# === Detect anomaly ===
def is_anomalous(model, scaler, sequence, threshold=ANOMALY_THRESHOLD):
    scaled_seq = scaler.transform(sequence.reshape(-1, 1))
    input_seq = np.expand_dims(scaled_seq, axis=0)
    reconstructed = model.predict(input_seq, verbose=0)
    loss = np.mean(np.square(reconstructed - input_seq))
    return bool(loss > threshold), float(loss)

# === Save result ===
def save_forecast(is_anomaly, loss):
    with open(FORECAST_PATH, 'w') as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "is_anomaly": is_anomaly,
            "reconstruction_loss": round(loss, 6),
            "note": "Anomaly = possible leak or unknown water usage"
        }, f, indent=2)

# === Main workflow ===
def main():
    print("📡 Fetching data...")
    df = fetch_data()
    if df.empty:
        print("⚠️ No data!")
        save_forecast(False, 0.0)
        return

    print("🧹 Preprocessing...")
    series = preprocess_data(df)

    if len(series) < SEQUENCE_LENGTH * 2:
        print("⚠️ Not enough data!")
        save_forecast(False, 0.0)
        return

    print("📦 Preparing sequences...")
    X, scaler = prepare_data(series.values)

    print("🧠 Training autoencoder...")
    model = build_autoencoder(SEQUENCE_LENGTH)
    model.fit(X, X, epochs=10, batch_size=16, verbose=0)

    print("🔎 Checking for anomaly in latest pattern...")
    last_sequence = series.values[-SEQUENCE_LENGTH:]
    is_anomaly, loss = is_anomalous(model, scaler, last_sequence)

    print(f"🔍 Anomaly Detected: {is_anomaly}, Loss: {loss}")
    save_forecast(is_anomaly, loss)

    print("✅ Done!")

if __name__ == "__main__":
    main()
