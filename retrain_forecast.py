import pandas as pd
import numpy as np
import requests
import json
from tensorflow import keras

# Config
THINGSPEAK_CHANNEL_ID = '2501725'
THINGSPEAK_API_KEY = 'JMHIQT5X1F9HCM95'
inlet_field = 'field1'
outlet_field = 'field2'
THINGSPEAK_URL = f'https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json?api_key={THINGSPEAK_API_KEY}&results=100'

def fetch_thingspeak_data():
    print("Fetching data from ThingSpeak...")
    response = requests.get(THINGSPEAK_URL)
    data = response.json()
    feeds = data['feeds']
    
    # Create DataFrame
    df = pd.DataFrame(feeds)
    
    if 'created_at' not in df or inlet_field not in df or outlet_field not in df:
        raise KeyError("Expected fields missing in ThingSpeak data.")

    df['timestamp'] = pd.to_datetime(df['created_at'])
    df['inlet'] = pd.to_numeric(df[inlet_field], errors='coerce')
    df['outlet'] = pd.to_numeric(df[outlet_field], errors='coerce')

    df = df[['timestamp', 'inlet', 'outlet']].dropna()

    return df

def train_forecast_model(df):
    print("Training forecast model...")
    # Simple model: predict next "inlet" based on previous ones
    X = []
    y = []
    WINDOW_SIZE = 5
    
    values = df['inlet'].values
    for i in range(len(values) - WINDOW_SIZE):
        X.append(values[i:i+WINDOW_SIZE])
        y.append(values[i+WINDOW_SIZE])
    
    X = np.array(X)
    y = np.array(y)

    # Define a simple model
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(WINDOW_SIZE,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    return model

def predict_future_usage(model, recent_data):
    print("Predicting future water usage...")
    X_input = np.array(recent_data[-5:]).reshape(1, -1)
    prediction = model.predict(X_input)
    return float(prediction[0][0])

def save_forecast(predicted_value):
    forecast = {
        "predicted_water_usage": predicted_value
    }
    with open('forecast.json', 'w') as f:
        json.dump(forecast, f, indent=4)
    print("Forecast saved to forecast.json")

def main():
    df = fetch_thingspeak_data()
    model = train_forecast_model(df)
    predicted_usage = predict_future_usage(model, df['inlet'].values)
    save_forecast(predicted_usage)

if __name__ == "__main__":
    main()
