import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

# Load dataset
df = pd.read_csv("trend_water_quality.csv")

# Drop Date column if exists
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])

# Drop NaNs if any
df = df.dropna()

# Load scaler
scaler = joblib.load("scaler.save")

# Normalize
scaled_data = scaler.transform(df)

# Parameters (same as training)
look_back = 5
look_forward = 5

# Use the latest look_back data
input_seq = scaled_data[-look_back:]
input_seq = np.expand_dims(input_seq, axis=0)  # Shape (1, look_back, num_features)

# Load model
model = load_model("rnn_water_model.h5", custom_objects={'mse': MeanSquaredError()})

# Predict
predicted_scaled = model.predict(input_seq)

# Reshape and inverse transform
predicted_scaled = predicted_scaled.reshape(look_forward, df.shape[1])
predicted = scaler.inverse_transform(predicted_scaled)

# Display predictions
predicted_df = pd.DataFrame(predicted, columns=df.columns)
print("\n📈 Predicted Water Quality for Next 5 Days:")
print(predicted_df)

# Optionally save predictions
predicted_df.to_csv("predicted_water_quality_next_5_days.csv", index=False)