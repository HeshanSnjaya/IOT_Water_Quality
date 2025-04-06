import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

# Load 5-day sample input
input_df = pd.read_csv("sample_5_days_input.csv")

scaler = joblib.load("scaler.save")

scaled_input = scaler.transform(input_df)

input_seq = np.expand_dims(scaled_input, axis=0) 

# Load trained model
model = load_model("rnn_water_model.h5", custom_objects={'mse': MeanSquaredError()})

# Predict next 5 days
predicted_scaled = model.predict(input_seq)
predicted_scaled = predicted_scaled.reshape(5, input_df.shape[1])

predicted = scaler.inverse_transform(predicted_scaled)

predicted_df = pd.DataFrame(predicted, columns=input_df.columns)
print("Predicted Water Quality for Next 5 Days:")
print(predicted_df)

# Save predictions
predicted_df.to_csv("predicted_next_5_days.csv", index=False)
