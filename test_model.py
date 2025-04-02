import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

model = load_model("water_suitability_trend_model.h5")

# input_data = np.array([
#     [7.0, 800, 22, 80, 5.6, 0.8],  
#     [7.2, 850, 23, 85, 5.7, 0.85],  
#     [7.1, 870, 24, 90, 5.8, 0.87],  
#     [7.3, 890, 25, 95, 5.9, 0.89],  
#     [7.4, 910, 26, 100, 6.0, 0.91]  
# ])

input_data = np.array([
    [7.89, 900.68, 25.50, 76.05, 4.40, 0.90],  # Day 0
    [9.19, 1333.70, 24.54, 114.14, 4.78, 1.33],  # Day 1
    [5.85, 903.55, 23.95, 29.61, 5.02, 0.90],  # Day 2
    [6.28, 589.67, 30.50, 103.90, 2.40, 0.59],  # Day 3
    [5.68, 1138.43, 25.54, 43.09, 4.39, 1.14]   # Day 4
])

scaler = StandardScaler()
trend_dataset = pd.read_csv("trend_water_quality.csv")
X_sample = trend_dataset.drop(columns=["Future_Suitability"]).values
scaler.fit(X_sample)
input_data = scaler.transform(input_data.flatten().reshape(1, -1))

sequence_length = 5
num_features = input_data.shape[1] // sequence_length
input_data = input_data.reshape(1, sequence_length, num_features)

prediction = model.predict(input_data)[0][0]
predicted_label = "Suitable" if prediction > 0.5 else "Unsuitable"

print(f"Predicted Suitability Trend for Next Day: {predicted_label}")
