import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("trend_water_quality.csv")

X = df.drop(columns=["Future_Suitability"]).values
y = df["Future_Suitability"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

sequence_length = 5
num_features = X.shape[1] // sequence_length
X = X.reshape(-1, sequence_length, num_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM Model
model = Sequential([
    LSTM(64, activation="relu", return_sequences=True, input_shape=(sequence_length, num_features)),
    Dropout(0.2),
    LSTM(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model and scaler
model.save("water_suitability_trend_model.h5")
np.save("scaler.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)

print("Model training complete. Saved model as 'water_suitability_trend_model.h5'.")
print("Run 'evaluate_model.py' to test accuracy.")
