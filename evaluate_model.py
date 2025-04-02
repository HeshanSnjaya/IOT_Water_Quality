import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load data
df = pd.read_csv("trend_water_quality.csv")

X = df.drop(columns=["Future_Suitability"]).values
y = df["Future_Suitability"].values

# Load the saved scaler
scaler = StandardScaler()
scaler.mean_ = np.load("scaler.npy")
scaler.scale_ = np.load("scaler_scale.npy")
X = scaler.transform(X)

sequence_length = 5
num_features = X.shape[1] // sequence_length
X = X.reshape(-1, sequence_length, num_features)

# Split test data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = load_model("water_suitability_trend_model.h5")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save results to a text file
with open("evaluation_results.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
print("Results saved to 'evaluation_results.txt'.")
