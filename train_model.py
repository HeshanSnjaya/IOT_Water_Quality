import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


df = pd.read_csv("trend_water_quality.csv")

if 'Date' in df.columns:
    df = df.drop(columns=['Date'])

df = df.dropna()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

look_back = 5
look_forward = 5

X, y = [], []
for i in range(len(scaled_data) - look_back - look_forward + 1):
    X.append(scaled_data[i:i+look_back])
    y.append(scaled_data[i+look_back:i+look_back+look_forward].reshape(-1))
X = np.array(X)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential()
model.add(SimpleRNN(64, activation='tanh', input_shape=(look_back, df.shape[1])))
model.add(Dense(look_forward * df.shape[1]))  
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1)

# Evaluate
loss, mae = model.evaluate(X_test, y_test)
predicted = model.predict(X_test)
mse = mean_squared_error(y_test, predicted)
print(f"Model Evaluation:\nMSE: {mse:.4f}, MAE: {mae:.4f}")

# Save model and scaler
model.save("rnn_water_model.h5")
import joblib
joblib.dump(scaler, "scaler.save")
