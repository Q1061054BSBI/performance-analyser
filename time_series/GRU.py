import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Load the data
file_path = './processed_data.csv'
data = pd.read_csv(file_path)

# Convert timestamp to datetime and aggregate by minute
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data = data.resample('1T').mean()

# Remove outliers
def remove_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

for col in ['response_time_ms', 'cpu_usage_avg', 'memory_usage_mb']:
    data = remove_outliers(data, col)

# Normalize data
scaler = MinMaxScaler()
data[['response_time_ms', 'cpu_usage_avg', 'memory_usage_mb']] = scaler.fit_transform(
    data[['response_time_ms', 'cpu_usage_avg', 'memory_usage_mb']]
)

# Create sequences for GRU
def create_sequences(data, target_column, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i + sequence_length][['cpu_usage_avg', 'memory_usage_mb']].values
        target = data.iloc[i + sequence_length][target_column]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

sequence_length = 30  # Last 30 minutes
X, y = create_sequences(data, 'response_time_ms', sequence_length)

# Split data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build GRU model
model = Sequential([
    GRU(64, input_shape=(sequence_length, 2), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predictions
predictions = model.predict(X_test)

# Inverse transform predictions
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
predictions = scaler.inverse_transform(predictions)

# Plot predictions vs actual values
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title('GRU Predictions vs Actual')
plt.show()
