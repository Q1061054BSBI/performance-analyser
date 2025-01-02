import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

# Load data
file_path = './processed_data.csv'
data = pd.read_csv(file_path)

# Convert timestamp to datetime and aggregate by minute
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data = data.resample('1T').mean()

# Focus on the response_time_ms column
response_time = data['response_time_ms'].dropna()

# Check stationarity
from statsmodels.tsa.stattools import adfuller
adf_test = adfuller(response_time)
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")

# Differencing if not stationary
if adf_test[1] > 0.05:
    response_time = response_time.diff().dropna()

# Autocorrelation plot
autocorrelation_plot(response_time)
plt.show()

# Fit ARIMA model
model = ARIMA(response_time, order=(5, 1, 0))  # p, d, q
fitted_model = model.fit()

# Forecast
forecast = fitted_model.forecast(steps=10)
print(f"Forecast: {forecast}")

# Plot forecast vs actual
plt.figure(figsize=(10, 6))
plt.plot(response_time[-50:], label='Actual')
plt.plot(range(len(response_time), len(response_time) + len(forecast)), forecast, label='Forecast')
plt.legend()
plt.title('ARIMA Forecast vs Actual')
plt.show()
