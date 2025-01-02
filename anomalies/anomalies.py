#Isolation Forest
#One-Class SVM
#Autoencoder с малыми параметрами

#Цель:
#Идентифицировать аномальные события, которые негативно влияют на производительность системы, и предоставить рекомендации по устранению их причин.

# Постановка задач для моделей
# Общее описание:
# Модели будут анализировать метрики производительности (например, response_time_ms, cpu_usage_avg, memory_usage_mb) и выявлять отклонения от нормального поведения.
# Обнаруженные аномалии помогут понять, какие аспекты системы требуют оптимизации (например, высокое потребление памяти или резкий рост времени ответа).
# 1.1. Isolation Forest
# Цель:
# Обнаружить аномальные значения метрик, которые сильно отличаются от нормального диапазона.
# Особенности:
# Модель хорошо работает с многомерными данными и не требует нормального распределения признаков.
# Может использоваться для больших объемов данных.
# Пример задачи:

# Выявить события с аномально высоким response_time_ms или cpu_usage_avg, которые сигнализируют о перегрузке системы.
# 1.2. One-Class SVM
# Цель:
# Построить гиперплоскость, разделяющую нормальные и аномальные события, чтобы выделить необычные комбинации метрик.
# Особенности:
# Подходит для задач, где нормальные данные хорошо описываются одним классом.
# Эффективен для высокоразмерных данных.
# Пример задачи:

# Обнаружить комбинации метрик (например, высокая память + низкая загрузка CPU), которые указывают на узкие места в системе.
# 1.3. Autoencoder
# Цель:
# Научиться воспроизводить нормальные данные, чтобы выявлять события, которые плохо реконструируются (аномалии).
# Особенности:
# Может работать с малым числом параметров для экономии ресурсов.
# Идеален для выявления сложных нелинейных аномалий.
# Пример задачи:

# Определить редкие события, где response_time_ms значительно выше нормы при стабильной cpu_usage_avg и memory_usage_mb.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the data
file_path = './processed_data.csv'
data = pd.read_csv(file_path)

# Normalize data
scaler = MinMaxScaler()
data[['response_time_ms', 'cpu_usage_avg', 'memory_usage_mb']] = scaler.fit_transform(
    data[['response_time_ms', 'cpu_usage_avg', 'memory_usage_mb']]
)

# Features for anomaly detection
features = ['response_time_ms', 'cpu_usage_avg', 'memory_usage_mb']
X = data[features]

# Function to plot correlation heatmap
def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# Function to plot anomalies
def plot_anomalies(data, column, anomaly_column, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data.index, data[column], c=data[anomaly_column], cmap='coolwarm', label='Anomalies')
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.legend()
    plt.show()

# 1. Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X)
iso_labels = iso_forest.predict(X)
data['IsolationForest_Anomaly'] = (iso_labels == -1).astype(int)

# Plot Isolation Forest Anomalies
plot_anomalies(data, 'response_time_ms', 'IsolationForest_Anomaly', 'Isolation Forest Anomalies')

# 2. One-Class SVM
svm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
svm_model.fit(X)
svm_labels = svm_model.predict(X)
data['OneClassSVM_Anomaly'] = (svm_labels == -1).astype(int)

# Plot One-Class SVM Anomalies
plot_anomalies(data, 'response_time_ms', 'OneClassSVM_Anomaly', 'One-Class SVM Anomalies')

# 3. Autoencoder
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
autoencoder = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'),
    Dense(X.shape[1], activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

reconstructed = autoencoder.predict(X)
mse = np.mean(np.square(X - reconstructed), axis=1)
threshold = np.percentile(mse, 95)  # Top 5% as anomalies
data['Autoencoder_Anomaly'] = (mse > threshold).astype(int)

# Plot Autoencoder Anomalies
plot_anomalies(data, 'response_time_ms', 'Autoencoder_Anomaly', 'Autoencoder Anomalies')

# Correlation Heatmap
plot_correlation_matrix(data[features])

# Print Summary of Detected Anomalies
print("Isolation Forest Anomalies:", data['IsolationForest_Anomaly'].sum())
print("One-Class SVM Anomalies:", data['OneClassSVM_Anomaly'].sum())
print("Autoencoder Anomalies:", data['Autoencoder_Anomaly'].sum())
