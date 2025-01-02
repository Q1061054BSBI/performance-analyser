#Многословный перцептрон
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the data
file_path = './processed_data.csv'
data = pd.read_csv(file_path)

# Data cleaning function
def clean_data(data, target_column):
    return data[(data['level_info_'] != 0) & (data[target_column] != 0)]

# Remove outliers using IQR
def remove_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Normalize data
def normalize_data(data, feature_columns):
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

# Prepare data for modeling
def prepare_data(target_column, feature_columns, data):
    # Clean and preprocess
    cleaned_data = clean_data(data, target_column)
    cleaned_data = remove_outliers(cleaned_data, target_column)
    cleaned_data = normalize_data(cleaned_data, feature_columns)
    
    # Split data into train and test sets
    X = cleaned_data[feature_columns]
    y = cleaned_data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_and_evaluate_model(model, target, features, data, model_name):
    X_train, X_test, y_train, y_test = prepare_data(target, features, data)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    print(f"Model: {model_name}")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    print("-" * 40)

# 1. MLP Model
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

# 2. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Targets and features
tasks = [
    {
        "target": "total_initial_rendering_time",
        "features": [
            "primary_CLS", "primary_FCP", "primary_LCP",
            "secondary_CLS", "secondary_FCP", "secondary_LCP",
            "cpu_usage_avg", "memory_usage_mb", "server_response_time"
        ],
        "model_name": "Total Initial Rendering Time"
    },
    {
        "target": "total_rerender_time",
        "features": [
            "secondary_CLS", "secondary_FCP", "secondary_LCP",
            "cpu_usage_avg", "memory_usage_mb", "server_response_time"
        ],
        "model_name": "Total Rerender Time"
    },
    {
        "target": "server_response_time",
        "features": ["cpu_usage_avg", "memory_usage_mb"],
        "model_name": "Server Response Time"
    },
    {
        "target": "primary_LCP",
        "features": ["primary_CLS", "primary_FCP"],
        "model_name": "Primary LCP"
    },
    {
        "target": "secondary_LCP",
        "features": ["secondary_CLS", "secondary_FCP"],
        "model_name": "Secondary LCP"
    }
]

# Train and evaluate for each task
for task in tasks:
    # Train MLP
    train_and_evaluate_model(
        mlp_model,
        target=task["target"],
        features=task["features"],
        data=data,
        model_name=f"MLP: {task['model_name']}"
    )
    
    # Train Random Forest
    train_and_evaluate_model(
        rf_model,
        target=task["target"],
        features=task["features"],
        data=data,
        model_name=f"Random Forest: {task['model_name']}"
    )