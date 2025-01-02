import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Load the data
file_path = './processed_data.csv'  # Specify the path to your file
data = pd.read_csv(file_path)


def plot_correlation_matrix(data):
    # Compute the correlation matrix
    correlation_matrix = data.corr()

    # Display the matrix
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Overall Correlation Heatmap")
    plt.show()

# Function to plot target distribution
def plot_distribution(data, target_column, title):
    plt.figure(figsize=(10, 6))
    data[target_column].hist(bins=30, edgecolor='k', alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel("Number of records", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Function to remove outliers using IQR
def remove_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Data cleaning function
def clean_data(data, target_column):
    # Remove records where level_info = 0 or target = 0
    cleaned_data = data[(data['level_info_'] != 0) & (data[target_column] != 0)]
    return cleaned_data

# Data normalization function
def normalize_data(data, feature_columns):
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
    return data

# Function to prepare the dataset
def prepare_data(target_column, feature_columns, data):
    # Plot distribution before outlier removal
    # plot_distribution(data, target_column, f"Distribution of {target_column} before outlier removal")

    # Clean the data
    cleaned_data = clean_data(data, target_column)

    # Remove outliers
    cleaned_data = remove_outliers(cleaned_data, target_column)

    # Plot distribution after outlier removal
    plot_distribution(cleaned_data, target_column, f"Distribution of {target_column} after outlier removal")
    

    # Normalize the data
    cleaned_data = normalize_data(cleaned_data, feature_columns)

    print(f"Correlation Analysis for Target: {target_column}")
    plot_correlation_matrix(cleaned_data[feature_columns + [target_column]])
    # Split data into features (X) and target variable (y)
    X = cleaned_data[feature_columns]
    y = cleaned_data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate the model
def train_and_evaluate_model(target, features, data, model_name):
    X_train, X_test, y_train, y_test = prepare_data(target, features, data)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print(f"Model: {model_name}")
    print(f"R² Score: {r2:.3f}")
    print("Feature Importance (Coefficients):")
    for feature, coef in zip(features, model.coef_):
        print(f"  {feature}: {coef:.3f}")
    print("-" * 40)


# 1. Total initial rendering time
train_and_evaluate_model(
    target="total_initial_rendering_time",
    features=[
        "primary_CLS", "primary_FCP", "primary_LCP",
        "secondary_CLS", "secondary_FCP", "secondary_LCP",
        "cpu_usage_avg", "memory_usage_mb", "server_response_time"
    ],
    data=data,
    model_name="Total Initial Rendering Time"
)

# 2. Total rerender time
train_and_evaluate_model(
    target="total_rerender_time",
    features=[
        "secondary_CLS", "secondary_FCP", "secondary_LCP",
        "cpu_usage_avg", "memory_usage_mb", "server_response_time"
    ],
    data=data,
    model_name="Total Rerender Time"
)

# 3. Server response time
train_and_evaluate_model(
    target="server_response_time",
    features=["cpu_usage_avg", "memory_usage_mb"],
    data=data,
    model_name="Server Response Time"
)

# 4. Primary LCP
train_and_evaluate_model(
    target="primary_LCP",
    features=["primary_CLS", "primary_FCP"],
    data=data,
    model_name="Primary LCP"
)

# 5. Secondary LCP
train_and_evaluate_model(
    target="secondary_LCP",
    features=["secondary_CLS", "secondary_FCP"],
    data=data,
    model_name="Secondary LCP"
)