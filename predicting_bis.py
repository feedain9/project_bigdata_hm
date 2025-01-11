import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to data
clean_combined_csv = 'clean/combined_clean_data.csv'  # Combined cleaned data
classroom_file = 'sources/Classroom_list_FK10.xlsx'  # Excel file with room and sensor mapping

# Load data
def load_data(clean_file, classroom_file):
    """
    Load the cleaned temperature data and the classroom mapping file.
    """
    # Load combined cleaned temperature data
    clean_data = pd.read_csv(clean_file)
    clean_data = clean_data[clean_data['name'] == 'Room Temperature']
    
    # Load classroom mapping
    classroom_data = pd.read_excel(classroom_file)
    
    return clean_data, classroom_data

# Handle missing values
def preprocess_data(data):
    """
    Preprocess data by handling missing values.
    """
    # Drop rows with missing phenomenonTime
    data = data.dropna(subset=['phenomenonTime'])
    
    # Convert phenomenonTime to datetime and extract features
    data['phenomenonTime'] = pd.to_datetime(data['phenomenonTime'], errors='coerce')
    data['hour'] = data['phenomenonTime'].dt.hour
    data['day'] = data['phenomenonTime'].dt.day
    data['month'] = data['phenomenonTime'].dt.month

    # Fill missing values in features with median values
    data['hour'] = data['hour'].fillna(data['hour'].median())
    data['day'] = data['day'].fillna(data['day'].median())
    data['month'] = data['month'].fillna(data['month'].median())
    
    return data

# Train Random Forest model by room
def train_random_forest(clean_data, classroom_data):
    """
    Train a Random Forest model for predicting temperature by room.
    """
    # Preprocess clean_data
    clean_data = preprocess_data(clean_data)

    # Merge with classroom data
    merged_data = pd.merge(
        clean_data,
        classroom_data,
        left_on='thing_id',
        right_on='Thing ID',
        how='inner'
    )

    # Loop through each room and train a model
    room_models = {}
    for room in merged_data['Room'].unique():
        print(f"Training model for Room: {room}")
        
        # Filter data for the room
        room_data = merged_data[merged_data['Room'] == room]
        
        # Define features and target
        X = room_data[['hour', 'day', 'month', 'Room Volume']]
        y = room_data['result']
        
        # Drop rows with any remaining NaN values
        X = X.dropna()
        y = y[X.index]  # Ensure target matches the filtered features
        
        # Check if there are enough samples to train
        if len(X) < 10:
            print(f"Not enough data for Room: {room}. Skipping...")
            continue
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Room: {room} | RMSE: {rmse:.2f} | R²: {r2:.2f}")
        
        # Save the model
        room_models[room] = model
    
    return room_models

# Visualize predictions vs actual values
def plot_predictions(y_test, y_pred, room):
    """
    Plot the predicted vs actual values for a specific room.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Diagonal line
    plt.title(f"Predictions vs Actual Values for Room: {room}")
    plt.xlabel("Actual Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'predictions_vs_actual_{room}.png')
    # plt.show()

# Update the train_random_forest function to include visualizations
def train_random_forest_with_visualizations(clean_data, classroom_data):
    """
    Train a Random Forest model for predicting temperature by room and include visualizations.
    """
    clean_data = preprocess_data(clean_data)
    merged_data = pd.merge(
        clean_data,
        classroom_data,
        left_on='thing_id',
        right_on='Thing ID',
        how='inner'
    )

    room_models = {}
    for room in merged_data['Room'].unique():
        print(f"Training model for Room: {room}")
        room_data = merged_data[merged_data['Room'] == room]
        X = room_data[['Room Volume']]
        y = room_data['result']
        X = X.dropna()
        y = y[X.index]

        if len(X) < 10:
            print(f"Not enough data for Room: {room}. Skipping...")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Room: {room} | RMSE: {rmse:.2f} | R²: {r2:.2f}")

        # Save the model
        room_models[room] = model

        # Plot predictions vs actual values
        plot_predictions(y_test, y_pred, room)

    return room_models

# Visualize feature importance
def plot_feature_importance(model, features, room):
    """
    Plot the feature importance for a specific room's Random Forest model.
    """
    importance = model.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance, y=features, palette="viridis")
    plt.title(f"Feature Importance for Room: {room}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'feature_importance_{room}.png')
    # plt.show()

# Update train_random_forest_with_visualizations function to include feature importance plots
def train_random_forest_with_visualizations_and_feature_importance(clean_data, classroom_data):
    """
    Train a Random Forest model for predicting temperature by room, 
    include visualizations for predictions vs actual values and feature importance.
    """
    clean_data = preprocess_data(clean_data)
    merged_data = pd.merge(
        clean_data,
        classroom_data,
        left_on='thing_id',
        right_on='Thing ID',
        how='inner'
    )

    room_models = {}
    for room in merged_data['Room'].unique():
        print(f"Training model for Room: {room}")
        room_data = merged_data[merged_data['Room'] == room]
        X = room_data[['hour', 'day', 'month', 'Room Volume', 'name']]
        y = room_data['result']
        X = X.dropna()
        y = y[X.index]

        if len(X) < 10:
            print(f"Not enough data for Room: {room}. Skipping...")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Room: {room} | RMSE: {rmse:.2f} | R²: {r2:.2f}")

        # Save the model
        room_models[room] = model

        # Plot predictions vs actual values
        plot_predictions(y_test, y_pred, room)

        # Plot feature importance
        plot_feature_importance(model, X.columns, room)

    return room_models


# Main execution
if __name__ == "__main__":
    # Load data
    clean_data, classroom_data = load_data(clean_combined_csv, classroom_file)

    # Train Random Forest models
    # room_models = train_random_forest_with_visualizations(clean_data, classroom_data)
    room_models = train_random_forest_with_visualizations_and_feature_importance(clean_data, classroom_data)