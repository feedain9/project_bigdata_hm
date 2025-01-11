import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = "average_temp_per_room.csv"  # Replace with your dataset
data = pd.read_csv(file_path)

# Feature engineering
data["phenomenon_time"] = pd.to_datetime(data["phenomenon_time"])
data["hour"] = data["phenomenon_time"].dt.hour  # Extract hour of the day
data["day_of_week"] = data["phenomenon_time"].dt.dayofweek  # Day of the week (0=Monday)

# Encode 'Room_x' as categorical
data = pd.get_dummies(data, columns=["Room_x"], drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=["temperature", "phenomenon_time"])  # Explanatory variables
y = data["temperature"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f} °C")

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title("Actual vs Predicted Temperatures")
plt.xlabel("Actual Temperatures (°C)")
plt.ylabel("Predicted Temperatures (°C)")
plt.grid()
plt.show()