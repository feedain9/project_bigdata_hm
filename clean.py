# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load the CSV file efficiently (only load required columns and parse dates)
# Adjust 'file_path' to match your dataset
file_path = "sources/140.csv"
df = pd.read_csv(file_path, parse_dates=['phenomenonTime'])

# Display the first few rows to understand the structure
print(df.head())

# Rename columns for easier access
df.columns = ['Timestamp', 'Temperature', 'SensorID', 'Name', 'RoomID']

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Summary statistics to identify outliers
print("Summary statistics:")
print(df.describe())


# Check the distribution of temperature values
print("Distribution of temperature values:", df['Temperature'].describe())

# Show the highest temperature values
print("Highest temperature values:", df[df['Temperature'] > df['Temperature'].quantile(0.99)])

# Show the lowest temperature values
print("Lowest temperature values:", df[df['Temperature'] < df['Temperature'].quantile(0.01)])

# Plot the temperature distribution to identify potential outliers
plt.figure(figsize=(10, 5))
plt.hist(df['Temperature'], bins=50, color='blue', alpha=0.7)
plt.title("Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.show()

# Handling and cleaning outliers
# Remove outliers based on the (knowing that the sensor can measure temperatures between 0 and 50 degrees Celsius)
df_cleaned = df[(df['Temperature'] >= 0) & (df['Temperature'] <= 50)]

# Check the distribution of temperature values after cleaning
print("Distribution of temperature values after cleaning:", df_cleaned['Temperature'].describe())

# Visualize the cleaned temperature data
plt.figure(figsize=(10, 5))
plt.hist(df_cleaned['Temperature'], bins=50, color='green', alpha=0.7)
plt.title("Temperature Distribution (Cleaned)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.show()

# Resample the data to daily frequency
df_cleaned['Date'] = df_cleaned['Timestamp'].dt.date
df_daily = df_cleaned.groupby('Date').agg({'Temperature': 'mean'}).reset_index()

# Display the daily average temperature data
print("Daily Average Temperature:", df_daily.head())

# export the cleaned data to a new CSV file
# Adjust 'output_file_path' to the desired location
output_file_path = f"cleaned_data_{datetime.now().timestamp()}.csv"
# df_daily.to_csv(output_file_path, index=False)
df_cleaned.to_csv(output_file_path, index=False)