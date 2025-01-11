# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file "average_temp_per_room.csv" into a DataFrame
data = pd.read_csv("average_temp_per_room.csv")

# Quick exploration of the data
print("Dataset Overview:")
print(data.head())

# Rename columns for easier handling
data.columns = ["Room", "phenomenon_time", "Average Temperature"]

# Convert 'phenomenon_time' to datetime
data["phenomenon_time"] = pd.to_datetime(data["phenomenon_time"], errors='coerce')

# calculate the average temperature per room
average_temp_per_room = data.groupby("Room")["Average Temperature"].mean().reset_index()

# export the DataFrame to a Excel file
average_temp_per_room.to_excel("avg_temp_per_room.xlsx", index=False)

# Plot the average temperature per room
plt.figure(figsize=(12, 6))
sns.barplot(x="Room", y="Average Temperature", data=average_temp_per_room)
plt.title("Average Temperature per Room")
plt.xlabel("Room")
plt.ylabel("Average Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# plot the cloud of points 