# we need to create a heatmaps to show the average temperature per room per month

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file "average_temp_per_room.csv", heere are the headers Room_x,phenomenon_time,temperature
data = pd.read_csv("average_temp_per_room.csv")

# Convert 'phenomenon_time' to datetime
data["phenomenon_time"] = pd.to_datetime(data["phenomenon_time"])

# Extract the month from 'phenomenon_time'
data["month"] = data["phenomenon_time"].dt.month

# Extract the year from 'phenomenon_time'
data["year"] = data["phenomenon_time"].dt.year

# Create a pivot table to calculate the average temperature per room per month
pivot_table = data.pivot_table(index="Room_x", columns=["year", "month"], values="temperature", aggfunc="mean")

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Average Temperature per Room per Month")
plt.xlabel("Month")
plt.ylabel("Room")
plt.savefig("average_temp_per_room_heatmap.png")
plt.show()