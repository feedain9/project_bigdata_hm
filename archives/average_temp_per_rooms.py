# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# create a list of the files which are in the "sources" folder and merge them into one DataFrame
files = os.listdir("cleansources")

# Create an empty list to store the DataFrames
dfs = []

for file in files:
    if file.endswith(".csv"):
        try:
            # Load the CSV file
            data = pd.read_csv(f"cleansources/{file}")

            # Quick exploration of the data
            # print("Dataset Overview:")
            # print(data.head())
            # print(data.info())

            # Rename columns for easier handling
            data.columns = ["phenomenon_time", "temperature", "iot_id", "name", "thing_id", "FK", "Room", "Sensor_ID", "Thing ID", "Room Volume", "Comments"]

            # Convert 'phenomenon_time' to datetime
            data["phenomenon_time"] = pd.to_datetime(data["phenomenon_time"], errors='coerce')

            # Check for missing values
            missing_values = data.isnull().sum()
            print("\nMissing Values:")
            print(missing_values)

            # Detecting and removing outliers (e.g., temperatures outside the 0-50°C range)
            valid_temperature_range = (data["temperature"] >= 0) & (data["temperature"] <= 50)
            data_cleaned = data[valid_temperature_range]

            # Show the number of removed outliers
            outliers_count = len(data) - len(data_cleaned)
            print("Thing ID:", file.replace(".csv", ""))
            print("Intial data size: ", len(data))
            print("Cleaned data size: ", len(data_cleaned))
            print(f"Outliers Removed: {outliers_count}\n\n")

            # Load room details from Excel file
            excel_path = "cleansources/Classroom_List_FK10.xlsx"
            room_details = pd.read_excel(excel_path)

            # Merge sensor data with room details
            merged_data = data_cleaned.merge(room_details, left_on="thing_id", right_on="Thing ID", how="left")

            # Convert 'phenomenon_time' to datetime if not already done
            merged_data["phenomenon_time"] = pd.to_datetime(merged_data["phenomenon_time"])

            # Ensure 'temperature' is numeric
            merged_data["temperature"] = pd.to_numeric(merged_data["temperature"], errors='coerce')

            # Drop rows with NaN values in 'temperature'
            merged_data = merged_data.dropna(subset=["temperature"])

            # Select only numeric columns for resampling
            dfs.append(merged_data)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Merge all DataFrames in the list
all_data = pd.concat(dfs)

print(all_data.head())

# Resample the data to get the average temperature per room per hour
average_temp_per_room = all_data.groupby(["Room_x", pd.Grouper(key="phenomenon_time", freq="H")])["temperature"].mean().reset_index()

# save the data to a CSV file
average_temp_per_room.to_csv("average_temp_per_room.csv", index=False)

# Plot the average temperature per room
plt.figure(figsize=(12, 6))
sns.lineplot(data=average_temp_per_room, x="phenomenon_time", y="temperature", hue="Room_x")
plt.title("Average Temperature per Room per Hour")
plt.xlabel("Date")
plt.ylabel("Average Temperature (°C)")
plt.legend(title="Room")

# Save the plot as an image
plt.savefig("average_temp_per_room.png")

# Show the plot
plt.show()