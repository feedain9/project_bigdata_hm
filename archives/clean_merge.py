# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the CSV file
# create a list of the files which are in the "sources" folder
files = os.listdir("sources")

for file in files:
    if file.endswith(".csv"):
        try:
            # Load the CSV file
            data = pd.read_csv(f"sources/{file}")

            # Quick exploration of the data
            # print("Dataset Overview:")
            # print(data.head())
            # print(data.info())

            # Rename columns for easier handling
            data.columns = ["phenomenon_time", "temperature", "iot_id", "name", "thing_id"]

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
            excel_path = "sources/Classroom_List_FK10.xlsx"
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
            numeric_columns = merged_data.select_dtypes(include=['number']).columns
            daily_data = merged_data.resample("D", on="phenomenon_time")[numeric_columns].mean()

            # Plot daily temperature trends
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=daily_data, x=daily_data.index, y="temperature")
            plt.title("Daily Temperature Trends")
            plt.xlabel("Date")
            plt.ylabel("Average Temperature (°C)")
            plt.grid(True)
            plt.savefig(f"images/{file.replace('.csv', '')}_daily_temperature_trends.png")  # Save plot for report

            # Plot weekly temperature trends
            weekly_data = daily_data.resample("W").mean()
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=weekly_data, x=weekly_data.index, y="temperature")
            plt.title("Weekly Temperature Trends")
            plt.xlabel("Week")
            plt.ylabel("Average Temperature (°C)")
            plt.grid(True)
            plt.savefig(f"images/{file.replace('.csv', '')}_weekly_temperature_trends.png")  # Save plot for report

            # Plot monthly temperature trends
            monthly_data = daily_data.resample("M").mean()
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=monthly_data, x=monthly_data.index, y="temperature")
            plt.title("Monthly Temperature Trends")
            plt.xlabel("Month")
            plt.ylabel("Average Temperature (°C)")
            plt.grid(True)
            plt.savefig(f"images/{file.replace('.csv', '')}_monthly_temperature_trends.png")  # Save plot for report
            
            # plot the distribution of the temperature values
            plt.figure(figsize=(10, 6))
            sns.histplot(data=merged_data, x="temperature", bins=30, kde=True)
            plt.title("Temperature Distribution")
            plt.xlabel("Temperature (°C)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.savefig(f"images/{file.replace('.csv', '')}_temperature_distribution.png")
            # plt.show()

            # Save the cleaned and merged data to a new CSV file in the 'cleanedsources' folder
            cleaned_file_path = f"cleansources/{file}"
            merged_data.to_csv(cleaned_file_path, index=False)
            # print(f"\nCleaned and merged data saved to: {cleaned_file_path}\n")
        except Exception as e:
            print(f"Error processing file: {file}")
            print(e)
            continue