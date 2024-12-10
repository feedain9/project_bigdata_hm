# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Load the CSV file
# create a list of the files which are in the "sources" folder
files = os.listdir("cleansources")
for file in files:
    if file.endswith(".csv"):
        print(f"Processing file: {file}")
        try:
            # Load the CSV file
            data = pd.read_csv(f"cleansources/{file}")

            # find anomalies in the temperatures
            # Calculate the z-score for each temperature value
            data["z_score"] = np.abs((data["temperature"] - data["temperature"].mean()) / data["temperature"].std())

            anomalies = data[data["z_score"] > 3]

            print(f"Anomalies in {file}:", len(anomalies))

            # save anomalies to a new CSV file
            anomalies.to_csv(f"anomalies/anomalies_{file}", index=False)

            # Plot temperature over time
            plt.figure(figsize=(12, 6))
            plt.plot(data["phenomenon_time"], data["temperature"], label="Temperature")

            # Highlight anomalies
            plt.scatter(anomalies["phenomenon_time"], anomalies["temperature"], color="red", label="Anomalies")

            plt.title(f"Temperature Anomalies in {file.replace('.csv', '')}")
            plt.xlabel("Time")
            plt.ylabel("Temperature (°C)")
            plt.legend()
            plt.xticks(rotation=45)
            # plt.tight_layout()
            plt.savefig(f"images/anomalies_{file.replace('.csv', '')}.png")
            plt.show()

            print("Anomalies detected and saved successfully.\n")

        except Exception as e:
            print(f"Error processing file: {file}")
            print(e)
            continue