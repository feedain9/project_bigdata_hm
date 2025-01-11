import pandas as pd
import os
from datetime import datetime

# Define paths and variables
data_folder = 'sources'  # Replace with the folder where your raw CSV files are located
clean_folder = 'clean'  # Folder where cleaned files will be stored
sensor_info = {
    "Room Temperature": (0, 50),  # Temperature in °C
    "CO2": (0, 10000),  # CO2 in ppm
    "Humidity": (0, 85),  # Humidity in % RH
    "Light": (4, 2000),  # Illuminance in LUX
    "Motion": (0, 256),  # Movements per interval
    # VDD is undefined; will ignore it
}
weekend_days = [5, 6]  # Saturday (5) and Sunday (6)

# Load all CSV files into a single DataFrame
def load_data(folder):
    """
    Load all CSV files from the given folder and combine them into a single DataFrame.
    """
    all_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]
    dataframes = []
    for file in all_files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)  # Add a column for source file ID
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

# Clean the dataset and calculate outliers
def clean_data(df):
    """
    Clean the dataset:
    - Parse phenomenonTime into datetime format.
    - Remove rows with invalid results based on sensor bounds.
    - Exclude weekend data.
    - Count outliers for each sensor type.
    """
    # Parse datetime
    df['phenomenonTime'] = pd.to_datetime(df['phenomenonTime'], errors='coerce')

    # Initialize storage for cleaned data and outlier counts
    cleaned_data = pd.DataFrame()
    outlier_counts = {}

    # Filter data based on sensor bounds
    for sensor, (lower, upper) in sensor_info.items():
        sensor_df = df[df['name'] == sensor]
        outliers = sensor_df[(sensor_df['result'] < lower) | (sensor_df['result'] > upper)]
        outlier_counts[sensor] = len(outliers)
        valid_data = sensor_df[(sensor_df['result'] >= lower) & (sensor_df['result'] <= upper)]
        cleaned_data = pd.concat([cleaned_data, valid_data])

    # Exclude weekend data
    cleaned_data['day_of_week'] = cleaned_data['phenomenonTime'].dt.dayofweek
    cleaned_data = cleaned_data[~cleaned_data['day_of_week'].isin(weekend_days)]

    return cleaned_data.reset_index(drop=True), outlier_counts

# Export cleaned data to separate CSV files
def export_cleaned_data(cleaned_df, output_folder):
    """
    Export cleaned data to separate CSV files based on source_file.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    for source_file in cleaned_df['source_file'].unique():
        file_data = cleaned_df[cleaned_df['source_file'] == source_file]
        output_path = os.path.join(output_folder, source_file)
        file_data.to_csv(output_path, index=False)

# Main pipeline
if __name__ == "__main__":
    # Step 1: Load data
    print("Loading data...")
    raw_data = load_data(data_folder)
    
    # Step 2: Clean data
    print("Cleaning data...")
    cleaned_data, outliers = clean_data(raw_data)
    
    # Step 3: Export cleaned data
    print(f"Exporting cleaned data to '{clean_folder}'...")
    export_cleaned_data(cleaned_data, clean_folder)
    
    # Step 4: Display outlier counts
    print("\nOutlier counts by sensor type:")
    for sensor, count in outliers.items():
        print(f"{sensor}: {count}")
